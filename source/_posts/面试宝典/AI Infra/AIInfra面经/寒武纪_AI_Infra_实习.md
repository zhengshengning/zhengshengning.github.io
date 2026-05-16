---
title: '寒武纪 AI Infra 实习'
date: 2026-04-16 12:19:48
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: GEMM中为什么一个线程算8x8的块？

在高性能 GEMM kernel 中，每个线程负责计算输出矩阵的一个 8x8（或类似大小如 4x8、8x4）的子块，这是多个硬件约束和性能目标平衡的结果：

**1. 最大化数据复用（算术强度）**：
- 每个线程在 K 维度循环中，每次加载 A 的 8 个元素和 B 的 8 个元素（共 16 次 load）
- 但执行 8*8=64 次 FMA（乘加）运算
- 计算/访存比 = 64/16 = 4，远高于 1x1 块的比值 1/2
- 更高的算术强度意味着更不容易受访存带宽限制

**2. 寄存器利用最优化**：
- 8x8 = 64 个 FP32 累加器需要 64 个寄存器
- 加上 A 的 8 个元素 + B 的 8 个元素 + 循环控制等 ≈ 80-100 个寄存器
- GPU 每线程最多 255 个寄存器（Ampere/Hopper），80-100 个是合理的使用量
- 若增加到 16x16=256 个累加器，可能导致 register spill 到 Local Memory（性能剧降 ~100x）
- 若减少到 4x4=16 个累加器，数据复用率下降，性能不够

**3. 控制线程总数和调度开销**：
- 输出矩阵 M*N 若每线程算 1 个元素需要 M*N 个线程
- 每线程算 8*8=64 个元素只需 M*N/64 个线程
- 减少线程数 -> 减少 warp 调度开销、减少 shared memory 上的同步压力
- 同时保证足够的线程数（occupancy）隐藏延迟

**4. 匹配硬件计算单元**：
- NVIDIA Tensor Core 每次执行 16x8x8（Ampere）或 16x8x16（Hopper）的矩阵乘加
- 每个线程的 8x8 tile 配合 warp-level 的协作，恰好填满 Tensor Core 的输入尺寸
- 如果 tile 太小（如 2x2），无法有效利用 Tensor Core 的宽操作

**5. 与存储层次的匹配**：
- Block-level tile（如 128x128）存 Shared Memory（A: 128*BK, B: BK*128）
- Warp-level tile（如 64x32）在 Shared Memory 内索引
- Thread-level tile（8x8）在寄存器中完成计算
- 这三级 tiling 形成 DRAM -> SMEM -> Register 的层次化数据流

**实际示例（CUTLASS）**：ThreadblockTile=128x128x32，WarpTile=64x64x32，InstructionTile=16x8x8。每个 block 有 4 个 warp，每 warp 内线程协作计算 64x64 的输出。

### Q: 用CUDA实现算子的难点在哪（开放性问题）？

CUDA 算子开发看似简单（写个 kernel launch 就行），但要写出高性能算子极具挑战：

**1. 并行分解（最核心的设计决策）**：
- 如何将问题映射为大量并行线程的工作是首要难题
- 不同的分解方式对性能影响可达 10-100x
- 例如 reduction：naive（相邻线程归约有 divergence）vs 连续线程归约 vs warp shuffle，性能差 5-10x
- 需要同时考虑：线程间独立性（减少同步）、数据局部性（减少访存）、负载均衡

**2. 多级内存层次管理**：
- GPU 有 4-5 级存储：Register(~1 cycle) -> Shared Memory(~20 cycles) -> L1/L2 Cache -> Global Memory(~400 cycles)
- 程序员需要显式管理 Global -> Shared -> Register 的数据搬运
- 需要精确计算每级的容量约束：shared memory per block 限制 tile 大小，寄存器数量限制每线程工作量
- 错误的数据搬运策略（如频繁访问 Global Memory）导致 kernel 受带宽限制

**3. 避免同步瓶颈**：
- `__syncthreads()` 是 block 级 barrier，如果频繁调用（如每次 shared memory 读写都同步）会严重限制指令级并行
- 原子操作（`atomicAdd`）在高竞争时串行化，延迟可达几百周期
- 策略：warp shuffle 替代部分 shared memory 操作（warp 内天然同步）、分层 reduction 减少原子操作

**4. 边界处理与正确性**：
- 实际 shape 通常不是 tile 大小的整数倍（如 seq_len=1537 对 tile=128）
- 边界线程需要特殊处理（mask/predicate），增加代码复杂度
- 边界检查的 branch 可能导致 warp divergence
- 权衡：padding 输入消除边界 vs 增加 mask 逻辑

**5. 跨架构适配**：
- 不同 GPU 的 SM 数（108@A100 vs 132@H100）、shared memory 大小（164KB@A100 vs 228KB@H100）、寄存器数、warp 调度策略不同
- 在一代 GPU 上最优的参数配置在另一代可能很差
- 需要 auto-tuning 或条件编译适配不同硬件

**6. 调试困难**：
- 数千个线程并发执行，race condition 难以复现
- printf 调试有缓冲区限制且影响性能
- 数值精度问题（FP16 累加误差）难以定位
- compute-sanitizer 检测速度慢（10x 开销）

**7. 性能调优的"黑魔法"**：
- 指令排布（instruction scheduling）影响流水线效率
- 寄存器 bank conflict（是的，寄存器也有 bank）
- 编译器行为不可预测（可能自动 unroll/inline 破坏精心设计的策略）
- 有时需要 PTX/SASS 级别的手动调优

### Q: Bank Conflict的概念？如何减少？

**概念**：

GPU 的 Shared Memory 物理上被组织为 32 个 bank，每个 bank 宽度为 4 字节（32 位）。连续的 4 字节地址依次映射到 bank 0、bank 1、...、bank 31、bank 0（循环）。

一个 warp（32 线程）中的所有线程在同一时钟周期发起 shared memory 访问。如果多个线程访问同一个 bank 的不同地址，这些访问必须串行化执行——这就是 bank conflict。

**冲突程度**：
- 无冲突（最优）：32 线程访问 32 个不同 bank -> 一个周期完成
- N-way conflict：N 个线程访问同一 bank 的 N 个不同地址 -> 需要 N 个周期
- Broadcast（特例）：多线程访问同一 bank 的**同一地址** -> 无冲突，硬件广播一次完成

**减少方法**：

**1. 保持 stride=1 连续访问（最佳模式）**：
```cuda
// 无冲突：线程 i 访问地址 i（各在不同 bank）
shared[threadIdx.x] = data;  // stride=1, 0 conflict
```

**2. Padding 打破冲突模式**：
```cuda
// 有冲突：列访问 shared[i][threadIdx.x] 中 stride=32
__shared__ float smem[32][32];  // 列访问 32-way conflict

// 修复：padding 一列
__shared__ float smem[32][33];  // 列访问 stride=33, 无冲突（33 mod 32 = 1）
```
- 原理：padding 后 stride 从 32（32 的倍数）变为 33（与 32 互质），打破了所有线程落在同一 bank 的模式
- 代价：浪费约 3% 的 shared memory 空间

**3. Swizzle 布局**：
- 用 XOR 等位操作将逻辑地址映射为物理地址，使任何规律性访问模式都分散到不同 bank
- CUTLASS 中广泛使用：`physical_addr = logical_addr ^ (logical_addr >> shift)`
- 优点：不浪费存储空间；缺点：增加地址计算的指令

**4. 调整访问粒度**：
- 使用 `float4` 128-bit 加载：一次访问跨越 4 个 bank，某些冲突模式被自然消除
- 但要注意 128-bit 访问的对齐要求

**诊断**：Nsight Compute -> Memory Workload Analysis -> Shared Memory -> Bank Conflicts/Wavefronts。如果 wavefronts/request > 1 说明存在冲突。

### Q: Little's Law在GPU中的应用（访存延迟和计算延迟相关）？

**Little's Law** 是排队论的基本定律：L = λ * W（系统中的请求数 = 到达率 * 平均延迟）。应用到 GPU 的内存子系统：

**GPU 内存系统中的形式**：
```
需要的并发内存请求数 = 内存带宽 × 内存延迟
```
或等价地：
```
Bytes_in_flight = Bandwidth × Latency
```

**具体含义**：要充分利用 HBM 带宽（如 A100 的 2TB/s），需要在任意时刻有足够多的"正在飞行中"（in-flight）的内存请求。

**数值示例**（A100）：
- HBM 延迟 ~400 周期（~300ns @ 1.4GHz）
- HBM 带宽 2TB/s = 2 bytes/ns
- 需要的并发数据量 = 2 bytes/ns * 300ns = 600 bytes 同时在飞行中
- 每个内存请求 128 bytes（一个 cache line），需要 ~5 个并发请求/SM
- 108 个 SM 总共需要 ~540 个并发内存事务来饱和带宽

**与 Occupancy 的关系**：
- 每个 warp 可以发起独立的内存请求（warp 在等待内存返回时让出执行单元给其他 warp）
- 更多活跃 warp（高 occupancy）= 更多并发内存请求 = 更好的带宽利用
- 这就是为什么 memory-bound kernel 需要高 occupancy（>50%）来隐藏延迟

**提高并发度的方法**（当 occupancy 受限时）：
1. **增加每线程的 ILP**：一个线程内发起多个独立的内存请求（循环展开/预取）
   ```cuda
   float a = smem[i];      // 请求 1
   float b = smem[i+32];   // 请求 2（与请求 1 独立，可并发）
   ```
2. **增加 warp 数**：减少每线程寄存器使用 -> 更高 occupancy -> 更多 warp 交替执行
3. **使用异步拷贝**：`cp.async` 指令不阻塞当前 warp，允许 warp 继续执行其他指令

**计算延迟的 Little's Law**：
- 类似地，Tensor Core 有流水线延迟（~8 周期），需要足够的独立矩阵乘指令填充流水线
- 如果 warp 只有一条 MMA 指令，每 8 周期才能发一次 -> 利用率 12.5%
- 需要 8 条独立 MMA 指令（软件流水线/register double buffer）饱和 Tensor Core

**实践意义**：Little's Law 解释了为什么 "增加 occupancy" 和 "增加 ILP" 是 GPU kernel 优化的两大基本手段——本质上都是增加并发请求数来隐藏延迟。

### Q: CUDA实现前缀和（Prefix Sum）的思路？

前缀和（Prefix Sum / Scan）是许多并行算法的基础原语（排序、稀疏矩阵、Stream Compaction 等都依赖它）：

**问题定义**：给定数组 [a0, a1, ..., an-1]，计算 [a0, a0+a1, a0+a1+a2, ..., sum_all]（inclusive scan）

**Blelloch 算法（经典双阶段方法）**：

**Phase 1: Up-sweep（Reduce/归约阶段）**：
- 自底向上，每层将相邻元素对累加到后一个位置
- 层 0：a[1]+=a[0], a[3]+=a[2], a[5]+=a[4], ...
- 层 1：a[3]+=a[1], a[7]+=a[5], a[11]+=a[9], ...
- 层 k：stride = 2^(k+1)，活跃线程间隔翻倍
- 完成后 a[n-1] 包含总和

**Phase 2: Down-sweep（分发阶段）**：
- 将根节点置 0
- 自顶向下，每层将父节点的值分发给子节点
- 每步操作：temp = a[left]; a[left] = a[right]; a[right] += temp
- 完成后数组变为 exclusive prefix sum

**复杂度**：O(n) work, O(log n) span, 需要 O(n) 空间

**CUDA 实现的工程挑战**：

**1. Bank Conflict**：
- Up-sweep 中 stride 翻倍，访问模式容易产生 bank conflict
- 解决：padding shared memory（声明时 +1）或使用 conflict-free 的索引计算

**2. 大数组的多 Block 协调**：
- 单 Block（最多 1024 线程）只能处理有限元素（如 2048 个）
- 大数组方案：
  - Block 内局部 scan -> 收集每 Block 的总和 -> 对总和做 scan -> 加回各 Block
  - 三个 kernel：local_scan -> block_sum_scan -> add_block_offset
- 或使用 decoupled lookback（单 kernel 方案）：每 Block 完成后将总和写入全局标志，后续 Block lookback 获取前缀

**3. Warp-level Scan 优化**：
- Warp 内 32 线程可用 `__shfl_up_sync` 做 scan（无需 shared memory 和 syncthreads）
- 5 步 shuffle 完成 32 元素的 inclusive scan
- Block 内先做 warp-level scan，再用 shared memory 汇总各 warp 的总和

**4. 向量化**：每线程处理多个元素（如 4 个 float），减少线程数和同步开销，同时用 float4 加载。

**CUB 库的实现**：NVIDIA 的 CUB（`cub::DeviceScan`）是生产级实现，使用 decoupled lookback + warp scan + 向量化加载，是当前最优的单 GPU prefix sum 实现。
