---
title: '快手 AI Infra 校招 (1)'
date: 2026-04-16 12:07:12
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 算子优化中的访存优化有哪些方法？

GPU 算子性能很大程度取决于内存访问效率。对于 memory-bound 算子，访存优化可带来数倍加速：

**1. 合并访存（Coalesced Access）**：
- 同一 warp 的 32 个线程访问连续地址时，硬件将多个请求合并为 1-4 次内存事务
- 最优模式：thread_i 访问 base + i（连续地址）
- 最差模式：thread_i 访问 base + i×stride（stride > 1 时无法合并，带宽浪费）
- 效果：合并 vs 非合并可以差 8-32x 带宽利用率

**2. 向量化加载**：
- 使用 `float4`/`int4` 一次加载 128-bit（4 个 float 或 8 个 half）
- 减少内存事务数量和指令数
- 要求：加载地址必须 16 字节对齐
- 效果：吞吐提升 2-4x

**3. Shared Memory Tiling**：
- Global Memory 带宽 ~2-3 TB/s，Shared Memory 带宽 ~19 TB/s（A100）
- 将数据分块加载到 Shared Memory，在 block 内多次复用
- 数据复用率 = block 内访问次数 / 从 global 加载次数
- GEMM 中：复用率 ∝ tile 大小（BM+BN 次复用 BK 的数据）

**4. 数据布局优化**：
- 选择对当前访问模式友好的内存布局
- 例：CNN 推理用 NHWC（channel 连续）比 NCHW 更适合合并访存
- Transformer 中：确保 head_dim 维度连续，利于 attention 计算

**5. Padding 消除 Bank Conflict**：
- Shared Memory 32 个 bank，每 4 字节一个 bank
- 多个线程访问同一 bank 的不同地址时串行化
- 解决：数组每行末尾 pad 1 个元素（`float s[N][N+1]`），错开 bank 映射

**6. 预取（Prefetch / Double Buffering）**：
- 分配两份 buffer，计算当前 tile 时异步加载下一 tile
- `cp.async`（Ampere+）支持硬件异步从 global→shared
- 彻底隐藏 global memory 加载延迟（400-800 cycles）

**7. 减少冗余读取（算子融合）**：
- 两个相邻算子的中间结果不写回 HBM，留在寄存器/shared memory 传递
- 消除中间 tensor 的一次 write + 一次 read（节省 2x 数据量的 HBM 带宽）

### Q: 算子融合的实现难点？

算子融合是将多个连续 kernel 合并为一个 kernel 执行。理论收益明确（减少 kernel launch + 中间 HBM 读写），但工程实现有诸多挑战：

**1. 依赖分析**：
- 融合的前提：两个算子之间是简单的生产者-消费者关系（单消费者）
- 如果中间结果被多个下游算子使用（fan-out），融合后需要在 kernel 内部生成多份输出或选择性融合部分分支
- 需要精确的数据流分析判断融合安全性

**2. 寄存器/Shared Memory 压力**：
- 融合后单个 kernel 需要更多寄存器存储中间结果
- 寄存器超限 → register spilling → 性能反而下降
- Shared Memory 用量增加 → 每 SM 能驻留的 block 减少 → occupancy 降低
- 需要在融合收益和资源压力间找平衡点

**3. 不同算子的并行模式差异**：
- Elementwise（逐元素）：每线程处理一个元素
- Reduction（归约）：需要线程间协作（shared memory reduce）
- GEMM：Tensor Core + 复杂 tiling
- 融合 elementwise + reduction 需要在同一 kernel 中处理两种并行范式

**4. Shape 动态性**：
- 动态 shape 下融合 kernel 的 grid 配置需要运行时决定
- 同一融合 pattern 对不同 shape 可能需要不同的实现策略
- JIT 编译方案（如 Triton）可部分缓解

**5. 调试困难**：
- 融合 kernel 内部状态不可直接 printf/profiling
- Bug 可能来自中间状态不正确、同步缺失等
- 单独验证融合正确性需要与 reference 实现逐点对比

**6. 通用性 vs 性能的权衡**：
- 手写融合 kernel：性能最优，但每个新 pattern 都要重写，维护成本高
- 自动融合（compiler-based，如 XLA/Triton/TVM）：通用但性能可能达不到手写的 100%
- 实践中：热点 pattern 手写 + 长尾 pattern 自动融合

### Q: GPU 并行计算中的负载均衡问题如何解决？

负载不均衡导致部分 SM idle 等待其他 SM，降低 GPU 利用率：

**1. 均匀划分工作**：
- 最简单方案：将 N 个元素平均分给所有线程/block
- Grid 大小 = ceil(N / block_size)
- 适用于规则数据（dense tensor），不适用于稀疏/变长数据

**2. 动态任务分配（Work-Stealing）**：
- 维护全局任务队列（用 atomic 计数器实现）
- 每个 block 完成当前任务后通过 `atomicAdd` 取下一个
- 适合任务粒度差异大的场景（如稀疏矩阵、不规则图）

**3. Padding/对齐**：
- 将变长数据 pad 到固定长度（如 seq_len pad 到 max_seq_len）
- 简化并行化但浪费计算（对 padding 部分做无效计算）
- 适合长度差异不大的场景

**4. Persistent Kernel**：
- 不按传统的"一个 block 处理一个 tile"模式
- 而是 launch 固定数量的 block（= SM 数量），每个 block 内部循环取任务
- 消除 block 间工作量不均的问题
- FlashAttention-3 和 cuBLAS 中广泛使用

**5. Block 粒度调整**：
- Block 过大：最后一个 block 可能只有少量有效线程
- Block 过小：kernel launch 和调度开销增大
- 选择原则：确保 Grid 中 block 数量 >> SM 数量（如 4x+），让 SM 调度器能均衡分配

**6. 避免 Warp Divergence**：
- 同一 warp 内 if-else 分支导致不同线程执行不同路径，两条路径串行执行
- 解决：将不同工作类型的数据分组，确保同一 warp 的线程执行相同路径
- 或使用 predication 替代分支（简单条件下编译器自动优化）

### Q: 如何判断优化是否已到瓶颈？

通过 **Roofline 模型**判断算子是否已接近硬件理论上限：

**Roofline 模型核心思想**：
```
实际性能 ≤ min(Peak_FLOPS, Arithmetic_Intensity × Peak_Bandwidth)

其中：Arithmetic Intensity (AI) = FLOPs / Bytes_accessed
```

**分析步骤**：
1. 计算算子的 AI（如 GEMM 的 AI ≈ M×N×K×2 / (M×K + K×N + M×N)×bytes）
2. 对比硬件拐点 = Peak_FLOPS / Peak_Bandwidth（A100: 312T / 2T = 156 FLOP/Byte）
3. AI > 拐点 → Compute-bound；AI < 拐点 → Memory-bound

**判断瓶颈的具体指标**：

| 瓶颈类型 | 特征 | 阈值 | 优化方向 |
|---|---|---|---|
| Compute-bound | SM 利用率高, Tensor Core 利用率高 | SOL > 80% | 更高效的计算（Tensor Core, 减少冗余计算） |
| Memory-bound | 带宽利用率高 | 带宽利用 > 80% Peak | 减少数据搬运（融合, 量化, tiling） |
| Latency-bound | kernel 时间短但数量多 | kernel < 10μs | 算子融合减少 kernel 数 |

**工具使用**：
- **Nsight Systems**：全局时间线 → 找热点 kernel 和 idle gap
- **Nsight Compute（NCU）**：深入分析单个 kernel：
  - SOL（Speed of Light）指标：计算/内存各达到峰值的百分比
  - Roofline 图：直观显示算子在 roofline 上的位置
  - Warp Stall 原因分布：指出瓶颈是内存等待还是执行依赖

**实践经验**：
- SOL > 80% 基本到达瓶颈，进一步优化收益递减
- 大多数初始实现只有 30-50% SOL，有 2-3x 优化空间

### Q: 常见的量化策略有哪些？

**按量化粒度分类**：

| 粒度 | 说明 | scale 数量 | 精度 | 适用场景 |
|---|---|---|---|---|
| Per-tensor | 整个 tensor 一个 scale | 1 | 最差 | 要求最简 |
| Per-channel | 每个 output channel 一个 scale | C_out | 好 | 权重量化标配 |
| Per-group | 每 G 个连续元素共享 scale | N/G | 更好 | W4 量化常用(G=128) |
| Per-token | 每个 token 一个 scale | seq_len | 好 | 激活量化 |

**按时机分类**：
- **PTQ（训练后量化）**：模型训练好后直接量化，成本低但精度可能损失
- **QAT（量化感知训练）**：训练中模拟量化噪声，精度更高但需要重训练
- **动态量化**：运行时根据实际数据计算 scale，无需校准但有统计开销

**按对象分类**：
- **权重量化（W8/W4）**：最常见，权重是静态的容易量化
- **激活量化（A8/A16）**：动态范围大、outlier 多，量化更难
- **KV Cache 量化**：减少推理时显存占用

**代表性方案对比**：

| 方案 | 配置 | 核心技术 | 精度-速度权衡 |
|---|---|---|---|
| GPTQ | W4A16 | 逐列量化 + Hessian 误差补偿 | 高精度，量化慢 |
| AWQ | W4A16 | 保护重要权重 channel | 精度好，量化快 |
| SmoothQuant | W8A8 | 平滑激活 outlier | 全 INT8 计算加速 |
| FP8 | W8A8 | Hopper 原生支持 | 精度损失极小 |
| GGUF | 多种 | per-block + super-block | CPU 推理友好 |
