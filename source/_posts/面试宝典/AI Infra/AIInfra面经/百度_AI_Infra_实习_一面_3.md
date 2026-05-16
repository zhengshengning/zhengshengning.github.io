---
title: '百度 AI Infra 实习 一面 (3)'
date: 2026-04-16 12:18:11
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: TP、PP、DP分别是什么？具体流程？

三种基础并行策略，通常组合使用（3D 并行）来训练大模型：

**TP（Tensor Parallelism，张量并行）**：
- 将单层的权重矩阵按行或列切分到多卡。如 Linear(4096, 4096) 切为 4 卡各持有 Linear(4096, 1024)
- 前向：每卡计算部分结果，通过 AllReduce（列切分）或 AllGather（行切分）聚合完整输出
- 通信：每层需要 2 次 AllReduce（attention 后 + FFN 后），通信量 = 2 * batch * seq * hidden * dtype_size / 层
- 特点：通信频率高但单次量不大，要求高带宽低延迟互联（NVLink 600GB/s）

**PP（Pipeline Parallelism，流水线并行）**：
- 将模型不同层分配到不同卡（stage），数据以 micro-batch 流水方式通过各 stage
- 调度策略：GPipe（先全部前向再全部反向，气泡大）、1F1B（交替前向反向，气泡 (p-1)/m）、Interleaved 1F1B（多虚拟 stage 进一步减少气泡）
- 通信：只在 stage 边界传递激活值（batch * seq * hidden），频率低
- 特点：通信量小容忍高延迟，适合跨节点；但存在流水线气泡

**DP（Data Parallelism，数据并行）**：
- 每卡持有完整模型副本，切分数据并行训练
- 前向各自计算，反向后梯度 AllReduce 同步
- 通信：每步一次 AllReduce（全部参数的梯度），可与反向计算重叠
- 特点：最简单的并行方式，通信量 = 参数量 * dtype_size。DDP/FSDP 是其变种

### Q: 如何根据TP和PP的通信量进行取舍？

**核心原则：通信频率与互联带宽匹配**。

| 对比 | TP | PP |
|------|----|----|
| 通信频率 | 极高（每层2次） | 低（每stage边界1次） |
| 单次通信量 | O(batch * seq * hidden) | O(batch * seq * hidden) |
| 延迟敏感度 | 高（在计算关键路径上） | 低（可流水线隐藏） |
| 带宽需求 | 极高 | 中等 |

**典型配置策略**：
- **节点内**用 TP：NVLink 提供 600-900 GB/s 带宽，满足 TP 的高频通信需求。通常 TP=4 或 8（一台机器的 GPU 数）。
- **节点间**用 PP：跨节点网络带宽有限（InfiniBand 400Gbps ≈ 50GB/s），PP 通信频率低可容忍。
- **全局**用 DP：梯度 AllReduce 可与反向计算重叠（communication-computation overlap），对延迟不敏感。

**实际案例**：LLaMA-2 70B 训练使用 TP=8（节点内）+ PP=4（跨节点）+ DP=16，共 512 GPU。

### Q: 量化中per-tensor、per-channel、per-group的区别？为什么group-wise能降低量化误差？

三者的区别在于量化参数（scale/zero_point）的共享粒度：

**per-tensor**：整个张量（如一个权重矩阵 [out_ch, in_ch]）共享一组 scale/zp。最粗粒度，如果张量内值域差异大（如某些通道值域 [-1,1] 而另一些 [-100,100]），小范围通道的量化精度被浪费。

**per-channel**：每个输出通道独立一组 scale/zp。一个 [128, 4096] 的权重矩阵有 128 组参数。能适应不同通道的值域差异，精度显著优于 per-tensor。

**per-group**：将通道内元素再分组（如每 128 个连续元素一组），每组独立参数。一个 [128, 4096] 的矩阵有 128*32=4096 组参数。

**group-wise 降低误差的原因**：
1. **更窄的值域**：组内 128 个元素的 min-max 范围远小于整个通道的范围，量化步长（scale）更小
2. **异常值影响被限制**：一个 outlier 只影响其所在组的 128 个元素的精度，而非整个通道
3. **量化区间利用率更高**：INT4 只有 16 个有效值，range 越小每个值的分辨率越高

**代价**：每组需要额外存储 scale（FP16，2B）和可能的 zero_point，增加元数据开销。GPTQ/AWQ 等方法通常使用 group_size=128。

### Q: 不同量化方法的区别及如何降低开销？

**主流 PTQ 量化方法对比**：

| 方法 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| **GPTQ** | 逐层用 Hessian 信息补偿量化误差 | 精度高（最优的W4质量） | 需要校准数据计算Hessian，一次性开销大 |
| **AWQ** | 按激活分布识别重要通道并保护 | 简单高效，代码量少 | 保护重要通道的方式较粗糙 |
| **SmoothQuant** | 等价数学变换将激活异常值转移到权重 | W和A都容易量化（W8A8） | 仅适用于weight+activation同时量化 |
| **GGUF/llama.cpp** | 混合精度per-layer量化 | 支持CPU推理，灵活 | 无Hessian优化，精度略低 |

**降低推理开销的方法**：
1. **高效量化 Kernel**：W4A16 GEMM 在计算时实时反量化权重，避免预先展开的内存开销
2. **减少反量化频率**：将反量化融合到 GEMM kernel 中（在 shared memory 层面做）
3. **权重预打包**：将量化权重按 kernel 计算顺序排列，减少运行时地址计算
4. **INT8 GEMM 直接计算**：使用 Tensor Core 的 INT8 模式直接做矩阵乘（如 FP8 E4M3）

### Q: 量化中的异常值问题及消除方法？

**问题描述**：大模型的某些激活通道存在极端异常值（outlier），幅度可达正常值的 100 倍以上。这导致量化范围（scale）被异常值主导，绝大部分正常值被压缩到极少的量化级中，精度严重损失。

**异常值特征**（来自 LLM.int8() 论文）：
- 出现在固定的通道位置（"emergent features"）
- 模型越大越明显（6B+ 参数后显著）
- 只占 0.1-1% 的通道，但去掉后模型崩溃

**消除/应对方法**：

1. **SmoothQuant**：在 LayerNorm 和 Linear 之间插入等价缩放变换 s：Y = (X * diag(s)^{-1}) * (diag(s) * W)。将激活的大值通道缩小（除以 s），对应权重通道放大（乘以 s）。数学等价但平滑了激活分布。s 的选择基于激活和权重的 per-channel max 的几何平均。

2. **LLM.int8()**（Mixed-precision decomposition）：识别异常值通道（如幅度>6的）保持 FP16 计算，其余通道做 INT8 量化。两部分结果相加。精度无损但实现稍复杂（需要混合精度 GEMM）。

3. **Clipping/Percentile**：截断极端值到 99.9% 分位数。简单但有信息损失。适合异常值不那么极端的情况。

4. **QuIP/QuIP#**：通过正交变换使权重分布更均匀后再量化，从数学上消除异常值。

### Q: FlashAttention为什么能加速？FlashAttention 1和2的区别？

**加速原因**（IO 视角）：
标准 Attention 需要将 N*N 的 score 矩阵写回 HBM 再读回（Softmax 需要全行信息）。对于 N=4096, d=128，计算量 = O(N^2*d) ≈ 2G FLOPs，但 IO 量 = O(N^2) ≈ 32MB 的 score 矩阵读写。在 A100 上这使得 Attention 是 memory-bound 的。

FlashAttention 通过 tiling 将 Q/K/V 分块加载到 SRAM（~200KB），在片上完成所有计算（online softmax 避免全局 materialization），只读写 O(N) 的最终输出。算术强度从 O(1) 提升到 O(d)（等于 head_dim），变为 compute-bound。

**FA1 vs FA2 的关键区别**：

| 方面 | FlashAttention 1 | FlashAttention 2 |
|------|------------------|------------------|
| 外层循环 | 遍历 KV 块 | 遍历 Q 块 |
| 内层循环 | 遍历 Q 块 | 遍历 KV 块 |
| 并行度 | 在 batch*heads 上并行 | 在 batch*heads*seq(Q) 上并行 |
| Warp 通信 | warp 间需 reduce O 和 m/l | warp 间无需通信（各自处理不同 Q 行） |
| 非矩阵乘 FLOPs | 较多（rescale output） | 更少（延迟 rescale） |
| 性能 | baseline | 约 2x 快 |

FA2 的核心改进：将外层循环改为遍历 Q 使得每个 thread block 的输出是最终的（无需全局 reduce），天然支持更多并行；减少 warp 间同步和非 matmul 计算。

### Q: FlashAttention中Bc块的切分思路？1-loop FlashAttention？

**Bc（K/V 块大小）的确定**：
受 SRAM 容量约束。一个 thread block 需要同时存放：
- K 块：Bc * d（FP16）
- V 块：Bc * d
- Q 块：Br * d
- 中间结果（score/O/m/l）

总容量 <= SRAM 大小 M。简化计算：Bc = M / (4 * d)（假设各块大小相近）。对 A100 M=192KB, d=128，Bc ≈ 192。实际取 128（2 的幂方便对齐）。

**1-loop FlashAttention（FA2 的核心改进）**：
- 外层只遍历 Q 块（每个 thread block 处理 Br 行 Q）
- 内层遍历所有 KV 块，逐块累加到该 Q 块的输出 O 上
- 好处：每个 Q 块的计算完全独立，不同 Q 块天然并行（可分配到不同 thread block/SM）
- 无需跨 block 的全局 reduce，输出直接写回 HBM

这与 FA1 的 2-loop（外层 KV，内层 Q，需要全局累加）形成对比。

### Q: PagedAttention的思路？

PagedAttention（vLLM 提出）借鉴**操作系统虚拟内存分页**思想管理 KV Cache：

**问题**：传统 KV Cache 为每个请求预分配 max_seq_len 的连续显存，导致：
1. 内部碎片：实际生成长度远小于预分配大小
2. 无法动态调整：不同请求实际长度差异大
3. 无法共享：相同前缀的请求各自存一份

**PagedAttention 方案**：
- 将 KV Cache 按固定大小的 block（如 16 个 token）分页存储
- 维护 block table（类似页表）记录每个请求各位置的物理 block 地址
- 物理 block 不要求连续，按需分配（类似虚拟内存的按需分页）

**收益**：
- 消除内部碎片：只分配实际使用的 block 数量
- 支持 Prefix Caching：共享相同 system prompt 的请求可共享前缀 block（Copy-on-Write）
- 灵活的内存管理：block 粒度的分配/释放/复用

**代价**：Attention kernel 需要按 block table 做间接寻址读取 KV，比连续访问略慢。但显存利用率从 ~50% 提升到 >95%，支持同时服务更多请求。

### Q: Prefill和Decoding阶段的区别？

自回归 LLM 推理分为两个性质完全不同的阶段：

| 对比 | Prefill | Decoding |
|------|---------|----------|
| **输入** | 完整 prompt（如 1000 token） | 1 个新 token |
| **输出** | 所有位置的 KV Cache | 1 个 next token |
| **瓶颈** | 计算密集（Compute-bound） | 访存密集（Memory-bound） |
| **矩阵乘形状** | [seq, hidden] * [hidden, hidden]（大矩阵） | [1, hidden] * [hidden, hidden]（向量-矩阵） |
| **GPU利用率** | 高（大batch矩阵乘，接近峰值TFLOPS） | 低（算术强度低，受限于HBM带宽） |
| **耗时占比** | 小（一次性完成） | 大（生成N个token需要N步） |

**系统优化启示**：
- Prefill 适合大 batch 并行处理（如 continuous batching 中新请求的 prefill 与旧请求的 decode 混合）
- Decoding 应尽量增大 batch size 提高 GPU 利用率（但受 KV Cache 显存限制）
- Splitwise/Disaggregation：将 Prefill 和 Decode 分离到不同 GPU 上独立优化

### Q: Prefill阶段的FlashAttention在Decoding阶段有什么问题？Flash Decoding的思路？

**问题**：Decoding 时 Q 只有 1 个 token（形状 [1, d]），但 KV Cache 可能有数千个 token。标准 FlashAttention 的外层循环遍历 Q 块——当 Q 只有 1 行时，只有一个 thread block 参与计算，GPU 大量 SM 闲置，并行度严重不足。

**Flash Decoding 思路**：
不再按 Q 维度并行（只有 1 行），而是**沿 KV 的序列长度维度并行切分**：
1. 将 KV Cache 按 seq 维度切分为多段（如 8-32 段）
2. 每段分配一个 thread block，独立计算该段的局部 attention 输出和 online softmax 统计量（local_O, local_m, local_l）
3. 最后用一个额外的 reduce kernel 合并所有段的结果（利用 online softmax 的修正公式）

**效果**：即使 Q 只有 1 行，也能启动数十个 thread block 并行计算，充分利用 GPU。在长序列（seq>4096）场景下 decode 速度提升 3-8x。

### Q: RMSNorm是如何实现的（CUDA kernel层面）？

**计算流程**：
1. 输入 x 维度 [hidden_size]，计算 x^2 的均值：RMS^2 = (1/n) * sum(x_i^2)
2. 计算 rsqrt：inv_rms = rsqrt(RMS^2 + eps)
3. 归一化并乘以可学习参数：y_i = x_i * inv_rms * gamma_i

**CUDA 优化要点**：
- **Block-level reduction**：一个 block 处理一行（hidden_size 个元素）。每个线程负责部分元素的平方和，通过 shared memory + warp shuffle 做 block 内 reduction
- **Warp Shuffle 优化**：使用 `__shfl_down_sync` 做 warp 内归约，减少 shared memory 使用和 `__syncthreads` 同步次数
- **向量化加载**：使用 `float4` 或 `half2` 加载连续元素，一次读 128bit，提升内存带宽利用
- **Kernel 融合**：RMSNorm 是 memory-bound 操作（算术强度极低），通常与前后的算子融合（如 RMSNorm + Linear 的输入预处理）

### Q: 如何优化一个CUDA kernel的思路？

**系统化优化方法论**（由宏观到微观）：

1. **分析瓶颈**：用 Nsight Compute 判断 kernel 是 compute-bound 还是 memory-bound
   - Compute-bound：SM 利用率高但计算 TFLOPS 未达峰值 -> 优化计算
   - Memory-bound：HBM 带宽接近峰值 -> 减少访存或提高数据复用

2. **内存优化**（通常最有效）：
   - 合并访存（coalesced access）：同一 warp 的线程访问连续地址
   - Shared memory 缓存：将重复访问的数据加载到片上
   - 减少 bank conflict：padding 或调整访问模式
   - 减少全局内存事务数：向量化加载（float4）

3. **计算优化**：
   - 增加 ILP（指令级并行）：每个线程处理多个元素，让 pipeline 填满
   - 利用 Tensor Core：WMMA/MMA 指令加速矩阵乘
   - 数学优化：rsqrt 代替 sqrt+div，fast math intrinsics

4. **并行度优化**：
   - 确保足够的 block 数占满所有 SM（如 A100 有 108 SM）
   - 合理设置 occupancy（不是越高越好，需要平衡寄存器/shared memory）
   - 减少 warp divergence（连续线程走相同分支）

5. **减少开销**：
   - 减少 `__syncthreads` 同步点
   - 减少原子操作
   - Kernel fusion 消除中间读写

### Q: Conv2D的参数量和FLOPs计算？

以输入 [C_in=64, H=128, W=64]，3x3 卷积，输出 [C_out=128, H=128, W=64]（stride=1, padding=1）为例：

**参数量** = C_out * C_in * K_h * K_w + C_out（bias）
= 128 * 64 * 3 * 3 + 128 = 73,856（约 74K 参数）

**FLOPs**（乘加各算一次操作）= 2 * C_out * C_in * K_h * K_w * H_out * W_out
= 2 * 128 * 64 * 3 * 3 * 128 * 64 ≈ **1.21 GFLOPs**

**算术强度** = FLOPs / 内存访问字节数。对于大 feature map，Conv2D 通常是 compute-bound；小 batch/小 feature map 时可能变为 memory-bound。

### Q: CPU算子优化方法（AVX512等）？

CPU 算子优化的核心是充分利用现代 CPU 的并行能力：

1. **SIMD 向量化**：使用 AVX512 指令一次处理 16 个 float32（512bit 寄存器）。关键是保证数据对齐（64B 对齐）和消除依赖链。编译器 auto-vectorization 或手写 intrinsics（`_mm512_fmadd_ps` 等）。

2. **循环展开 + 软件流水线**：展开内层循环 4-8 次，让多条指令并行执行（ILP）。编译器 `#pragma unroll` 或手动展开。

3. **Cache 分块（Tiling）**：将计算分块使工作集适配 L1/L2 Cache。如 GEMM 分块：tile_M * tile_N 的输出块适配 L1，tile_K 的迭代适配 L2。

4. **数据预取（Prefetch）**：`_mm_prefetch` 在使用前提前将数据加载到 Cache，隐藏内存延迟。

5. **多线程并行**：OpenMP 在外层循环并行化。注意 NUMA 感知（绑核、local memory）。

6. **内存布局优化**：AoS -> SoA 变换使 SIMD 加载连续数据；矩阵按计算顺序排列（如 GEMM 的 B 矩阵转置/packed 布局）。

### Q: C++智能指针？

shared_ptr（引用计数共享，原子操作保证计数线程安全，make_shared 一次分配更高效）、unique_ptr（独占零开销，delete 拷贝只允许 move，默认首选）、weak_ptr（弱引用不增计数，lock() 提升，打破循环引用/缓存探测）。

### Q: 手撕：CUDA实现LayerNorm？

（编程题）
