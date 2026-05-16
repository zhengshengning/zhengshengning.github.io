---
title: '蚂蚁 AI Infra 实习 一面 (2)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 如何判断一条推理链路中最有效的优化方式是什么？

**两大最有效方向**（投入产出比最高）：

**1. 高频基础算子的单算子优化**：
- 找到整网中**出现次数最多**的基础算子（如 GEMM 每层 7 个 = 80 层×7 = 560 次、LayerNorm 每层 2 个 = 160 次）
- 对其做 10% 的单点性能优化 → 整网收益 = 10% × 频次占比
- GEMM 通常占推理时间的 70-85%，是第一优化目标

**2. 算子融合**：
- 找到**既重、又成链、又高频**的算子组合做融合
- 典型高收益 pattern：QKV projection 融合、LayerNorm + Residual + Linear、GEMM + Bias + Activation
- 每次融合消除 1 次 kernel launch（~5μs）+ 中间 tensor 的 HBM 读+写（2×tensor_size）

**系统化定位方法**：
```
Nsight Systems 时间线分析
    ↓ 识别耗时 Top-N kernel
    ↓ 判断瓶颈类型（NCU Roofline）
    ↓ 选择最高 ROI 的优化路径
```

### Q: 优化后仍和目标性能有 gap，差距来自哪？

**三层分析框架**：

**1. 算法/实现层面（可优化）**：
- 算子实现未达 SOL 上限（如 Tensor Core 利用率只有 60%）
- 调度不够紧凑（kernel 间有 gap、不必要的同步）
- 内存管理效率低（碎片、频繁分配）
- 编译器未生成最优指令（需手动 PTX 优化）

**2. 硬件理论上限（不可突破）**：
- 受制于算力上限：即使 100% SOL 也只能达到 Peak FLOPS
- 受制于带宽上限：memory-bound 算子最多跑满 HBM 带宽
- 受制于互联带宽：多卡通信受 NVLink/IB 带宽限制

**3. 系统开销（可减少但不可消除）**：
- Kernel Launch 延迟（~5μs/kernel，融合可减少）
- Host-Device 同步（CUDA event、cudaDeviceSynchronize）
- 内存分配/释放（通过 memory pool 减少）
- Python/框架开销（通过 torch.compile/C++ inference 消除）
- 通信协议开销（NCCL setup、protocol negotiation）

**判断 gap 来源**：
- NCU SOL < 80% → 实现还有优化空间
- NCU SOL > 80% → 接近硬件极限，需要更本质的改变（如算法换硬件更匹配的方案）

### Q: 如何判断哪些算子链适合融合？

**四个判断维度**（按优先级排列）：

**1. 依赖关系（最关键）**：
- 理想：`A → B → C`（简单链，无分支）
- 可融合：中间结果只有一个消费者
- 困难：中间结果被多个分支消费（如 residual connection 中间的激活既要送下一层又要做 skip）
- 解决方案：融合后在 kernel 内部写两份输出（一份给融合链后续，一份给外部分支）

**2. 链条长度和频次**：
- 链越长：融合消除的 HBM 读写次数越多（每消除一个中间节点 = 节省 2×tensor_size 带宽）
- 出现频次越多：整网总收益 = 单次收益 × 频次
- 例：`LayerNorm → Linear → SiLU → Linear` 出现 80 次/模型

**3. 算子类型兼容性**：
- **纯 Elementwise 链**：最容易融合（逐元素操作，天然并行一致）
- **Elementwise + GEMM**：中等（GEMM 的后处理如 Bias+Activation 可以融合进去）
- **Reduction + Elementwise**：较难（reduce 需要线程协作，与 elementwise 的并行模式不同）
- **GEMM + Reduction**：难（如 GEMM + LayerNorm，两者的并行维度不同）

**4. 资源约束**：
- 融合后寄存器压力是否超限（→ register spilling）
- 融合后 shared memory 是否超限（→ occupancy 下降）
- 需要评估融合后单 kernel 的资源消耗

### Q: Profiling 时重点看什么指标？

**五大核心指标**：

1. **总执行时间拆分**：
   - 启动开销（CPU→GPU dispatch）vs 计算时间 vs 数据搬运时间
   - 如果 launch 开销占比 > 20%：需要算子融合
   - 如果数据搬运占比大：考虑 overlap 或减少传输

2. **Warp 执行效率**：
   - Stall 原因分布：Memory Dependency（等内存返回）、Execution Dependency（等计算完成）、Synchronization（等 `__syncthreads`）
   - Memory stall 占主导 → memory-bound，优化访存
   - Execution stall 占主导 → 指令级并行不足

3. **带宽利用率**：
   - Global Memory Throughput vs 峰值（A100: 2.0 TB/s）
   - Shared Memory Throughput vs 峰值（~19 TB/s）
   - 接近峰值说明访存模式已经很好

4. **计算利用率**：
   - SM Active Cycles / Total Cycles
   - Tensor Core utilization（矩阵乘应该 > 80%）

5. **多流场景**：
   - 是否形成合理流水（H2D / Compute / D2H 重叠）
   - 有无不必要的同步（cudaDeviceSynchronize 会打断流水）

### Q: W8A8 和 W4A16 分别代表什么？为什么激活要保留更高精度？

**配置含义**：
- **W8A8**：权重 INT8 + 激活 INT8。需要 SmoothQuant 等方法处理激活 outlier
- **W4A16**：权重 INT4 + 激活 FP16。只量化权重，激活保持高精度

**为什么激活更难量化、应保持高精度**：

| 维度 | 权重 | 激活 |
|---|---|---|
| 分布 | 静态固定，训练后不变 | 动态变化，每个输入不同 |
| Outlier | 少，分布平稳 | 多，某些 channel 极端值 |
| 校准 | 可离线精确统计 | 只能动态估计或用代表性数据 |
| 误差影响 | 误差在一次 GEMM 中影响 | 误差在后续层**累积放大** |
| 量化方案 | per-channel scale 效果好 | 需要 per-token 或更细粒度 |

**误差累积的直觉**：激活是层与层之间的"信号"，如果信号本身被污染，后续每一层都基于错误的输入计算，误差逐层放大。而权重误差只影响当前层的计算。

### Q: 均匀量化 vs 非均匀量化的取舍？

| 维度 | 均匀量化 (INT8/INT4) | 非均匀量化 (FP8/NF4/LUT) |
|---|---|---|
| Bin 分布 | 等间距 | 对数/自定义间距 |
| 硬件加速 | **强**（INT8 Tensor Core） | 弱或无（通用 CUDA Core） |
| 适配正态分布 | 较差（大量 bin 在尾部浪费） | 好（0 附近 bin 更密） |
| 实现复杂度 | 简单（乘/除 scale） | 复杂（lookup table / 特殊编解码）|
| 存储开销 | scale + zp（少量）| 可能需要 LUT 或额外元数据 |
| 推理吞吐 | 高 | 低（无硬件加速） |

**工程实践的主流选择**：
- **均匀量化 + per-group scale** = 折中最优
  - 均匀格式利用 INT8/INT4 Tensor Core
  - Per-group（group_size=128）使每组内分布相对均匀
  - 这是 GPTQ/AWQ 的做法

- **FP8**：在 Hopper 上有原生 Tensor Core 支持，兼具非均匀优势和硬件加速
- **NF4**：QLoRA 使用，4-bit 内最优地适配正态分布权重，但无硬件加速

### Q: CUDA 实现 Histogram 算子，如何优化写冲突？

**问题**：多个线程同时对同一个 bin 做 `atomicAdd`，大量原子操作竞争同一地址导致串行化。

**朴素实现**：
```cuda
__global__ void histogram_naive(int* data, int* hist, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&hist[data[idx]], 1);  // 大量线程竞争！
}
```

**优化方案（从简到复杂）**：

**1. Block-local Histogram（最常用）**：
```cuda
__shared__ int local_hist[NUM_BINS];  // block 内局部 histogram
// 初始化为 0
// block 内用 shared memory atomic（冲突概率低 32x）
atomicAdd(&local_hist[data[idx]], 1);
__syncthreads();
// 最后汇总到 global
if (threadIdx.x < NUM_BINS) atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
```
- 冲突从 grid 级别降低到 block 级别（256 线程 vs 百万线程）

**2. 每线程处理多个元素**：
- 每个线程循环处理 16-32 个数据，本地统计后一次性更新
- 减少总 atomic 操作次数

**3. Warp-level 预聚合（最激进优化）**：
```
warp 内：用 __ballot_sync 收集所有线程中目标 bin 相同的线程
         → __popc 统计个数
         → 由一个线程做一次 atomicAdd（32 次 → 1 次）
```

**4. 数据分区（减少 bin 冲突概率）**：
- 如果 bin 数量少（如 256），按 bin 范围将数据分区
- 每个 block 只处理特定范围的数据，进一步减少对同一 bin 的竞争
