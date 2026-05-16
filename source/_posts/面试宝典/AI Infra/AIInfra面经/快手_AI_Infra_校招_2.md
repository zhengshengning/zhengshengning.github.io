---
title: '快手 AI Infra 校招 (2)'
date: 2026-04-16 12:12:39
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 训练优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: H100 相比 A100 有哪些改进？

| 特性 | A100 (Ampere) | H100 (Hopper) | 提升幅度 |
|---|---|---|---|
| FP8 支持 | 无 | 原生 Tensor Core 支持 | 全新能力 |
| FP16 算力 | 312 TFLOPS | 989 TFLOPS | **3.2x** |
| FP8 算力 | N/A | 1978 TFLOPS | — |
| HBM | HBM2e 80GB, 2.0 TB/s | HBM3 80GB, 3.35 TB/s | 带宽 **1.67x** |
| NVLink | NVLink 3, 600 GB/s | NVLink 4, 900 GB/s | **1.5x** |
| TMA | 无 | Tensor Memory Accelerator | 全新能力 |
| DSMEM | 无 | 分布式共享内存 | 全新能力 |
| Cluster | 无 | Thread Block Cluster | 全新能力 |
| Transformer Engine | 无 | 自动 FP8 混合精度管理 | 全新能力 |
| SM 数 | 108 | 132 | **1.22x** |
| L2 Cache | 40 MB | 50 MB | 1.25x |

**三大核心提升**：

1. **FP8 支持**：算力直接翻倍。对于 LLM 训练，FP8 Transformer Engine 可在几乎无精度损失的情况下获得 2x 吞吐提升

2. **TMA（Tensor Memory Accelerator）**：
   - 硬件加速的异步多维数据搬运单元
   - 传统做法：SM 需要花线程计算地址 + 发起加载 → 浪费 SM 计算资源
   - TMA：由专用硬件计算地址和执行搬运，SM 完全腾出来做计算
   - 支持多维 tensor 直接搬运（无需 flatten），自动处理 padding/stride

3. **HBM3 带宽提升 67%**：对 memory-bound 的 LLM Decode 阶段直接转化为 1.67x 加速

### Q: 介绍 Warp 这个概念？

Warp 是 NVIDIA GPU 中**最小的调度和执行单位**，由 **32 个线程**组成，以 SIMT（Single Instruction, Multiple Threads）方式执行：

**核心特性**：

1. **锁步执行（Lockstep）**：同一 warp 内所有线程在每个时钟周期执行相同的指令。不同的是每个线程操作的数据不同（SIMD 的推广）

2. **分支分歧（Warp Divergence）**：
   - 当 warp 内线程遇到 if-else 时，两条路径**串行执行**
   - 先执行 if 路径（不满足条件的线程 mask 掉），再执行 else 路径
   - 效率降为 50%（最坏情况 32 条不同路径则 1/32）
   - 优化：尽量让同一 warp 的线程走相同分支

3. **内存合并基本单位**：
   - 同一 warp 32 线程访问连续 128 字节 → 1 次 32B/128B 事务
   - 访问分散地址 → 多次事务，带宽浪费

4. **Warp 切换隐藏延迟**：
   - 当一个 warp 等待内存返回时（~400 cycles），SM 切换到另一个就绪 warp 执行
   - 切换开销为 **0 cycle**（所有 warp 的寄存器同时驻留，无需保存/恢复上下文）
   - 这就是 GPU 用大量线程隐藏延迟的核心机制

5. **Warp 级原语**（高效 warp 内通信）：
   - `__shfl_sync`：warp 内线程间直接交换寄存器值（无需 shared memory）
   - `__ballot_sync`：收集 warp 内所有线程的 predicate 为 32-bit mask
   - `__reduce_sync`：warp 内直接做 reduce（Hopper 新增）
   - 延迟仅 1 cycle，远快于 shared memory 中转

### Q: DP、TP-SP 的计算通信重叠原理？具体是什么通信和什么计算重叠？

**DP（Data Parallel）重叠——边计算梯度边通信**：

```
反向传播方向: Layer_N → Layer_{N-1} → ... → Layer_1

时间线：
Layer_N: [计算梯度_N] [AllReduce_N 开始]
Layer_{N-1}:            [计算梯度_{N-1}] [AllReduce_{N-1} 开始]
Layer_{N-2}:                             [计算梯度_{N-2}] ...
```

- 通信：已完成层的梯度 AllReduce
- 计算：后续层的反向传播
- 实现：PyTorch DDP 的 Gradient Bucketing——将参数分桶，一桶梯度算完就发起 AllReduce
- 效果：通信几乎完全被计算隐藏（前提是计算时间 > 通信时间）

**TP-SP（Tensor Parallel + Sequence Parallel）重叠**：

SP 将 LayerNorm/Dropout 沿序列维度切分，TP 切分线性层权重。两者转换需要集合通信：
- TP→SP 转换：**ReduceScatter**（将 TP 的全量输出归约并分片）
- SP→TP 转换：**AllGather**（将 SP 的分片收集为全量输入）

**重叠策略**：

前向：
```
AllGather(input_shard) 与 GEMM 计算重叠
↓
将 AllGather 分成多个 chunk：
chunk_0 到达 → 立即开始计算 GEMM(chunk_0)
chunk_1 到达 → 开始计算 GEMM(chunk_1)
...
通信和计算流水化
```

反向类似：ReduceScatter 与下一层的反向计算重叠。

**关键配置**：`CUDA_DEVICE_MAX_CONNECTIONS` 设为 1 确保通信和计算在同一 queue 中有序执行，避免乱序导致依赖错误。

### Q: FlashAttention 深入知识点？

**核心机制详解**：

**Tiling 策略**：
- Q 分 T_q = ceil(N/B_q) 个 outer block
- K/V 分 T_kv = ceil(N/B_kv) 个 inner block
- 对每个 Q block，遍历所有 K/V block 计算局部 attention 并累积

**Online Softmax 数学推导**：
```
处理第 j 个 K/V block 时：
  S_j = Q_block × K_j^T            # 局部 score
  m_new = max(m_old, rowmax(S_j))  # 更新 running max
  P_j = exp(S_j - m_new)           # 安全 exp
  l_new = l_old × exp(m_old - m_new) + rowsum(P_j)  # 更新 running sum
  O_new = O_old × (l_old × exp(m_old - m_new) / l_new) + P_j × V_j / l_new
```
关键洞察：之前累积的 O_old 需要乘以一个修正因子 `l_old × exp(m_old - m_new) / l_new`，因为 max 变了之后之前的 softmax 权重也变了。

**反向传播**：
- 不存储 S = QK^T 矩阵（节省 O(N²) 显存）
- 保存 logsumexp = m + log(l) 用于反向重计算
- 反向时重新前向计算 S 矩阵的对应块，用保存的 logsumexp 恢复正确的梯度

**FlashAttention-2 改进**：
1. 减少非矩阵乘 FLOPs：将 rescaling（`O × l_old/l_new`）延迟到循环结束做一次
2. 增加序列维度并行：V1 只在 batch×head 并行；V2 在 Q 的 block 间也并行
3. 优化 warp 工作划分：V1 用 split-K 需要 warp 间同步；V2 每个 warp 独立处理完整的 K/V block

**FlashAttention-3（Hopper）**：
- 利用 TMA 做异步数据搬运（解放 SM）
- Warp-specialization：producer warp 负责数据加载，consumer warp 负责计算
- FP8 Tensor Core 支持（吞吐再翻倍）
- Pipeline overlap：多级流水掩盖所有延迟

### Q: 使用流水线并行和不使用 PP 并行，显存峰值一样吗？

**不一样**，两者的显存构成有本质区别：

**不使用 PP（所有层在一张卡）**：
```
显存 = 全部参数(2N) + 全部梯度(2N) + 优化器状态(12N) + 一份激活值
     = 16N + activations(1 micro-batch)
```

**使用 PP（每卡只有 L/P 层，P 为 PP degree）**：
```
每卡参数显存 = 16N/P（只存部分层）  ← 大幅减少
额外开销 = M 个 micro-batch 的中间激活缓存  ← 额外增加
```

**PP 的额外激活开销来源**：
- 1F1B 调度中，每个 stage 需要缓存最多 P-1 个 micro-batch 的激活值（等待反向到来）
- micro-batch 数 M 越多，气泡越小（bubble = (P-1)/M），但激活缓存越大
- 可用 activation checkpoint 减少缓存（只保留每层输入，反向时重算）

**总体比较**：
- PP 的参数显存节省 = 16N × (1 - 1/P)
- PP 的额外激活开销 ≈ (P-1) × per_layer_activation × layers_per_stage
- 通常参数节省 >> 激活增加，所以 **PP 的显存峰值更低**
- 典型场景：70B 模型 PP=8，每卡只存 ~10 层参数 vs 全部 80 层

### Q: CUDA_DEVICE_MAX_CONNECTIONS 的含义？

该环境变量控制每个 GPU 设备的**最大并发 hardware work queue 数量**。

**底层机制**：
- GPU 有多个 hardware queue（channel），每个 queue 独立串行执行其中的操作
- 不同 queue 中的 kernel/memcpy 可以并发执行（如果 SM 资源允许）
- 每个 CUDA Stream 映射到一个 hardware queue

**不同设置的影响**：

| 设置 | 行为 | 适用场景 |
|---|---|---|
| =1 | 所有 stream 共享 1 个 queue，严格串行 | TP 训练（保证通信计算顺序） |
| =32（默认） | 最多 32 个并发 queue | 需要多 stream 并发的推理 |

**为什么 TP 训练设为 1**：
- TP 中 GEMM 计算和 AllReduce 通信有严格的先后依赖
- 如果多个 queue 并发，可能出现 AllReduce 在 GEMM 完成前就开始（读到不完整的数据）
- 设为 1 后所有操作严格串行，保证正确性
- 看似限制了并行度，但 TP 场景中计算和通信本就有依赖，串行是正确的

**需要计算通信重叠时**：
- 不能简单设为 1（那样无法重叠）
- 需要手动用 CUDA event 管理依赖关系
- 或使用更精细的 stream 设计确保安全重叠

### Q: Launch Bound 是什么？H2D 和 D2H 可以重叠吗？

**Launch Bound（`__launch_bounds__`）**：

```cuda
__global__ void __launch_bounds__(256, 2) kernel() { ... }
// 含义：每 block 最多 256 线程，每 SM 至少 2 个 block 驻留
```

**作用原理**：
- 编译器根据 launch_bounds 决定每个线程可用的寄存器上限
- `maxRegs = total_regs_per_SM / (maxThreads × minBlocks)`
- 如果 kernel 实际需要更多寄存器，编译器会做 register spilling（溢出到 local memory，很慢）
- 好处：控制 occupancy——确保 SM 上有足够 block 驻留以隐藏延迟

**何时使用**：
- kernel 寄存器使用过多导致 occupancy 过低时
- 明确知道最优 block 配置时
- 配合 `--maxrregcount` 编译选项使用

---

**H2D 和 D2H 可以重叠吗？—— 可以！**

GPU 通常有**独立的 DMA 引擎**：
- 1 个 H2D Copy Engine（Host → Device）
- 1 个 D2H Copy Engine（Device → Host）
- 1 个 Compute Engine（SM 执行 kernel）

**三者可以同时进行**：
```
Stream 1: [H2D copy A]  →  [Compute A]  →  [D2H copy A]
Stream 2:     [H2D copy B]  →  [Compute B]  →  [D2H copy B]

实际执行时间线：
H2D Engine:  [copy A][copy B]
Compute:          [kernel A][kernel B]
D2H Engine:             [copy A result][copy B result]
```

**实现要求**：
- 使用不同的 CUDA Stream
- 使用 **Pinned Memory**（否则 cudaMemcpyAsync 退化为同步）
- 数据分成多个 chunk 才能形成流水

### Q: 手撕：LRU Cache？

（编程题）

### Q: 手撕：Online Softmax 和 FlashAttention 伪代码？

（编程题）
