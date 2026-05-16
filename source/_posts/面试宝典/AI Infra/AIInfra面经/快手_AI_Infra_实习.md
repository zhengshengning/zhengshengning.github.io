---
title: '快手 AI Infra 实习'
date: 2026-04-16 12:15:31
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 除了你做过的量化方法，还知道其他量化方法吗？

**主流量化方法全景**：

**PTQ（训练后量化）—— 无需重新训练**：
- **GPTQ**：逐列量化 + Hessian-based 误差补偿，精度最高的 PTQ 方案。按列顺序量化，每列误差分摊到未量化列。耗时较长（需计算 Hessian）
- **AWQ**：激活感知权重量化，保护与大激活对应的重要权重 channel。简单快速，精度接近 GPTQ
- **SmoothQuant**：通过 per-channel smooth 将激活 outlier 迁移到权重，实现 W8A8。适合需要激活也量化的场景
- **QuIP/QuIP#**：基于不相关投影（Incoherence Processing），先对权重做随机旋转使分布更均匀，再量化。2-bit 下精度显著优于 GPTQ
- **GGUF（llama.cpp）**：支持 Q2_K 到 Q8_0 多种格式，混合 per-block 和 per-super-block 的 scale，面向 CPU 推理优化

**QAT（量化感知训练）—— 训练中模拟量化**：
- 在前向中插入 fake-quantize 节点（quantize → dequantize），让模型学会适应量化误差
- 精度比 PTQ 高 0.3-1.0 PPL，但需要完整训练流程，成本高
- 适合对精度要求极高的生产场景

**动态量化**：
- 推理时根据实际输入数据的 min/max 动态计算 scale
- 无需校准数据，但每次推理有额外的统计开销
- PyTorch 的 `torch.quantization.quantize_dynamic` 即此方式

**混合精度量化**：
- 不同层/不同 tensor 使用不同比特数
- 对量化敏感的层（如 first/last layer、attention output）保持高精度
- 自动搜索最优 bit 分配（如 Mixed-bit quantization）

**FP8 量化**：
- E4M3（前向）/ E5M2（反向）浮点格式
- Hopper 架构原生 Tensor Core 支持，无需 dequant 即可直接计算
- Transformer Engine 自动管理 scale

---

### Q: 对于量化误差而言，数据应该怎样分布较好？

**对均匀量化（INT8/INT4）而言：数据均匀分布最好**。

原因分析：
- 均匀量化将 [min, max] 区间等分为 2^b 个 bin
- 每个 bin 的宽度 = (max - min) / (2^b - 1)
- 如果数据集中在某个小范围，大部分 bin 是空的（浪费），少数 bin 承载所有数据（有效精度低）
- 数据均匀铺满 [min, max] 时，每个 bin 承载等量数据，量化误差（= bin 宽度/2）被最小化

**直觉**：量化本质是用有限的"格点"近似连续值。格点等距分布时，希望数据也等距分布，这样每个数据到最近格点的距离最小。

**对于有异常值的情况**：
- 异常值将 [min, max] 撑得很大，大部分数据挤在很小的区间内
- 解决方案：clip（截断异常值）、per-group 量化（小范围内统计）、或使用非均匀量化格式

---

### Q: 量化到 FP4 时为什么数据均匀分布不一定最好？

**FP4 是非均匀量化格式**，其可表示的值是对数间隔的（不是等间距）：

```
FP4 可表示的正值（E2M1 为例）：
0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
   ↑    ↑    ↑    ↑    ↑    ↑    ↑
  靠近0的格点更密集，远离0的更稀疏
```

**为什么均匀分布不是最好**：
- FP4 在 0 附近有更多可表示的值（bin 密集），远离 0 处 bin 稀疏
- 如果数据均匀分布在 [-max, max]，远端有很多数据但 bin 少，量化误差大
- 如果数据接近正态分布（大部分集中在 0 附近），数据分布恰好匹配 bin 密度，误差最小

**总结**：
| 量化格式 | 最优数据分布 | 原因 |
|---|---|---|
| INT4/INT8（均匀量化） | 均匀分布 | bin 等距，数据等距最好 |
| FP4/FP8（浮点量化） | 正态/对数正态分布 | bin 靠近0密集，数据也应集中在0附近 |
| NF4（NormalFloat4） | 标准正态分布 | 格点按正态分位数设计 |

**NF4 的巧妙设计**：QLoRA 中使用的 NF4 格式，其 16 个可表示值恰好是标准正态分布的 16 个等概率分位数。这使得对于正态分布的权重，量化误差在信息论意义上最优。

---

### Q: 如何优化 GEMM？Shared Memory 大小怎么取？

**GEMM 优化——六层次递进**：

对于 C[M,N] = A[M,K] × B[K,N]：

**1. Tiling（分块计算）**：
- 每个 thread block 负责输出矩阵 C 的一个 BM×BN 的 tile
- 从 A 和 B 分别加载 BM×BK 和 BK×BN 的 tile 到 shared memory
- 沿 K 维循环，每次加载一个 BK 宽的 tile

**2. 双缓冲（Double Buffering）**：
- 分配两份 shared memory buffer（buf0 和 buf1）
- 当 SM 在计算 buf0 中的数据时，DMA 异步加载下一个 tile 到 buf1
- 下一轮切换：计算 buf1，加载到 buf0
- 效果：隐藏 global→shared memory 的加载延迟（约 200-400 cycles）

**3. 向量化访存**：
- 使用 `float4`（128-bit）一次加载 4 个 float，减少指令数和内存事务数 4x
- 确保加载地址对齐 16 字节

**4. 寄存器 Tiling**：
- 每个线程负责输出 TM×TN 个元素（如 8×8=64 个）
- 累加结果保存在寄存器中（register file 带宽无限大）
- 数据复用：从 shared memory 读一个 A 行/B 列，用于计算多个输出元素

**5. 避免 Bank Conflict**：
- Shared memory 有 32 个 bank（4 字节/bank 交错）
- 同一 warp 访问同一 bank 的不同地址会串行化
- 解决：padding（如 `float smem[BM][BK+1]`）或 swizzle

**6. 利用 Tensor Core**：
- WMMA API：`wmma::mma_sync` 执行 16×16×16 的矩阵乘加
- MMA PTX 指令：更底层控制，灵活度更高
- 条件：矩阵维度需对齐 Tensor Core 要求（16 的倍数）

---

**Shared Memory 大小选取原则**：

受限条件：
- 每个 SM 的 shared memory 总量（A100 最多 164KB 可配置为 shared）
- 需要平衡 tile 大小与 SM 上活跃 block 数（occupancy）

计算方式：
```
shared_memory_per_block = (BM × BK + BK × BN) × sizeof(element) × buffer_count

示例：BM=BN=128, BK=32, FP16, 双缓冲
= (128×32 + 32×128) × 2 × 2 = 32 KB/block

164KB SM / 32KB = 最多 5 个 block/SM（但还受寄存器限制）
```

**经验值**：
- BM=BN=128，BK=8-32 是常见起点
- 追求高 occupancy：减小 tile 让更多 block 驻留
- 追求高复用率：增大 tile 但 occupancy 降低
- 最优点通过 benchmark 确定（通常 occupancy 50-75% 时性能最好）

---

### Q: 手撕 CUDA：实现一个 Norm 算子（LayerNorm）？

（编程题）
