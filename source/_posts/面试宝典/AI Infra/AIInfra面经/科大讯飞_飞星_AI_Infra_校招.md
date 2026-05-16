---
title: '科大讯飞 飞星 AI Infra 校招'
date: 2026-04-16 12:09:03
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 面经]
---


<!-- more -->

---

### Q: NVIDIA 的卡对 FP8 有哪些独特的优化？

Hopper 架构（H100）是首款原生支持 FP8 计算的 GPU，做了全方位的硬件优化：

**1. Tensor Core 原生支持**：
- FP8 矩阵乘法直接在第四代 Tensor Core 上执行
- 吞吐对比：FP8 = 1978 TFLOPS，FP16 = 989 TFLOPS——**2x 吞吐提升**
- 支持 FP8 × FP8 → FP16/FP32 的混合精度累加，保证数值精度

**2. Transformer Engine**：
- NVIDIA 提供的软件层，自动管理 FP8 训练中的**动态缩放（Dynamic Scaling）**
- 根据激活/权重/梯度的统计信息自动选择最优格式：前向用 E4M3（精度更高），反向用 E5M2（范围更大）
- Per-tensor 粒度的 scale factor，每层独立维护
- 与 PyTorch/JAX 深度集成，用户代码几乎不需修改

**3. 硬件 Format 转换**：
- 支持 FP8 ↔ FP16/BF16/FP32 的高效类型转换
- 转换通过专用硬件单元完成，不占用 CUDA Core/Tensor Core

**4. 延迟缩放（Delayed Scaling）**：
- 传统做法：当前 step 计算 amax → 计算 scale → 量化。这需要额外的同步操作
- 延迟缩放：利用**上一步**的 amax 统计进行当前步的缩放
- 减少了实时统计的同步开销，对训练精度影响可忽略（相邻 step 的分布变化很小）

**5. TMA（Tensor Memory Accelerator）配合**：
- TMA 可以在数据从 HBM 搬到 SRAM 时自动做格式转换
- FP8 数据在 HBM 中占用更少带宽，加载时由 TMA 转为 FP16 计算

---

### Q: FP8 的两种格式（E4M3 和 E5M2）有什么区别？分别怎么计算？

| 属性 | E4M3 | E5M2 |
|---|---|---|
| 指数位 | 4 | 5 |
| 尾数位 | 3 | 2 |
| 偏置（bias） | 7 | 15 |
| 最大正值 | 448 | 57344 |
| 最小正 subnormal | 2^(-9) ≈ 0.00195 | 2^(-16) ≈ 0.0000153 |
| 动态范围 | ~10^2.7 | ~10^4.8 |
| 精度级数 | 8 级（2^3） | 4 级（2^2） |
| 特殊值 | 有 NaN，无 Inf | 有 NaN 和 Inf |
| 适用场景 | 前向（权重、激活） | 反向（梯度） |

**计算方式**（与标准 IEEE 浮点相同）：
```
value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa/2^m)

E4M3 示例：bits = 0_0110_101
  sign = 0, exp = 6, mantissa = 5/8
  value = 2^(6-7) × (1 + 5/8) = 0.5 × 1.625 = 0.8125

E5M2 示例：bits = 0_01110_10
  sign = 0, exp = 14, mantissa = 2/4
  value = 2^(14-15) × (1 + 0.5) = 0.5 × 1.5 = 0.75
```

**为什么分两种格式**：
- **前向用 E4M3**：权重和激活的分布相对集中，需要更高精度（8 级尾数）来区分相近的值，动态范围要求不高
- **反向用 E5M2**：梯度分布跨度大（尤其是深层网络），需要更大动态范围防止下溢/上溢，精度要求可以放低（梯度方向比精确值更重要）

---

### Q: 如何计算 FP8 的 KV Cache 大小？

**公式**：
```
KV Cache = 2(K和V) × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes_per_element
```

**具体计算示例**（LLaMA-70B：80 层、8 KV heads（GQA）、128 head_dim）：

| 配置 | 计算 | 结果 |
|---|---|---|
| FP16, bs=1, seq=4096 | 2×80×8×128×4096×1×2 | **1.0 GB** |
| FP8, bs=1, seq=4096 | 2×80×8×128×4096×1×1 | **0.5 GB** |
| FP8, bs=16, seq=4096 | 2×80×8×128×4096×16×1 | **8.0 GB** |
| FP8, bs=1, seq=32768 | 2×80×8×128×32768×1×1 | **4.0 GB** |

**FP8 KV Cache 的意义**：
- 相比 FP16 减少 **50% 显存**
- 可以支持 2x 更长序列或 2x 更大 batch
- 对精度影响：attention score 的精度损失约 0.1-0.3 PPL，多数场景可接受
- 进一步优化：结合 GQA（减少 head 数）+ FP8（减少每个元素字节数），KV Cache 可以降低到 MHA+FP16 的 1/16

---

### Q: H100 有什么新设计，和 L40 有什么不一样？

**H100（Hopper 架构）核心新特性**：

| 特性 | 说明 |
|---|---|
| FP8 Tensor Core | 原生支持，算力 1978 TFLOPS（FP8），是 A100 FP16 的 6x |
| TMA | Tensor Memory Accelerator，硬件异步数据搬运，解放 SM 不再做地址计算和搬运 |
| DSMEM | 分布式共享内存，SM 间直接访问对方 shared memory，无需经过 L2 |
| Thread Block Cluster | 多个 Block 组成 Cluster 协同工作，共享 DSMEM |
| NVLink 4.0 | 900 GB/s 双向带宽（A100 为 600 GB/s） |
| HBM3 | 80 GB，3.35 TB/s 带宽（A100 HBM2e 为 2.0 TB/s） |
| 132 SM | 比 A100（108 SM）多 22% |

**H100 vs L40 对比**：

| 维度 | H100 (Hopper) | L40 (Ada Lovelace) |
|---|---|---|
| 定位 | 数据中心训练+推理 | 推理/图形工作站 |
| 显存类型 | HBM3 80GB | GDDR6X 48GB |
| 显存带宽 | 3.35 TB/s | 864 GB/s |
| NVLink | 支持（900 GB/s） | **不支持** |
| FP8 算力 | 1978 TFLOPS | 362 TFLOPS |
| FP16 算力 | 989 TFLOPS | 181 TFLOPS |
| TDP 功耗 | 700W | 300W |
| 价格 | ~$30,000+ | ~$7,000 |
| 适用场景 | 大规模训练/推理 | 单卡推理/渲染 |

**核心差异总结**：L40 无 HBM（带宽只有 H100 的 1/4）、无 NVLink（不能高速多卡互联）、算力远低，但功耗和成本也低很多。L40 适合**单卡推理**小模型，H100 适合**多卡训练**和大模型推理。

---

### Q: 最近推理方面有什么新的 tricks？

**2024-2025 推理优化前沿方向**：

**解码加速**：
- **投机解码（Speculative Decoding）**：小模型草稿 + 大模型验证，不改变输出分布的前提下加速 2-3x
- **EAGLE/EAGLE-2**：特征级别投机，用隐藏状态预测下一token特征而非直接预测token，接受率更高
- **Medusa**：在最后一层加多个预测头，并行预测多个未来 token
- **Lookahead Decoding**：利用 Jacobi 迭代并行解码

**KV Cache 优化**：
- **MLA（Multi-head Latent Attention）**：DeepSeek 的低秩 KV 压缩，压缩比 5-10x
- **StreamingLLM**：attention sink + 滑动窗口，支持无限长度推理
- **H2O（Heavy Hitter Oracle）**：动态保留高累积 attention score 的 token KV
- **SGLang RadixAttention**：前缀树自动复用公共前缀的 KV Cache

**系统级优化**：
- **PD 分离 + Chunked Prefill**：Prefill/Decode 异构部署 + chunk 交错执行
- **MoE + Expert Parallel**：稀疏激活降低计算量，EP 实现专家分布式部署
- **FP8 推理**：Hopper 原生支持，吞吐翻倍，精度损失可控
- **DeepEP**：DeepSeek 开源的高性能 MoE All-to-All 通信库

**量化/压缩**：
- **FP4/NF4**：4-bit 浮点量化，比 INT4 更适配正态分布权重
- **W4A8**：权重 INT4 + 激活 INT8，兼顾压缩率和计算加速
