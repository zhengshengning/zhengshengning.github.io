---
title: '蚂蚁 AI Infra 校招 一面'
date: 2026-04-16 12:10:26
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: MLA 为什么比 MHA 好？

MLA（Multi-head Latent Attention）是 DeepSeek-V2/V3/R1 提出的注意力机制，其核心思想是**将 KV Cache 投影到低维 latent 空间存储**，推理时通过 up-projection 恢复。相比 MHA 的优势主要体现在以下维度：

**1. KV Cache 大幅减少（核心优势）**：

| 注意力方式 | KV Cache 维度/token | 典型值（128 heads × 128 dim） |
|---|---|---|
| MHA | 2 × num_heads × head_dim | 2 × 128 × 128 = 32768 |
| GQA (8 KV heads) | 2 × num_kv_heads × head_dim | 2 × 8 × 128 = 2048 |
| MLA | latent_dim + rope_dim | 512 + 64 = 576 |

MLA 相比 MHA 压缩了约 **57 倍**，相比 GQA 还能再压缩 **3-4 倍**。这意味着同等显存下可以支持更多并发请求或更长序列。

**2. 精度保持更好**：
- GQA 通过简单的 KV head 共享来减少 cache，本质是"丢信息"——多个 Q head 被迫使用相同的 K/V
- MLA 通过**可学习的低秩投影**压缩，latent vector 保留了最有信息量的方向（类似 PCA 保留主成分）
- 实验证明 MLA 在同等压缩比下精度损失更小

**3. 推理显存释放**：
- 长序列 + 大 batch 时 KV Cache 是主要显存瓶颈（如 LLaMA-70B，bs=16，seq=4096 时 KV Cache 约 16 GB）
- MLA 直接从根源减少 KV Cache 大小，使得同等显存可支持 3-4 倍的并发量

**4. 支持权重吸收（Weight Absorption）**：
```
标准计算：Q × K^T = Q × (W_up_K × c_kv)^T
权重吸收：(Q × W_up_K^T) × c_kv^T = Q_new × c_kv^T
```
推理时可以将 up-projection 矩阵吸收进 Q_proj 和 O_proj 中，无需在推理时显式执行 up-projection，直接对 latent vector 做 attention 计算，进一步减少计算量。

**5. 与 GQA 的本质差异**：
- GQA：在 head 维度做"硬共享"（discrete grouping）
- MLA：在特征空间做"软压缩"（continuous projection）
- 类比：GQA 像把多张图片缩略为几张代表图，MLA 像对所有图片做 PCA 压缩

---

### Q: MLA 中权重吸收会遇到什么问题？

权重吸收是 MLA 实现高效推理的关键技巧，但实际工程中会遇到几个挑战：

**1. RoPE 不兼容问题（最核心的挑战）**：

RoPE 需要对 K 的每个位置施加与位置相关的旋转矩阵：
```
K_pos = RoPE(K, position)
```
但 MLA 缓存的是压缩后的 latent vector `c_kv`，它是 K 和 V 的混合低维表示。问题在于：
- RoPE 旋转是**位置相关的**，每个位置的旋转矩阵不同
- 如果先压缩再施加 RoPE，则无法将 W_up_K 吸收（因为 RoPE 矩阵在 W_up_K 和 c_kv 之间，阻断了吸收路径）
- 如果先施加 RoPE 再压缩，则每个位置的 K 不同，无法共享压缩矩阵

**DeepSeek 的解决方案——分离 RoPE 维度**：
```
K = [K_nope (压缩到 latent, 无 RoPE) | K_rope (不压缩, 施加 RoPE)]
```
- 大部分维度（如 448-d）通过 latent 压缩存储，不施加 RoPE
- 少量维度（如 64-d）不压缩，直接缓存并施加 RoPE
- 最终 KV Cache = latent_dim + rope_dim（如 512 + 64 = 576 dim/token）

**2. 吸收后矩阵维度变化**：
```
原始: Q [bs, seq, num_heads, head_dim] × W_up_K [latent_dim, num_heads × head_dim]
吸收后: Q_new [bs, seq, num_heads, latent_dim] = Q × W_up_K^T
```
- 吸收后的 Q_new 维度可能变为 `[num_heads, latent_dim]`，如果 latent_dim 较大，Q_new 的计算量反而增加
- 需要精心设计 latent_dim 使得总计算量仍有收益
- 实践中 latent_dim 通常设为 head_dim 的 2-4 倍（如 512 vs 128），但因为消除了 up-projection 的在线计算，总体仍有收益

**3. 数值精度累积**：
- 原始路径：Q × (W_up × c_kv)^T——两次矩阵乘
- 吸收路径：(Q × W_up^T) × c_kv^T——同样两次矩阵乘，但中间结果维度不同
- 浮点运算的结合律不成立：`(A×B)×C ≠ A×(B×C)` 在浮点数下
- 实践中精度差异很小（< 1e-5），但在混合精度训练中需要验证数值一致性

**4. 多头并行效率**：
- 吸收后每个 head 的 Q_new 维度变为 latent_dim（可能比原始 head_dim 大）
- 这改变了 GEMM 的形状，可能不是硬件最优的 tile 大小
- 需要针对吸收后的矩阵形状重新调优 kernel

---

### Q: KV Cache 的离线计算与不常用 KV Cache 的卸载加载？

**离线计算（Prefix Caching / Prompt Caching）**：

核心观察：很多请求共享相同的前缀（如 system prompt、few-shot examples），这些前缀对应的 KV Cache 是确定性的（相同输入 → 相同 KV）。

**离线计算流程**：
```
离线阶段:
  常用 prompt → Prefill 计算 → KV Cache → 序列化存储到 SSD/CPU 内存

在线阶段:
  新请求到来 → 检查前缀是否命中缓存
    → 命中: 加载预计算的 KV Cache (跳过 Prefill, TTFT 大幅降低)
    → 未命中: 正常 Prefill 计算
```

**收益分析**：
- System prompt 通常 500-2000 tokens，Prefill 计算 ~50-200ms
- 缓存命中后 TTFT 减少到加载时间（PCIe/NVLink 传输 ~5-20ms）
- 在多轮对话场景（历史上下文相同）收益尤其大

**实际实现**：
- **vLLM Automatic Prefix Caching**：自动检测请求间的公共前缀，共享 KV Cache block
- **SGLang RadixAttention**：用前缀树（Radix Tree）索引所有缓存的 KV，O(n) 最长前缀匹配
- **磁盘持久化**：对超长 system prompt（如 RAG 的文档上下文），可将 KV Cache 持久化到 SSD

---

**卸载/加载（KV Cache Offloading）**：

**动机**：在高并发场景下，GPU 显存不足以容纳所有活跃请求的 KV Cache。需要一种机制将暂时不活跃的请求"换出"。

**两种策略**（vLLM Preemption）：

| 策略 | 机制 | 恢复延迟 | 适用场景 |
|---|---|---|---|
| **Swap** | KV Cache 通过 PCIe 传输到 CPU 内存 | ~10-50ms（取决于大小） | CPU 内存充足、请求暂停时间短 |
| **Recompute** | 释放 KV Cache，恢复时重新 Prefill | ~50-500ms（取决于序列长度） | CPU 内存也紧张、序列较短 |

**Swap 的实现细节**：
```
Preempt（显存不足）:
  1. 选择优先级最低的请求（通常 FCFS：最早到达的先抢占）
  2. 异步将其 KV Cache blocks 从 GPU → CPU（pinned memory）
  3. 释放 GPU 上的 physical blocks
  4. 请求状态标记为 SWAPPED

Resume（显存释放）:
  1. 检查 SWAPPED 队列
  2. 异步将 KV Cache blocks 从 CPU → GPU
  3. 恢复请求状态为 RUNNING
```

**性能考量**：
- PCIe 4.0 带宽 ~32 GB/s（双向），单个请求的 KV Cache（如 256 MB）swap 耗时 ~8ms
- 通过 CUDA Stream 异步传输，可以与其他请求的 decode 计算重叠
- NVLink 连接的 CPU-GPU 环境（如 Grace Hopper）带宽更高（~900 GB/s），swap 成本极低

---

### Q: 还有哪些 KV Cache 优化的 tricks？

KV Cache 优化是 LLM 推理优化的核心方向之一，方法可分为多个层次：

**1. 窗口/稀疏策略（减少存储的 token 数）**：

| 方法 | 策略 | 保留 token 数 | 精度影响 |
|---|---|---|---|
| Sliding Window | 只保留最近 W 个 token 的 KV | W（固定） | 丢失早期上下文 |
| StreamingLLM | Attention Sink（前 4 个） + 滑动窗口 | 4 + W | 支持无限长推理 |
| H2O | 保留累积 attention score 最高的 token（Heavy Hitter） | Top-K + 最近窗口 | 20% token 保 95%+ 精度 |
| Scissorhands | 跨层一致的重要性判断，一次性驱逐 | 动态 | 高效全局决策 |

**StreamingLLM 的关键发现**：
- 前几个 token（称为 attention sink）无论语义是否重要，都吸收了大量 attention score
- 原因：softmax 需要一个"汇聚点"来分配剩余概率质量
- 去掉 attention sink 会导致生成质量崩溃，保留后即可稳定生成

**2. Token 合并/压缩（减少每个 token 的表示量）**：
- **Token Merging**：合并相似 token 的 KV 表示（如将语义接近的相邻 token 平均）
- **分层压缩**：底层（靠近输入）保留更多 KV，高层（靠近输出）更积极压缩
  - 直觉：底层 attention 更分散，高层 attention 更集中在少数 token
- **渐进精度**：最近 token FP16，中等距离 FP8，远处 INT4

**3. 前缀复用（减少重复计算）**：
- **Prefix Caching**（vLLM）：相同 system prompt 的请求共享 KV Cache blocks
- **RadixAttention**（SGLang）：用前缀树管理所有缓存的 KV，自动匹配最长公共前缀
  ```
  Radix Tree:
    "You are a helpful assistant" → [KV blocks 0-31]
      ├── "Answer in JSON" → [KV blocks 32-35]
      └── "Answer in Chinese" → [KV blocks 32-36]
  ```
- **Multi-turn 缓存**：多轮对话中，前轮的 KV 不释放，新轮只增量计算新 token 的 KV

**4. 投机解码配合**：
- Draft model 生成候选 token 时使用自己的小 KV Cache
- 只有 verify 通过的 token 才追加到 target model 的 KV Cache
- 如果 draft model 被 reject，不需要回滚 target model 的 KV Cache（因为还没写入）

**5. 量化（减少每个元素的存储位宽）**：
- **FP8 KV Cache**：减少 50% 存储，精度损失极小（PPL +0.1-0.2）
- **INT4 KV Cache**：减少 75%，需要 per-head/per-token scale，精度需仔细验证
- **KV Cache 量化 vs 权重量化的区别**：KV Cache 是动态生成的，scale 需要在线计算（dynamic quantization），而权重 scale 可以离线预计算

**6. 架构级优化（训练时决定）**：
- **GQA**：减少 KV head 数量（LLaMA-2/3、Qwen-2）
- **MLA**：低秩 latent 压缩（DeepSeek-V2/V3/R1）
- **Cross-layer KV sharing**：相邻层共享 KV Cache（部分 token 无需每层重算 KV）

**综合实践建议**：
```
显存极度紧张 → MLA/GQA（架构级） + INT4 KV Cache + H2O 稀疏
显存中等紧张 → GQA + FP8 KV Cache + Prefix Caching
显存充足追求吞吐 → GQA + Prefix Caching + Continuous Batching
```
