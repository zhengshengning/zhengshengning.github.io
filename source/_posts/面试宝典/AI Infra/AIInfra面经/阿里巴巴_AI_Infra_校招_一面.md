---
title: '阿里巴巴 AI Infra 校招 一面'
date: 2026-04-16 12:16:35
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Transformer为什么比RNN好？有没有Scaling Law？

**Transformer相比RNN的核心优势**：

**1. 并行计算能力**
- RNN：必须按时间步串行处理——t时刻的隐状态依赖t-1时刻，无法并行
- Transformer：attention对所有位置做矩阵运算，**所有位置的表示可以一次性并行计算**
- 实际影响：训练RNN时GPU利用率通常<30%，Transformer可达50-70%
- 这使得Transformer能充分利用GPU的大规模并行计算能力，训练吞吐量高出数倍

**2. 长距离依赖建模**
- RNN：信息需要通过隐状态链式传递——位置i到位置j的信号需要经过|i-j|步，梯度消失/爆炸使得有效建模长度通常<500 token
- LSTM/GRU通过门控缓解但不能根本解决
- Transformer：attention直接建立**任意两个位置**之间的连接，信息传递路径长度O(1)

**3. 可扩展性（Scaling）**
- Transformer性能随参数/数据/算力增长稳定提升，且提升曲线可预测
- RNN的scaling效果差——增大模型后训练困难（梯度问题加剧）、收益递减

**4. 硬件友好性**
- Transformer的核心操作是矩阵乘（GEMM）——GPU/TPU的最优化操作
- RNN涉及大量element-wise操作和状态依赖，硬件利用效率低

**Transformer的代价**：
- Attention的时间/空间复杂度O(N^2)：序列长度N增大时计算量爆炸
- 推理时自回归生成仍是串行的（训练并行，推理串行）
- 没有内建的递归归纳偏置（需要更多数据来学习顺序关系）

**Scaling Law**：

**有！** 且这正是Transformer统治的关键原因。Kaplan等2020年（OpenAI）发现：

模型性能（交叉熵loss L）与三个变量呈**幂律关系**：
```
L(N) ∝ N^(-0.076)    # N = 非embedding参数量
L(D) ∝ D^(-0.095)    # D = 训练数据token数
L(C) ∝ C^(-0.050)    # C = 训练计算量(FLOPs)
```

**关键推论**：
- 模型大小增加10倍 → loss下降约17%（0.076的幂律）
- 数据增加10倍 → loss下降约20%
- 参数和数据应**等比例增长**（Chinchilla Scaling：最优数据量 ≈ 20 × 参数量）
- 可以用小规模实验**准确预测**大规模模型的性能——这使得Scaling成为一种可靠的工程方法

**后续发展**：
- Chinchilla（2022）修正了最优计算分配策略：更多数据、更小模型性价比更高
- 推理时Scaling（Test-time Compute Scaling）：2024年发现通过增加推理时的思考步骤也能scaling提升

### Q: 推理优化中的Flash Attention和KV Cache分别是什么？

**Flash Attention — IO感知的精确注意力算法**：

**核心问题**：标准attention需要将N*N的attention矩阵写入HBM（高带宽内存），当N很大时IO成为瓶颈而非计算。

**解决方案**：通过**tiling（分块）+ 在线softmax重计算**避免物化完整attention矩阵：

```
标准Attention: Q(N,d) × K^T(d,N) → S(N,N) → softmax → P(N,N) × V(N,d) → O(N,d)
                                    ↓ 写入HBM（O(N^2)）
                                    
Flash Attention: 将Q/K/V按block分块，在SRAM中逐块计算：
  for each Q_block:
    for each K_block, V_block:
      compute local attention (small S block in SRAM)
      update running softmax (online algorithm)
      update O accumulator
  → 最终O与标准attention数学等价
```

**性能提升**：
- IO复杂度：O(N^2 * d / SRAM_size) vs 标准 O(N^2 + N*d)
- 实际加速：A100上对GPT-2 (seq=1024) 加速**2-4倍**
- 显存节省：不需要存储N*N的attention矩阵，显存从O(N^2)降为O(N)
- **不是近似算法**——数学结果与标准attention完全相同（bit-wise identical）

**Flash Attention v1 vs v2 vs v3**：
- v1：基本分块+在线softmax
- v2：优化warp级并行（减少non-matmul FLOPs、更好的work partitioning）→额外加速2x
- v3：利用H100的异步特性（TMA + WGMMA + Ping-pong scheduling）→再加速1.5-2x

**KV Cache — 避免自回归推理中的重复计算**：

**核心问题**：自回归生成时，生成第t个token需要与前面所有t-1个token做attention。如果不做缓存，每生成一个新token就要重新计算所有历史token的K和V。

**解决方案**：缓存已计算的K和V矩阵，生成新token时只计算新token的Q、K、V：
```
第t步: 
  new_q, new_k, new_v = compute(input_t)  # 只计算当前token
  K_cache.append(new_k)  # 追加到缓存
  V_cache.append(new_v)
  output = attention(new_q, K_cache, V_cache)  # 新Q与所有历史KV做attention
```

**效果**：
- 计算量从O(t^2 * d)降为O(t * d)（每步只做一次向量-矩阵乘）
- 代价：需要O(t * d * layers * 2)的显存存储KV Cache
- 对于LLaMA-70B（80层，8KV头，128维，FP16），4K序列的KV Cache约10GB

**两者的关系**：
- Flash Attention优化的是**Prefill阶段**（一次性处理完整prompt的attention计算）
- KV Cache优化的是**Decode阶段**（逐token生成时避免重复计算）
- 二者互补：Prefill用Flash Attention高效计算，Decode用KV Cache避免重复

### Q: KV Cache出现在训练中还是推理中？为什么训练中不使用？

KV Cache**只用于推理**（自回归生成阶段）。

**为什么训练中不使用？**

**1. Teacher Forcing——训练时不需要逐token生成**
- 训练时输入完整序列（如"The cat sat on the mat"），标签是右移一位的序列
- 所有位置的Q/K/V可以**一次性并行计算**——因为完整输入已知
- Attention计算：`O = softmax(Q @ K^T / √d) @ V`，这是一次完整的矩阵乘
- 不存在"生成第i个token时还不知道第i+1个token"的情况

**2. 训练的因果性由Causal Mask保证**
```python
# 训练时的Causal Attention（并行计算所有位置）
mask = torch.tril(torch.ones(N, N))  # 下三角mask
attn = softmax(Q @ K.T / sqrt(d) + mask * (-inf))  @ V
# 所有位置一次完成，mask确保位置i只能attend到≤i的位置
```
- Causal Mask让位置i只能看到位置≤i的KV——等效于自回归的信息约束
- 但**计算**是完全并行的——所有Q同时与对应范围的K/V做attention

**3. 效率对比**
- 训练（并行）：一次GEMM计算所有位置，时间O(N^2 * d)，GPU利用率高
- 如果训练也逐token（用KV Cache）：需要N步串行计算，每步O(N * d)，总时间相同但GPU利用率极低
- 结论：KV Cache是为解决推理时的串行约束设计的，训练没有这个约束

**4. 反向传播的需求**
- 训练需要保存中间激活用于反向传播
- KV Cache的设计是为了避免存储中间结果（空间换时间），与训练的需求矛盾

### Q: 注意力机制的优化方法有哪些？

注意力优化是LLM推理效率的核心，可从**计算、存储、精度**三个维度分类：

**计算优化（减少FLOPs）**：

| 方法 | 原理 | 复杂度 | 精度 |
|------|------|--------|------|
| 标准Attention | 全量Q×K计算 | O(N^2*d) | 精确 |
| 稀疏Attention | 只计算部分位置对 | O(N*k*d) | 近似 |
| Linear Attention | 核函数近似softmax | O(N*d^2) | 近似 |

- **稀疏Attention的具体形态**：
  - Sliding Window（Mistral）：只关注局部窗口内的token
  - Block Sparse：预定义稀疏pattern（如BigBird的local+global+random）
  - Dynamic Sparse：根据Q/K的内积动态选择重要位置（如Reformer的LSH）
- **Linear Attention**：
  - 将softmax(QK^T)V分解为φ(Q)(φ(K)^T V)，改变计算顺序后复杂度降为O(N*d^2)
  - 代表：Performer（Random Feature）、RWKV/Mamba（RNN-like linear recurrence）
  - 缺点：近似质量通常不如精确attention

**IO/显存优化（不改变数学结果）**：

- **Flash Attention**：分块计算+在线softmax，减少HBM读写
  - 硬件亲和：利用SRAM（19TB/s）代替HBM（2TB/s）做中间计算
  - 是目前最重要的attention优化，已成为标配
- **PagedAttention（vLLM）**：
  - 问题：不同请求的KV Cache长度不同→预分配连续内存造成严重碎片（浪费60-80%显存）
  - 方案：类似OS虚拟内存的分页管理——KV Cache按固定大小page分配，逻辑连续但物理不连续
  - 效果：显存利用率从20-40%提升到>95%，同等显存可服务更多请求
- **CUDA Graph**：将attention kernel的launch开销固化（对小batch/短序列的decode特别有效）

**KV Cache压缩（减少存储量）**：

- **GQA/MQA**：减少KV head数量（结构化减少）
  - MQA: 1个KV head → 32倍压缩（但质量损失较大）
  - GQA-8: 8个KV head → 4倍压缩（质量接近MHA）
- **MLA**：低秩投影压缩KV到latent空间
  - 压缩比可达10-20倍（latent_dim << num_heads * head_dim）
- **KV Cache量化**：
  - INT8量化：每个KV值从FP16(2B)→INT8(1B)，2倍压缩
  - INT4量化：4倍压缩，但质量损失明显（per-channel quantization缓解）
  - FP8：Hopper GPU原生支持，质量好于INT8
- **驱逐策略**：
  - H2O：保留attention score累计最高的token（Heavy Hitter）
  - Scissorhands：利用attention pattern跨层一致性做驱逐决策
  - 保留attention sink（前几个token）+ 近期token + 重要token
- **前缀共享（Prefix Caching）**：
  - 相同system prompt的多个请求共享KV Cache
  - 节省大量重复计算和存储（system prompt通常占输入的50%+）
  - vLLM/SGLang的自动Prefix Caching
- **Offload**：
  - 将不活跃的KV Cache卸载到CPU内存/NVMe SSD
  - 需要时异步预取到GPU
  - 适合超长序列但低频访问的场景

### Q: 手撕：接雨水？

（编程题）
