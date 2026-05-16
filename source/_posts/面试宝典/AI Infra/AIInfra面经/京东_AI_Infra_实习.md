---
title: '京东 AI Infra 实习'
date: 2026-04-16 11:45:38
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 训练优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 介绍一下 Qwen 系列模型的架构和训练方法？

Qwen 系列基于 Decoder-only Transformer 架构，在原始 Transformer 的基础上做了多项现代化改进：

**架构细节**：
- **位置编码**：采用 RoPE（Rotary Position Embedding），通过旋转矩阵对 Q/K 注入位置信息，天然支持长度外推。Qwen2 通过 YaRN 插值扩展到 128K 上下文
- **激活函数**：FFN 使用 SwiGLU（Swish + Gated Linear Unit），相比 ReLU 在同等参数量下效果更好，结构为 `SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)`
- **归一化**：Pre-RMSNorm，去掉均值中心化只保留缩放，减少计算量约 15%
- **注意力**：Qwen2 引入 GQA（Grouped Query Attention），将 KV head 数减少到 Q head 的 1/4-1/8，显著降低 KV Cache 开销。例如 Qwen2-72B 使用 64 个 Q head + 8 个 KV head
- **MoE 扩展**：Qwen2.5-MoE 使用稀疏 MoE 架构，总参数量大但每 token 只激活部分 Expert，实现更高效率

**训练流程**：
1. **预训练**：在数万亿 token 的多语言多模态语料上做 next-token prediction，使用 BF16 混合精度 + 3D 并行
2. **SFT（Supervised Fine-Tuning）**：用高质量指令数据训练模型遵循指令
3. **RLHF 对齐**：Qwen2.5 使用 GRPO（Group Relative Policy Optimization）替代传统 PPO，降低训练资源需求

---

### Q: PPO、DPO 和 GRPO 的区别？

这三种方法都是 LLM 对齐（Alignment）的策略优化算法，但复杂度和资源需求递减：

**PPO（Proximal Policy Optimization）**：
- 需要**四个模型**同时在线：policy model（正在训练的策略）、reference model（初始策略，用于 KL 约束）、reward model（打分）、critic model（估计状态价值）
- 通过 GAE（Generalized Advantage Estimation）计算优势函数 A_t
- 使用 clip 机制约束策略更新幅度：`L = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)`，防止策略突变
- 训练开销最大（4 个模型的显存 + critic 需要多步回报估计），但理论上最通用

**DPO（Direct Preference Optimization）**：
- 核心创新：证明 reward model 可以用 policy 本身的 log probability 隐式表达
- 只需**两个模型**：policy 和 reference（且 reference 只做前向，可冻结）
- 损失函数直接基于偏好对 (chosen, rejected)：`L = -log σ(β(log π/π_ref(chosen) - log π/π_ref(rejected)))`
- 优点：训练简单，无需单独训练 reward model
- 缺点：严重依赖偏好数据质量；无法像 PPO 那样在线探索新策略

**GRPO（Group Relative Policy Optimization）**：
- DeepSeek 提出，**去掉 critic model**
- 对同一 prompt 采样一组回答（如 8-16 个），用外部 reward model（或规则）给每个回答打分
- 用组内奖励的相对排名计算优势函数：`A_i = (r_i - mean(r)) / std(r)`，无需 critic 估计基线
- 显存需求：只需 policy + reference + reward（reward 可以是规则，不占模型显存）
- 特别适合可验证任务（数学、代码）：用结果正确性做 reward，无需训练 reward model

| 维度 | PPO | DPO | GRPO |
|---|---|---|---|
| 模型数量 | 4 | 2 | 2-3 |
| 数据需求 | 在线采样 | 离线偏好对 | 在线采样+规则reward |
| 训练稳定性 | 需要调参 | 较稳定 | 较稳定 |
| 典型应用 | InstructGPT | Zephyr, LLaMA2-Chat | DeepSeek-R1 |

---

### Q: 熵、交叉熵和 KL 散度的联系？

这三个概念是信息论的核心度量，在机器学习中无处不在：

**熵 H(P)**：
- 公式：`H(P) = -Σ P(x) log P(x)`
- 含义：分布 P 自身的不确定性/信息量。均匀分布熵最大，确定性分布熵为 0
- 直觉：最优编码下的平均码长

**交叉熵 H(P, Q)**：
- 公式：`H(P,Q) = -Σ P(x) log Q(x)`
- 含义：用分布 Q 编码服从分布 P 的数据时的平均编码长度
- 在分类任务中：P 是 one-hot 标签，Q 是 softmax 输出，H(P,Q) = -log Q(正确类)

**KL 散度 D_KL(P||Q)**：
- 公式：`D_KL(P||Q) = Σ P(x) log(P(x)/Q(x)) = H(P,Q) - H(P)`
- 含义：用 Q 近似 P 所产生的额外信息损失（非对称！）
- 性质：D_KL ≥ 0，当且仅当 P = Q 时取等

**三者关系**：
```
交叉熵 = 熵 + KL 散度
H(P,Q) = H(P) + D_KL(P||Q)
```

当 P 固定时（如分类任务中标签固定），最小化交叉熵等价于最小化 KL 散度。这就是为什么分类任务用交叉熵损失等同于让模型输出分布逼近真实分布。

---

### Q: 讲一下 DeepSpeed、DDP 和 FlashAttention？

这三者解决的是大模型训练中不同层面的问题：

**DDP（DistributedDataParallel）**：
- PyTorch 原生数据并行方案，每张卡持有**完整模型副本**
- 各卡处理不同 mini-batch，前向独立计算
- 反向时通过 Ring AllReduce 同步梯度（Gradient Bucketing：将参数分桶，边算梯度边通信，实现重叠）
- 通信量：每次 AllReduce 传输 2 × model_size（与卡数无关）
- 局限：模型必须放得进单卡显存。对于 7B+ 模型，单卡 Adam 优化器状态就需要 ~12×参数量 bytes ≈ 84 GB

**DeepSpeed ZeRO（Zero Redundancy Optimizer）**：
- 解决 DDP 中每张卡都冗余存储优化器状态/梯度/参数的问题

| 级别 | 分片内容 | 每卡显存（Adam + FP16，P 卡） | 通信量 |
|---|---|---|---|
| ZeRO-1 | 优化器状态 | 4N + 12N/P | 同 DDP |
| ZeRO-2 | +梯度 | 2N + 14N/P | 同 DDP |
| ZeRO-3 | +参数 | 16N/P | 1.5x DDP |

- ZeRO-Offload：将优化器状态/参数 offload 到 CPU/NVMe，进一步突破 GPU 显存限制
- 实际效果：ZeRO-3 可在 8 张 A100 上训练 30B+ 模型

**FlashAttention**：
- 解决标准 Attention 的内存瓶颈（O(N^2) 显存存储 attention 矩阵）
- 核心技术：**Tiling + Online Softmax**
  - 将 Q/K/V 分块加载到 GPU SRAM（~20 MB）中计算
  - 使用 Online Softmax 增量更新：维护 running max `m` 和 running sum `l`，每处理一个 K/V block 更新一次
  - `m_new = max(m_old, max(current_block))`
  - `l_new = l_old × exp(m_old - m_new) + sum(exp(current_block - m_new))`
- 效果：显存从 O(N^2) 降至 O(N)，同时因减少 HBM 读写次数实现 2-4x 加速
- HBM 读写分析：标准实现读写 O(N^2)，FlashAttention 只读写 O(N^2·d/M)（M 为 SRAM 大小），当 M >> d 时显著减少

---

### Q: 为什么分类任务不用 MSE 而用交叉熵损失？

从**梯度**和**优化**两个角度理解：

**梯度角度**：
- MSE 对 softmax 输出求梯度：`∂L/∂z_i = (ŷ_i - y_i) · σ'(z_i)`，其中 σ' = ŷ(1-ŷ)
- 当 softmax 输出接近 0 或 1 时（预测很确定但错误），σ'(z) 趋近于 0，**梯度消失**，模型几乎无法修正错误
- 交叉熵对 softmax 输出求梯度：`∂L/∂z_i = ŷ_i - y_i`
- 梯度直接正比于预测误差，预测越错误梯度越大，收敛更快

**概率论角度**：
- 分类问题本质是估计条件概率 P(class|input)
- 交叉熵损失等价于最大似然估计（MLE），是概率空间中最自然的优化目标
- MSE 隐含假设输出误差服从高斯分布，这对概率输出（[0,1] 之间）不合理

**实际效果对比**：
- 相同模型和数据，交叉熵通常收敛速度快 3-5 倍
- 交叉熵对类别不平衡更鲁棒（可通过 weighted CE/Focal Loss 调整）

---

### Q: 计算 Qwen3-8B 推理时需要多少显存？

以 FP16 推理为例，需要计算三部分：

**1. 模型参数**：
- 8B × 2 bytes/param = **16 GB**

**2. KV Cache**（假设 Qwen3-8B：32 层，128 head_dim，8 个 KV head（GQA），batch_size=1，seq_len=2048）：
- 公式：`2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes`
- = 2 × 32 × 8 × 128 × 2048 × 1 × 2 bytes
- = **256 MB ≈ 0.25 GB**
- 注意：seq_len=8192 时 KV Cache 翻 4 倍到 ~1 GB；batch_size=8 时再翻 8 倍

**3. 激活值 + 框架开销**：
- 前向计算中间激活（不需要反向传播，所以不需保留全部激活）：~1-2 GB
- CUDA context + PyTorch caching allocator：~1-2 GB

**总计**：约 **18-20 GB**（FP16，batch=1，seq=2048）

**量化场景**：
| 量化方式 | 参数显存 | 总计约 |
|---|---|---|
| FP16 | 16 GB | 18-20 GB |
| INT8 | 8 GB | 10-12 GB |
| INT4 (GPTQ/AWQ) | 4 GB | 6-8 GB |

INT4 量化后单张 RTX 4090（24GB）即可运行。

---

### Q: RAG 的流程以及可以优化的策略？

**完整流程**：

```
文档集 → 分块(Chunking) → Embedding 编码 → 存入向量数据库
                                                    ↓
用户 Query → Query Embedding → 向量检索 Top-K → 拼接 Context → LLM 生成回答
```

**各环节优化策略**：

**1. 分块优化**：
- 语义分块（按段落/标题切分）优于固定长度切分
- 重叠分块（chunk 间 overlap 10-20%）避免语义截断
- 分层索引：摘要索引 + 原文索引，先粗后细
- 典型 chunk size：512-1024 tokens

**2. 检索优化**：
- **混合检索**：向量检索（语义）+ BM25（关键词），加权融合分数
- **Reranker 重排序**：用 cross-encoder 对 Top-K 结果精排（如 BGE-Reranker）
- **查询改写/扩展**：HyDE（让 LLM 先生成假设答案再用答案检索）、Multi-Query（生成多个检索 query）
- **多轮迭代检索**：根据初次回答的不足，再次检索补充信息

**3. 上下文优化**：
- 压缩检索文档（去冗余、提取关键句）
- 长文档摘要后再拼入 context
- 控制 context 长度不超过模型有效窗口

**4. 生成优化**：
- 引用溯源：回答中标注来源文档
- 事实校验：检测生成内容与检索文档的一致性
- Self-RAG：模型自主判断是否需要检索

---

### Q: PPO 中优势函数如何计算？Critic 模型如何更新？

**优势函数计算（GAE - Generalized Advantage Estimation）**：

核心思想是在 bias 和 variance 之间取折中：

```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)      # TD 误差
A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}  # GAE 优势估计
```

- `γ`（折扣因子，通常 0.99）：衡量未来奖励的重要性
- `λ`（GAE 参数，通常 0.95）：控制 bias-variance 权衡
  - λ=0：A_t = δ_t（1步 TD，低方差高偏差）
  - λ=1：A_t = R_t - V(s_t)（Monte Carlo，高方差低偏差）

在 LLM RLHF 中，每个 token position 对应一个 timestep，reward 通常只在序列末尾给出（sparse reward），中间步骤的 reward 为 0。

**Critic 模型更新**：
- 目标：让 V(s_t) 准确估计从 t 步开始的期望回报
- 损失函数：`L_critic = E[(V_θ(s_t) - R_t)²]`，其中 R_t 是 discounted return
- PPO 中 Critic 也可加 clip：`L = max((V-R)², (clip(V, V_old±ε) - R)²)`，防止价值函数更新过大导致优势估计不稳定
- 在 LLM 场景中，Critic 通常与 Policy 共享底层 Transformer backbone，只在最后加一个 scalar head

---

### Q: GRPO 中冷启动是如何处理的？

GRPO 的冷启动策略针对的是模型在强化学习阶段初期**不具备基本推理格式**的问题：

**DeepSeek-R1 的冷启动流程**：

1. **SFT 冷启动阶段**：
   - 收集少量（数千条）高质量长链推理样本（包含完整的 `<think>...</think>` 推理过程）
   - 在 base 模型上做 SFT，让模型学会基本的 CoT 推理格式
   - 此阶段不追求推理质量，只要格式正确

2. **GRPO 强化学习阶段**：
   - 在冷启动后的模型上开始 GRPO 训练
   - 使用规则 reward（数学正确性、格式完整性等）
   - 模型逐渐学会更高质量的推理

**为什么需要冷启动**：
- 如果直接从 base 模型开始 GRPO，模型的输出格式混乱，reward signal 噪声大
- 冷启动让模型进入"正确的搜索空间"——至少知道要生成推理过程，GRPO 再优化推理质量
- 类似 PPO 中需要一个好的 behavior policy 起点

**替代方案**：
- DeepSeek-R1-Zero 尝试完全不做 SFT 冷启动直接 RL，但需要更多训练步数且格式不够稳定
- 也可以用蒸馏数据（从更强模型获取推理样本）做冷启动

---

### Q: 手撕 GQA（Grouped Query Attention）？

（编程题）
