---
title: '百度 AI Infra 一面 (1)'
date: 2026-04-16 12:09:31
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 解释Attention机制？

Attention 是 Transformer 的核心组件，实现了序列中任意位置间的直接信息交互。

**计算流程**：
1. 输入 X 经三个线性投影得到 Q（Query）、K（Key）、V（Value）
2. 计算注意力权重：scores = QK^T / sqrt(d_k)（缩放防止梯度消失）
3. 归一化：weights = softmax(scores)（转为概率分布）
4. 加权聚合：output = weights * V

**Multi-Head Attention**：将 QKV 投影到 h 个独立的子空间（head），各 head 并行计算 attention 后拼接再投影。好处：不同 head 可以关注不同位置/不同语义模式（如有的 head 学习语法关系，有的学习共指关系）。

**直觉理解**：Q 是"想找什么"，K 是"我是什么"，V 是"我能提供什么"。Q 和 K 的点积衡量"匹配度"，高匹配度的位置的 V 被更多地聚合到输出中。

**复杂度**：O(N^2 * d)，N 为序列长度，d 为维度。N^2 是 Transformer 处理长序列的主要瓶颈。

---

### Q: QKV为什么需要K，直接用V不行吗？

**核心原因：解耦"匹配函数"和"信息内容"**。

如果直接用 V 计算相似度：score = Q * V^T，则同一个向量既要表达"我与 query 有多相关"，又要表达"相关时返回什么信息"。这两个目标可能冲突——某个位置可能很相关（应该获得高权重）但实际提供的信息有限，或者反过来。

**K 和 V 分离后**：
- K 专门学习"这个位置与 query 的相关性度量"
- V 专门学习"相关时提供什么信息"
- 两者可以独立优化，互不干扰

**类比**：图书馆检索。K 是书的索引标签（用于匹配搜索词），V 是书的实际内容（匹配后返回的信息）。好的索引标签和好的内容是两个独立的维度。

**实验验证**：如果强制 K=V（将 Attention 退化为 K/V 共享），模型表达力下降 2-5%，尤其在需要细粒度注意力区分的任务上。

---

### Q: BN、LN、RMSNorm的区别？

三种归一化方法的核心区别在于**沿哪个维度计算统计量**：

| 方法 | 归一化维度 | 依赖 batch | 适用场景 |
|------|-----------|-----------|---------|
| **BN** | 沿 batch 维度（对每个特征通道） | 是 | CNN（图像分类/检测） |
| **LN** | 沿特征维度（对每个样本） | 否 | Transformer/RNN |
| **RMSNorm** | 沿特征维度（LN的简化版） | 否 | 现代 LLM |

**BatchNorm**：对 batch 内所有样本的同一特征位置计算 mean/var。问题：小 batch 时统计量不准确；推理时需维护 running mean/var；不适合变长序列（不同长度的 padding 影响统计）。

**LayerNorm**：对单个样本的所有特征维度计算 mean/var。与 batch 大小无关，适合 NLP。公式：y = (x - mean) / sqrt(var + eps) * gamma + beta。

**RMSNorm**：去掉 LN 中的 mean subtraction，只做 RMS 归一化：y = x / RMS(x) * gamma。计算更少（省一次 reduction 和减法），实验证明均值中心化贡献可忽略。LLaMA/Mistral/Qwen 等现代 LLM 全部采用。

---

### Q: 位置编码有哪些方式？

位置编码为 Transformer 提供序列顺序信息（attention 本身是排列不变的）：

**绝对位置编码**：
- **Sinusoidal**（原始 Transformer）：用不同频率的正弦/余弦函数编码位置，无需学习。理论上可泛化到任意长度但实际外推能力有限。
- **可学习 embedding**：每个位置一个可训练向量（GPT-2 采用）。简单但长度固定。

**相对位置编码**：
- **ALiBi**：不加位置向量，在 attention score 上减去线性偏置 |i-j| * slope。不同 head 用不同 slope。天然支持外推。
- **T5 Relative Bias**：可学习的相对位置偏置矩阵加到 attention score 上。

**旋转位置编码（RoPE）**（当前主流）：
- 对 Q/K 按维度对分组，每组施加位置相关的 2D 旋转
- 使内积 <q_m, k_n> 天然只依赖相对位置 (m-n)
- 优点：相对位置 + 无额外参数 + 实现简单（element-wise 乘法）
- LLaMA/Mistral/GPT-NeoX 等均采用

---

### Q: PPO、DPO、GRPO的区别？

三者都是 LLM 对齐算法，但机制和适用场景不同：

**PPO（Proximal Policy Optimization）**：
- 需要单独训练 Reward Model + Critic/Value Network
- 流程：生成 -> RM 打分 -> 用 PPO 更新策略（clip 限制更新幅度）
- 优点：通用性强，支持在线学习
- 缺点：训练复杂（4 个模型：policy/ref/RM/critic），不稳定，超参敏感

**DPO（Direct Preference Optimization）**：
- 无需 Reward Model，直接用偏好数据对 (chosen, rejected) 优化
- 核心推导：将 RLHF 的 RM+PPO 简化为一个 closed-form loss
- Loss: -log(sigma(beta * (log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected))))
- 优点：实现简单（只需 policy + ref model）、训练稳定
- 缺点：offline 方法，不能利用在线探索；对数据质量敏感

**GRPO（Group Relative Policy Optimization）**：
- DeepSeek 提出，为数学/推理任务设计
- 对每个 prompt 采样一组回答（如 8 个），组内按 reward 排序计算相对优势
- 无需 Critic 网络（用组内排名代替基线估计）
- 适合 outcome-based reward（如数学答案对错）
- 优点：简单高效，在推理类任务上效果好

---

### Q: 为什么Decoder-only模型比Encoder-Decoder用得多？

现代 LLM 几乎统一采用 Decoder-only 架构，核心原因：

1. **统一性**：一个架构同时处理理解和生成。用户输入和模型输出在同一个序列中处理，无需设计 cross-attention 的交互方式。简化了训练、推理和 serving 的工程复杂度。

2. **参数效率**：全部参数都用于提升生成能力。Encoder-Decoder 中 Encoder 的参数不直接贡献生成能力，等效浪费了 ~50% 参数。

3. **In-context Learning 友好**：自回归的序列建模方式天然支持 few-shot prompting——示例和任务无缝拼接在同一个序列中，不需要额外的编码机制。

4. **训练效率**：CLM（Causal Language Modeling）中每个位置都有预测目标（next token），loss 信号密度 100%。相比 Encoder 的 MLM 只有被 mask 的 15% 位置有 loss，训练同等数据时 CLM 的学习信号量多 ~7x。

5. **KV Cache 优化友好**：纯自回归结构的 KV Cache 管理更简单（只需追加），无需处理 encoder 输出的 cross-attention cache。
