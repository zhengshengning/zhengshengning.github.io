---
title: '字节跳动 抖音电商 AI Infra'
date: 2026-04-16 11:43:18
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 训练优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: SFT阶段的训练流程？数据处理和框架选择？

**SFT（Supervised Fine-Tuning）流程**：
1. **数据来源**：人工标注、模型蒸馏（强模型生成）、开源数据集筛选。
2. **数据处理**：格式化为instruction-response对、去重、质量过滤（长度/困惑度/毒性）、多样性采样。
3. **训练框架**：DeepSpeed ZeRO（大模型分片）、Megatron-LM（3D并行）、LLaMA-Factory/Axolotl（轻量微调工具）。
4. **训练配置**：学习率1e-5~5e-5、cosine schedule、FP16/BF16混合精度。

---

### Q: PPO训练流程？包括数据、奖励模型训练、各模型的Loss？

**PPO（Proximal Policy Optimization）在RLHF中的流程**：
1. **奖励模型训练**：用人类偏好数据（chosen/rejected对）训练RM，Loss = -log(sigmoid(r_chosen - r_rejected))。
2. **PPO训练循环**：
   - Actor（策略模型）生成response。
   - RM打分作为reward。
   - Critic（价值模型）估计baseline V(s)。
   - 计算advantage A = R - V(s)。
   - Actor Loss = -min(ratio×A, clip(ratio, 1-eps, 1+eps)×A)（PPO-Clip）。
   - Critic Loss = MSE(V_pred, R_target)。
   - 加KL惩罚防止策略偏离参考模型太远。
3. **训练资源**：通常需要4个模型（Actor/Critic/RM/Reference），多卡训练。

---

### Q: DPO训练流程？PPO和DPO的区别？

**DPO（Direct Preference Optimization）**：
- 直接用偏好数据训练策略模型，无需训练奖励模型。
- Loss = -log(sigmoid(beta × (log_pi(chosen)/log_ref(chosen) - log_pi(rejected)/log_ref(rejected))))。
- 隐式将偏好学习转化为一个分类问题。

**区别**：
- PPO需要4个模型（开销大），DPO只需2个（策略+参考）。
- PPO是在线学习（生成新数据），DPO是离线学习（固定偏好数据集）。
- PPO效果通常更好（在线探索）但训练不稳定，DPO简单稳定但探索能力弱。
- 多轮DPO（Iterative DPO）可弥补离线局限。

---

### Q: 怎么评估微调后模型有提升？

1. **自动指标**：特定任务的benchmark得分（如MMLU/GSM8K/HumanEval）。
2. **人工评估**：人类评分胜率（vs 基线模型）。
3. **LLM-as-Judge**：用强模型（如GPT-4）对比评分。
4. **A/B测试**：线上真实用户流量对比。
5. **领域专项评估**：针对具体应用的测试集。
6. **安全评估**：毒性/偏见/越狱测试。

---

### Q: PPO为什么效果比DPO强？多轮DPO为什么有提升？

**PPO更强因为**：
- 在线生成数据，能探索策略空间中新的response，学到更好的行为。
- 奖励模型提供细粒度的reward信号（连续值），比DPO的二分类更丰富。
- 能优化分布外的response，DPO只能优化训练数据中出现的pair。

**多轮DPO提升因为**：
- 每轮用当前策略生成新response构建偏好数据，近似在线学习效果。
- 逐轮缩小on-policy/off-policy gap。
- 类似self-play的迭代改进。

---

### Q: CLIP训练原理？

CLIP（Contrastive Language-Image Pre-training）：
- 同时训练图像编码器和文本编码器，使匹配的图文对embedding接近，不匹配的远离。
- Loss：对称的InfoNCE/对比损失。一个batch内N个图文对形成N×N相似度矩阵，对角线为正样本，其余为负样本。
- 训练数据：400M互联网图文对。
- 推理：计算图像和文本embedding的余弦相似度，实现零样本分类/检索。

---

### Q: 讲几种优化器？

- **Adam**：自适应学习率，结合一阶矩（动量）和二阶矩（RMSProp），加bias correction。
- **AdamW**：解耦weight decay，L2正则不参与自适应计算。
- **SGD+Momentum**：传统方法，泛化性好，大模型微调时也常用。
- **LAMB**：Layer-wise Adaptive Moments for Batch training，大batch训练用。
- **Adafactor**：内存高效的Adam变体，分解二阶矩为行列，省显存。

---

### Q: MHA（Multi-Head Attention）原理？

1. 输入X通过三组线性投影得到Q、K、V（每组分为h个头）。
2. 每个头独立计算：Attention_i = softmax(Q_i × K_i^T / sqrt(d_k)) × V_i。
3. 多头结果拼接：Concat(head_1, ..., head_h)。
4. 最终线性投影：Output = Concat × W_O。

多头的好处：不同头可以关注不同位置/不同语义子空间的信息，增强表达能力。

---

### Q: GRPO（Group Relative Policy Optimization）原理？

GRPO是DeepSeek提出的RLHF算法，改进点：
1. **无需Critic模型**：通过组内相对比较估计baseline，省去价值模型。
2. **组采样**：对每个prompt生成一组（如G=8个）response，组内response的reward均值作为baseline。
3. **优势估计**：A_i = (R_i - mean(R_group)) / std(R_group)，归一化后的相对优势。
4. **PPO-Clip更新**：使用标准PPO loss但用组相对优势替代传统advantage。

优势：减少一半模型显存（无Critic），训练更稳定（组归一化减少reward scale敏感性）。

---

### Q: LoRA原理？r大小对模型训练的影响？

**LoRA（Low-Rank Adaptation）原理**：
- 冻结原始权重W，添加低秩旁路：W' = W + BA，其中B∈R^{d×r}，A∈R^{r×k}，r << min(d,k)。
- 训练时只更新A和B，参数量极少。
- 推理时可合并回原权重：W_merged = W + BA，无额外延迟。

**r大小的影响**：
- r越大：拟合能力越强（接近全参数微调），但参数量增加、过拟合风险。
- r越小：参数少训练快，但可能欠拟合复杂任务。
- 经验值：通用任务r=8-16足够；复杂任务（如代码生成）可用r=64-128。
- QLoRA用NF4量化基础模型+FP16 LoRA，进一步省显存。

---

### Q: ViT训练原理？

**ViT（Vision Transformer）**：
1. 将图像切分为固定大小的patch（如16×16）。
2. 每个patch线性投影为embedding（相当于token）。
3. 添加位置编码（learnable或sinusoidal）。
4. 添加[CLS] token。
5. 经过标准Transformer Encoder（Multi-Head Attention + FFN）。
6. [CLS] token的输出经MLP head做分类。

训练：通常需要大规模数据预训练（如ImageNet-21K或JFT-300M），小数据上不如CNN。DeiT通过知识蒸馏和数据增强让ViT在ImageNet上也有好效果。

---

### Q: Swin Transformer原理？

Swin Transformer引入层次化结构和局部注意力：
1. **窗口注意力**：将特征图划分为不重叠的窗口（如7×7），每个窗口内做self-attention。计算量从O(n^2)降为O(n×w^2)。
2. **移位窗口（Shifted Window）**：相邻层交替使用原始和移位的窗口划分，实现跨窗口信息交流。
3. **层次化**：通过Patch Merging逐层降采样，类似CNN的多尺度特征金字塔。
4. **相对位置编码**：不用绝对位置，用窗口内的相对位置bias。

优势：线性复杂度（相对于序列长度）、多尺度特征、适用于检测/分割等密集预测任务。

---

### Q: Qwen3快思考和慢思考的原理？

Qwen3支持两种推理模式：
- **快思考（Fast/Non-thinking）**：直接生成答案，类似传统LLM。延迟低、token消耗少。
- **慢思考（Slow/Thinking）**：先在特殊thinking标签内进行推理链（Chain-of-Thought），再输出最终答案。精度更高但消耗更多token。

实现原理：通过训练让模型学会何时需要深入思考。用户可通过参数控制（enable_thinking=True/False）。模型内部通过特殊token（如<think>...</think>）区分思考过程和最终输出。本质是将test-time compute scaling的理念融入模型：简单问题直接答，复杂问题先推理再答。
