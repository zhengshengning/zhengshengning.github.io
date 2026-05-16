---
title: '字节跳动 豆包 AI Infra 实习 二面'
date: 2026-04-16 11:43:10
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 训练优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Qwen2的模型结构？Qwen2相比Qwen1做了哪些改进？

**Qwen2 架构**：标准 Decoder-only Transformer，融合了多项现代化设计：

| 组件 | Qwen2 选择 | 说明 |
|------|-----------|------|
| 注意力 | GQA | Q head 数 >> KV head 数，降低 KV Cache 开销 |
| FFN | SwiGLU | `SwiGLU(x) = Swish(xW_gate) ⊙ (xW_up)`，相比 ReLU 同等参数下效果更好 |
| 位置编码 | RoPE | 旋转位置编码，天然支持长度外推 |
| 归一化 | Pre-RMSNorm | 去掉均值中心化只保留缩放，计算量减少约 15% |

**相比 Qwen1 的关键改进**：

1. **GQA 替代 MHA**：这是最核心的架构改变。Qwen2-72B 使用 64 个 Q head + 8 个 KV head，KV Cache 内存降低 8 倍。对于 LLM 推理场景（decode 阶段 memory-bound），这直接提升了吞吐量和可服务的并发数。

2. **上下文窗口大幅扩展**：从 Qwen1 的 8K 扩展到 32K/128K。技术上通过 YaRN（Yet another RoPE ExtensioN）插值实现长度外推，核心是对 RoPE 的不同频率分量施加不同的缩放因子。

3. **更大词表和改进的 Tokenizer**：词表从 ~150K 扩展到 ~152K，改善了多语言编码效率，特别是代码和数学符号的覆盖。

4. **训练数据和质量**：预训练语料从万亿级扩展到数万亿 token，增加了更多高质量代码和数学数据，这直接体现在下游 benchmark 的提升。

5. **MoE 变体（Qwen2-MoE）**：引入稀疏 MoE 架构，总参数 57B 但每 token 只激活约 14B，在同等推理成本下获得更强能力。

---

### Q: 为什么Decoder-only成为大模型主流架构？

这是一个从 GPT-3 到 LLaMA 到 Qwen 再到 DeepSeek 的行业共识，原因涉及多个层面：

**1. 训练效率（最核心原因）**：
- Decoder-only 的 next-token prediction 中，**每个 token 位置都参与 loss 计算**。一个 N token 的序列贡献 N-1 个训练信号。
- Encoder-Decoder（如 T5）中 encoder 输入不计算 loss，等量数据的训练信号更少。
- 在大规模预训练中（万亿 token），这个差异被极度放大。

**2. Scaling Law 表现好**：
- 结构简单统一（只有 decoder blocks 堆叠），参数量和数据量增加带来的能力提升更可预测。
- 对比 Encoder-Decoder：encoder 和 decoder 的参数分配比例本身就是额外的超参数，且不易优化。

**3. 推理效率**：
- **KV Cache** 天然适配自回归生成：每步只计算新 token 的 Q 与缓存的 K/V 做 attention，增量计算。
- Encoder-Decoder 需要先完整跑完 encoder（无法流式处理输入），再自回归 decode。
- Decoder-only 的 Continuous Batching + PagedAttention 等优化已非常成熟。

**4. 统一的能力范式**：
- 将所有任务（理解、生成、推理、对话）统一为 "给定前缀，续写后续" 的生成范式。
- In-context learning / few-shot 能力自然涌现，无需为不同任务设计不同的输入输出格式。

**5. 工程简单性**：
- 单一结构便于工程优化：TP/PP/ZeRO 并行策略、KV Cache 管理、量化方案都只需针对一种结构适配。
- 开源生态（vLLM、TGI、SGLang 等推理框架）主要围绕 Decoder-only 优化。

**反面观点**：Encoder-Decoder 在某些场景仍有优势——如翻译、摘要等明确的 "输入→输出" 任务中，encoder 可以用双向 attention 更充分地理解输入。但这些优势在 scale up 后被 Decoder-only 的通用能力追平。

---

### Q: 模型训练推理优化方法有哪些？

这是 AI Infra 的核心问题域。训练和推理的优化目标不同：训练追求**更大模型/更快收敛**，推理追求**更低延迟/更高吞吐**。

**训练优化**：

| 技术 | 解决的问题 | 关键原理 |
|------|-----------|---------|
| 混合精度（BF16/FP16） | 计算加速+省显存 | 前向/反向用低精度，master weight 保留 FP32 |
| ZeRO（1/2/3） | 显存冗余 | 将 optimizer state/gradient/param 分片到多卡 |
| 3D 并行（TP+PP+DP） | 单卡放不下 | TP 切 attention/FFN，PP 切层，DP 切数据 |
| 梯度检查点 | 激活显存 | 前向不保存中间激活，反向时重算，用时间换空间 |
| FlashAttention | Attention 显存+速度 | Tiling+Online Softmax，O(N²)→O(N) 显存 |
| 梯度累积 | 大 batch 需求 | 多步小 batch 的梯度累积后统一更新 |
| 通信优化 | 多卡同步开销 | Gradient Bucketing、计算通信 overlap |

**推理优化**：

| 技术 | 解决的问题 | 关键原理 |
|------|-----------|---------|
| 量化（W4/W8/FP8） | 显存+带宽 | 低精度存储权重，decode 阶段 memory-bound 下直接加速 |
| KV Cache + PagedAttention | KV 显存管理 | 分页管理避免碎片，显存利用率从 20-40% 提升到 >95% |
| FlashAttention/FlashDecoding | Attention 速度 | Tiling 减少 HBM 访问；FlashDecoding 对短 Q 长 KV 优化 |
| 投机解码 | 自回归延迟 | 小模型 draft + 大模型 verify，一步验证多个 token |
| Continuous Batching | 吞吐 | 请求级调度，完成一个立即补入新请求，避免 batch 空洞 |
| 算子融合 | Kernel launch 开销 | 将多个 element-wise op 融合到一个 kernel |
| PD 分离 | 资源利用率 | Prefill（计算密集）和 Decode（访存密集）分到不同 GPU 集群 |

**关键认知**：训练瓶颈通常在显存和通信，推理瓶颈通常在带宽和延迟。Decode 阶段是典型的 memory-bound（算术强度 < 1 FLOP/byte），所以量化（减少读取量）和 batching（摊分权重读取）是最直接的加速手段。

---

### Q: 为什么有了SFT之后还需要RLHF？

这个问题的本质是：**模仿学习（SFT）和强化学习（RLHF）各解决对齐的什么层面**。

**SFT 的能力和局限**：
- SFT 通过监督学习让模型学会"说人话"——遵循指令格式、理解对话意图、产生流畅输出。
- **局限 1**：SFT 只能学到训练数据的分布。对于同一个 prompt 的多种合理回答，SFT 无法区分哪个"更好"，它只会最大化所有标注数据的似然。
- **局限 2**：数据天花板。高质量 SFT 数据生产成本高，且标注者水平参差不齐。
- **局限 3**：负面行为抑制困难。"不要做什么"比"要做什么"更难通过正例学习。

**RLHF 补充了什么**：
1. **偏好排序能力**：Reward Model 学习人类对回答的偏好排序（A > B），而非绝对质量。这比标注"正确答案"容易得多——人类判断"哪个更好"比写出"最好的回答"简单。
2. **隐性标准编码**：有帮助(helpful)、无害(harmless)、诚实(honest) 这些标准难以通过 SFT 数据穷举，但 RM 可以从偏好对中隐式学到。
3. **策略空间探索**：PPO/GRPO 在策略空间中采样和探索，可以发现 SFT 数据中未出现的更优回答模式。
4. **安全对齐**：拒绝有害请求、承认不确定性、减少幻觉——这些"边界"行为需要 RL 的负反馈信号来强化。

**直觉类比**：SFT 像教一个人看范文学写作，RLHF 像给他批改作文——前者学格式和基础能力，后者学什么是好品味。

---

### Q: PPO和DPO的主要思想是什么？

**PPO（Proximal Policy Optimization）**：

核心思路是**在线 RL + 稳定更新**：
1. Policy model 对 prompt 生成 response
2. Reward model 给 response 打分
3. Critic model 估计每个 token 位置的状态价值 V(s)
4. 计算优势函数（GAE）：`A_t = Σ (γλ)^l · [r_t + γV(s_{t+1}) - V(s_t)]`
5. 用 clip 机制限制策略更新幅度：

```
L_PPO = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)
其中 r(θ) = π_new(a|s) / π_old(a|s)
```

- `ε` 通常取 0.2，防止单步更新过大导致策略崩溃
- 需要 **4 个模型**同时在线：Actor（策略）、Critic（价值）、RM（奖励）、Reference（KL 约束基准）
- 显存需求大：4 × 模型大小 + 激活值

**DPO（Direct Preference Optimization）**：

核心思路是**将 RL 问题转化为监督学习**：
- 数学推导证明：最优 RL 策略可以用 policy 本身的 log probability 隐式表达 reward
- 直接从偏好数据对 (chosen, rejected) 学习，无需显式 reward model

损失函数：
```
L_DPO = -E[log σ(β · (log π/π_ref(y_w|x) - log π/π_ref(y_l|x)))]
```
其中 y_w 是 chosen，y_l 是 rejected，β 控制偏离 reference policy 的惩罚力度。

- 只需 **2 个模型**：Policy + Reference（reference 冻结只做前向）
- 无需在线采样和 reward model，训练更简单稳定

**关键对比**：

| 维度 | PPO | DPO |
|------|-----|-----|
| 模型数 | 4 | 2 |
| 数据 | 在线采样 | 离线偏好对 |
| 显存 | ~4x 模型大小 | ~2x 模型大小 |
| 探索能力 | 强（在线采样新 response）| 弱（受限于离线数据分布）|
| 调参难度 | 高（reward scale、KL系数、GAE参数）| 低（主要调 β）|
| 实际效果 | 略优（尤其复杂任务）| 略逊但性价比高 |
| 代表应用 | InstructGPT, Claude | Zephyr, LLaMA2-Chat |

**趋势**：GRPO（DeepSeek）去掉 Critic 用组内相对奖励，兼顾在线探索和低资源需求，可能成为下一代主流。

---

### Q: 手撕：合并K个升序链表（LeetCode 23）？

（编程题）
