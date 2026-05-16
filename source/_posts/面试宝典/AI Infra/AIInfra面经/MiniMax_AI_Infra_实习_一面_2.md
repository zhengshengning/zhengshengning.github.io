---
title: 'MiniMax AI Infra 实习 一面 (2)'
date: 2026-04-16 11:44:46
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 训练优化, 面经]
---


<!-- more -->

---

### Q: 介绍你熟悉的大模型架构？

主流大模型架构以 **Decoder-only Transformer** 为主（GPT系列的选择被后续工作验证为最优），关键变体包括：

**LLaMA系列（Meta）**：
- **RoPE位置编码**：旋转位置编码，通过旋转矩阵将位置信息注入attention，天然支持相对位置且可外推
- **RMSNorm**（替代LayerNorm）：去掉均值中心化只做方差归一化，减少15%计算量且训练更稳定
- **SwiGLU激活**：`SwiGLU(x) = Swish(xW₁) ⊙ (xW₃)`，比ReLU/GELU效果更好但多一个权重矩阵
- **GQA（Grouped Query Attention）**：多个query head共享一组KV head，LLaMA-2用8组KV head（32/4=8）
- Pre-Norm（Norm在attention/FFN之前）

**DeepSeek-V3（DeepSeek）**：
- **MLA（Multi-head Latent Attention）**：将KV投影到低维latent空间存储，推理时解压。KV Cache大小仅为 `latent_dim * seq_len`（如512维 vs GQA的数千维）
- **DeepSeekMoE**：细粒度专家（256个小专家）+ 共享专家（1-2个始终激活），每token选top-8路由专家
- **Multi-Token Prediction**：辅助训练头同时预测未来多个token，提升训练效率，推理时可复用做speculative decoding
- **FP8混合精度训练**：大规模使用FP8加速训练
- **Auxiliary-loss-free load balancing**：通过bias项而非loss项实现路由均衡

**Qwen（阿里）**：
- 类LLaMA架构基础 + 多模态扩展（Qwen-VL/Qwen-Audio）
- 支持超长上下文（Qwen-2.5支持128K）
- 使用YaRN做RoPE长度外推

**核心架构选择维度对比**：

| 模块 | 选项 | 代表模型 | 特点 |
|------|------|---------|------|
| 注意力 | MHA/GQA/MQA/MLA | GPT-4/LLaMA/PaLM/DeepSeek | MLA最省显存，GQA最平衡 |
| FFN | 标准/SwiGLU/MoE | GPT/LLaMA/Mixtral | MoE容量大计算少 |
| 位置编码 | RoPE/ALiBi/Learned | LLaMA/BLOOM/GPT | RoPE是当前主流 |
| 归一化 | LayerNorm/RMSNorm | GPT/LLaMA | RMSNorm更快更稳定 |
| 归一化位置 | Pre-Norm/Post-Norm | 主流/原始Transformer | Pre-Norm训练更稳定 |

### Q: 为什么MoE架构能在参数规模继续扩大时保持训练效率？

MoE的核心优势在于**解耦了模型容量（总参数量）和计算成本（激活参数量）**：

**计算成本分析**：
- 假设模型有E个专家，每token激活top-k个：实际计算的FLOPs ≈ Dense模型的 `k/E` 倍
- 例如DeepSeek-V3：256个专家选8个，FFN部分计算量仅为Dense的 8/256 = 3.125%
- 加上共享专家和非FFN部分，总计算量约为等参数Dense模型的 **30-40%**

**为什么这行得通（不损失质量）？**
1. **条件计算（Conditional Computation）**：不同token激活不同专家，每个专家专精于特定类型的输入，ensemble效应提升质量
2. **参数冗余假设**：Dense模型中大部分参数对特定输入是"休眠"的，MoE只是让这种稀疏性显式化
3. **Scaling Law的延续**：Google的研究表明MoE模型的loss与总参数量和计算量都满足power law，但相同loss下计算量更少

**具体数值对比**：
- GPT-3 175B Dense：训练需要 3.14 × 10²³ FLOPs
- Mixtral 8x7B（46.7B总参/12.9B激活）：质量接近LLaMA-70B，但训练/推理计算量仅约等于13B Dense模型
- DeepSeek-V3 671B（37B激活）：训练仅用2.788M H800 GPU小时，远低于同等质量的Dense模型

**MoE的代价**：
- 显存占用仍然与总参数量成正比（所有专家都需要加载）
- AllToAll通信开销（Expert Parallelism）
- 路由不均衡导致部分GPU负载高、部分空闲
- 微调困难（活跃专家子集不确定）

### Q: MoE的路由机制和负载不均问题？

**路由机制的实现**：
```python
# 简化的路由过程
router_logits = x @ W_gate  # [batch*seq, num_experts]
router_probs = softmax(router_logits)
topk_probs, topk_indices = topk(router_probs, k)  # 选top-k个专家
# 将token dispatch到对应专家，计算后按权重加权合并
output = sum(topk_probs[i] * expert_i(x) for i in topk_indices)
```

**负载不均的根本原因——马太效应（Rich-get-richer）**：
1. 初始阶段某些专家因随机初始化在某些token上表现稍好
2. 路由网络学到这个偏好→更多token被路由到这些专家
3. 更多训练数据→这些专家进一步优化→更强
4. 最终部分专家处理绝大多数token（过载），部分专家几乎不被选择（塌缩/死亡专家）

**负载不均的危害**：
- **计算效率低**：Expert Parallelism中，负载高的GPU成为瓶颈，其他GPU空等
- **容量浪费**：死亡专家占用显存但不贡献模型能力
- **训练不稳定**：严重不均时loss可能spike

**量化评估**：通常用 `Load Balance Factor = max_tokens_per_expert / avg_tokens_per_expert` 衡量，理想值为1.0

### Q: 如何优化低专家利用率的路由策略？

**1. 辅助负载均衡Loss（Auxiliary Load Balancing Loss）**：
- Switch Transformer提出的经典方法：`L_aux = α * N * Σᵢ(fᵢ * pᵢ)`
  - fᵢ = 被分配到专家i的token比例
  - pᵢ = 路由到专家i的平均概率
- 惩罚路由概率与实际分配的乘积偏离均匀分布
- α通常取0.01-0.1，过大会损害模型质量

**2. Expert Choice路由（EC路由）**：
- 反转选择方向：不是token选专家，而是**专家选token**
- 每个专家选择affinity最高的top-k个token处理
- **天然保证均匀负载**（每个专家处理固定数量的token）
- 缺点：某些token可能不被任何专家选中（需要特殊处理）

**3. 无辅助损失方法（Auxiliary-loss-free，DeepSeek-V3）**：
- 为每个专家维护一个可学习的bias项，加到路由分数上
- 训练过程中根据负载情况动态调整bias：负载低的专家bias增大，负载高的减小
- 不引入额外loss项，避免对主任务优化的干扰

**4. 容量因子（Capacity Factor）+ 溢出重路由**：
- 设定每个专家的最大容量 = `CF * (总token数 / 专家数)`，CF通常取1.0-1.5
- 超出容量的token被丢弃或重新路由到次优专家
- 简单有效但可能丢失信息

**5. 其他策略**：
- **Random routing**：在top-k之外随机选择一个专家（增加探索）
- **Hash routing**：用确定性hash函数分配token（完美均衡但不可学习）
- **Shared Expert**：始终激活的共享专家处理通用模式，路由专家处理特化模式

### Q: SFT和RLHF的本质区别？为什么SFT后仍需RLHF？

**本质区别**：

| 维度 | SFT | RLHF |
|------|-----|------|
| 学习方式 | 最大似然（模仿学习） | 策略优化（奖励最大化） |
| 目标 | "像人一样说" | "说得让人满意" |
| 信号 | token级（每个位置的正确答案） | 序列级（整条回答的质量评分） |
| 数据需求 | 高质量问答对 | 偏好对比数据（好 vs 坏） |
| 覆盖范围 | 只学见过的回答方式 | 能泛化到新场景的偏好判断 |

**为什么SFT后仍需RLHF？**

1. **分布偏移（Exposure Bias）**：SFT在teacher forcing下训练，但推理时是自回归的。训练和推理的分布不同导致误差累积。RLHF直接在自回归生成的分布上优化

2. **无法覆盖所有场景**：SFT数据再多也无法穷举所有可能的输入-输出对。RLHF通过奖励模型学到通用的偏好判断，能泛化到训练数据未覆盖的场景

3. **多样性与安全性的平衡**：SFT倾向于学习数据中最频繁的模式（mode-seeking），可能学到有害内容。RLHF可以明确惩罚有害输出

4. **开放式任务的评价困难**：创意写作、代码生成等任务没有唯一正确答案，SFT的最大似然目标不适合。RLHF通过人类偏好信号指导这类开放式优化

5. **拒绝/安全行为**：SFT教模型"如何回答"，RLHF教模型"什么时候拒绝回答"——后者本质上是一个reward optimization问题

### Q: RLHF中PPO的核心优化目标和目标函数解释？

**PPO目标函数**：
```
L_PPO = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - β * KL(π_θ || π_ref)
```

**各项详解**：

**r(θ) = π_θ(a|s) / π_old(a|s)**：新旧策略的概率比
- r(θ) > 1：新策略比旧策略更倾向于选择该动作
- r(θ) < 1：新策略比旧策略更不倾向于选择该动作
- r(θ) = 1：新旧策略相同

**A（Advantage）**：优势函数估计
- A > 0：该动作比平均好→应增大其概率
- A < 0：该动作比平均差→应减小其概率
- 通常用GAE(λ)计算：`A_t = Σ(γλ)^l * δ_{t+l}`，δ为TD error

**clip(r(θ), 1-ε, 1+ε)**：裁剪机制（ε通常取0.1-0.2）
- 当A > 0时：即使新策略大幅增加了好动作的概率（r >> 1），也被clip到1+ε
- 当A < 0时：即使新策略大幅减少了坏动作的概率（r << 1），也被clip到1-ε
- **效果**：形成信赖域（trust region），防止策略单步更新过大导致崩溃

**min操作的含义**：取裁剪前后目标的较小值
- 形成"悲观"估计：对好的更新保守（clip上界），对坏的更新也保守（clip下界）
- 确保无论哪个方向的更新都不会太激进

**β * KL(π_θ || π_ref)**：KL惩罚项
- π_ref是SFT后的初始策略（冻结的参考模型）
- 防止策略偏离参考模型太远导致**reward hacking**（找到奖励模型的漏洞获得高分但实际质量差）
- β通常取0.01-0.1，可自适应调整

**在LLM RLHF中的具体形式**：
- State = 已生成的前缀 tokens
- Action = 下一个token的选择
- Reward = 奖励模型对完整回答的打分（通常只在最后一个token处给出）
- Episode = 一次完整的回答生成

### Q: 手撕：实现滑动窗口最大值？

（编程题）
