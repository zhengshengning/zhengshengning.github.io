---
title: '阿里巴巴 淘天 AI Infra 一面 (2)'
date: 2026-04-16 11:44:02
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: GRPO和PPO的区别？

GRPO（Group Relative Policy Optimization）是DeepSeek提出的PPO简化版，核心区别在于**如何估计Advantage（优势函数）**：

**架构差异**：

| 组件 | PPO | GRPO |
|------|-----|------|
| 策略模型 (Actor) | 需要 | 需要 |
| 参考模型 (Reference) | 需要 | 需要 |
| 奖励模型 (Reward) | 需要 | 需要 |
| 价值模型 (Critic/Value) | **需要** | **不需要** |

**Advantage估计方法的差异**：

**PPO**：使用Value模型 + GAE(λ) 估计逐token的advantage
```python
# PPO: 需要Critic网络V(s)
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  # TD error
A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_l  # GAE
```
- 逐token级别的value估计→advantage估计更精细
- 但Critic网络本身需要训练（额外计算+显存+可能的训练不稳定）

**GRPO**：对同一prompt采样一组回答，用组内奖励的统计量作为baseline
```python
# GRPO: 无需Critic，用组内比较
responses = [sample(policy, prompt) for _ in range(G)]  # 采样G条回答
rewards = [reward_model(prompt, r) for r in responses]
# 组内标准化作为advantage
A_i = (rewards[i] - mean(rewards)) / std(rewards)
```
- 整条回答一个奖励分数→一个advantage值（非逐token）
- Baseline自然来自同组其他回答（相对好坏）

**GRPO的优势**：
1. **显存减少**：无需加载和维护Value模型（减少~25%总显存占用）
2. **训练更稳定**：Value模型训练本身可能不稳定，移除后整体更简单
3. **实现更容易**：不需要GAE计算、Value loss设计等
4. **无Value模型偏差**：Value模型估计不准会mislead策略更新

**PPO的优势**：
1. **逐token信号**：能识别一条回答中哪些token是好的、哪些是坏的
2. **样本效率**：不需要对每个prompt采样多条回答（GRPO需要G=64-256条）
3. **更通用**：理论上适用于更广泛的RL场景

**适用场景**：
- GRPO更适合：有明确奖励信号的任务（数学推理、代码生成——答案对错明确）
- PPO更适合：奖励信号模糊的任务、需要精细token级别反馈的场景

---

### Q: PPO中clip操作的作用？

**clip的具体机制**：

```python
ratio = pi_new(a|s) / pi_old(a|s)  # 新旧策略概率比
clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)  # epsilon通常=0.2
loss = -min(ratio * A, clipped_ratio * A)
```

**直觉理解——通过case分析**：

**Case 1：A > 0（好动作，应该增大概率）**
- ratio增大（新策略确实增大了好动作概率）→ ratio*A增大
- 但clip限制ratio最大为1+ε → 即使新策略非常看好这个动作，目标函数也不会因此获得过大的梯度
- **效果**：防止"过度兴奋"——一步就把好动作概率拉到极高

**Case 2：A < 0（坏动作，应该减小概率）**
- ratio减小（新策略减小了坏动作概率）→ ratio*|A|增大（loss减小是好的）
- 但clip限制ratio最小为1-ε → 即使新策略极度避免这个动作，目标函数的收益也有上界
- **效果**：防止"过度恐惧"——一步就把坏动作概率降到接近零

**Case 3：ratio > 1+ε 且 A > 0**
- clipped_ratio * A < ratio * A
- min取clipped版本 → 梯度被截断，**不再鼓励**继续增大概率
- 直觉：已经够好了，别再往这个方向走了

**Case 4：ratio < 1-ε 且 A < 0**
- 类似地被截断 → 不再鼓励继续减小概率

**为什么需要clip？**

1. **信赖域约束的近似**：TRPO严格约束KL(π_new||π_old) < δ，但需要复杂的二阶优化。PPO的clip是一阶的近似替代——简单且效果好
2. **防止策略崩溃**：没有约束时，一个outlier样本可能导致策略在一步内发生剧烈变化（catastrophic forgetting），之后的采样完全偏离→训练崩溃
3. **允许多epoch复用数据**：有了clip保护，可以对同一批数据训练多个epoch而不会更新过度。这大幅提高了数据利用效率

**ε的选择**：
- 通常ε = 0.1-0.2
- ε太大：约束太松，可能训练不稳定
- ε太小：约束太紧，学习速度慢（每步更新量小）
- LLM RLHF中常用ε = 0.2

---

### Q: 重要性采样的作用？除了与clip结合限制更新幅度还有什么用？与KL散度限制更新的区别？

**重要性采样（Importance Sampling）的本质作用**：

在策略梯度中，我们想估计新策略π_θ下的期望回报，但数据是从旧策略π_old收集的。重要性采样提供了数学上正确的修正：

```
E_{a~π_θ}[f(a)] = E_{a~π_old}[π_θ(a)/π_old(a) * f(a)]
```

**作用1：Off-policy修正（最核心）**
- **问题**：在线策略梯度（REINFORCE）每次更新策略后旧数据失效，需要重新采样→数据利用极低
- **解决**：通过重要性权重`w = π_new/π_old`修正分布偏差，允许**用旧策略采集的数据训练新策略**
- **收益**：数据利用效率提升K倍（一批数据可以训练K个epoch），这对LLM RLHF极其关键（采样成本高）

**作用2：方差控制**
- 重要性权重的方差与两个分布的差异成正比
- 当π_new和π_old差距太大时，少数高ratio样本主导估计→高方差→训练不稳定
- 这也解释了为什么需要clip/KL约束

**作用3：信用分配**
- 通过per-token的importance ratio，可以识别哪些token的生成概率变化了
- 配合advantage，实现"精准"的策略更新（只调整需要调整的token概率）

**重要性采样+Clip vs KL散度约束——两种"信赖域"实现**：

| 维度 | Clip + IS (PPO) | KL惩罚 (KL-PPO/TRPO) |
|------|----------------|---------------------|
| 约束方式 | 硬截断：ratio超出[1-ε,1+ε]时梯度为0 | 软惩罚：β*KL(π_θ\|\|π_ref) |
| 实现复杂度 | 简单（一行clip） | 中等（需要计算KL） |
| 调参难度 | ε相对鲁棒（0.2通常work） | β需要自适应调整（手动调很难） |
| 约束精度 | 粗糙（只看ratio，不看整体分布） | 精确（直接约束分布距离） |
| 计算成本 | 低 | KL计算有额外开销 |
| 实际效果 | PPO在大多数场景表现最好 | TRPO理论更优美但实现复杂 |

**TRPO vs PPO vs KL-PPO的关系**：
```
TRPO: max E[r(θ)*A]  s.t. KL(π_θ||π_old) < δ  (约束优化，需要二阶方法)
KL-PPO: max E[r(θ)*A] - β*KL(π_θ||π_old)       (拉格朗日松弛，β自适应)
PPO:   max E[min(r(θ)*A, clip(r(θ))*A)]         (clip近似信赖域)
```

**在LLM RLHF中的实践**：
- 通常同时使用clip **和** KL惩罚：clip防止单步过大更新，KL防止整体偏离参考模型
- clip约束的是相对于π_old（同一training iteration内的旧策略）
- KL约束的是相对于π_ref（SFT后的初始策略）——防止reward hacking

---

### Q: 马尔可夫性质是什么？

**马尔可夫性（Markov Property）的严格定义**：

```
P(s_{t+1} | s_t, s_{t-1}, ..., s_0) = P(s_{t+1} | s_t)
```

即：**未来状态只依赖于当前状态，与如何到达当前状态的历史路径无关**。当前状态包含了做最优决策所需的全部信息。

**直觉理解**：
- 棋类游戏：只需要看当前棋盘局面就能做最优决策，不需要知道前面每步是怎么走的
- 自回归LLM：当前时刻的状态 = KV Cache中存储的所有历史信息，包含了做next token prediction所需的全部context

**在强化学习中的意义**：
1. **MDP（马尔可夫决策过程）的基础假设**：PPO/GRPO等RL算法都基于MDP框架
2. **状态表示的充分性**：如果状态设计不满足马尔可夫性（遗漏了关键历史信息），RL算法的理论保证不成立
3. **Value函数的存在性**：只有满足马尔可夫性，V(s)和Q(s,a)才是well-defined的

**LLM生成中的马尔可夫性**：
- **Decode阶段**：给定已生成的所有token（编码在KV Cache中），下一个token的分布完全确定→满足马尔可夫性
- **本质上**：KV Cache就是"状态"的完整表示——它包含了从attention机制角度做prediction所需的所有历史信息
- **但有subtlety**：如果从"LLM权重+input"角度看，每步生成的"状态"是(prompt + all_generated_tokens)，而attention mask确保了当前输出只依赖前面所有token→天然满足

**不满足马尔可夫性的例子**：
- 速度：当前位置不够做决策，还需要知道位置变化率（速度）。解决：将(位置,速度)作为状态
- 部分可观测（POMDP）：Agent只能看到环境的部分信息→当前观测不满足马尔可夫性。解决：用历史观测的聚合作为状态（如用RNN/Transformer编码历史）

---

### Q: 从策略梯度到GRPO的发展脉络？哪些模块保留了，哪些丢掉了？

**RL for LLM的演进路线**：

```
REINFORCE (1992) → Actor-Critic (AC) → PPO (2017) → GRPO (2024)
  ↓                  ↓                  ↓              ↓
策略梯度基础        引入Critic         引入Clip+IS     去掉Critic
高方差              降低方差           稳定+数据复用   简化+高效
```

**各阶段详细对比**：

**REINFORCE（最原始的策略梯度）**：
```python
# ∇J = E[Σ_t ∇log π(a_t|s_t) * G_t]   G_t = 累计回报
# 整条轨迹的回报作为每步的"奖励信号"
loss = -log_prob(action) * total_return
```
- **保留至今**：策略梯度的核心思想——好动作增大概率，坏动作减小概率
- **问题**：方差极高（total return噪声大），学习不稳定

**Actor-Critic（引入Baseline降方差）**：
```python
# 引入Value网络V(s)作为baseline
# A_t = G_t - V(s_t)  或 A_t = r_t + γV(s_{t+1}) - V(s_t) (TD)
loss_actor = -log_prob(action) * advantage
loss_critic = (V(s) - target)^2
```
- **保留至今**：Advantage的概念（相对好坏而非绝对好坏）
- **在GRPO中丢弃**：Value网络 → 用组内奖励替代

**PPO（引入Clip + 重要性采样）**：
```python
# 允许多epoch复用数据 + clip防止过度更新
ratio = pi_new / pi_old  # 重要性采样
loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```
- **保留至今**：重要性采样（off-policy修正）、Clip机制、多epoch数据复用
- **问题**：4个模型（Actor + Critic + Reward + Reference）显存压力大

**GRPO（去掉Critic，组相对奖励）**：
```python
# 对同一prompt采样G条回答
# 组内标准化奖励作为Advantage
responses = sample(policy, prompt, num=G)
rewards = [reward_model(resp) for resp in responses]
advantages = (rewards - mean(rewards)) / std(rewards)
# 仍用clip + IS更新
loss = -min(ratio * A, clip(ratio) * A) + β * KL
```
- **丢弃了**：Value/Critic网络、GAE(λ)计算、逐token的value估计
- **保留了**：策略梯度核心、重要性采样、Clip机制、KL约束

**总结——每个阶段的保留与丢弃**：

| 模块 | REINFORCE | AC | PPO | GRPO |
|------|-----------|-----|-----|------|
| 策略梯度 ∇log π * signal | 保留 | 保留 | 保留 | 保留 |
| Baseline/Advantage | 无(高方差) | V(s)作baseline | GAE(λ) | 组相对奖励 |
| Value/Critic网络 | 无 | 引入 | 保留 | **丢弃** |
| 重要性采样 | 无(on-policy) | 通常无 | 引入 | 保留 |
| Clip机制 | 无 | 无 | 引入 | 保留 |
| KL约束 | 无 | 无 | 引入 | 保留 |
| 逐token value | 无 | 有 | 有 | **丢弃**(序列级) |
| 多epoch复用 | 否 | 否 | 是 | 是 |

**设计哲学的演进**：
- REINFORCE → AC：引入结构化的方差降低方法
- AC → PPO：从on-policy到off-policy（数据效率），加入稳定性保证
- PPO → GRPO：针对LLM场景的简化——用"比较"代替"估计"，反映了"大力出奇迹"的思路（采样足够多就不需要精细估计）

---

### Q: 手撕：LeetCode 173 二叉搜索树迭代器？

（编程题）
