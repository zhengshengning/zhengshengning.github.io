---
title: '百度 AI Infra 一面 (3)'
date: 2026-04-16 11:44:18
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 训练优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Trust Region和PPO的关系？

**TRPO（Trust Region Policy Optimization）**：通过约束策略更新步长保证单调改进。核心约束：KL(pi_old || pi_new) <= delta。求解需要二阶优化（Fisher 信息矩阵 + 共轭梯度法），计算复杂度高且实现困难。

**PPO 是 TRPO 的简化近似**：
- 用 clip 机制代替 KL 硬约束：将比率 r(theta) = pi_new(a|s) / pi_old(a|s) 裁剪到 [1-eps, 1+eps]（eps 通常 0.2）
- Loss = min(r * A, clip(r, 1-eps, 1+eps) * A)
- 效果上实现了类似的"不要更新太远"的目标，但无需求解约束优化
- 计算只需一阶梯度（标准 SGD），实现简单且效果接近 TRPO

**关系总结**：PPO 是 TRPO 的一阶近似 + 工程友好版本。牺牲了理论上的单调改进保证，换来了实现简单和训练稳定。在实践中 PPO 已完全取代 TRPO。

---

### Q: PPO是on-policy还是off-policy？

**On-policy**。PPO 要求训练数据来自当前策略（或非常接近当前策略），通过 clip 机制保证 on-policy 假设的有效性。

**具体解释**：
- 每次迭代：用当前策略收集一批轨迹（rollout）-> 计算优势函数 -> 用这批数据更新策略多个 epoch（通常 3-5 个 epoch）
- 多 epoch 复用同一批数据是为了提高样本效率，但 clip 确保策略不会偏离数据采集时的策略太远
- 更新完成后旧数据作废，需要重新采样

**与 off-policy 的区别**：off-policy 方法（如 DQN、SAC）可以使用任意旧策略产生的数据训练（通过 replay buffer），样本效率更高但训练稳定性通常更差。

---

### Q: 为什么PPO需要Importance Sampling？

**原因**：在实际实现中，数据采集（rollout worker）和策略更新（learner）之间存在时间差：
1. 采集数据时用的是"行为策略" mu（当时的模型参数）
2. 更新时模型参数已经变化为"目标策略" pi（经过部分 gradient step）
3. 二者不完全相同，直接用数据估计 pi 的梯度会有偏

**重要性采样修正**：
```
gradient ≈ E_{a~mu} [pi(a|s)/mu(a|s) * A * grad(log pi(a|s))]
```
比率 pi/mu 修正了采样分布的偏差，使得用 mu 采集的数据也能无偏估计 pi 的梯度。

**PPO 的处理**：直接在 loss 中使用比率 r = pi_new/pi_old，clip 操作同时限制了 IS 比率的范围，防止方差爆炸。这是 PPO 简洁且有效的核心设计。

---

### Q: PPO中clip如何根据优势函数A的正负限制上下界？

PPO 的 clip 机制针对 A 的正负分别施加不同方向的约束：

**A > 0（好动作，应该鼓励）**：
- 无 clip 时 loss = r * A，梯度倾向于增大 r（增大该动作概率）
- clip 限制 r 的上界为 1+eps：min(r*A, (1+eps)*A) = min(r, 1+eps) * A
- 效果：防止一步更新中过度增大好动作的概率（避免过拟合到单个好轨迹）

**A < 0（坏动作，应该抑制）**：
- 无 clip 时 loss = r * A（A<0），梯度倾向于减小 r（减小该动作概率）
- clip 限制 r 的下界为 1-eps：min(r*A, (1-eps)*A) = max(r, 1-eps) * A（注意 A<0 翻转不等号方向）
- 效果：防止一步更新中过度减小坏动作的概率

**双向 clip 的整体效果**：无论好动作还是坏动作，策略更新幅度都被限制在 [1-eps, 1+eps] 范围内（约 +-20%），保证了训练稳定性。

---

### Q: PPO的损失函数怎么计算？GAE中lambda的作用？

**PPO 总 Loss = L_clip + c1 * L_value + c2 * L_entropy**

- L_clip：策略损失，clip 后的 surrogate objective
- L_value：价值函数损失，MSE(V_pred, V_target)，训练 critic
- L_entropy：策略熵 bonus，鼓励探索防止过早收敛。c2 通常 0.01

**GAE（Generalized Advantage Estimation）**：

优势函数 A_t 的估计方式决定了方差-偏差 trade-off。GAE 通过 lambda 参数平滑插值：

```
A_t^GAE = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)  （TD error）
```

**Lambda 的作用**：
- lambda = 0：A_t = delta_t（单步 TD），低方差但高偏差（只看一步）
- lambda = 1：A_t = R_t - V(s_t)（Monte Carlo return），低偏差但高方差（看完整轨迹）
- lambda = 0.95（常用）：折中方案，兼顾方差和偏差

**直觉**：lambda 控制"看多远的未来"。小 lambda 只关注即时反馈（稳定但短视），大 lambda 考虑长远收益（准确但波动）。

---

### Q: GRPO的损失计算？序列级别损失如何给到每个token？

**GRPO 的核心流程**：
1. 对每个 prompt x 采样 G 个回答 {y_1, ..., y_G}
2. 对每个回答计算 reward r_i（如数学答案正确性）
3. 组内归一化：advantage_i = (r_i - mean(r)) / std(r)
4. 策略梯度：对每个 token 的 log probability 乘以序列的 advantage

**序列级 reward 到 token 级 loss 的传递**：
```
L = -E_x [1/G * sum_i (advantage_i * sum_t log(pi(y_i_t | x, y_i_{<t})) / |y_i|)]
```

- 每个 token y_i_t 的 loss weight = advantage_i（该序列的归一化奖励）
- 除以 |y_i|（序列长度）做长度归一化，防止长序列主导梯度
- 加 KL 惩罚项：beta * KL(pi || pi_ref) 防止偏离参考策略过远

**关键设计**：无需 Critic/Value 网络（用组内相对排名代替绝对 baseline 估计），大幅简化了训练 pipeline。

---

### Q: GRPO有哪些变体？

| 变体 | 核心改进 | 解决的问题 |
|------|---------|-----------|
| **DAPO** | 动态分配采样资源，对难 prompt 多采样 | 样本效率低（简单 prompt 浪费算力） |
| **Dr.GRPO** | 去除长度归一化中的隐式长度偏差 | GRPO 倾向生成长回答（长=更多 reward 信号） |
| **RLOO** | Leave-One-Out 基线估计代替组均值 | 组均值估计方差大（组内只有 4-8 个样本） |
| **GSPO** | 组内排序生成 pairwise 对做 DPO-style 优化 | 更充分利用组内对比信息 |
| **Online DPO** | 在线采样 chosen/rejected 对 | 结合 GRPO 的在线采样和 DPO 的稳定训练 |

---

### Q: 分布式训练中优化器/梯度/模型参数的显存比例？FSDP和DeepSpeed Zero-1/2/3？

**显存构成**（以 7B 模型 + Adam + FP16 混合精度为例）：

| 组件 | 每参数字节数 | 7B模型总计 |
|------|------------|-----------|
| FP16 模型参数 | 2B | 14 GB |
| FP16 梯度 | 2B | 14 GB |
| FP32 master weight | 4B | 28 GB |
| FP32 momentum | 4B | 28 GB |
| FP32 variance | 4B | 28 GB |
| **总计** | 16B | 112 GB |

优化器状态（12B/参数）占总显存的 75%，是最大的开销来源。

**ZeRO / FSDP 分级切分策略**：

| 策略 | 切分内容 | 显存节省 | 额外通信 |
|------|---------|---------|---------|
| **ZeRO-1** | 优化器状态 | ~4x（每卡 84->42 GB） | 梯度 AllReduce + 参数 AllGather |
| **ZeRO-2** | 优化器 + 梯度 | ~8x（每卡->28 GB） | 梯度 ReduceScatter + 参数 AllGather |
| **ZeRO-3/FSDP** | 全部（optimizer+grad+param） | ~N倍（N=卡数） | 前向/反向时 AllGather 参数，用完释放 |

ZeRO-3 通信量最大（每次前反向都需要 AllGather 全部参数），但显存最省。FSDP 是 PyTorch 对 ZeRO-3 的原生实现。

---

### Q: Agentic RL是什么？

将 LLM Agent 的多步决策过程（工具调用、搜索、代码执行等）建模为**强化学习问题**，通过 RL 训练优化整个决策轨迹。

**形式化**：
- State：当前对话历史 + 环境反馈
- Action：LLM 生成的文本（可能包含工具调用指令）
- Reward：任务完成度（如代码测试通过率、网页任务成功率）
- Policy：LLM 本身

**与标准 RLHF 的区别**：
- 标准 RLHF：单轮 response 的质量评估（immediate reward）
- Agentic RL：多步交互的最终结果评估（delayed reward），需要 credit assignment

**代表工作**：
- SWE-Agent + RL：训练 agent 自动修复代码 bug
- WebAgent + RL：训练 agent 操作网页完成任务
- DeepSeek-R1：在推理过程中通过 RL 优化 CoT 策略

**挑战**：环境交互成本高（每步需要执行工具/代码）、reward 稀疏（只有最终结果有信号）、动作空间巨大（自然语言生成）。

---

### Q: 手撕：二叉树的层次遍历，并记录每个节点在第几层？

（编程题）
