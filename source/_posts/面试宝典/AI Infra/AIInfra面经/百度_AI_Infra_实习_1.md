---
title: '百度 AI Infra 实习 (1)'
date: 2026-04-16 12:14:24
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 异步RL训练中对损失函数的修正方法？

异步 RL（如 IMPALA、APE-X）中，数据采集（actor）和策略更新（learner）使用不同版本的策略参数，导致训练数据的分布与当前策略不一致（off-policy 偏差）。如果不修正，梯度估计有偏，可能导致训练不稳定或收敛到次优策略。

**修正方法**：

1. **重要性采样（IS）修正**：用行为策略 mu 与目标策略 pi 的概率比加权 loss：
   - 比率 rho_t = pi(a_t|s_t) / mu(a_t|s_t)
   - 修正后的梯度 = rho_t * 原始梯度
   - 问题：当 pi 和 mu 差异大时比率方差极高，训练不稳定

2. **V-trace**（IMPALA 使用）：截断重要性采样比，限制方差：
   - 引入截断参数 c_bar 和 rho_bar（通常取 1.0）
   - 对超出阈值的比率进行 clip，牺牲少量偏差换取方差大幅降低
   - 实现 actor-learner 高度异步时仍能稳定训练

3. **PPO-clip**：通过 clip ratio 限制新旧策略的偏差：
   - L = min(r * A, clip(r, 1-eps, 1+eps) * A)
   - 不需要显式计算 IS 比率的截断，而是直接限制策略更新幅度
   - 实现简单，是当前最广泛使用的方案

4. **Off-policy correction（KL 惩罚）**：在 loss 中加入 KL(pi_old || pi_new) 惩罚项，约束策略更新幅度不要偏离行为策略太远。TRPO 用硬约束，PPO 用 clip 近似。

**实践选择**：小规模异步用 PPO-clip 即可；大规模分布式（数百 actor）用 V-trace 更稳定。

---

### Q: DualPipe的原理？

DualPipe 是 DeepSeek 在其 MoE 模型训练中提出的**双向流水线并行**方案，旨在大幅降低流水线气泡率。

**核心问题**：传统 PP（如 1F1B）存在不可消除的流水线气泡——各 stage 在等待上/下游传递激活值或梯度时处于空闲状态。气泡比例约为 (p-1)/m（p 为 stage 数，m 为 micro-batch 数）。

**DualPipe 的核心思想**：
1. 将 micro-batch 分为两组，一组从前向后执行（forward direction），另一组从后向前执行（backward direction）
2. 双向调度使得每个 stage 在等待某一方向的通信时，可以执行另一方向的计算
3. 前向计算和反向计算的通信被重叠（overlap），而非串行等待

**具体做法**：
- 将一个 batch 的 micro-batches 分为两个子集
- 子集 A 按正常的前向/反向流程从 stage 0 -> stage N 执行
- 子集 B 按反向顺序从 stage N -> stage 0 执行
- 每个 stage 交替处理两个方向的 micro-batch，使通信延迟被计算隐藏

**效果**：在 DeepSeek-V3 训练中，DualPipe 将通信开销几乎完全隐藏在计算中，流水线利用率显著高于传统 1F1B 方案，尤其适合通信带宽受限的跨节点 PP 场景。

---

### Q: 手撕：合并K个升序链表？

（编程题）

---

### Q: 手撕：二叉树的右视图？

（编程题）
