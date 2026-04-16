---
title: '百度 AI Infra 一面 (3)'
date: 2026-04-16 11:44:18
categories:
  - [求职面试, AI Infra, 大厂真题]
tags: [AIInfra, 训练优化, 高性能计算, 大厂面经, 面经, 牛客]
---

# 百度 AI Infra 一面 (3)

<!-- more -->

> 原文链接: https://www.nowcoder.com/discuss/863890669662674944
> 来源: 牛客网

---

百度 大模型后训练 一面

📍面试公司：百度 文心一言

🕐面试时间：2026.03.18

💻面试岗位：大模型RL后训练

❓面试问题：

自我介绍，教育背景，项目经历
项目介绍，基于大语言模型的信号灯控制，问题是什么，如何结合熵，怎么评测，指标结果如何
强化学习理论内容，trust-range和PPO的关系
PPO是off-policy or on-policy? on-policy
为什么会有importance sampling，采样的策略模型和要训练的策略模型有偏差，重要性采样加以修正
PPO的clip在优势A 正/负时 限制上/下届，A为正限制上届，A为负限制下届
PPO的损失函数怎么计算的？广义优势估计是怎么计算？GAE中lambda的作用，该值大小和GAE 方差/偏差的关系
GRPO的损失计算，在序列级别的损失上，损失如何给到每一个token上？序列级别平均 或 批次级别平均
其它GRPO变体？DAPO，GSPO，GFPO等
分布式训练中，优化器/梯度/模型参数占用的显存比例，FSDP和DeepSpeed的Zero-1/2/3
Agentic RL
代码手撕：二叉树的层次遍历，如何记录每个节点在第几层？

🙌面试感想：

leetcode刷少了，非递归方式没有整出来，节点第几层的问题没有做出来
