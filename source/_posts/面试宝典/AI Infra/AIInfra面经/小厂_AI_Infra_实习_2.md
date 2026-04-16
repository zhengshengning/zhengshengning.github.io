---
title: '小厂 AI Infra 实习 (2)'
date: 2026-04-16 12:07:24
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 面经]
---


<!-- more -->

---

1.详细说明大模型rl全流程，涉及到哪些模型，ppo/grpo有什么区别；
2.rl里rollout耗时占比大概百分之多少，policy mfu大概多少，mfu计算公式，6Nd公式是什么；
3.rl里rollout有哪些优化点（rollout量化 异步rollout等）；
4.介绍rl中如何把预训练权重同步到推理引擎；
5.Megatron，tp是怎么切分的，mlp中第一个矩阵和第二个矩阵分别是行切还是列切，通信分别是什么算子；
6.预训练和sft loss、数据集有什么区别；
7.预训练优化，介绍流水线并行，说明一下1f1b，dualpipe；
8.说一下DeepSeek论文里觉得印象深刻的点（fp8训练）；
9.介绍一下vllm/sglang里的continuous batching；
