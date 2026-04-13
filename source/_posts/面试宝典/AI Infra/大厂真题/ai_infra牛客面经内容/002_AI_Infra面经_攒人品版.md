---
title: AI Infra面经 攒人品版
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, AI Infra面经]
tags: [AI Infra, 面经, 牛客]
---

# AI Infra面经 攒人品版

<!-- more -->

**来源**: 牛客网
**链接**: https://www.nowcoder.com/feed/main/detail/d695614a06424c148c04586ac3a66e78

---

继续来分享下之前的面经~欢迎友好讨论，信息共享

---

## 1️⃣ 算法题

- 手撕内存池（要求支持类似new Foo[], delete []功能)
- C++如何比较两个float是否相等
- LRU
- 岛屿个数
- 二叉树的层序遍历
- Hamming weight
- K-coverage intervals

---

## 2️⃣ Torch手撕题

- LoRA adapter

---

## 3️⃣ CUDA手撕题

1. 支持torch broadcast的4D tensor的elementwise mul

2. A: (1, 256), B: (256, 128), C: (128, 256)，计算 (A * B) * C

3. Embedding Sparse Feature Pooling：
   - A是100万个离散ID（0~999）
   - B是100万个float
   - 计算长度为1000的float数组C
   - C[i] = Σ_{j s.t. A[j] = i} B[j]

---

## 4️⃣ AI Infra或算法八股

1. LLM的知识蒸馏放在预训练做是否合适？

2. Hopper TMA的优点，调用方式，是否需要经过L1？

3. Flash Attention v2为什么外层对Q循环，Flash Decoding的combine kernel耗时占比大概是多少？

4. Mooncake kv-cache centric的PD分离

5. DiT的推理框架设计思路和LLM的有什么异同？

6. 分析MLA decode的计算访存比，它和seqlen、batch size是否相关？

7. Diffusion model的训练和推理步骤，推理num_inference_steps为40时，为什么训练的timesteps仍要设成1000？

8. 介绍dLLM，如何看待它和AR的区别？

9. torch.repeat 和 torch.expand的区别？

10. torchrun的启动参数有哪些，如何在Linux上批量kill包含torchrun的进程？
