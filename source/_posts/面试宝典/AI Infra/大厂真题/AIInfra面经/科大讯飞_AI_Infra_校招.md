---
title: '科大讯飞 AI Infra 校招'
date: 2026-04-16 12:09:58
categories:
  - [求职面试, AI Infra, 大厂真题]
tags: [AIInfra, 高性能计算, 面经, 牛客]
---

# 科大讯飞 AI Infra 校招

<!-- more -->

> 原文链接: https://www.nowcoder.com/feed/main/detail/140a45bc0b314798a0d94b512cb7ea90
> 来源: 牛客网

---

给我面没招了，感觉自己好菜、面试很难，还是要多多练习
项目深挖
1. Flash Attention：核心优化点是什么？（分块加载QKV、Online Softmax、显存复杂度O(N^2)->O(N)）
2. Self-Attention：为什么要除以 √d？（防止点积过大导致Softmax梯度消失）
3. 回调函数怎么实现？
4. 显存越界怎么排查？
