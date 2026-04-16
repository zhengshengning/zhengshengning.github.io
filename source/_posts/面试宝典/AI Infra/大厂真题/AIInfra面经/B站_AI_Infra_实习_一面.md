---
title: 'B站 AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, AI Infra, 大厂真题]
tags: [AIInfra, 算子优化, 高性能计算, 大厂面经, 面经, 牛客]
---

# B站 AI Infra 实习 一面

<!-- more -->

> 原文链接: https://www.nowcoder.com/feed/main/detail/f43b38c668044b31ac4689462303fa0c
> 来源: 牛客网

---

发些面经攒攒人品
1. GPU算子优化通用方法论：profiling定性（memory/compute-bound）
2. 针对性优化（访存连续性/计算简化/block size调整）
3. 项目深挖，问得比较细，具体的优化过的部分都有问到
4. 分布式通信原语理解：all-reduce / all-gather / all-to-all 语义区分
5. 手撕CUDA编程：large array reduce sum 实现（shared memory归约 + 分层kernel设计）
6. 系统基础：进程/线程/协程概念
7. CPU调度粒度（进程级 vs 线程级公平性）
