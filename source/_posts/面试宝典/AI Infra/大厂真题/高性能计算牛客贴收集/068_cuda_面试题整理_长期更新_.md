---
title: cuda 面试题整理（长期更新）
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, 高性能计算面经]
tags: [高性能计算, 面经, 牛客]
---

# cuda 面试题整理（长期更新）

<!-- more -->

> 来源：https://www.nowcoder.com/feed/main/detail/99ab6b35f4a141a799a60160fa7f1fe2
> 获取时间：2026-04-12
> 获取方式：web_fetch

---

## CUDA面试问题

### 基础问题

1. 对一个CUDA kernel进行优化可以从哪些角度入手
2. GPU L1/L2缓存介绍
3. 同步stream和异步stream的理解

### 手撕题目

1. 矩阵乘法
2. Softmax规约
3. NCHW转NHWC
4. 长度为n的数组，统计每个元素出现的频率（元素大小为0-256）
5. 将数组里面的奇数位置的数放在左边，偶数位置的数放在右边，原地操作

## 经验总结

1. CUDA优化需要从内存访问、计算效率、并行度等方面考虑
2. 熟悉GPU内存层次结构（寄存器、共享内存、全局内存）
3. Bank conflict、coalesced access是常见考点
4. 需要大量练习手撕常见算子