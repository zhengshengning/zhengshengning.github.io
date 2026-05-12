---
title: "第2章：CUDA 性能优化基础"
description: "掌握 Warp 执行模型、内存访问优化、Occupancy 调优和同步机制，建立 CUDA 性能优化的核心方法论"
pubDate: 2026-04-16
category: "cuda-optimization"
order: 11
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程基础]
tags: ["CUDA", "Warp", "合并访问", "Occupancy", "性能优化"]
---

## 本章简介

写出能跑的 CUDA 代码只是起点，写出跑得快的代码才是 AI Infra 工程师的核心能力。本章建立 CUDA 性能优化的核心方法论。

**Warp 与执行模型**详解 SIMT 执行模式、Warp Divergence 导致的性能损失，以及 Warp Shuffle 指令实现线程间高效数据交换。

**内存访问优化**是性能提升最大的杠杆：Coalesced Access（合并访问）决定全局内存效率，Bank Conflict 影响共享内存性能（Padding 技巧解决），向量化加载（float4/int4）进一步提升带宽利用率。

**Occupancy 与资源分配**解释 Occupancy 的定义与意义，分析影响因素（寄存器数、共享内存、Block 大小），强调 Occupancy 不是越高越好——需要在 Latency Hiding 和 Resource Utilization 之间找平衡。

**同步与原子操作**涵盖 `__syncthreads()` 块内同步和原子操作的使用与性能影响。
