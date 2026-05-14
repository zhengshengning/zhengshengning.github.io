---
title: "第5章：经典算子实现—Softmax 与算子融合"
description: "实现数值稳定的 Softmax 和 Online Softmax，掌握算子融合的原理与实践"
pubDate: 2026-04-16
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程进阶]
order: 14
tags: ["CUDA", "Softmax", "Online Softmax", "算子融合", "Kernel Fusion"]
---

## 本章简介

Softmax 是 Transformer 中最关键的非线性操作之一，也是理解 FlashAttention 的前置知识。本章从数值稳定性问题出发，逐步优化到 Online Softmax，并引入算子融合的思想。

**1. Softmax 的数值稳定实现**：分析朴素实现的数值溢出问题，引入 Safe Softmax（减最大值技巧），并分析三遍扫描（max → exp-sum → normalize）的性能开销。

**2. Online Softmax**：讲解 Online normalizer calculation 原理，推导一遍扫描完成 Softmax 的算法，并在 GPU 上高效实现——这是 FlashAttention 的核心前置知识。

**3. 算子融合**：解释为什么需要融合（减少 Kernel Launch 和全局内存读写），介绍常见融合模式（Bias + Activation、LayerNorm + Dropout）以及手动融合与编译器自动融合的对比。

**4. 动手实验**：实现 Online Softmax CUDA Kernel，将 Softmax + Scale 融合为一个 Kernel 并对比性能。
