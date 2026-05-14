---
title: "第4章：经典算子实现—GEMM"
description: "从朴素矩阵乘法到 Shared Memory Tiling、寄存器 Tiling、Tensor Core，逐步逼近 cuBLAS 性能"
pubDate: 2026-04-16
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程进阶]
order: 13
tags: ["CUDA", "GEMM", "矩阵乘法", "Tiling", "Tensor Core", "cuBLAS"]
---

## 本章简介

GEMM（通用矩阵乘法）是深度学习中最核心的算子——线性层、Attention 的 QKV 投影、FFN 的计算本质上都是 GEMM。本章从零实现高性能 GEMM，理解连接硬件和上层框架的桥梁。

**矩阵乘法基础**从朴素三重循环出发，分析为什么 GEMM 是深度学习的核心算子。

**Shared Memory Tiling**详解分块策略：将大矩阵切成 Tile 载入共享内存，从数学上分析全局内存访问的减少量。

**进一步优化**覆盖向量化加载（float4）、寄存器 Tiling、双缓冲预取和 Tensor Core（WMMA API）。

**与 cuBLAS 对比**学习 cuBLAS 的使用方法，对比手写 GEMM 的性能差距，理解实际项目中通常调用 cuBLAS 的原因。

**动手实验**：实现 Shared Memory Tiling GEMM，在 1024×1024 矩阵上达到 cuBLAS 50%+ 性能，用 Nsight Compute 分析瓶颈。
