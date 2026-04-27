---
title: "第1章：CUDA 编程入门"
description: "搭建 CUDA 开发环境，理解 Grid/Block/Thread 编程模型和内存模型，编写第一个实用 CUDA Kernel"
pubDate: 2026-04-16
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程基础]
order: 10
tags: ["CUDA", "编程模型", "内存模型", "Kernel"]
---

## 本章简介

本章是 CUDA 编程的起点，带你从零搭建开发环境并写出第一个高性能 Kernel。

**开发环境搭建**涵盖 CUDA Toolkit 安装与版本管理、nvcc 编译工具链和 CMake 集成，以及编写第一个 Hello World kernel。

**编程模型**部分详解 Grid/Block/Thread 三级线程层次、线程索引计算（threadIdx/blockIdx/blockDim/gridDim）、Kernel Launch 语法 `<<<gridDim, blockDim>>>` 以及 Block 大小的选择策略。

**内存模型**部分介绍全局内存、共享内存、寄存器、常量内存和统一内存的特性与适用场景。

**第一个实用 Kernel**以向量加法为例，走通 cudaMalloc → cudaMemcpy → kernel launch → 错误检查的完整流程，并对比 CPU 和 GPU 耗时。
