---
title: CUDA编程基础
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, 推理优化面经]
tags: [推理优化, 面经, 牛客]
---

# CUDA编程基础

<!-- more -->

## 背景

CUDA是NVIDIA GPU的并行计算平台

## 基础概念

### 执行模型

- Grid、Block、Thread
- Warp执行
- SIMT架构

### 内存层次

- 全局内存
- 共享内存
- 寄存器

### 编程模型

- Kernel函数
- 内存管理
- 同步机制

## 优化技巧

1. 内存合并访问
2. 共享内存使用
3. 避免分支发散
4. 指令优化

来源：牛客网
