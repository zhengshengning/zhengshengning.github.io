---
title: CUDA算子手撕与面试
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, 高性能计算面经]
tags: [高性能计算, 面经, 牛客]
---

# CUDA算子手撕与面试

<!-- more -->

> 来源：https://www.nowcoder.com/discuss/697901950464954368
> 获取时间：2026-04-12

---

## 项目介绍

本项目是 CUDA 算子手撕与面试指南：

1. 汇总了面试高频的 CUDA 算子题目和优化策略，包含面试高频算子的编写示例
2. 项目从算子 naive 实现到优化版本均包含完整代码，便于调试与性能分析
3. 每个算子附有相关的 GPU 知识点，帮助求职者高效备战 CUDA 编程面试

## 目前覆盖以下 CUDA 常见算子及其优化版本：

| 文件夹 | 描述 | 内容 |
|--------|------|------|
| example | 一些简单的例子 | / |
| elementwise | 数组对应元素计算 | add |
| gemv | 矩阵乘向量 | sgemv |
| reduce | 归约计算优化 | sum, max, softmax, softmax_matrix |
| sgemm | 矩阵乘优化 | naive, blocktile, threadtile, ... |
| transpose | 矩阵转置优化 | naive, 优化访存并解决bank conflict |

## 算子手撕说明

面试时不会提供 CUDA 运行环境，也不会要求完整写出可以运行的代码，通常只需要写出 CUDA 算子函数（大部分情况只需要写这个），block_size，grid_size 和函数调用。