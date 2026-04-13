---
title: CUDA面经
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, 高性能计算面经]
tags: [高性能计算, 面经, 牛客]
---

# CUDA面经

<!-- more -->

> 来源：https://www.nowcoder.com/feed/main/detail/d8c2f7acc55745ea8caa24257d8da1c9
> 获取时间：2026-04-12
> 获取方式：web_fetch

---

## 面试背景

实习和秋招都面了一些高性能计算的岗，分享一点被问过的CUDA八股。

## CUDA八股问题

1. SIMT是什么
2. occupancy和什么有关，怎么控制
3. bank conflict粒度
4. GEMM分块大小受什么影响
5. float4读写gmem为什么更快
6. block能否被调度到不同SM上
7. 常用卡的cache是多大
8. divergency对性能的影响
9. nvidia gpu的指令级并行

## 手撕题目

1. 矩阵转置
2. 向量外积

## 评论问答

**问：这个赛道很卷吗？**

答：市面上会CUDA的应该很少，挺好找工作的。

**问：float4为什么会更快？**

答：float4是一次性读写128位数据，可以提高内存带宽利用率。

**问：点技能树的时候，会学TVM推理引擎、训练框架这些吗？**

答：建议学习TVM、TensorRT等推理引擎。

## 经验总结

1. CUDA是高性能计算的核心技能
2. 需要深入理解GPU架构
3. 多练习常见算子的手撕实现
4. 了解推理引擎和训练框架