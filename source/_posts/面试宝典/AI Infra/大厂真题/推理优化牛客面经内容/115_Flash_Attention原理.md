---
title: Flash Attention原理
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, 推理优化面经]
tags: [推理优化, 面经, 牛客]
---

# Flash Attention原理

<!-- more -->

## 背景

Flash Attention是高效注意力计算方法

## 核心思想

### 分块计算

- 将注意力矩阵分块
- 减少内存访问
- 提高计算效率

### IO感知

- 优化GPU内存访问
- 减少HBM读写
- 利用SRAM缓存

## 性能提升

- 显存占用减少
- 计算速度提升
- 支持更长序列

## 应用

- 大模型训练
- 大模型推理
- 长文本处理

来源：牛客网
