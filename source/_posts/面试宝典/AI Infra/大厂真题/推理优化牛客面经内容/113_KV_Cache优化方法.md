---
title: KV Cache优化方法
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, 推理优化面经]
tags: [推理优化, 面经, 牛客]
---

# KV Cache优化方法

<!-- more -->

## 背景

KV Cache是大模型推理的重要优化技术

## 优化方法

### 压缩方法

1. 量化压缩
2. 剪枝压缩
3. 低秩压缩

### 存储优化

1. Paged Attention
2. 分页管理
3. 内存复用

### 计算优化

1. Flash Attention
2. 稀疏注意力
3. 窗口注意力

## 技术要点

KV Cache优化是提升推理性能的关键技术之一。

来源：牛客网
