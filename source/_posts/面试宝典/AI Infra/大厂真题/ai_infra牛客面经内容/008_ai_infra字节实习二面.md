---
title: ai infra字节实习二面
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, AI Infra面经]
tags: [AI Infra, 面经, 牛客]
---

# ai infra字节实习二面

<!-- more -->

**来源**: 牛客网
**链接**: https://www.nowcoder.com/feed/main/detail/eaea5cf9e9e44c5bb5fecf3f1d8243ce

---

## 相关面经问题

### 快手 Ai infra一面拷打

1. 拷打项目
2. 有没有了解过AF分离，他是为了解决什么问题，既然有PD分离了，为什么还要AF分离？
3. 有没有读过flash attention的代码，V2比起V1做了哪些改进？细聊一下他是怎么改进的。有没有了解最近的V4版本？
4. 大模型的一层有几个线性层？TP的时候怎么切的？这样子做的原因是什么？有什么思路优化中间的allreduce吗？
5. 看过ray的底层实现吗？它有什么特性，你的课题研究中是怎么使用ray的？
6. 聊一下你所找到的cuda gemm的优化方法
7. leetcode 单词接龙

### 智谱Ai infra一面面经

1. 实习拷打
2. 简述一下minmax和percentile有什么不同？
3. 你还知道什么其他校准算法吗？回答kl和mse，简单讲了一下中心思想
4. 在上家公司做vla的量化的时候说用了smoothquant，awq。按照量化粒度说明一下smoothquant是做的什么粒度的？了解gptq吗？他们分别的作用流程
5. 上家公司一般量化到什么格式，聊到fp8,nvfp4
6. nvfp4的原理是什么样的，怎么做缩放的，在哪个维度缩放？保存的格式等
7. per-tensor/channel/group，哪个粒度更细？
8. 代码实现一下minmax和percentile

### 美团实习ai infra一面分享

1. 项目拷打
2. 在实习的时候有人带你吗？你是怎么开始上手的？
3. 在你们做量化的过程中，一般是直接应用组内已有工具就能达到可交付的效果吗？要是达不到预期，一般要怎么调量化呢？
4. 然后问了有关vla稍微大点的模型，一般用的多大的模型，讲了awq.smoothquant.gptq原理之类的。假如直接应用这些算法效果不好的话，你对优化这些算法有什么想法吗？有哪些可能可以优化的路径
5. 剪枝稀疏化一般可以分为哪些种类，大概讲一下
6. 对常见的模型架构了解多少？比如说deepseek.qwen.llava他们有什么差异
7. 经常遇到工程问题一般是咋debug的
8. 手撕：最短编辑距离和树的层级遍历
