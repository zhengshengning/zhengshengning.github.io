---
title: 阿里国际AI Infra实习凉经
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, AI Infra面经]
tags: [AI Infra, 面经, 牛客]
---

# 阿里国际AI Infra实习凉经

<!-- more -->

**来源**: 牛客网  
**链接**: https://www.nowcoder.com/feed/main/detail/6cbfd441972d4a96ae47e1cdf54a3fef

**作者**: 找工小蓝莓  
**发布时间**: 01-30 05:40

---

## 面经内容

### 面试问题

1. 实习介绍
2. 针对项目提问
3. FlashAttention原理，Online softmax，有没有看过cuda kernel，FlashAttention V1,V2,V3，FlashDecoding原理
4. 推理优化的思路（方法）有哪些
5. 用没用过vllm/SGLang，原理
6. 有没有听说过Dynamic Batching

### Coding

7. 写一个reduce，用block，优化版：用warp shuffle，能不能再优化？

### 深度问题

8. 加载到shared memory和直接从HBM取input比为什么更快
9. 有没有听说过shared memory的bank conflict

### 反问

10. 在哪些地方可以继续提升自己

---

## 评论区讨论

**嵌入式的小白**: 你这都没手撕啊，你是啥时候面的  
**李橙子**: 加载到shared memory和直接从HBM取input比为什么更快咋回答的  
**GoldenPotato**: 看你使用场景吧，如果你的数据读进来立刻参与计算且只用一次的话，和先读到shared memory再计算应该没特别大区别。加载到shared memory一般是为了pipeline化，把访存耗时和计算重叠起来；此外，有可能全局的layout和实际计算所需的layout不一致，为了合并全局访存，可以用不同的线程组织方式把数据先读到shared memory做中转，然后再用另外一种组织方式计算

---

**浏览**: 1443  
**评论**: 3