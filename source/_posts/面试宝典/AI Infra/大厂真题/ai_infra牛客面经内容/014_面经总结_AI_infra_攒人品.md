---
title: 面经总结 AI infra 攒人品
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, AI Infra面经]
tags: [AI Infra, 面经, 牛客]
---

# 面经总结 AI infra 攒人品

<!-- more -->

**来源**: 牛客网
**链接**: https://www.nowcoder.com/feed/main/detail/1fc2981b52cc4166929772b98189b508

---

## 算法题

1. 快排，寻找两个正序数组的中位数，下一个排列，二叉树中的最大路径和，Path Sum III

2. 给定若干点的数轴坐标数组和固定数量的等长线段，问该线段最少要多长才能覆盖所有点

3. 前k个高频字符串，词频一样时按字典序升序排列

4. 给定初始字符串s，每次将字符串向右旋转一次，并将旋转后的字符串拼接到原字符串的末尾，每次操作都会使字符串的长度变为原来的两倍，求计算出无限扩展后的字符串中第N个位置的字符

5. 两根手指放在26个小写字母组成的键盘上，最少移动多少距离才能敲出给定的字符串s

---

## Torch手撕题

- MHA * 3
- Flash Attention v1
- Flow matching model采样的伪代码

---

## AI Infra或算法八股

1. Flow matching模型预测的是什么，怎么理解conditional velocity (conditioned on data sample x0)？

2. 如何计算QwenImage的time shift？

3. 介绍Flash Attention的原理和实现思路

4. GPU matrix transpose使用shared memory的好处

5. CPU按列遍历一个行优先的矩阵相比按行遍历为什么性能会变差，具体是因为哪个性能指标变差导致的？

6. Weight-only量化有哪些，实现weight-only量化cuda kernel时如何优化访存，是否了解Marlin kernel？

7. Megatron SP的实现方式

8. DeepSpeed ZeRO stage1和stage 2的通信量区别，论文和代码实现有没有gap？

9. 多GPU通信时NVSHMEM和NVLink的区别
