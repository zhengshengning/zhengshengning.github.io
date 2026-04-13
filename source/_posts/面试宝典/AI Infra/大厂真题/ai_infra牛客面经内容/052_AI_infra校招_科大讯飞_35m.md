---
title: AI infra校招 科大讯飞 35m
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, AI Infra面经]
tags: [AI Infra, 面经, 牛客]
---

# AI infra校招 科大讯飞 35m

<!-- more -->

**来源**: 牛客网
**链接**: https://www.nowcoder.com/feed/main/detail/140a45bc0b314798a0d94b512cb7ea90

**作者**: 起床全靠室友 / 妈妈叫醒
**发布时间**: 2025-03-02 17:55
**学校专业**: 门头沟学院 机器学习

---

## 面经内容

面试很难，还是要多多练习

### 项目深挖问题

1. **Flash Attention**：核心优化点是什么？
   - 分块加载QKV
   - Online Softmax
   - 显存复杂度 O(N²) -> O(N)

2. **Self-Attention**：为什么要除以 √d？
   - 防止点积过大导致Softmax梯度消失

3. 回调函数怎么实现？

4. 显存越界怎么排查？

---

**浏览数**: 1007
**评论数**: 4