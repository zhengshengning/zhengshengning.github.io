---
title: oppo AI infra实习二面 好难
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, AI Infra面经]
tags: [AI Infra, 面经, 牛客]
---

# oppo AI infra实习二面 好难

<!-- more -->

**来源**: 牛客网
**链接**: https://www.nowcoder.com/feed/main/detail/101e205200ab480db5677ee2852e7d91

**作者**: JulIus
**发布时间**: 2025-03-02 19:20
**学校专业**: 门头沟学院 机器学习

---

## 面经内容

发一下问题给大家参考，攒攒人品！有面试过同岗的朋友欢迎评论区交流

### 项目拷打

**1. 数据布局详解：NHWC vs NCHW：在训练/推理中怎么选？**

**2. 何时应该关闭 Shared Memory？**
- 当出现 Bank Conflict 严重或收益不如直接访问 L2 时

**3. 特定 Shape 导致使用 Shared Memory 时结果异常如何排查？**

**4. Thread/Warp/Block/SM/Grid 的映射关系**

**5. 如何确定最优线程数？**

**6. 异步设计：CUDA Stream 的使用前提**
- 无内存访问重叠

**7. 算子融合决策：什么场景适合融合？**

---

**浏览数**: 807
**评论数**: 1