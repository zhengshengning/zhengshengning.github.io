---
title: '腾讯 AI Infra 实习 一面 (1)'
date: 2026-04-16 12:06:39
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

攒攒人品！有面试过同岗的朋友欢迎评论区交流
1. 聊项目
2. 聊一下chunk prefill，他是为了解决什么问题而提出的
3. 说一下reduce-scatter和all-to-all通信
4. 怎么减少launch kernel overhead
5. cuda编程中bank conflict是什么，怎么解决？
6. 场景题：一个大集群中有节点内有nvlink，节点间部分机器有rdma，怎么去设计你的分布式推理方案
代码题：k个一组翻转链表
