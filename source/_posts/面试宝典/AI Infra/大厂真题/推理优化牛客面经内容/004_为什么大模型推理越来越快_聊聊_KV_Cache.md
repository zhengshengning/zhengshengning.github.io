---
title: 为什么大模型推理越来越快？聊聊 KV Cache
date: 2026-04-13 12:00:00
categories:
  - [求职面试, AI Infra, 推理优化面经]
tags: [推理优化, 面经, 牛客]
---

# 为什么大模型推理越来越快？聊聊 KV Cache

<!-- more -->

> 来源：https://www.nowcoder.com/feed/main/detail/39ddecfddef849b0b17338e5d7ebc8b1
> 获取时间：2026-04-12

---

最近复盘了 KV Cache，给面试/实战一个好记版本：

1）Decoder 生成第 k 个 token 时，历史 token 的 K/V 不必重算，缓存后直接复用；

2）不做缓存会反复算历史注意力，长度一长延迟明显；

3）KV Cache 省算力但吃显存，长上下文时显存压力会成为瓶颈；

4）MHA→MQA→GQA→MLA，本质都在做"少缓存/更聪明缓存"，其中 GQA 是当前工程里很常见的平衡点。

一句话：KV Cache 是速度的来源，注意力变体是显存账本。