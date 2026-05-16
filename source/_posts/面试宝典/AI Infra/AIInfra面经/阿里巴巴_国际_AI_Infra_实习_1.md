---
title: '阿里巴巴 国际 AI Infra 实习 (1)'
date: 2026-04-16 12:08:40
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: FlashAttention的原理？Online Softmax是什么？FlashAttention V1/V2/V3和FlashDecoding的区别？

**原理**：通过tiling将Q/K/V分块加载到SRAM计算，避免在HBM上存储N*N注意力矩阵。

**Online Softmax**：分块计算softmax的关键技术。每处理一个KV块时更新running max和running sum，最终等价于全局softmax结果。公式：`m_new = max(m_old, m_block)`, `l_new = l_old * exp(m_old - m_new) + l_block * exp(m_block - m_new)`。

**版本演进**：
- **V1**：外层遍历KV，内层遍历Q，需要跨block归约。
- **V2**：外层遍历Q（每个Q block独立完成计算），提高并行度；减少非matmul FLOPs。
- **V3**：利用Hopper架构（TMA、Warp Specialization、FP8）。
- **FlashDecoding**：针对decode阶段（batch小seq长），将KV序列维度split到多block并行，最后combine归约。

---

### Q: 推理优化的方法有哪些？

- KV Cache（避免重复计算）。
- 量化（W8A8/W4A16降低计算和访存）。
- Flash Attention/Decoding（IO优化）。
- Speculative Decoding（小模型草稿+大模型验证）。
- Continuous Batching（动态组batch）。
- PagedAttention（KV Cache内存管理）。
- 算子融合（减少kernel launch和中间存储）。
- CUDA Graph（减少CPU开销）。
- Prefix Caching（前缀KV复用）。

---

### Q: vLLM/SGLang的原理？

**vLLM**：核心是PagedAttention，将KV Cache按page管理（类似OS虚拟内存），解决KV Cache碎片问题。配合Continuous Batching实现高效调度。

**SGLang**：在vLLM基础上引入RadixAttention，用Radix Tree管理KV Cache前缀。相同前缀的请求自动共享KV Cache，减少重复计算。还提供高级编程接口（fork/join）支持复杂推理模式。

---

### Q: 什么是Dynamic Batching？

Dynamic Batching是推理服务中动态组batch的策略：不同请求到达时间不同、序列长度不同，系统在满足延迟约束的前提下，将多个请求组成一个batch一起计算以提高GPU利用率。与Continuous Batching结合：已完成的请求及时退出、新请求随时加入，无需等待batch内所有请求完成。

---

### Q: 手撕：CUDA写一个reduce（用block实现），优化版用warp shuffle？

（编程题）

---

### Q: 加载数据到shared memory比直接从HBM取为什么更快？

HBM（全局内存）延迟约400-800个时钟周期，带宽约2TB/s但被所有SM共享。Shared Memory位于SM片上，延迟仅约20-30个时钟周期（快约20倍），且每个SM有独立的带宽不互相竞争。关键场景：当同一数据被多个线程重复访问时，加载一次到shared memory后多次读取的总延迟远低于多次访问HBM。

---

### Q: 什么是Shared Memory的Bank Conflict？

共享内存被划分为32个bank（4字节宽），同一Warp内多个线程同时访问同一bank的不同地址时产生冲突，访问被串行化（N-way conflict需N次访问）。解决：padding使列访问错开bank、重新设计访问模式、利用broadcast（同一地址多线程访问不冲突）。
