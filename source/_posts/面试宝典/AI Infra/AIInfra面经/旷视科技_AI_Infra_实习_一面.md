---
title: '旷视科技 AI Infra 实习 一面'
date: 2026-04-16 12:16:31
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: vLLM 推理框架的核心机制？

vLLM 是面向 LLM 的高吞吐推理框架，其核心设计围绕**最大化 GPU 利用率和并发请求数**：

**PagedAttention（核心创新）**：
- KV Cache 按固定大小 block（16 tokens）分页管理
- Block Table 维护逻辑→物理映射，类似 OS 页表
- 消除内部/外部碎片：传统方案预留 max_seq_len 连续空间造成 60-80% 浪费
- 利用率从 20-40% 提升到接近 100%，等效于同等显存可服务 **2-4x 更多并发请求**
- 支持 Copy-on-Write：beam search 中多个 beam 共享公共前缀 block

**Continuous Batching（动态批处理）**：
- 不等 batch 全部完成，每个 forward iteration 动态加入/移除请求
- 短请求完成后立即释放资源给等待队列中的新请求
- 相比 static batching 吞吐提升 2-10x（取决于请求长度分布）

**Prefix Caching（前缀共享）**：
- 相同 system prompt 的请求自动共享已计算的 KV Cache
- 场景：多轮对话中系统提示词 / RAG 中相同上下文
- 节省重复 Prefill 计算

**Scheduler + Preemption（调度与抢占）**：
- FCFS（先来先服务）基本调度
- 显存不足时抢占低优先级请求：swap（KV Cache 换出到 CPU）或 recompute（释放后重新 prefill）
- 保证高优先级请求的 SLA

**分布式支持**：
- Tensor Parallel（单节点多卡）
- Pipeline Parallel（多节点）
- 基于 Ray 的 worker 管理

---

### Q: CUDA 算子优化常用的方法有哪些？

系统化的 CUDA 算子优化方法，按重要性排序：

**1. 合并访存（Coalesced Access）**——最基础：
- 确保 warp 内 32 线程访问连续 128 字节
- 非合并访问的带宽损失可达 8-32x
- 检查方法：NCU 的 Memory Workload Analysis 中 Sector Efficiency

**2. Shared Memory Tiling**——减少重复访问：
- Global Memory（~2 TB/s）→ Shared Memory（~19 TB/s）
- 数据加载到 Shared Memory 后 block 内多次复用
- 典型场景：GEMM、Conv、Softmax

**3. 向量化加载（float4 = 128-bit）**：
- 一条指令加载 4 个 float 或 8 个 half
- 减少内存事务数和指令数
- 要求加载地址 16 字节对齐

**4. 减少 Bank Conflict**：
- Shared Memory 32 bank 交错，同 bank 不同地址访问串行
- 解决：Padding（每行末加 1 元素）或 Swizzle（重映射地址）

**5. 循环展开（`#pragma unroll`）**：
- 展开消除循环控制开销（比较+跳转指令）
- 给编译器更多指令级并行的优化空间
- 注意：过度展开增加寄存器压力

**6. Warp 级原语**：
- `__shfl_sync`：warp 内寄存器直接交换（1 cycle，比 shared memory 快）
- 适合 warp 内 reduce/broadcast/scan 操作
- 避免 shared memory 的 load/store/sync 开销

**7. 算子融合**：
- 消除中间 tensor 的 HBM 读写（每次融合节省 2×tensor_size 的带宽）
- 减少 kernel launch 开销（~5-10μs/launch）
- 典型 pattern：GEMM+Bias+Activation、LayerNorm+Residual

**8. 提高 Occupancy**：
- 控制寄存器使用（`__launch_bounds__`、`--maxrregcount`）
- 控制 shared memory 使用
- 但 occupancy 不是越高越好——需要与数据复用率平衡

**9. 异步拷贝（`cp.async`/TMA）**：
- Ampere+ 支持 Global→Shared 的异步拷贝，不占用 SM 计算资源
- Hopper 的 TMA 更强：支持多维 tensor 直接异步搬运
- 实现 compute 与 memory 流水化

---

### Q: 手撕：链表倒数第 K 个节点？

（编程题）
