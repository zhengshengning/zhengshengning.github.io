---
title: '快手 AI Infra 实习 一面 (2)'
date: 2026-04-16 12:12:47
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 推理框架中算子如何优化？如何构建 Memory Pool？如何对框架进行测试？

**算子优化——系统化方法论**：

1. **利用 Tensor Core 加速矩阵运算**：
   - 使用 WMMA（Warp Matrix Multiply-Accumulate）或 MMA PTX 指令
   - 确保矩阵维度对齐 Tensor Core 要求（通常 16 的倍数）
   - A100 上 FP16 Tensor Core 吞吐是 CUDA Core 的 16x

2. **共享内存 Tiling**：
   - 将 Global Memory 数据分块加载到 Shared Memory（带宽 ~19 TB/s vs HBM ~2 TB/s）
   - 每个 block 内多次复用同一份数据（数据复用率 = tile 大小相关）
   - 典型 tile size：128×128 或 64×64

3. **算子融合**：
   - 识别可融合的 pattern：GEMM + Bias + Activation、LayerNorm + Linear
   - 减少 kernel launch 开销（每次 ~5-10μs）和中间 tensor 的 HBM 读写
   - 融合后中间结果留在寄存器/Shared Memory 中

4. **向量化访存**：
   - 使用 `float4`/`int4` 一次加载 128-bit（4 个 float32 或 8 个 float16）
   - 减少内存事务数量 4x，提升有效带宽

5. **避免 Bank Conflict 和 Warp Divergence**：
   - Shared Memory padding 消除 bank conflict
   - 减少分支（if-else）保证 warp 内线程执行相同路径

---

**Memory Pool 构建**：

设计目标：避免频繁 `cudaMalloc`/`cudaFree`（每次耗时约 1ms），实现微秒级分配释放。

**架构设计**：
```
┌─────────────────────────────────────────┐
│  预分配的大块连续显存（如 4GB）            │
├────┬────┬────┬────┬────┬────┬────┬────┤
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │  固定大小 Block
└────┴────┴────┴────┴────┴────┴────┴────┘
         ↕
┌─────────────────┐
│ Free Block List  │ → B1 → B4 → B6 → ...
└─────────────────┘
```

**关键设计决策**：
- **分级管理（Slab Allocator）**：预定义几种 block 大小（如 1MB、4MB、16MB、64MB），每种大小独立管理，减少碎片
- **Buddy System**：按 2 的幂次管理，支持分裂和合并，适合变长分配
- **延迟释放**：释放时不立即归还 OS，保留在 pool 中备用
- **对齐**：所有分配对齐到 256 bytes（GPU 访存对齐要求）

---

**框架测试策略**：

1. **正确性测试**：与 PyTorch 参考实现对比输出
   - Cosine Similarity > 0.999
   - Max Absolute Error < 1e-3（FP16）
   - 不同输入 shape（边界情况：batch=1、seq=1、非对齐维度）

2. **性能测试**：
   - 吞吐：tokens/s（Decode）、prompt_tokens/s（Prefill）
   - 延迟：TTFT（首 token 延迟）、TPOT（每 token 耗时）
   - 对比基准：vLLM/TensorRT-LLM/HuggingFace

3. **压力测试**：长序列（128K）、大 batch（256+）、多并发请求、长时间运行稳定性

4. **精度测试**：不同量化配置下的 PPL、MMLU、GSM8K 等 benchmark 指标

---

### Q: vLLM 的 PagedAttention、Chunked Prefill 和 Continuous Batching？

**PagedAttention**：

核心思想：将 KV Cache 按固定大小 block（如 16 tokens）分页管理，类似 OS 的虚拟内存。

- **Block Table**：每个请求维护逻辑 block → 物理 block 的映射表
- **按需分配**：只在序列增长时分配新 block，不预留 max_seq_len
- **消除碎片**：物理 block 可以不连续，彻底消除内部碎片和外部碎片
- **显存利用率**：从传统方案的 20-40% 提升到接近 100%（浪费仅来自最后一个 block 的未填满部分）
- **COW 共享**：Beam search 中多个 beam 共享公共前缀 block

**Chunked Prefill**：

解决的问题：长 prompt（如 8K tokens）的 Prefill 是一个大 GEMM，可能独占 GPU 200-500ms，导致正在 Decode 的请求 TPOT 飙升。

解决方案：
- 将长 prompt 分成 chunk（如每 chunk 512 tokens）
- 每个 chunk 计算完后，将当前 batch 中其他请求的 Decode token **一起计算**
- 形成交错：`[Prefill chunk₁ + Decode tokens] → [Prefill chunk₂ + Decode tokens] → ...`
- 关键：chunk 内 Prefill tokens 和 Decode tokens 可以组成一个**混合 batch** 一起执行 GEMM，利用 GPU 并行性

**Continuous Batching**：

传统 Static Batching 的问题：一个 batch 内所有请求必须同时开始、同时结束。短请求生成完成后 GPU idle 等待长请求。

Continuous Batching 改进：
- **每个 iteration（一次 forward）动态调整 batch 组成**
- 已完成的请求立即移除，腾出的 KV Cache 空间立即给等待队列中的新请求
- 新请求可以在任意 iteration 加入（无需等整个 batch 完成）
- 效果：GPU 始终满载，吞吐提升 2-10x（取决于请求长度分布的不均匀程度）

---

### Q: 手撕：二维数组每行有序，求第 K 小的元素？

（编程题）
