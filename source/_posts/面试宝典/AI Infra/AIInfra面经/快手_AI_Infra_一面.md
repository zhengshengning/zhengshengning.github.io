---
title: '快手 AI Infra 一面'
date: 2026-04-16 12:06:24
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: AF 分离是什么？为什么有了 PD 分离还要 AF 分离？

**AF 分离（Attention-FFN Disaggregation）**：将 Transformer 层中的 Attention 部分和 FFN（MLP）部分分别部署在不同的硬件节点上。

**为什么 PD 分离不够——需要更细粒度的分离**：

PD 分离解决的是**阶段级别**的异构性：
- Prefill = 大 batch GEMM → compute-bound
- Decode = batch=1 GEMV → memory-bound

AF 分离解决的是**同一阶段（Decode）内部**的异构性：
- **Attention**：需要读取大量 KV Cache（memory-bound），对**带宽**要求高
- **FFN**：纯矩阵乘（compute-bound），对**算力**要求高

| 组件 | 瓶颈 | 最佳硬件 |
|---|---|---|
| Attention (Decode) | KV Cache 读取带宽 | 高带宽 GPU（HBM3, 3.35TB/s） |
| FFN (Decode) | 矩阵乘计算量 | 高算力 GPU（多 Tensor Core） |

**AF 分离的典型架构**：
```
Decode 请求 → Attention Node (高带宽) ← KV Cache 存储
              ↓ 中间结果传输
              FFN Node (高算力) → 输出 token
              ↓
              Attention Node → ... (循环)
```

**特别适合 MoE 模型**：
- FFN 中有多个 Expert，需要 All-to-All 通信做 token dispatch
- 如果 Attention 和 FFN 绑定在一起，All-to-All 会阻塞 Attention 计算
- AF 分离后 FFN 节点独立做 Expert Parallel + All-to-All，不影响 Attention 节点

**挑战**：Attention→FFN 之间的中间激活传输延迟，需要高速网络（NVLink/RDMA）。

---

### Q: FlashAttention V2 相比 V1 做了哪些改进？V4 版本了解吗？

**FlashAttention-2 的四大改进**：

**1. 减少非矩阵乘 FLOPs**：
- V1 每处理一个 K/V block 都要对输出做 rescaling：`O = O × (l_old/l_new)`
- V2 将 rescaling **延迟到循环结束**：累加时不归一化，最后统一除以 l_final
- 减少了 O(N/B) 次 elementwise 操作，这些操作不能利用 Tensor Core

**2. 增加序列维度并行**：
- V1 只在 batch × num_heads 维度并行（每个 block 处理一个 head 的全部 seq）
- V2 增加 Q block 间的并行：不同 Q block 分配给不同 thread block
- 短 batch / 少 head 时（batch=1, heads=8），V1 只能利用 8 个 block；V2 可利用 8×(N/B_q) 个 block

**3. 优化 Warp 间工作划分**：
- V1 使用 split-K：一个 warp 处理 K/V 的一部分，最后多个 warp 需要同步合并结果
- V2 让每个 warp 处理**完整的 K/V block**，消除 warp 间同步开销
- 同一 block 内不同 warp 处理不同的 Q 行

**4. 支持更大 head_dim**：V1 只支持 head_dim ≤ 128，V2 支持 256

**性能提升**：V2 在长序列上比 V1 快约 2x（主要来自更好的并行和更少的非矩阵乘操作）。

---

**FlashAttention-3（Hopper 架构，2024）**：

利用 Hopper 的三个新硬件特性：
- **TMA 异步加载**：由硬件完成地址计算+数据搬运，SM 不参与
- **Warp-specialization**：producer warp 专门做数据加载，consumer warp 专门做计算，流水化
- **FP8 Tensor Core**：支持 FP8 attention 计算，吞吐再翻倍

注意：目前公开文献中没有 "FlashAttention-4" 的正式版本（截至 2025）。

---

### Q: 大模型一层有几个线性层？TP 怎么切？为什么这样切？如何优化中间的 AllReduce？

**线性层数量**（以 LLaMA 架构为例）：

每个 Transformer 层有 **7 个线性层**（可融合为 4-5 个）：
- Attention：Q_proj、K_proj、V_proj（通常融合为 QKV_proj）、O_proj
- MLP：gate_proj、up_proj（通常融合为 gate_up_proj）、down_proj

**TP 切法和原因**：

```
                    Column Parallel                    Row Parallel
输入 x ──→ [Q/K/V/gate/up_proj] ──→ 局部计算 ──→ [O/down_proj] ──→ AllReduce ──→ 输出
              每卡存部分列                              每卡存部分行
```

- **Q/K/V/gate/up_proj：列切分（Column Parallel）**
  - 权重 [hidden, out_dim] 按 out_dim 切分为 [hidden, out_dim/P]
  - 每卡计算部分输出特征：`y_partial = x × W_shard`
  - 不需要通信！输入 x 每卡相同，各卡独立计算即可

- **O/down_proj：行切分（Row Parallel）**
  - 权重 [in_dim, hidden] 按 in_dim 切分为 [in_dim/P, hidden]
  - 每卡处理部分输入特征：`z_partial = y_shard × W_shard`
  - 输出是 hidden 维度的**部分和**，需要 AllReduce 求和

**为什么这样配对**：列切分的输出恰好是行切分需要的输入分片，中间无需通信，一对列+行切分只引入 1 次 AllReduce。

**优化 AllReduce 的方法**：

1. **计算通信重叠**：将 AllReduce 拆分为 ReduceScatter + AllGather，与下一层计算重叠
2. **Sequence Parallel（SP）**：
   - 用 ReduceScatter 替代 AllReduce 后半段，结果按 seq 维度分片
   - LayerNorm/Dropout 在分片数据上计算（每卡只处理 seq/P 的 token）
   - 下一层开始前做 AllGather 恢复全量
   - 好处：LayerNorm 计算也被分摊，activation memory 减少
3. **通信压缩**：AllReduce 的数据用 FP8 传输，接收端再转回 FP16/BF16
4. **Ring → Tree AllReduce**：小消息时 Tree 延迟更低

---

### Q: Ray 的底层实现和特性？

Ray 是一个通用分布式计算框架，在 AI Infra 中广泛用于推理服务编排（vLLM）和 RLHF 训练编排（OpenRLHF/veRL）。

**核心架构**：

```
┌─────────────────────────────────────────┐
│           GCS (Global Control Store)     │ ← 集中式元数据管理
├─────────────────────────────────────────┤
│  Local Scheduler  │  Local Scheduler    │ ← 每节点一个
├───────────────────┼─────────────────────┤
│  Worker Process   │  Worker Process     │ ← 执行 task/actor
│  Object Store     │  Object Store       │ ← 共享内存数据存储
└───────────────────┴─────────────────────┘
```

**核心特性**：

1. **Task-based 并行**：`@ray.remote` 将 Python 函数/类变为分布式 task/actor
   ```python
   @ray.remote
   def compute(x): return x * 2
   futures = [compute.remote(i) for i in range(100)]
   results = ray.get(futures)
   ```

2. **Object Store（基于 Apache Arrow）**：
   - 每个节点有共享内存的 object store
   - 同节点 task 间数据传递 = 零拷贝（shared memory）
   - 跨节点传递通过 gRPC/RDMA

3. **调度器（分层）**：
   - Local Scheduler：优先在本节点分配 task（数据局部性）
   - Global Scheduler：跨节点负载均衡
   - 支持依赖感知调度（DAG 中的任务按拓扑序执行）

4. **Fault Tolerance**：task 失败自动重试，actor 崩溃可恢复 checkpoint

5. **适用场景**：
   - vLLM 用 Ray 做多节点推理的 worker 管理和请求分发
   - OpenRLHF/veRL 用 Ray 编排 generation + reward + training 多组件
   - 超参搜索（Ray Tune）

---

### Q: CUDA GEMM 的优化方法有哪些？

从基础到极致，GEMM 优化有 9 个层次：

**1. Tiling（Block → Thread → Warp 多级分块）**：
- Grid-level：每 block 负责 C 的 BM×BN tile
- Thread-level：每线程负责 TM×TN 输出元素
- Warp-level：Tensor Core MMA 操作的基本粒度（16×16×16）

**2. Shared Memory 缓存**：
- Global Memory 延迟 ~400-800 cycles，Shared Memory ~5-30 cycles
- 从 HBM 加载 tile 到 Shared Memory，block 内多次复用
- 数据复用率 ∝ min(BM, BN) / BK

**3. 双缓冲/多缓冲（Prefetch）**：
- 分配 2+ 份 shared memory buffer
- 计算 buf[i] 时异步加载 buf[i+1]
- 彻底隐藏 global→shared 加载延迟

**4. 向量化访存**：
- `float4`/`LDS.128` 一次 128-bit 加载
- 减少内存事务数 4x，提升有效带宽
- 要求地址 16 字节对齐

**5. Register Tiling**：
- 每线程在寄存器中维护 TM×TN 的累加器
- 寄存器带宽无限，是最高效的数据复用层级
- TM×TN 越大复用越好，但寄存器用量增加

**6. Tensor Core（WMMA/MMA）**：
- 硬件加速的 16×16×16 矩阵乘加
- FP16 吞吐是 CUDA Core 的 16x（A100）
- 需要确保数据格式和对齐满足 Tensor Core 要求

**7. 避免 Bank Conflict**：
- Shared Memory 32 bank × 4 bytes 交错
- Padding：`smem[BM][BK+1]` 错开 bank 映射
- Swizzle：重新映射地址避免冲突

**8. Software Pipelining**：
- 将 load/compute/store 分解为独立阶段
- 不同阶段对应不同硬件单元（DMA/Tensor Core/Store Queue）
- 三者流水化执行，最大化硬件利用率

**9. Split-K**：
- 当 M×N 太小时（如 GEMV），block 数不足以填满所有 SM
- 将 K 维也切分到多个 block，每个 block 计算部分 K 的累加
- 最后用额外 kernel 做 K 维归约
- 适合 Decode 阶段的瘦矩阵乘

---

### Q: 手撕：LeetCode 单词接龙（BFS 最短路径）？

（编程题）
