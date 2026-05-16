---
title: '蔚来 AI Infra 实习 一面 (2)'
date: 2026-04-16 12:16:07
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 手撕：CUDA实现reduce_sum？

（编程题）

### Q: 手撕：CUDA实现conv2d？

（编程题）

### Q: 手撕：循环buffer（环形缓冲区）？

（编程题）

### Q: 手撕：单链表冒泡排序？

（编程题）

### Q: 大模型推理优化有哪些方法？

大模型推理优化涵盖从**算法层到系统层**的多维度技术栈：

**1. 注意力计算优化（减少核心计算/IO瓶颈）：**

| 技术 | 核心原理 | 效果 |
|------|---------|------|
| KV-Cache | 缓存历史token的K/V，decode时只算新token | 避免O(n)重复计算 |
| FlashAttention | 分块在SRAM计算，避免N×N矩阵落地HBM | 速度2-4x，显存O(N²)→O(N) |
| GQA/MQA | 减少KV head数量 | KV-Cache显存减少4-32倍 |
| PagedAttention | KV-Cache按block分页管理 | 显存利用率接近100% |
| Prefix Caching | 共享前缀复用KV-Cache(如system prompt) | 减少重复prefill |

**2. 量化优化（降低计算/带宽需求）：**
- **Weight-Only量化(INT4/INT8)**：Decode阶段(memory-bound)有效，权重带宽减半/四分之一
- **W8A8量化**：权重+激活都量化，Prefill阶段(compute-bound)也加速
- **KV-Cache量化(INT8/FP8)**：减少长序列KV存储和读取带宽
- **FP8推理(H100)**：计算吞吐翻倍，精度损失极小

**3. 生成加速（减少decode步数）：**
- **投机解码(Speculative Decoding)**：小模型draft K个token + 大模型一次verify → 加速1.5-3x
- **Medusa/EAGLE**：多头并行预测多个候选token
- **Multi-Token Prediction**：模型直接预测多个token

**4. 服务系统优化（提高吞吐）：**
- **Continuous Batching**：iteration级动态调度，消除短请求被长请求阻塞
- **Chunked Prefill**：长prefill分块与decode穿插，降低P99延迟
- **PD分离**：Prefill(compute-bound)和Decode(memory-bound)分到不同GPU
- **CUDA Graph**：消除decode阶段的kernel launch开销

**5. 分布式推理（降低延迟/支持大模型）：**
- 张量并行(TP)：同节点多卡切分单层，减少单次推理延迟
- 流水线并行(PP)：跨节点按层切分
- 专家并行(EP)：MoE模型专家分布到不同卡

### Q: vLLM的核心原理和特点？

vLLM是由UC Berkeley开发的高性能LLM推理引擎，核心创新是**PagedAttention**。

**PagedAttention——借鉴OS虚拟内存分页思想：**

```
传统方式: 为每个请求预分配max_seq_len的连续KV-Cache空间
  → 问题: 大量内部碎片(实际长度远小于max时浪费)，外部碎片(不同请求间空隙)

PagedAttention: KV-Cache分成固定大小的block(如16 tokens/block)
  → 物理block可以分散在显存任意位置
  → 用block_table(类似页表)记录逻辑→物理映射
  → 利用率接近100%（只在最后一个block有少量内部碎片）
```

**vLLM核心特性：**

| 特性 | 原理 | 收益 |
|------|------|------|
| PagedAttention | 非连续KV-Cache分页管理 | 显存浪费从60-80%降到<4% |
| Continuous Batching | 每步动态增删请求 | 吞吐提升2-5x |
| Copy-on-Write | beam search/parallel sampling共享block | 显存节省(p-1)/p |
| Prefix Caching | 共同前缀的请求复用KV block | 减少重复prefill |
| 抢占(Preemption) | 显存不足时swap/recompute | 避免OOM，保证服务可用 |

**架构设计：**
```
Python调度器(Scheduler)
  ├── 管理请求队列(waiting/running/swapped)
  ├── 显存预算管理（决定哪些请求可以run）
  └── 抢占策略（swap to CPU / recompute）
        ↓
C++/CUDA执行引擎(Worker)
  ├── Model Runner: 执行前向计算
  ├── PagedAttention Kernel: 非连续KV读取
  └── Block Manager: 物理block分配/释放
```

### Q: 模型量化相关（量化原理和方法）？

**量化基本原理：**

将连续的浮点值映射到离散的低精度整数表示：
```
量化:   q = round(x / scale) + zero_point
反量化: x_hat = (q - zero_point) × scale
误差:   |x - x_hat| ≤ scale / 2
```

**量化分类维度：**

| 维度 | 选项 | 说明 |
|------|------|------|
| 时机 | PTQ(训练后) / QAT(训练时) | PTQ快但精度可能差；QAT精度好但需重训练 |
| 对象 | Weight-only / W+A / KV-Cache | 不同对象影响不同瓶颈 |
| 粒度 | Per-tensor / Per-channel / Per-group | 粒度越细精度越好但开销越大 |
| 精度 | INT8 / INT4 / FP8 / FP4 | 精度与压缩率的权衡 |

**主流量化方法：**

**1. GPTQ (逐层最优权重量化):**
```
核心: 基于Hessian信息(二阶导数)衡量权重重要性
流程: 按列顺序量化权重，同时调整未量化的权重补偿误差
      min ||WX - Ŵ_q X||² (用Hessian的逆做最优补偿)
适用: 权重量化到INT4/INT3, 精度损失小, 无需训练数据
```

**2. AWQ (激活感知权重量化):**
```
核心: 观察到少量"重要"权重通道对精度影响极大(由激活值大小决定)
流程: 找到重要通道(激活值大的通道) → 对这些通道用更高精度/更小scale
适用: INT4权重量化, 比GPTQ更快(不需要逐列回归)
```

**3. SmoothQuant (平滑量化):**
```
核心: 激活值中有outlier(极大值)导致量化困难
      通过per-channel缩放将难度从激活"转移"到权重
流程: Y = (X / diag(s)) × (diag(s) × W)
      s的选择平衡了X和W的量化难度
适用: W8A8量化(权重和激活同时INT8), 实现真正的INT8 GEMM加速
```

**量化方法选择指南：**
- 追求最小模型体积 → GPTQ/AWQ做INT4权重量化
- 追求计算加速(Prefill) → SmoothQuant做W8A8
- H100推理 → FP8量化（硬件原生支持，精度最好）
- 长上下文推理 → KV-Cache INT8量化（减少显存和带宽）
