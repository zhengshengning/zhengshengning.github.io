---
title: GPU架构与存储体系
date: 2026-03-30 10:00:00
categories:
  - [AI Infra, 前置知识, GPU硬件概述]
order: 502
chapter: 5
tags: [GPU, NVIDIA, Tensor Core, HBM, Memory Wall]
---

在动手写 CUDA kernel 之前，先搞清楚 GPU 的"硬件说明书"是必修课——不了解工厂有多少工人、车间怎么布局、仓库带宽多大，写出来的程序大概率在"等数据"中浪费生命。本文从 GPU 内部的计算单元讲到存储层次，再到制约性能的 Memory Wall，最后横向对比主流数据中心 GPU 的关键参数，帮你建立一套从硬件视角思考性能问题的直觉。

<!-- more -->

## 📑 目录

- [1. GPU 计算单元：SM、CUDA Core 与 Tensor Core](#1-GPU-计算单元：SM、CUDA-Core-与-Tensor-Core)
- [2. 存储层次结构](#2-存储层次结构)
- [3. Memory Wall：为什么带宽往往比算力先成为瓶颈](#3-Memory-Wall：为什么带宽往往比算力先成为瓶颈)
- [4. 数据中心 GPU 规格对比与选型](#4-数据中心-GPU-规格对比与选型)
- [5. 总结](#5-总结)
- [自我检验清单](#🎯-自我检验清单)
- [参考资料](#📚-参考资料)

---

## 1. GPU 计算单元：SM、CUDA Core 与 Tensor Core

### 1.1 SM：GPU 的基本调度单元

SM（Streaming Multiprocessor，流多处理器）是 GPU 最核心的组织单元。如果把整块 GPU 比作一座大型工业园区，SM 就是园区里的一间间标准化车间——每间车间配备了自己的工人（计算单元）、工具柜（寄存器）、临时物料架（共享内存），以及一位调度组长（Warp Scheduler）负责安排工人干活。数十上百个车间同时开工，整座园区的产能就是这些车间产能的总和。

具体来说，每个 SM 内部包含：

- **CUDA Core**：执行标量浮点和整数运算的通用计算单元
- **Tensor Core**：执行矩阵乘累加运算的专用加速单元
- **Shared Memory / L1 Cache**：片上高速缓存，同一线程块内的线程可共享访问
- **Register File**：每个线程独享的最快存储
- **Warp Scheduler**：以 Warp（32 个线程为一组）为粒度调度执行

一块 GPU 拥有的 SM 数量直接决定了它的并行规模：

| GPU 型号 | 架构代号 | SM 数量 |
|---------|---------|:------:|
| A100 SXM | Ampere | 108 |
| H100 SXM | Hopper | 132 |
| H200 SXM | Hopper | 132 |
| B200 SXM | Blackwell | 148 |

每个 SM 同一时刻可以管理多达 64 个 Warp（共 2048 个线程）。当某个 Warp 因为等待数据而暂停时，调度器会立刻切换到另一个就绪的 Warp 继续执行——这种**延迟隐藏**机制是 GPU 高吞吐的关键：与其让工人干等材料送来，不如先派他去干别的活。

### 1.2 CUDA Core：通用计算的基本力量

CUDA Core 是 GPU 上最基础的标量运算单元。每个 CUDA Core 在一个时钟周期内完成一次浮点或整数运算，适合处理激活函数（ReLU、GELU）、LayerNorm、逐元素加减等操作。

CUDA Core 遵循 **SIMT（Single Instruction, Multiple Threads）** 执行模型：一条指令驱动 32 个线程（一个 Warp）同步执行相同操作，每个线程处理不同的数据。这比 CPU 的 SIMD 向量指令更灵活——线程可以走不同的分支（虽然效率会降低）。

以 H100 SXM 为例：
- 每个 SM 包含 128 个 FP32 CUDA Core
- 全片 132 个 SM，共计 16,896 个 CUDA Core
- FP32 理论峰值 = 16896 核 x 2 FMA/cycle x 1830 MHz ≈ **67 TFLOPS**

### 1.3 Tensor Core：矩阵运算的专用引擎

如果 CUDA Core 是"逐个计算的计算器"，Tensor Core 就是"一次处理一整块矩阵的批量引擎"。Tensor Core 专为**矩阵乘累加（MMA, Matrix Multiply-Accumulate）** 设计，一条指令就能完成一个小矩阵块（如 16x16）的乘加运算，效率比 CUDA Core 高出一个数量级。

深度学习的核心运算——全连接层（GEMM）和注意力机制（Attention）——本质都是大规模矩阵乘法，天然适配 Tensor Core。可以说，**Tensor Core 是大模型训练和推理的真正主力**，CUDA Core 更多负责"边角料"工作。

**Tensor Core 随架构演进不断扩展支持精度**：

| 架构 | 代表 GPU | 新增精度支持 | 标志性特性 |
|------|---------|------------|----------|
| Volta | V100 | FP16 | 首次引入 Tensor Core |
| Ampere | A100 | BF16, TF32, INT8, INT4 | 2:4 结构化稀疏（算力翻倍） |
| Hopper | H100 | FP8 | Transformer Engine 自动精度切换 |
| Blackwell | B200 | FP4 | 第五代 Tensor Core |

**H100 SXM 在不同精度下的峰值算力**：

| 精度 | 算力 | 稀疏加速后 |
|------|:----:|:---------:|
| FP64 | 34 TFLOPS | - |
| FP32 (CUDA Core) | 67 TFLOPS | - |
| TF32 (Tensor Core) | 495 TFLOPS | 989 TFLOPS |
| BF16/FP16 (Tensor Core) | 989 TFLOPS | 1,979 TFLOPS |
| FP8 (Tensor Core) | 1,979 TFLOPS | 3,958 TFLOPS |

一个关键数字：**Tensor Core FP8 的算力是 CUDA Core FP32 的约 60 倍**。这解释了为什么业界积极推动低精度训练（BF16/FP8）——同样的芯片面积，算力差距是数量级的。

### 1.4 三者的协作全景

把 GPU 的组织结构画成一张图：

```
GPU（以 H100 SXM 为例）
├── SM 0
│   ├── CUDA Cores x128 ── 通用标量运算（激活函数、LayerNorm等）
│   ├── Tensor Cores x4  ── 矩阵乘累加（GEMM、Attention）
│   ├── Shared Memory / L1 Cache ── 片上高速缓存（最多 228KB）
│   ├── Register File ── 每线程私有（每 SM 256KB）
│   └── Warp Schedulers ── 以 Warp 为粒度调度线程
├── SM 1 ~ SM 131（共 132 个 SM，结构同上）
├── L2 Cache ── 全局共享，50MB
└── HBM3 ── 80GB 显存，3.35 TB/s 带宽
```

简单的记忆法则：
- **SM** 负责调度和资源隔离——它是 GPU 并行的基本粒度
- **CUDA Core** 干"杂活"——逐元素运算（激活、Norm、Softmax 等）
- **Tensor Core** 干"主活"——大规模矩阵乘法（占训练/推理 80%+ 的时间）

---

## 2. 存储层次结构

GPU 的存储层次与 CPU 类似，遵循"越靠近计算单元，容量越小、速度越快"的金字塔结构。理解这个层次是写好 CUDA kernel 的基础——因为优化 kernel 的本质就是**让数据尽量待在快的地方，少去慢的地方搬运**。

### 2.1 五级存储层次

从快到慢依次为：

```
寄存器 (Registers)
  │  最快，每线程私有，零延迟
  ▼
共享内存 (Shared Memory)
  │  片上 SRAM，同一 Block 内线程共享，~20-30 cycles
  ▼
L1 Cache / L2 Cache
  │  L1 与共享内存共享物理空间；L2 全局共享，~200 cycles
  ▼
HBM（High Bandwidth Memory）
  │  "显存"，容量大但延迟高，~400-600 cycles
  ▼
主机内存（CPU DRAM）
     需要跨 PCIe 传输，延迟极高
```

### 2.2 各级参数对比（H100 SXM）

| 存储层级 | 容量 | 带宽 | 延迟 | 可见范围 |
|---------|------|------|------|---------|
| 寄存器 | 每 SM 256KB | ~20 TB/s | ~0 cycle | 每线程私有 |
| 共享内存 | 每 SM 最多 228KB | ~20 TB/s | ~20-30 cycles | Thread Block 内共享 |
| L1 Cache | 与共享内存共享物理空间 | ~20 TB/s | ~30 cycles | 每 SM |
| L2 Cache | 50MB | ~12 TB/s | ~200 cycles | 全局共享 |
| HBM3 | 80GB | 3.35 TB/s | ~400-600 cycles | 全局共享 |
| CPU DRAM | 数百 GB ~ 数 TB | ~100 GB/s | 数千 cycles | 需跨 PCIe 传输 |

几个值得牢记的数字级差：
- 寄存器/共享内存的带宽比 HBM 高约 **6 倍**
- HBM 到共享内存的延迟差约 **20 倍**
- HBM 到 CPU DRAM 的带宽再降 **~30 倍**

### 2.3 存储层次与 AI 工作负载的关系

理解存储层次如何影响 AI 计算，能帮你建立优化直觉：

- **模型权重**常驻 HBM。推理时每生成一个 token，权重都要从 HBM 搬进计算单元——权重越大、搬运越慢，这就是"模型越大推理越慢"的物理原因
- **KV Cache** 也存放在 HBM 中。自回归生成时，每一步都要读取之前所有 token 的 KV，随着上下文变长，HBM 的带宽压力持续增加
- **FlashAttention** 的核心优化思路：把 Attention 的中间矩阵（QK^T 和 softmax 结果）分块搬到共享内存里计算，避免把完整的 N x N 矩阵写回 HBM，从而将 HBM 访问量从 O(N^2) 降到 O(N)
- **算子融合（Kernel Fusion）** 的本质：把多个小算子合并成一个大 kernel，中间结果保留在寄存器或共享内存中，省去"写回 HBM → 重新读取"的往返开销

---

## 3. Memory Wall：为什么带宽往往比算力先成为瓶颈

### 3.1 直觉理解

Memory Wall（内存墙）描述的是一个简单事实：**GPU 的计算速度增长得比内存搬运速度更快**，结果就是处理器经常"吃不饱"——不是因为算不过来，而是因为数据还没送到。

打个生活中的比方：一个食量惊人的大胃王参加吃播比赛，嘴巴每秒能吃掉 1000 TFLOPS 的食物。问题是传送带每秒只能送来 3.35 TB 的食材。如果每口食物需要的"咀嚼计算量"不够大（计算强度低），大胃王就只能嚼两下就干等下一口——传送带成了瓶颈。

### 3.2 用数据量化瓶颈

以 H100 SXM 的 FP16 Tensor Core 为例，计算**平衡点**（即计算和带宽恰好同时打满的临界点）：

```
FP16 Tensor Core 峰值算力 = 989 TFLOPS
HBM3 带宽              = 3.35 TB/s

平衡点 = 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/Byte
```

含义：**每从 HBM 搬运 1 字节数据，GPU 需要完成 295 次 FP16 浮点运算才能不让计算单元空闲。** 达不到这个强度的操作，就一定是在等数据。

常见 AI 操作的实际计算强度：

| 操作类型 | 典型计算强度 (FLOP/Byte) | 瓶颈类型 |
|---------|:----------------------:|---------|
| 大 GEMM（训练，大 batch） | 数百 ~ 数千 | Compute Bound |
| 小 GEMM（推理 decode，batch=1） | 数十 | **Memory Bound** |
| Attention Prefill（长序列） | 数百 | Compute Bound |
| Attention Decode（逐 token） | < 10 | **Memory Bound** |
| 逐元素操作（ReLU、LayerNorm） | 1 ~ 10 | **Memory Bound** |

**关键洞察**：
- 训练阶段 batch size 较大，主要的 GEMM 运算计算强度高，通常是 compute bound
- 推理的 decode 阶段每步只处理一个 token，矩阵退化成向量，几乎所有操作都是 **memory bound**
- 这就是 FlashAttention、KV Cache 量化、算子融合等优化技术的物理动机——它们的共同目标都是"**少搬数据**"

### 3.3 Roofline Model：一图判定瓶颈

Roofline Model 是可视化 kernel 性能瓶颈的标准工具。横轴是操作的计算强度（FLOP/Byte），纵轴是实际能达到的性能（FLOPS）：

```
实际性能 (FLOPS)
    │
    │                       ┌────────── 算力天花板
    │                     ╱
    │                   ╱
    │                 ╱  ← 拐点 = 峰值算力 / HBM带宽
    │               ╱
    │             ╱
    │           ╱  ← 带宽斜坡（Memory Bound 区域）
    │         ╱
    │       ╱
    │     ╱
    │   ╱
    │ ╱
    └──────────────────────── 计算强度 (FLOP/Byte)
```

- **拐点左侧**：Memory Bound，性能随带宽线性增长。优化方向：减少数据搬运（算子融合、量化、Tiling）
- **拐点右侧**：Compute Bound，性能被算力封顶。优化方向：用 Tensor Core、提升 Occupancy、用更低精度

在 Nsight Compute 的 SOL（Speed of Light）面板中，可以直接看到某个 kernel 落在 Roofline 的哪个区域，快速判断优化方向。

---

## 4. 数据中心 GPU 规格对比与选型

### 4.1 核心参数一览

| 指标 | A100 SXM | H100 SXM | H200 SXM | B200 SXM |
|------|:--------:|:--------:|:--------:|:--------:|
| **架构** | Ampere | Hopper | Hopper | Blackwell |
| **工艺** | 7nm | 4nm | 4nm | 4nm |
| **SM 数** | 108 | 132 | 132 | 148 |
| **FP32 算力** | 19.5 TFLOPS | 67 TFLOPS | 67 TFLOPS | 90 TFLOPS |
| **FP16 Tensor** | 312 TFLOPS | 989 TFLOPS | 989 TFLOPS | 2,250 TFLOPS |
| **FP8 Tensor** | - | 1,979 TFLOPS | 1,979 TFLOPS | 4,500 TFLOPS |
| **显存类型** | HBM2e | HBM3 | HBM3e | HBM3e |
| **显存容量** | 80GB | 80GB | 141GB | 192GB |
| **HBM 带宽** | 2.0 TB/s | 3.35 TB/s | 4.8 TB/s | 8.0 TB/s |
| **NVLink 带宽** | 600 GB/s | 900 GB/s | 900 GB/s | 1,800 GB/s |
| **TDP** | 400W | 700W | 700W | 1000W |

### 4.2 代际增长趋势

| 指标 | A100 → H100 增幅 | H100 → B200 增幅 |
|------|:----------------:|:----------------:|
| FP16 Tensor 算力 | 3.2x | 2.3x |
| HBM 带宽 | 1.7x | 2.4x |
| 显存容量 | 1.0x (80→80GB) | 2.4x (80→192GB) |
| NVLink 带宽 | 1.5x | 2.0x |

几个值得注意的趋势：

1. **算力增长速度仍然快于带宽增长**——Memory Wall 问题在加剧，这意味着内存优化技术（FlashAttention、量化、算子融合）在未来只会越来越重要
2. **显存容量的跳跃式增长**：H200 的 141GB 和 B200 的 192GB 意味着越来越多的模型可以单卡放下。70B 模型 FP16 约 140GB，H200 勉强塞进，B200 则有余量
3. **NVLink 带宽与算力同步增长**：保证了多卡并行时通信不会成为新的瓶颈

### 4.3 选型参考

不同场景对 GPU 参数的侧重点不同：

| 使用场景 | 最关键的参数 | 推荐选择 | 原因 |
|---------|------------|---------|------|
| 大模型训练 (>100B) | 算力 + NVLink 带宽 | H100/B200 SXM | TP 依赖高带宽互联 |
| 大模型推理（高吞吐） | HBM 带宽 + 显存容量 | H200/B200 | decode 阶段是 memory bound |
| 大模型推理（低成本） | 性价比 | A100 / L40S | 够用即可 |
| 中小模型微调 | 显存容量 | A100 / H100 PCIe | 按需选择 |

**一个实用的快速判断**：如果你的场景主要卡在 decode 延迟（逐 token 生成慢），优先看 **HBM 带宽**；如果卡在 prefill（长 prompt 处理慢），优先看 **Tensor Core 算力**。

---

## 5. 总结

GPU 硬件知识对 AI Infra 工程师来说不是"nice to have"，而是分析性能问题的底层语言。核心要记住三件事：

1. **计算单元分工**：CUDA Core 处理逐元素运算，Tensor Core 处理矩阵乘法，大模型 80%+ 的时间花在后者上
2. **存储层次决定优化方向**：从寄存器到 HBM，带宽差 6 倍、延迟差数百倍。CUDA 优化的本质就是让数据尽量停在快的地方
3. **Memory Wall 是常态**：除了大 batch 的 GEMM，大多数 AI 操作都是 memory bound。判断一个操作的瓶颈类型，是选择优化方向的第一步

---

## 🎯 自我检验清单

- 能画出 GPU 的存储层次金字塔（寄存器 → 共享内存 → L1 → L2 → HBM → CPU DRAM），标注每一级的容量和带宽量级
- 能解释 SM、CUDA Core、Tensor Core 三者的关系与分工，说出 Tensor Core 在 AI 计算中的主导地位
- 能说出 H100 SXM 的关键参数：HBM 容量（80GB）、HBM 带宽（3.35 TB/s）、L2 大小（50MB）、共享内存上限（228KB/SM）
- 能计算 H100 的 Roofline 拐点（~295 FLOP/Byte），并据此判断 Attention Decode 是 memory bound
- 能解释"为什么 FP8 训练比 FP16 快"——不只是数据量减半，Tensor Core 的 FP8 吞吐量也是 FP16 的 2 倍
- 能对比 A100、H100、H200、B200 的核心规格差异，并针对不同场景给出选型建议
- 能解释 FlashAttention 和算子融合"为什么有效"——不是算法魔法，而是减少了对 HBM 的读写次数

## 📚 参考资料

- [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
