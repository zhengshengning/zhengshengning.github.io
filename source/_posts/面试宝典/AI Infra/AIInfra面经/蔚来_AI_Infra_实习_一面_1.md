---
title: '蔚来 AI Infra 实习 一面 (1)'
date: 2026-04-16 12:13:00
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 面经]
---


<!-- more -->

---

### Q: C++多线程和多进程有什么区别？

**核心对比：**

| 维度 | 多线程 | 多进程 |
|------|--------|--------|
| 地址空间 | 共享（同一进程内） | 独立（各进程独立） |
| 创建开销 | 小（~10us，不需要复制页表） | 大（~100us，fork需要COW页表） |
| 上下文切换 | 快（不需刷新TLB） | 慢（需要刷新TLB、切换页表） |
| 通信方式 | 直接访问共享内存（需同步） | 需要IPC(pipe/socket/shm) |
| 隔离性 | 差（一个线程崩溃影响整个进程） | 好（一个进程崩溃不影响其他） |
| 调试难度 | 高（数据竞争难复现） | 中（问题相对隔离） |
| Python GIL | 多线程无法并行（GIL限制） | 多进程可真正并行 |

**选择建议：**
- **计算密集 + 需要共享大量数据** → 多线程（如GPU kernel并行的host端管理）
- **计算密集 + Python** → 多进程（绕过GIL）
- **需要高隔离性/容错** → 多进程（如分布式训练中各worker独立进程）
- **IO密集** → 多线程或协程（切换快，GIL在IO时释放）
- **分布式系统** → 多进程（MPI的SPMD模型，每个rank一个进程）

**深度学习框架的实际选择：**
- PyTorch DDP：多进程（每个GPU一个进程，通过NCCL通信）
- DataLoader：多进程（worker进程预处理数据，通过共享内存传给主进程）
- CUDA Runtime：多线程（host端多线程管理多stream/多设备）

---

### Q: Attention解决了什么问题？还存在什么缺点？

**Attention解决的问题：**

**1. 长距离依赖(Long-range Dependencies)：**
```
RNN/LSTM: 信息必须逐步传递 → 距离远的信息衰减/遗忘
  token_0 → token_1 → ... → token_1000 (token_0的信息已经稀释)
  
Attention: 任意两个位置直接交互 → 无距离衰减
  token_0 ←→ token_1000 (一步可达，注意力权重直接决定信息流)
```

**2. 并行化能力：**
- RNN必须顺序计算(t步依赖t-1步的hidden state) → 无法GPU并行
- Attention的QK^T和AV都是矩阵乘法 → 完全并行，GPU利用率高
- 训练加速：相同模型size，Transformer训练速度远超RNN

**3. 可解释性：**
- 注意力权重矩阵可可视化（哪些token之间有强关联）
- 有助于模型行为理解和调试

**仍然存在的缺点：**

| 缺点 | 影响 | 缓解方案 |
|------|------|---------|
| O(n²)计算复杂度 | 长序列计算量爆炸 | FlashAttention, Linear Attention, Mamba |
| O(n²)显存(无优化) | 长序列显存不够 | FlashAttention(O(n)), 分块处理 |
| KV-Cache显存大 | 推理时长上下文OOM | GQA/MQA, PagedAttention, 量化 |
| 对位置不敏感 | 需额外位置编码 | RoPE, ALiBi |
| 局部模式效率低 | 捕获局部pattern不如CNN | 结合CNN(ConvNeXt), 局部注意力窗口 |
| 推理自回归慢 | 每步只生成1个token | 投机解码, 并行解码 |

---

### Q: GPU和CPU的区别？

**设计哲学的根本差异：**

```
CPU: 针对延迟优化(Latency-Optimized)
  → 少量强核心 + 大缓存 + 复杂控制逻辑
  → 目标: 尽快完成单个任务

GPU: 针对吞吐优化(Throughput-Optimized)  
  → 大量简单核心 + 高带宽 + 简单控制
  → 目标: 同时处理大量相同任务
```

**硬件规格对比（以Xeon 8380 vs A100为例）：**

| 维度 | CPU (Intel Xeon 8380) | GPU (NVIDIA A100) |
|------|----------------------|-------------------|
| 核心数 | 40核心 | 6912 CUDA Core + 432 Tensor Core |
| 时钟频率 | 2.3-3.4 GHz | 1.4 GHz |
| 缓存 | L1: 48KB/核, L2: 1.25MB/核, L3: 60MB | L1: 192KB/SM, L2: 40MB |
| 内存带宽 | ~51 GB/s (DDR4) | 2039 GB/s (HBM2e) |
| 内存容量 | 数百GB~TB | 80GB |
| 单精度算力 | ~3 TFLOPS | 19.5 TFLOPS(FP32), 312 TFLOPS(FP16 TC) |
| 控制逻辑 | 复杂(分支预测/乱序执行/推测执行) | 简单(SIMT) |
| 适合任务 | 串行逻辑、分支多、缓存友好 | 大规模数据并行、矩阵运算 |

**为什么深度学习用GPU？**
- 矩阵乘法是核心运算 → 天然适合GPU的SIMD/SIMT并行
- 数据量大且访问模式规律 → GPU高带宽内存优势明显
- 训练中同一个kernel要处理百万级元素 → GPU千核并行处理
- Tensor Core专门为矩阵乘优化 → FP16峰值算力是CPU的100倍

---

### Q: 手撕：旋转矩阵（Python实现）？

（编程题）
