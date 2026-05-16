---
title: '美团 北斗 AI Infra 校招'
date: 2026-04-16 12:09:07
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 介绍一下 Transformer 的架构，和传统 RNN 相比有何优势？

**现代 Decoder-only Transformer 架构**（以 LLaMA 为例）：

```
输入 tokens → Embedding → x

重复 N 层:
  ┌─────────────────────────────────────────┐
  │ x_norm = RMSNorm(x)                     │
  │ attn_out = MultiHeadAttention(x_norm)   │
  │   Q = x_norm × W_Q + RoPE              │
  │   K = x_norm × W_K + RoPE              │
  │   V = x_norm × W_V                     │
  │   Score = Q × K^T / √d_k + CausalMask  │
  │   Attn = softmax(Score) × V            │
  │   attn_out = Attn × W_O                │
  │ x = x + attn_out          (残差连接)     │
  │                                          │
  │ x_norm = RMSNorm(x)                     │
  │ ffn_out = SwiGLU_FFN(x_norm)           │
  │   gate = x_norm × W_gate               │
  │   up = x_norm × W_up                   │
  │   ffn_out = SiLU(gate) ⊙ up × W_down   │
  │ x = x + ffn_out           (残差连接)     │
  └─────────────────────────────────────────┘

x → RMSNorm → LM_Head → logits → softmax → next token probability
```

**Transformer vs RNN 全面对比**：

| 维度 | Transformer | RNN (LSTM/GRU) |
|---|---|---|
| **并行性** | 所有 token 同时计算 attention（训练时） | 严格串行（t 依赖 t-1 的隐状态） |
| **训练速度** | 快 10-100x（GPU 并行利用率高） | 慢（无法并行，GPU 利用率低） |
| **长距离依赖** | O(1) 路径（任意两 token 直接连接） | O(n) 路径（信息经 n 步衰减/遗忘） |
| **计算复杂度** | O(n²d)（attention 是 n² 的） | O(nd²)（每步 d×d 矩阵乘） |
| **内存** | O(n²)（attention matrix）或 O(n)（FlashAttention） | O(n)（只需当前隐状态） |
| **Scaling** | Scaling Law 明确，100B+ 参数仍有收益 | 难以扩展到大规模 |
| **推理效率** | 需要 KV Cache，decode 是 memory-bound | 每步计算固定，无需缓存历史 |
| **位置感知** | 需要额外位置编码（RoPE/绝对位置） | 天然感知顺序（串行结构） |

**Transformer 胜出的根本原因**：
1. **训练并行性**：现代 GPU 有数千个核心，Transformer 能充分利用，RNN 只能逐步串行
2. **梯度流**：残差连接 + 短路径使深层网络梯度畅通，RNN 即使有 LSTM 仍有梯度衰减
3. **注意力直连**：第 1 个 token 和第 1000 个 token 之间只需 1 次 attention 计算即可交互

**RNN 的现代回归**——Mamba/RWKV/RetNet**：
- 线性 attention/状态空间模型 (SSM)：保持 O(n) 复杂度 + 并行训练
- 推理时像 RNN：每步只需常数状态，无需 KV Cache
- 但目前在超大规模下仍不如 Transformer（注意力的动态路由能力难以替代）

### Q: Transformer 中参数分布在哪里？参数量和计算量最大的是哪部分？

**参数分布详细分析**（以 hidden_dim = d，FFN hidden_dim = 4d 为例）：

```
每层参数:
  Attention 层:
    W_Q: d × d
    W_K: d × d (GQA 下 = d × d/G)
    W_V: d × d (GQA 下 = d × d/G)
    W_O: d × d
    → Attention 总参数: 4d² (MHA) 或 约 2.5d² (GQA with G=8)

  FFN 层 (SwiGLU):
    W_gate: d × (8d/3)
    W_up:   d × (8d/3)
    W_down: (8d/3) × d
    → FFN 总参数: 3 × d × (8d/3) = 8d²

  RMSNorm: 2 × d (两个 norm 层，参数可忽略)
  
  每层总参数 ≈ 4d² + 8d² = 12d² (MHA, SwiGLU)
```

**全模型参数分布**（以 LLaMA-7B，d=4096，32 层为例）：

| 组件 | 参数量 | 占比 |
|---|---|---|
| Embedding | vocab_size × d = 32000 × 4096 = 131M | ~1.9% |
| Attention (32 层) | 32 × 4 × d² = 32 × 67M = 2147M | ~31% |
| FFN (32 层) | 32 × 8d² = 32 × 134M = 4295M | **~62%** |
| RMSNorm | 32 × 2 × d = 0.26M | ~0% |
| LM Head | 通常与 Embedding 共享权重 | 0% |
| **总计** | **~6.7B** | 100% |

**结论**：**FFN 层占参数量的约 2/3**，是模型的"知识存储"主体。

**计算量（FLOPs）分析**：

对于输入 [batch=B, seq=S]：
```
Attention 层:
  QKV 投影: 3 × 2BSd² = 6BSd² FLOPs
  QK^T:     2BS²d FLOPs (这一项随序列长度平方增长!)
  Attn×V:   2BS²d FLOPs
  O 投影:   2BSd² FLOPs
  → 总计: 8BSd² + 4BS²d

FFN 层:
  gate:  2BSd × (8d/3) FLOPs
  up:    2BSd × (8d/3) FLOPs
  down:  2BS × (8d/3) × d FLOPs
  → 总计: 16BSd²/3 × 3 = 16BSd² (SwiGLU 有三个矩阵)
```

**关键观察**：
- 短序列（S << d）：FFN 计算量主导（16BSd² vs 8BSd²）
- 长序列（S >> d）：Attention 的 O(S²d) 项主导
- 实际应用中（S=2048-8192, d=4096）：FFN 仍然是计算量最大的部分

### Q: GPU 的 CUDA Core 和 Tensor Core？常用 GPU 的显存和显存带宽？

**CUDA Core vs Tensor Core 本质区别**：

| 维度 | CUDA Core | Tensor Core |
|---|---|---|
| **计算粒度** | 标量：一次 1 个浮点 FMA（a×b+c） | 矩阵：一次 4×4×4 或 16×16×16 MMA |
| **吞吐比** | 基准 | FP16: 8-16x，INT8: 16-32x |
| **数据类型** | FP32, FP64, INT32 | FP16, BF16, TF32, FP8, INT8, INT4 |
| **精度** | 精确 | 混合精度（输入低精度，累加高精度） |
| **适用算子** | 逐元素操作、规约、控制流 | GEMM、卷积等可分解为矩阵乘的操作 |
| **编程方式** | 直接编程 | 通过 MMA PTX 指令或 WMMA API |

**Tensor Core 的工作方式（H100 FP16 为例）**：
```
一次 MMA 操作:
  D[16×16] = A[16×16] × B[16×16] + C[16×16]
  
  输入: A(FP16), B(FP16)
  累加器: C/D(FP32) — 保持精度
  
  一个 warp (32 线程) 协作完成一次 MMA
  每个线程持有 A/B 的几个元素 + D 的几个元素
```

**常用 GPU 规格对比（推理/训练相关）**：

| GPU | 架构 | 显存 | 显存带宽 | FP16 TC | INT8 TC | FP8 TC | TDP |
|---|---|---|---|---|---|---|---|
| **A100** | Ampere | 80GB HBM2e | **2.0 TB/s** | 312 TFLOPS | 624 TOPS | - | 400W |
| **H100 SXM** | Hopper | 80GB HBM3 | **3.35 TB/s** | 989 TFLOPS | 1979 TOPS | 1979 TFLOPS | 700W |
| **H200** | Hopper | 141GB HBM3e | **4.8 TB/s** | 989 TFLOPS | 1979 TOPS | 1979 TFLOPS | 700W |
| **L40S** | Ada | 48GB GDDR6X | 864 GB/s | 362 TFLOPS | 724 TOPS | 724 TFLOPS | 350W |
| **4090** | Ada | 24GB GDDR6X | 1.0 TB/s | 330 TFLOPS | 660 TOPS | 660 TFLOPS | 450W |
| **B200** | Blackwell | 192GB HBM3e | **8.0 TB/s** | 2250 TFLOPS | 4500 TOPS | 4500 TFLOPS | 1000W |

**关键性能指标解读**：
- **显存容量**决定能放下多大的模型（7B FP16 ≈ 14GB）
- **显存带宽**决定 Decode 阶段吞吐（memory-bound，每步读全部权重）
- **Tensor Core 算力**决定 Prefill 阶段速度（compute-bound，大矩阵乘）
- **Roofline 拐点** = 算力/带宽：A100 = 156 FLOP/B，H100 = 295 FLOP/B

### Q: 大模型量化和量化算子？

**量化的核心目标**：减少模型权重/激活的存储位宽，从而降低内存占用和带宽需求，加速推理。

**量化方法分类**：

| 分类维度 | 选项 | 说明 |
|---|---|---|
| 量化对象 | W-only (W4A16), W+A (W8A8) | 只量化权重 vs 权重+激活都量化 |
| 量化粒度 | per-tensor, per-channel, per-group | 粒度越细精度越好，scale 存储越多 |
| 量化时机 | PTQ (训练后), QAT (训练中) | PTQ 简单，QAT 精度更好 |
| 数据格式 | INT8, INT4, FP8 (E4M3/E5M2), NF4 | 均匀/非均匀量化 |

**量化算子的核心计算流程**：

```cuda
// W4A16 GEMM kernel 伪代码 (权重 INT4, 激活 FP16)
__global__ void quantized_gemm(
    half* activation,     // FP16 输入 [M, K]
    uint8_t* weight_q,    // INT4 权重 (packed, 2 per byte) [K, N/2]
    half* scale,          // FP16 scale [K/group_size, N]
    half* output          // FP16 输出 [M, N]
) {
    // 1. 从 Global Memory 加载 INT4 权重 (只需 0.5 byte/element!)
    uint8_t packed = weight_q[...];
    int4 w_low  = packed & 0x0F;        // 低 4 bit
    int4 w_high = (packed >> 4) & 0x0F; // 高 4 bit
    
    // 2. 在寄存器中 dequant 为 FP16
    half w_fp16_low  = __int2half_rn(w_low) * scale[group_idx];
    half w_fp16_high = __int2half_rn(w_high) * scale[group_idx];
    
    // 3. FP16 矩阵乘 (Tensor Core)
    output[m][n] += activation[m][k] * w_fp16;
}
```

**关键工程优化点**：
1. **Dequant 与 GEMM 融合**：在同一 kernel 中完成读取 INT4 → dequant → 计算，不产生额外 HBM 写读
2. **Pack/Unpack 高效实现**：2 个 INT4 打包为 1 个 INT8，用位运算解包
3. **Group 对齐**：group_size=128 对齐到 warp size 和 tile size，简化索引
4. **Scale 预取**：scale 数组较小，可预加载到 shared memory

**为什么量化能加速**：
- Decode 阶段是 memory-bound（AI ≈ 2/d ≈ 0.0005 << 拐点 156）
- 瓶颈 = 权重读取带宽
- INT4 权重 = FP16 的 1/4 数据量 → 读取时间减少 4x → 理论加速 4x
- Dequant 计算是 compute（寄存器内简单乘加），而 compute 资源本来就是闲置的

### Q: 详细讲 PD 分离？

**PD 分离（Prefill-Decode Disaggregation）** 是将 LLM 推理的两个阶段部署到不同硬件/集群的架构设计。

**为什么需要分离——两个阶段的特性差异**：

| 维度 | Prefill（首次处理 prompt） | Decode（逐 token 生成） |
|---|---|---|
| **计算模式** | 大矩阵乘 [seq_len × d] × [d × d] | GEMV [1 × d] × [d × d] |
| **瓶颈类型** | **Compute-bound**（算力决定速度） | **Memory-bound**（带宽决定速度） |
| **算术强度** | AI ≈ seq_len/2 ≈ 500-2000 | AI ≈ 2/d ≈ 0.0005 |
| **GPU 利用率** | Tensor Core 高利用（80%+） | Tensor Core 低利用（<10%） |
| **延迟需求** | 影响 TTFT | 影响 TPOT |
| **资源需求** | 高算力（Tensor Core FLOPS） | 高带宽（HBM bandwidth） |

**混合部署的问题**：
```
混合部署: 同一组 GPU 同时处理 Prefill 和 Decode 请求
  
  问题 1: Prefill 的大 GEMM 抢占了 Decode 的 SM → Decode 延迟抖动
  问题 2: GPU 的带宽和算力无法同时被两种工作负载充分利用
  问题 3: Decode 的 batch 小 → GPU 算力浪费; Prefill 打断 Decode → TPOT 不稳定
```

**PD 分离架构**：
```
                    ┌──────────────────────┐
                    │     调度器/Router      │
                    └──────┬──────┬─────────┘
                           │      │
              ┌────────────┘      └────────────┐
              ▼                                 ▼
    ┌─────────────────┐              ┌─────────────────┐
    │  Prefill 集群    │              │  Decode 集群     │
    │  (高算力 GPU)    │    KV Cache   │  (高带宽 GPU)    │
    │  H100/A100      │───传输───────→│  H100/A100      │
    │  大 batch 并行   │              │  大 batch 拼接   │
    └─────────────────┘              └─────────────────┘
```

**工作流程**：
1. 新请求到达 → Router 分配到 Prefill 节点
2. Prefill 节点计算 prompt 的 KV Cache
3. KV Cache 通过高速网络（NVLink/RDMA）传输到 Decode 节点
4. Decode 节点将请求加入 Continuous Batching，逐 token 生成
5. 生成完成 → 返回结果

**KV Cache 传输的挑战**：
- 传输量：LLaMA-70B, seq=2048, FP16 → 约 1 GB KV Cache
- NVLink 带宽 900 GB/s → 传输时间 ~1.1ms
- InfiniBand 200 Gbps → 传输时间 ~40ms
- 需要流水线化：传输第 N 层的 KV 同时计算第 N+1 层

**收益分析**：
- Prefill 集群：可以用大 batch（如 64-128 prompts 并行），Tensor Core 利用率 > 80%
- Decode 集群：可以用更大 batch（如 256-512 requests），带宽利用率提高
- 总体吞吐提升 **1.5-2x**，同时 TPOT 更稳定（不被 Prefill 打断）

**实际系统**：
- DistServe（OSDI 2024）、Splitwise（ISCA 2024）
- TensorRT-LLM 支持 PD 分离模式
- Mooncake（月之暗面）生产系统

### Q: 详细讲 PagedAttention？

PagedAttention 借鉴操作系统虚拟内存的分页机制，解决 KV Cache 的显存碎片和利用率问题。

**传统 KV Cache 管理的痛点**：

```
场景: 服务 3 个请求，最大序列长度 2048
  
传统预分配 (连续空间):
  |<---请求 A (预分配 2048, 实际用 512)--->|
  |<---请求 B (预分配 2048, 实际用 1800)->|
  |<---请求 C (预分配 2048, 实际用 200)---->|
  
  显存利用率 = (512+1800+200) / (2048×3) = 40.9%
  
  更糟糕: 如果请求 D 需要 1000 tokens 空间
    → 总空闲 = 2048×3 - (512+1800+200) = 3636 tokens 空间
    → 但没有连续的 2048 块可分配 → 无法接收! (外部碎片)
```

**PagedAttention 的设计**：

**核心概念**：
- **Physical Block**：固定大小的显存块（如 16 tokens × num_kv_heads × head_dim × 2(K+V) × dtype）
- **Logical Block**：每个请求的 KV 逻辑空间被分为等大的逻辑块
- **Block Table**：维护每个请求的逻辑块→物理块映射

```
PagedAttention 管理:
  Physical blocks pool: [0][1][2][3][4][5][6][7][8][9]...
  
  请求 A (实际 32 tokens = 2 blocks):
    Block Table A: {logical_0→phys_3, logical_1→phys_7}
  
  请求 B (实际 48 tokens = 3 blocks):
    Block Table B: {logical_0→phys_1, logical_1→phys_5, logical_2→phys_9}
  
  请求 C (实际 16 tokens = 1 block):
    Block Table C: {logical_0→phys_4}
  
  空闲 blocks: [0, 2, 6, 8, ...]
  利用率: 已分配 blocks 几乎全部有效使用，仅最后一个 block 可能有内部碎片
```

**Attention Kernel 的实现变化**：

标准 Attention kernel 假设 KV 连续存储：
```cuda
// 标准: KV 在显存中连续
K_ptr = kv_cache + request_offset;  // 简单偏移
for (int i = 0; i < seq_len; i++) {
    score += Q * K_ptr[i];
}
```

PagedAttention kernel 需要间接寻址：
```cuda
// PagedAttention: KV 分散在不同 physical block
for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int physical_block = block_table[request_id][block_idx];
    K_block_ptr = kv_cache + physical_block * block_size;
    for (int i = 0; i < tokens_in_block; i++) {
        score += Q * K_block_ptr[i];
    }
}
```

**Copy-on-Write (COW) 机制**：
```
Beam Search 场景 (beam_width=4):
  
  初始: 4 个 beam 共享相同前缀的 KV blocks
    Beam 0: [Block A] [Block B] [Block C]
    Beam 1: [Block A] [Block B] [Block C]  ← 共享，引用计数 = 4
    Beam 2: [Block A] [Block B] [Block C]
    Beam 3: [Block A] [Block B] [Block C]
  
  分叉: Beam 2 生成不同 token，需要修改最后一个 block
    → COW: 只复制 Block C → Block C'，前面的 block 继续共享
    Beam 2: [Block A] [Block B] [Block C']  ← 只有最后一个 block 不同
```

**性能数据**：
- KV Cache 利用率：传统 20-40% → PagedAttention **96-98%**
- 同等显存下并发量：提升 **2-4x**
- Attention 计算 overhead：~3-5%（间接寻址的额外 latency）
- 总吞吐提升：**2-4x**（因为能服务更多请求）

### Q: 手撕：K 个一组翻转链表？

（编程题）

### Q: 手撕 CUDA：前缀和算子，实现 base 版本并讲优化方法？

（编程题）

### Q: 手撕 CUDA：GEMM 算子，实现 base 版本并讲优化方法？

（编程题）
