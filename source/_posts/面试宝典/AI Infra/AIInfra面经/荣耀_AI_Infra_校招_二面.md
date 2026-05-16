---
title: '荣耀 AI Infra 校招 二面'
date: 2026-04-16 12:11:22
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 面经]
---


<!-- more -->

---

### Q: LLM 在 NPU 上的最大瓶颈在哪？

NPU（如华为昇腾 Ascend、寒武纪 MLU、海光 DCU）相比 NVIDIA GPU 生态在部署 LLM 时面临多层次瓶颈：

**1. 内存带宽瓶颈（最核心）**：

LLM Decode 阶段是 memory-bound——每步需读取全部模型权重：
```
Decode 每步数据读取: model_params × bytes_per_param
  7B FP16: 14 GB/step
  70B FP16: 140 GB/step

吞吐 = HBM 带宽 / 模型大小
  A100 (2.0 TB/s, 7B FP16): 2000/14 = 143 tokens/s
  Ascend 910B (~1.6 TB/s, 7B FP16): 1600/14 = 114 tokens/s
  差距: ~20% 带宽劣势直接转化为吞吐劣势
```

NPU 的 HBM 带宽通常低于同代 NVIDIA GPU（H100: 3.35 TB/s vs Ascend 910B: ~1.6 TB/s），这在 decode 场景下直接限制了单卡吞吐。

**2. 算子生态和适配问题**：

| 算子 | CUDA 生态 | NPU 生态 | 差距原因 |
|---|---|---|---|
| GEMM | cuBLAS (极致优化) | 自研 GEMM library | 优化迭代时间短 |
| FlashAttention | 原生支持 (FA1/2/3) | 需要重新适配 AI Core | 片上 buffer 架构不同 |
| RoPE | Triton/CUDA 实现丰富 | 需要手写 TIK/Ascend C | 开发者少 |
| Fused LayerNorm | 多种开源实现 | 需要自研 | 融合模式不同 |
| SwiGLU | 简单融合 kernel | 需要验证算子兼容性 | 算子库覆盖度低 |

LLM 有很多特殊算子组合（如 RoPE + GQA Attention + SwiGLU），需要针对 NPU 架构重新设计高效实现。

**3. 动态 shape 处理**：
- LLM 推理的 seq_len 和 batch_size 是动态变化的（Continuous Batching）
- NPU 编译器通常偏好静态 shape（可做更多编译期优化）
- 动态 shape 可能导致：频繁重编译、无法使用最优 tiling、算子选择不优

**4. KV Cache 管理灵活性**：
- PagedAttention 需要间接寻址（通过 block table 索引 KV blocks）
- NPU 的内存管理 API 可能不如 CUDA 灵活（如自定义 memory allocator）
- 分页管理的 kernel 需要重新适配 NPU 的存取模式

**5. 软件栈成熟度**：
```
NVIDIA 推理栈:                    NPU 推理栈:
  vLLM / TensorRT-LLM              自研框架（不够成熟）
  CUDA / cuBLAS / cuDNN             CANN / Ascend C / TIK
  Nsight Systems / Compute          profiling 工具有限
  丰富的开源生态                     社区较小
```

**实际应对策略**：
- 优先做 GEMM kernel 极致优化（占时间 70%+）
- 将 FlashAttention 适配 NPU 的片上 buffer（类比 shared memory）
- 实现类 PagedAttention 的 KV Cache 分页管理
- 利用 NPU 的特有优势（如某些 NPU 有更大的片上缓存）

---

### Q: Prefill 和 Decoding 的 Matmul 优化方法有什么不同？

两个阶段的矩阵乘法有本质不同的形状特征和瓶颈类型，需要完全不同的优化策略：

**Prefill 阶段——大矩阵乘（Compute-bound）**：
```
形状: [batch × seq_len, hidden] × [hidden, hidden]
  典型: [1 × 2048, 4096] × [4096, 4096] = [2048, 4096]
  
  M = 2048 (大), K = 4096, N = 4096
  算术强度 AI ≈ M/(1/K + 1/N) ≈ M = 2048 >> 拐点 (~156)
  → Compute-bound!
```

**Prefill Matmul 优化重点**：

| 优化方向 | 具体方法 | 效果 |
|---|---|---|
| 最大化 Tensor Core 利用率 | MMA 指令 + 合适的 tile 大小 | 利用率 > 80% |
| 大 Tile + 高数据复用 | BM=BN=128-256, BK=32-64 | 减少全局访存 |
| 双缓冲预取 | 异步加载下一个 tile 到 shared memory | 隐藏加载延迟 |
| 最优 Block 配置 | 保证足够 occupancy 隐藏计算延迟 | SM 利用率高 |
| Pipeline（软流水） | 加载、计算、存储三级流水 | 最大化吞吐 |
| TMA (H100) | Tensor Memory Accelerator 异步加载 | 释放计算 SM |

---

**Decode 阶段——瘦长矩阵乘/GEMV（Memory-bound）**：
```
形状: [batch × 1, hidden] × [hidden, hidden]
  典型: [1, 4096] × [4096, 4096] = [1, 4096] (纯 GEMV!)
  或 batch=32: [32, 4096] × [4096, 4096] = [32, 4096] (瘦 GEMM)
  
  M = 1~32 (极小), K = 4096, N = 4096
  算术强度 AI ≈ 2M / (M+N) ≈ 2×1/4096 ≈ 0.0005 << 拐点
  → Memory-bound!
```

**Decode Matmul 优化重点**：

| 优化方向 | 具体方法 | 效果 |
|---|---|---|
| **减少权重读取量** | INT4/INT8 量化（W4A16/W8A8） | 带宽需求减少 2-4x |
| **向量化访存** | float4/LDG.128 加载 | 减少内存事务数 |
| **Split-K 并行** | 沿 K 维切分增加并行度 | 更多 SM 参与（M 小时 SM 不够饱和） |
| **增大 batch** | Continuous Batching 凑大 batch | 提高 AI，从 memory→compute bound 过渡 |
| **Persistent Kernel** | 一个 kernel 处理多层 | 减少 launch 开销 |
| **权重预排列 (Packing)** | 按 tile 顺序重排权重 | 连续内存访问 |

**Split-K 策略详解**：
```
常规 GEMM (M=1, K=4096, N=4096):
  每个 block 计算一列输出 → 只需 N/BN = 4096/128 = 32 blocks
  A100 有 108 SM → 只有 32/108 = 30% SM 在工作!

Split-K (将 K 维切分):
  K 分为 8 份 → 每份 K=512
  总 blocks = 32 × 8 = 256 → 256/108 > 2x → SM 利用率更高
  最后用一个 reduce kernel 合并 8 份结果
```

**关键差异总结**：

| 维度 | Prefill (大 GEMM) | Decode (GEMV/瘦 GEMM) |
|---|---|---|
| 瓶颈 | Compute-bound | Memory-bound |
| 核心优化 | 最大化 TC 利用率 | 最大化带宽利用率 |
| Tiling 策略 | 大 tile 高复用 | 小 tile + Split-K |
| 量化收益 | FP8 加速计算（TC 吞吐 2x） | INT4 减少读取量（带宽收益 4x） |
| Batch 影响 | batch 越大越好（更大 M） | batch 增大可过渡到 compute-bound |

---

### Q: 分块策略如何保证数据在缓存中的连续性？

**问题本质**：Tiling 将大矩阵分块后，每个 tile 在原始矩阵中可能不是内存连续的（如行优先存储时列方向的 tile）。需要确保加载到片上缓存后的访问是高效的。

**1. 数据布局对齐（最基础）**：
```
行优先 (Row-major) 存储:
  A[M][K]: 第 i 行连续存储, A[i][0], A[i][1], ..., A[i][K-1]
  
  取 tile A[BM][BK]:
    如果 tile 跨行取 → 每行连续但行间不连续 → 需要 BM 次独立 load
    如果 tile 跨列取 → 同行连续 → 一次可以 load BK 个连续元素

  优化: 确保内循环的加载方向与存储方向一致
    → A 矩阵行方向加载 (连续), B 矩阵需要列方向 → B 提前转置为 B^T[N][K]
```

**2. Block 大小对齐缓存行**：
```
GPU 的 Global Memory 事务: 32 bytes (L2) 或 128 bytes (L1)

确保 tile 的行宽 = 缓存行整数倍:
  BK = 8 (float, 32 bytes) 或 BK = 32 (float, 128 bytes)
  → 每行的 load 请求恰好对齐一个/几个事务，无浪费

如果 BK = 5 (float, 20 bytes):
  → 每次 load 请求 32 bytes 事务但只用 20 bytes → 带宽浪费 37.5%
```

**3. DMA/TMA 预取策略**：
```
在 NPU/H100 上:
  TMA (Tensor Memory Accelerator) 异步加载:
    在计算当前 tile 的同时，TMA 异步加载下一个 tile 到 shared memory
    → 加载延迟被计算完全隐藏

双缓冲 (Double Buffering):
  Buffer A: [当前计算的 tile]
  Buffer B: [正在加载的下一个 tile]
  每次迭代交换 A/B 的角色
```

**4. 避免片上缓存的 Bank Conflict**：
```
Shared Memory 加载后的布局:
  原始: __shared__ float tile[BM][BK]  → BK=32 时每行恰好 32 bank → 列访问全冲突

  优化: __shared__ float tile[BM][BK + PADDING]  → 错开 bank
  或 swizzle: 重映射地址使列访问也无冲突
```

**5. 权重预排列 (Weight Pre-packing)**：
```
推理前一次性重排权重:
  原始: W[K][N] (行优先)
  重排: W_packed[N/BN][K/BK][BN][BK] (按 tile 顺序存储)
  
  效果: 加载一个 tile 时，数据在内存中完全连续
        → 一次大块 DMA 传输 → 最大化带宽利用率

cuBLAS/TensorRT 在初始化时会做 weight packing
```

**6. NPU 特殊考虑**：
- NPU 的片上 buffer 通常比 GPU shared memory 大（如 Ascend AI Core 有 ~MB 级 Local Memory）
- 更大的片上缓存意味着可以存更大的 tile → 更少的全局访存次数
- 但 DMA 传输与计算的并行需要手动编排（PIPE_M/PIPE_V 分离）

---

### Q: 昇腾 NPU 架构对 Transformer 友好吗？

**昇腾 NPU（如 Ascend 910B）的核心架构**：
```
Ascend 910B:
  32 个 AI Core (类比 SM)
  每个 AI Core:
    Cube Unit (矩阵乘加): 类比 Tensor Core, 支持 FP16/INT8 MMA
    Vector Unit: 类比 CUDA Core, 逐元素运算
    Scalar Unit: 控制流
    Local Memory (L1 Buffer): ~512 KB - 1 MB (远大于 GPU 的 shared memory)
    
  HBM2E: 64 GB, ~1.6 TB/s
  FP16 算力: ~320 TFLOPS (Cube Unit)
```

**对 Transformer 友好的方面**：

| 维度 | 友好原因 | 对应 Transformer 操作 |
|---|---|---|
| Cube Unit | 专为矩阵乘设计，FP16/INT8 高吞吐 | Q×K^T, Attn×V, FFN 的 Linear |
| 大 Local Memory | 可存更大的 tile（~MB 级 vs GPU ~KB 级） | FlashAttention 式 tiling 可用更大 block |
| 多核并行 | 32 AI Core 并行 | 多 head 并行、batch 并行 |
| AI Core 内流水 | Cube 和 Vector 可以流水执行 | GEMM + activation 可以 pipeline |

**不够友好的方面**：

| 维度 | 具体问题 | 影响的操作 |
|---|---|---|
| **编程模型** | TIK/Ascend C 不如 CUDA 灵活 | 自定义算子开发慢 |
| **动态 shape** | 编译器优化依赖静态 shape | Continuous Batching 的动态 seq/batch |
| **非 GEMM 算子** | Vector Unit 性能/工具不如 CUDA Core | Softmax, RoPE, LayerNorm |
| **生态** | 缺少 FlashAttention 等成熟实现 | 需要从头适配 |
| **Profiling** | 工具不如 Nsight 成熟 | 性能瓶颈定位困难 |
| **量化支持** | INT4/FP8 支持可能不完善 | 低精度推理加速受限 |

**实际适配建议**：
```
优先适配:
  1. 高效 GEMM kernel (利用 Cube Unit): 占 70%+ 时间
  2. FlashAttention 适配 (利用大 Local Memory 做更大的 tiling)
  3. Fused kernel (LayerNorm+Add, GEMM+Bias+Act)

利用特有优势:
  - 大 Local Memory: tiling 块可以更大 → 复用率更高 → 可能反超 GPU
  - 如果 Local Memory 足以放完整的 K/V block → 比 GPU 的 shared memory 更宽裕
```

---

### Q: 量化后的大模型运行时内存占用大概多少？

**模型权重内存**（最确定的部分）：

| 模型规模 | FP16 (2B/param) | INT8 (1B/param) | INT4 (0.5B/param) |
|---|---|---|---|
| 1.5B | 3 GB | 1.5 GB | 0.75 GB |
| 7B | **14 GB** | **7 GB** | **3.5 GB** |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 70B | 140 GB | 70 GB | 35 GB |
| 405B | 810 GB | 405 GB | 202 GB |

注：INT4 量化还需要额外存储 scale/zero_point（每 group 128 个参数存一组 FP16 scale），额外增加约 0.5-2%。

**KV Cache 内存**（动态部分，取决于 batch/seq）：

```
KV Cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes

7B (32 层, 8 KV heads, 128 dim, FP16):
  bs=1, seq=2048:   2×32×8×128×2048×1×2 = 256 MB
  bs=16, seq=4096:  2×32×8×128×4096×16×2 = 8 GB
  bs=64, seq=4096:  2×32×8×128×4096×64×2 = 32 GB ← 可能超过模型权重!

FP8 KV Cache: 上述值 ÷ 2
INT4 KV Cache: 上述值 ÷ 4
```

**其他内存开销**：

| 组件 | 大小 | 说明 |
|---|---|---|
| CUDA/框架运行时 | 1-2 GB | CUDA context、cuBLAS workspace |
| 激活值 (前向计算中间结果) | 0.5-2 GB | 取决于 batch 大小和算子融合程度 |
| 量化 scale/metadata | 模型的 0.5-2% | per-group scale + zero_point |
| Block Table (PagedAttention) | < 100 MB | 请求数 × max_blocks × sizeof(int) |

**总内存估算公式**：
```
总显存 ≈ 模型权重 + KV Cache + 框架开销 + 激活缓冲

7B INT4, bs=1, seq=4096:
  ≈ 3.5 GB + 0.5 GB + 1.5 GB + 0.5 GB = ~6 GB (可在 8GB 卡上运行)

7B FP16, bs=32, seq=4096:
  ≈ 14 GB + 8 GB + 1.5 GB + 1 GB = ~24.5 GB (需要 A100 40GB)

70B INT4, bs=16, seq=4096 (2 卡 TP):
  ≈ 35 GB + 8 GB + 1.5 GB + 1 GB = ~45.5 GB / 2 ≈ 23 GB/卡
```

**实际部署的显存规划建议**：
- 预留 10-20% 显存余量给内存碎片和突发峰值
- KV Cache 用 PagedAttention 按需分配，避免预分配最大值
- 量化 KV Cache（FP8）可以在大 batch 场景下翻倍并发量
