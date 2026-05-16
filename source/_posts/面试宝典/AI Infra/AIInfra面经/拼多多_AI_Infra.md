---
title: '拼多多 AI Infra'
date: 2026-04-16 12:08:44
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Transformer 相比 MLP 的优点？

Transformer 和纯 MLP 在处理序列数据时有本质区别，核心差异在于**动态 vs 静态**的信息处理方式：

**1. 动态权重 vs 静态权重（最核心区别）**：
```
MLP: output = input × W (固定权重矩阵)
  → 无论输入什么，变换方式相同
  → 无法"选择性关注"输入的某些部分

Transformer (Attention): output = softmax(QK^T/√d) × V
  → 权重矩阵 softmax(QK^T/√d) 是输入依赖的！
  → 每个 token 动态决定"关注谁、关注多少"
```

**2. 全局依赖 vs 固定维度**：
- Transformer：任意两个 token 直接交互（O(1) 信息路径）
- MLP：只能处理固定维度的拼接输入，无法处理变长序列
- 如果用 MLP 处理序列：需要将所有 token 拼接为固定长度输入 → 丢失位置/长度灵活性

**3. 可扩展性（Scaling Law）**：

| 模型 | Scaling 行为 | 原因 |
|---|---|---|
| Transformer | loss ∝ N^{-0.076}（持续下降） | 注意力的动态路由 + 残差结构使深层有效 |
| 纯 MLP (如 MLP-Mixer) | 收益递减较快 | 静态权重的表达能力有瓶颈 |

**4. 序列处理能力对比**：

| 维度 | Transformer | MLP |
|---|---|---|
| 变长序列 | 天然支持（attention 适配任意长度） | 需要 padding 到固定长度 |
| 位置感知 | 通过位置编码 (RoPE/APE) | 依赖拼接顺序（脆弱） |
| 长距离依赖 | O(1) 直连 | 多层堆叠间接传递（信息衰减） |
| 复杂度 | O(n²d) (attention) + O(nd²) (FFN) | O(nd²) |

**5. 多头并行的表达力**：
- 多头注意力在不同子空间捕获不同语义模式（语法、语义、位置等）
- MLP 的不同神经元也能学习不同特征，但缺乏"选择性聚合"能力
- 直觉：Attention 像"查询数据库"（按需检索），MLP 像"固定公式计算"

**6. In-Context Learning 能力**：
- Transformer 可以通过 few-shot prompt 学习新任务（无需梯度更新）
- 理论分析表明 attention 可以隐式实现梯度下降（attention as gradient descent）
- MLP 没有这种能力——权重固定后行为固定

### Q: 介绍 MHA、GQA 和 MLA？

三种注意力机制代表了 **KV Cache 效率** 与 **模型表达能力** 之间的不同权衡：

**MHA（Multi-Head Attention）——原始方案**：
```
每个 head 有独立的 Q、K、V 投影:
  Q_i = x × W_Q_i  [seq, head_dim]
  K_i = x × W_K_i  [seq, head_dim]
  V_i = x × W_V_i  [seq, head_dim]

KV Cache: 2 × num_heads × head_dim × seq_len × bytes
  例 (LLaMA-65B): 2 × 64 × 128 × 4096 × 2 bytes = 128 MB/请求
  
优点: 每个 head 完全独立，表达能力最强
缺点: KV Cache 随 head 数线性增长，长序列/大 batch 时显存瓶颈
```

**GQA（Grouped Query Attention）——分组共享方案**：
```
将 Q heads 分组，每组共享一组 KV:
  如 32 Q heads, 8 KV heads (G=4, 每 4 个 Q head 共享 1 组 KV)

  Q_i = x × W_Q_i   (32 个独立的 Q head)
  K_g = x × W_K_g   (只有 8 个 KV head)
  V_g = x × W_V_g

KV Cache: 2 × num_kv_heads × head_dim × seq_len × bytes
  = 原始 MHA 的 1/G (G=4 时减少 75%)
  
优点: 显著减少 KV Cache，推理效率高
缺点: 共享 KV 损失部分表达能力（但实验证明影响很小）
代表: LLaMA-2 70B (G=8), LLaMA-3 (G=4), Qwen-2
```

**MLA（Multi-head Latent Attention）——低秩压缩方案**：
```
将 KV 投影到低维 latent 空间:
  c_kv = x × W_down  [seq, latent_dim] (latent_dim << num_heads × head_dim)
  
  推理时恢复 (或通过权重吸收避免显式恢复):
  K' = c_kv × W_up_K
  V' = c_kv × W_up_V

KV Cache: (latent_dim + rope_dim) × seq_len × bytes
  例: (512 + 64) × 4096 × 2 bytes = 4.7 MB/请求 (vs MHA 的 128 MB!)

优点: 压缩比极高（相比 GQA 还能再压 3-4x），精度保持好（可学习投影）
缺点: 推理时需要权重吸收优化，实现复杂
代表: DeepSeek-V2/V3/R1
```

**三者对比总结**：

| 维度 | MHA | GQA (G=8) | MLA |
|---|---|---|---|
| KV 存储/token | 2×H×d (高) | 2×H/G×d (中) | latent_d + rope_d (低) |
| 典型值 (128h, 128d) | 32768 | 2048 | 576 |
| 相比 MHA 压缩 | 1x | 16x | **57x** |
| 训练时确定 | 是 | 是 | 是 |
| 精度保持 | 最好 | 好 | 很好（可学习投影） |
| 工程实现 | 简单 | 简单 | 复杂（权重吸收+RoPE 分离） |

### Q: 算法和 Infra 工作有什么不同？侧重点是什么？

**AI 算法工程师**：

| 维度 | 说明 |
|---|---|
| **核心目标** | 提升模型效果（accuracy/quality） |
| **工作内容** | 模型架构设计、训练 recipe、数据 pipeline、评估方法 |
| **关注指标** | Loss、PPL、MMLU/GSM8K 等 benchmark、人类评测 |
| **技能要求** | 数学（优化、概率）、深度学习理论、实验设计、论文阅读 |
| **日常工作** | 跑实验、调超参、分析 loss 曲线、写论文 |
| **产出** | 新模型/新方法/新 recipe、论文、训练好的模型权重 |

**AI Infra 工程师**：

| 维度 | 说明 |
|---|---|
| **核心目标** | 提升系统效率（speed/cost/throughput） |
| **工作内容** | 高性能 kernel 开发、推理引擎、训练框架、调度系统 |
| **关注指标** | 吞吐(tokens/s)、延迟(TTFT/TPOT)、显存利用率、GPU 利用率、成本 |
| **技能要求** | CUDA/C++、计算机体系结构、分布式系统、性能优化 |
| **日常工作** | Profiling、写 kernel、优化通信、设计调度算法 |
| **产出** | 高效 kernel/框架/系统、性能报告、部署方案 |

**两者的协作关系**：
```
算法: "我设计了 MLA 注意力机制，比 MHA 效果好"
Infra: "我来实现高效的 MLA kernel + 权重吸收优化 + 适配推理引擎"

算法: "我需要训练 671B MoE 模型"
Infra: "我来设计 EP + PP + DP 并行策略 + 通信库 + 调度方案"

算法: "量化到 W4 后 benchmark 下降 3%"
Infra: "我来实现更高效的 W4 kernel / 或用 per-group scale 减少精度损失"
```

### Q: 如何优化模型训练中的访存？

训练中的访存优化旨在减少 GPU HBM 的读写量和显存占用，提升训练吞吐。

**问题分析——训练的访存瓶颈**：
```
前向: 读权重 + 读输入 → 计算 → 写激活 (保存用于反向)
反向: 读权重 + 读激活 + 读输出梯度 → 计算 → 写权重梯度 + 写输入梯度

最大访存消耗:
  - 激活值存储: O(batch × seq × hidden × num_layers) → 可达数十 GB
  - 权重读取: 前向 + 反向各读一次，共 2 次
  - 优化器状态: FP32 主权重 + m + v = 12 bytes/param
```

**优化方法及其对访存的影响**：

| 方法 | 减少的访存 | 原理 | 代价 |
|---|---|---|---|
| **混合精度 (BF16)** | 权重+激活带宽减半 | FP16 数据 = FP32 的 1/2 | 需 FP32 主权重（显存不减） |
| **梯度 Checkpoint** | 激活存储减少 √N 倍 | 不保存中间激活，反向时重算 | 增加 ~33% 计算时间 |
| **FlashAttention** | Attention 的 HBM 读写减 10x | Tiling 到 SRAM 计算，不存 N×N 矩阵 | 无额外代价 |
| **算子融合** | 消除中间 tensor 的 HBM 读写 | 多个操作合并为一个 kernel | 开发复杂度 |
| **Sequence Parallel** | 激活分片到多 GPU | 沿 seq 维度分片 activation | 增加 All-Gather 通信 |
| **Zero-Offload** | 优化器状态移到 CPU | 利用 CPU 内存（更大更便宜） | CPU-GPU 传输开销 |
| **通信压缩** | 减少梯度通信量 | FP16/INT8 压缩梯度后 All-Reduce | 少量精度损失 |

**FlashAttention 对训练访存的具体优化**：
```
标准 Attention 训练:
  前向: 读 QKV → 写 S[N×N] → 读 S → 写 P[N×N] → 读 PV → 写 O
  反向: 读 dO, P, V → 写 dV, dP → 读 dP, S → 写 dS → 读 dS, Q, K → 写 dQ, dK
  HBM 访问: O(N² + Nd) per layer

FlashAttention 训练:
  前向: 读 QKV (SRAM tiling) → 写 O + logsumexp (仅 O(N) 大小)
  反向: 读 dO, Q, K, V, logsumexp → 写 dQ, dK, dV (在 SRAM 中重算 P)
  HBM 访问: O(N²d/M) per layer (M = SRAM 大小)
  
  节省: S 和 P 的 N×N 矩阵从不写入 HBM
```

**实践建议**：
```
训练 7B 模型:
  必选: BF16 + FlashAttention + Gradient Checkpoint (前 50% 层)
  可选: ZeRO-2 + 算子融合

训练 70B+ 模型:
  必选: BF16 + FlashAttention + ZeRO-3 + TP + PP
  可选: Sequence Parallel + FP8 训练 (H100) + Offload
```

### Q: 介绍下针对 KL 散度算子做了哪些优化？

KL 散度（Kullback-Leibler Divergence）在 RLHF/DPO 训练中频繁使用，计算公式：
```
KL(P || Q) = Σ p(x) × log(p(x) / q(x))
           = Σ p(x) × (log p(x) - log q(x))
```

**标准实现的问题**：
```python
# PyTorch 标准实现（多个 kernel）
log_p = F.log_softmax(logits_p, dim=-1)  # kernel 1: softmax + log
log_q = F.log_softmax(logits_q, dim=-1)  # kernel 2: softmax + log
p = log_p.exp()                           # kernel 3: exp
kl = (p * (log_p - log_q)).sum(-1)        # kernel 4: sub + mul + reduce

# 问题: 4 次 kernel launch + 3 个中间 tensor 的 HBM 读写
# 每个中间 tensor: batch × seq × vocab_size × 4 bytes
# vocab_size=128K 时: 单个中间 tensor = 512 KB/token → 访存极重
```

**优化方案**：

**1. 算子融合（最核心）**：
```cuda
// 将 log_softmax + exp + sub + mul + reduce 融合为单一 kernel
__global__ void fused_kl_divergence(
    float* logits_p, float* logits_q, float* output,
    int vocab_size
) {
    // 在寄存器/shared memory 中完成全部计算
    // Step 1: 两个 log_softmax（需要 max + sum reduction）
    float max_p = -INFINITY, max_q = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        max_p = fmaxf(max_p, logits_p[row * V + i]);
        max_q = fmaxf(max_q, logits_q[row * V + i]);
    }
    max_p = block_reduce_max(max_p);
    max_q = block_reduce_max(max_q);
    
    // Step 2: 计算 sum_exp 和 KL 同时进行
    float sum_exp_p = 0, sum_exp_q = 0, kl_sum = 0;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float lp = logits_p[...] - max_p;
        float lq = logits_q[...] - max_q;
        float exp_lp = expf(lp);
        sum_exp_p += exp_lp;
        sum_exp_q += expf(lq);
        kl_sum += exp_lp * (lp - lq);  // p * (log_p - log_q) 的分子部分
    }
    // 最终修正 log_sum_exp
    kl_sum = kl_sum / sum_exp_p - logf(sum_exp_q) + logf(sum_exp_p);
    // 这里利用了 log_softmax 的性质避免额外计算
}
```

融合后效果：
- 中间 tensor 的 HBM 读写完全消除
- Kernel launch 从 4 次降为 1 次
- 加速 3-5x（取决于 vocab_size）

**2. 向量化加载**：
```cuda
// 使用 float4 一次加载 128 bits
float4 logits_vec = reinterpret_cast<float4*>(logits_p)[idx];
// 减少内存事务数 4x
```

**3. 高效 Reduction**：
```cuda
// Warp-level reduction (无需 shared memory)
float val = ...;
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);

// Block-level reduction (用 shared memory)
__shared__ float warp_results[32];
if (lane_id == 0) warp_results[warp_id] = val;
__syncthreads();
if (warp_id == 0) {
    val = warp_results[lane_id];
    // warp reduce again
}
```

**4. 数值稳定性**：
```
// 直接计算 p * log(p/q) 有数值问题:
//   - p 接近 0 时 log(p) → -inf
//   - p/q 中 q 接近 0 时溢出

// 使用 log_softmax 形式:
//   KL = Σ exp(log_p) × (log_p - log_q)
//   其中 log_p = logits_p - log_sum_exp(logits_p)
// 这种形式数值稳定（log_sum_exp 通过减去 max 实现）
```

### Q: PagedAttention 的原理？

PagedAttention 借鉴 OS 虚拟内存的分页机制管理 KV Cache，核心解决 KV Cache 的**显存碎片**和**利用率低**的问题。

**传统方案的问题**：
```
为每个请求预分配最大序列长度的连续显存:
  → 内部碎片: 实际生成 200 tokens 但预分配了 4096 的空间，浪费 95%
  → 外部碎片: 释放后的碎片无法合并供新请求使用
  → 利用率: 仅 20-40%
```

**PagedAttention 方案**：
- KV Cache 按固定大小 **block**（如 16 tokens）分配物理显存
- **Block Table** 维护每个请求的逻辑 block → 物理 block 映射
- 物理 block 不需要连续，通过间接寻址访问
- 按需分配：生成新 token 时才分配新 block
- COW 共享：Beam Search / Parallel Sampling 的公共前缀共享 block

**性能数据**：
- 利用率从 20-40% 提升到 **96-98%**
- 同等显存下并发量提升 **2-4x**
- Attention 计算 overhead: ~3-5%（间接寻址开销）
- vLLM 的核心创新，被广泛采纳（TensorRT-LLM、SGLang 等）

### Q: Triton 和 CUDA 的区别？

Triton 是 OpenAI 推出的 GPU 编程语言，定位在 CUDA 之上的抽象层，目标是**降低 GPU kernel 开发门槛的同时保持高性能**。

**核心区别对比**：

| 维度 | CUDA | Triton |
|---|---|---|
| **抽象级别** | 线程级（per-thread 编程） | Block/Tile 级（per-block 编程） |
| **语言** | C/C++ + CUDA 扩展 | Python（装饰器 @triton.jit） |
| **内存管理** | 手动管理 shared memory、寄存器 | 编译器自动决定数据放哪里 |
| **并行表达** | 手动指定 grid/block/thread | 编写 tile 操作，编译器自动并行 |
| **Bank Conflict** | 手动 padding/swizzle 避免 | 编译器自动优化 |
| **性能上限** | 最高（完全控制硬件） | CUDA 的 80-95%（大部分场景足够） |
| **开发效率** | 低（数百行 C++ 实现一个 kernel） | 高（几十行 Python） |
| **可移植性** | 仅 NVIDIA GPU | 支持 NVIDIA + AMD GPU |
| **学习曲线** | 陡峭（需理解硬件细节） | 平缓（只需理解 tile 操作） |

**代码对比（矩阵乘法）**：

```python
# Triton GEMM (核心 ~40 行)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * K + (k + tl.arange(0, BLOCK_K))[None, :])
        b = tl.load(B + (k + tl.arange(0, BLOCK_K))[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)
```

```cuda
// CUDA GEMM (完整优化版 ~300+ 行)
// 需要手动处理: shared memory tiling, 双缓冲, bank conflict,
// 寄存器 tiling, 向量化加载, Tensor Core MMA...
```

**Triton 编译器做的优化**：
- 自动选择 shared memory vs registers 存放数据
- 自动处理 bank conflict（通过 swizzle）
- 自动选择 Tensor Core 指令（当检测到 tl.dot 时）
- 自动向量化内存访问
- 自动流水线化（double buffering）

**选择建议**：
```
选 Triton:
  - 快速原型验证新算子
  - Elementwise / Reduction / 简单 GEMM
  - 需要 AMD GPU 支持
  - 团队 CUDA 经验不足

选 CUDA:
  - 追求最极致性能（如 cuBLAS 级别的 GEMM）
  - 需要精细控制（如 persistent kernel、warp-level 技巧）
  - 非标准操作（如 PagedAttention 的间接寻址）
  - 需要与 NCCL/cuDNN 等 CUDA 生态深度集成
```

**实际案例**：
- FlashAttention: CUDA 实现（需要极致优化 + 复杂的 online softmax）
- torch.compile (Inductor): 默认生成 Triton kernel
- vLLM PagedAttention: CUDA 实现（需要间接寻址的精细控制）
- 普通 LayerNorm/RMSNorm: Triton 实现即可达到接近最优性能

### Q: 手撕：MHA 实现？

（编程题）

### Q: 手撕：C++ 编程题？

（编程题）
