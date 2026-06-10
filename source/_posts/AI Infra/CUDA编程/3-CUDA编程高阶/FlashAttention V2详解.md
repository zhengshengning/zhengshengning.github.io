---
title: FlashAttention V2详解
date: 2026-06-08 10:30:00
mathjax: true
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程高阶]
tags: [FlashAttention, Attention, CUDA, GPU优化, 并行计算]
---

本文深入剖析 FlashAttention V2 相比 V1 的核心改进：调换内外循环顺序、优化线程块内的工作分配、以及支持更多 Attention 变体，最终在 A100 上达到理论 FLOPS 的 50-73%。

<!-- more -->

## 📑 目录

- [1. V1 的性能瓶颈回顾](#1-v1-的性能瓶颈回顾)
- [2. 核心改进一：调换循环顺序](#2-核心改进一调换循环顺序)
- [3. 核心改进二：Warp 级工作分配优化](#3-核心改进二warp-级工作分配优化)
- [4. 核心改进三：减少非矩阵乘运算](#4-核心改进三减少非矩阵乘运算)
- [5. Causal Mask 优化](#5-causal-mask-优化)
- [6. 前向传播完整算法](#6-前向传播完整算法)
- [7. 反向传播优化](#7-反向传播优化)
- [8. 性能分析与对比](#8-性能分析与对比)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)
- [参考资料](#-参考资料)

---

## 1. V1 的性能瓶颈回顾

### 1.1 V1 的实际利用率

FlashAttention V1 在 A100 上只达到了理论 FLOPs 的 25-40%。对于一个理应是 Compute-Bound（通过减少 IO 后）的算子来说，这个利用率并不理想。

瓶颈主要来自三个方面：

| 📊 问题 | 📝 原因 | 💡 影响 |
|---------|---------|---------|
| 循环顺序不佳 | 外循环遍历 K/V，每个 Q 块需要频繁读写 HBM | 额外的 HBM 读写 |
| Warp 间工作分配不均 | 所有 Warp 都参与相同的工作，存在冗余同步 | 低并行效率 |
| 非 GEMM 运算占比高 | Softmax rescaling 中有大量逐元素操作 | Tensor Core 利用率低 |

### 1.2 硬件特性分析

A100 GPU 的关键参数：

- **FP16 Tensor Core 峰值**：312 TFLOPS
- **HBM 带宽**：2 TB/s
- **SRAM 容量**：192 KB/SM（Shared Memory，可配置最大 164 KB 供用户使用）
- **Warp 数量**：每个 Thread Block 可包含多个 Warp（通常 4-8 个）

要充分利用 Tensor Core，需要确保计算流水线不被非 GEMM 操作打断。

---

## 2. 核心改进一：调换循环顺序

### 2.1 V1 的循环结构问题

V1 中外循环遍历 K/V 块，内循环遍历 Q 块。这带来两个问题：

1. **Q 和 O 的重复读写**：每处理一个新的 K/V 块，所有 Q 块和对应的输出 O 都需要重新加载和写回 HBM
2. **难以并行化**：内循环的各 Q 块之间有写依赖（都在更新同一个 O），但跨 Q 块的并行性无法利用

### 2.2 V2 的新循环结构

V2 将循环顺序调换为**外循环遍历 Q 块，内循环遍历 K/V 块**：

```
外循环: for i = 1 to T_r (遍历 Q 块) — 分配给不同 Thread Block
    从 HBM 加载 Q_i 到 SRAM（只加载一次）
    初始化 O_i = 0, m_i = -inf, l_i = 0
    内循环: for j = 1 to T_c (遍历 K/V 块)
        从 HBM 加载 K_j, V_j 到 SRAM
        计算 S_ij，更新统计量和 O_i（全在 SRAM 中）
    将最终 O_i 写回 HBM（只写一次）
```

### 2.3 收益分析

这个看似简单的调换带来了三重收益：

**收益 1**：O 只在最后写回一次

在 V1 中，每处理一个 K/V 块，O 都要读取再写回（共 $T\_c$ 次）。V2 中 O 只在内循环结束后写回一次。

**收益 2**：外循环天然可并行

不同的 Q 块之间完全独立（每个 Q 块计算自己的输出行），可以直接映射到不同的 Thread Block，实现 GPU 级别的并行。

**收益 3**：Q 只加载一次

每个 Thread Block 只负责一个 Q 块，Q 在 Shared Memory 中驻留整个内循环的过程。

📌 **关键点**：V1 外循环 K/V 是因为想让 K/V 只加载一次。但 V2 发现让 Q 只加载一次更优——因为 Q 块同时关联着输出 O，减少 O 的读写比减少 K/V 的读写收益更大。

---

## 3. 核心改进二：Warp 级工作分配优化

### 3.1 V1 的 Warp 分配方案

V1 中，Thread Block 内的多个 Warp 协同计算同一个 $S\_{ij}$ 块，然后协同完成后续的 Softmax 和 $P \cdot V$ 运算。这要求频繁的 Warp 间同步和通信。

### 3.2 V2 的 Warp 分配：沿 Q 维度划分

V2 将 Q 块进一步细分给不同的 Warp：

假设 Thread Block 有 4 个 Warp，Q 块大小为 $B\_r = 128$：
- Warp 0 负责 Q 的第 0-31 行
- Warp 1 负责 Q 的第 32-63 行
- Warp 2 负责 Q 的第 64-95 行
- Warp 3 负责 Q 的第 96-127 行

每个 Warp 独立计算自己负责的 Q 行与完整 K/V 块的注意力，互不干扰。

### 3.3 对比两种分配策略

| ✅ V2 沿 Q 分 Warp | ❌ 沿 K/V 分 Warp（V1 方案） |
|-------------------|--------------------------|
| 每个 Warp 独立完成 Softmax | 多个 Warp 需要合并部分 Softmax 结果 |
| 无需 Warp 间同步 | 需要 shared memory 或 shuffle 通信 |
| 输出 O 的不同行由不同 Warp 独占 | 输出 O 的同一行被多个 Warp 共同更新 |

💡 **提示**：如果沿 K/V 维度分割给不同 Warp，每个 Warp 只看到 Softmax 输入的一部分，必须在 Warp 间通信来合并最大值和求和——这正是 V1 性能低下的原因之一。

---

## 4. 核心改进三：减少非矩阵乘运算

### 4.1 问题：非 GEMM 指令的瓶颈

在 Attention 的 Tiling 计算中，除了两次矩阵乘法（$QK^\top$ 和 $PV$），还有大量逐元素操作：

- 减去最大值：$S\_{ij} - m\_i$
- 求指数：$\exp(\cdot)$
- 行求和：$\text{rowsum}(\cdot)$
- Rescaling：乘以修正因子

这些操作无法使用 Tensor Core，只能用 CUDA Core，成为计算流水线中的瓶颈。

### 4.2 V2 的优化：延迟 Rescaling

V2 将 Softmax 的 rescaling 步骤延迟到内循环结束后统一执行：

**V1 的做法**（每个 K/V 块都 rescale）：

$$
O\_i^{(j)} = \frac{l\_i^{(j-1)} \cdot e^{m\_i^{(j-1)} - m\_i^{(j)}}}{l\_i^{(j)}} \cdot O\_i^{(j-1)} + \frac{e^{S\_{ij} - m\_i^{(j)}}}{l\_i^{(j)}} \cdot V\_j
$$

**V2 的做法**（只维护未归一化的累积）：

$$
\tilde{O}\_i^{(j)} = e^{m\_i^{(j-1)} - m\_i^{(j)}} \cdot \tilde{O}\_i^{(j-1)} + e^{S\_{ij} - m\_i^{(j)}} \cdot V\_j
$$

最终一次性归一化：

$$
O\_i = \tilde{O}\_i^{(T\_c)} / l\_i^{(T\_c)}
$$

### 4.3 收益量化

这个优化减少了内循环中每次迭代的两次除法操作（一次对旧 O 的 rescale 除以 $l^{(j)}$，一次对新贡献除以 $l^{(j)}$），以一次最终除法替代。当内循环迭代次数 $T\_c$ 较大时，节省相当可观。

---

## 5. Causal Mask 优化

### 5.1 Causal Mask 的作用

在自回归模型（如 GPT）中，位置 $i$ 只能 attend 到位置 $j \leq i$。这对应一个下三角掩码矩阵。

### 5.2 朴素实现的浪费

如果不做特殊处理，即使 mask 为 0 的位置也会执行完整的 GEMM 和 Softmax 计算，浪费约一半的算力。

### 5.3 V2 的块级跳过策略

V2 引入了块级（Block-level）的 Causal Mask 优化：

对于第 $i$ 个 Q 块（行范围 $[iB\_r, (i+1)B\_r)$）和第 $j$ 个 K 块（列范围 $[jB\_c, (j+1)B\_c)$）：

1. **完全在 mask 下方**（$iB\_r \geq (j+1)B\_c$）：这个块完全可见，正常计算
2. **完全在 mask 上方**（$(i+1)B\_r \leq jB\_c$）：这个块完全不可见，直接跳过
3. **跨越对角线**：块内需要逐元素应用 mask

```
对角线以下（全可见）    对角线块（部分mask）    对角线以上（全跳过）
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ ■ ■ ■ ■ ■ ■│      │ ■ ■ ■ □ □ □│      │ □ □ □ □ □ □│
│ ■ ■ ■ ■ ■ ■│      │ ■ ■ ■ ■ □ □│      │ □ □ □ □ □ □│
│ ■ ■ ■ ■ ■ ■│      │ ■ ■ ■ ■ ■ □│      │ □ □ □ □ □ □│
└─────────────┘      └─────────────┘      └─────────────┘
```

### 5.4 加速效果

对于 Causal Attention，V2 的块跳过可以减少约 50% 的无效计算。结合循环顺序调换后的外循环沿 Q 分配，每个 Thread Block 只需遍历 K/V 到对角线位置即可提前退出内循环。

---

## 6. 前向传播完整算法

### 6.1 算法伪代码

```python
def flash_attention_v2_forward(Q, K, V, B_r, B_c, causal=False):
    N, d = Q.shape
    O = zeros(N, d)
    L = zeros(N)  # logsumexp，用于反向传播

    # 外循环（并行化到不同 Thread Block）
    parallel for i in range(0, N, B_r):
        Q_i = Q[i:i+B_r]  # 加载到 SRAM，整个内循环驻留

        # 初始化累积量（SRAM/寄存器）
        m_i = full(B_r, -inf)
        l_i = zeros(B_r)
        O_tilde = zeros(B_r, d)  # 未归一化的累积输出

        # 内循环：遍历 K/V 块
        kv_end = min(N, (i + B_r)) if causal else N
        for j in range(0, kv_end, B_c):
            K_j = K[j:j+B_c]  # 从 HBM 加载
            V_j = V[j:j+B_c]  # 从 HBM 加载

            # 计算局部注意力分数
            S_ij = Q_i @ K_j.T / sqrt(d)

            # 应用 causal mask（仅对角线块需要）
            if causal and i * B_r <= j * B_c + B_c:
                apply_causal_mask(S_ij, i, j, B_r, B_c)

            # 更新统计量
            m_new = max(m_i, rowmax(S_ij))
            P_ij = exp(S_ij - m_new[:, None])
            l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)

            # 更新未归一化输出
            O_tilde = exp(m_i - m_new)[:, None] * O_tilde + P_ij @ V_j

            m_i = m_new
            l_i = l_new

        # 最终归一化并写回
        O[i:i+B_r] = O_tilde / l_i[:, None]
        L[i:i+B_r] = m_i + log(l_i)  # 保存 logsumexp 给反向

    return O, L
```

### 6.2 与 V1 的关键差异

| 📊 方面 | V1 | V2 |
|---------|----|----|
| 外循环 | K/V 块 | Q 块（并行） |
| O 的 HBM 读写次数 | $T\_c$ 次 | 1 次 |
| 内循环归一化 | 每步都除以 $l$ | 最后统一除 |
| Causal 优化 | 无 | 块级跳过 |
| 保存的统计量 | $m$ 和 $l$ | $L = m + \log(l)$（合并为 logsumexp） |

---

## 7. 反向传播优化

### 7.1 保存 logsumexp 而非分离的 m 和 l

V2 在前向传播中保存 $L\_i = m\_i + \log(l\_i)$（即行级 logsumexp），而非分别保存 $m\_i$ 和 $l\_i$。这有两个好处：
- 减少一半的统计量存储
- 反向传播中可以直接恢复 $P\_{ij}$：$P\_{ij} = \exp(S\_{ij} - L\_i)$

### 7.2 反向传播的并行策略

V2 的反向传播采用**外循环遍历 K/V 块**的策略（与前向不同），确保 $dK$ 和 $dV$ 的更新不需要原子操作：

- $dK\_j$ 和 $dV\_j$ 由单个 Thread Block 独占累积（该 Block 遍历所有 Q 块来累积梯度）
- $dQ\_i$ 需要跨 Thread Block 累加（通过原子加或分两遍扫描实现）

### 7.3 反向传播中的 D 向量

V2 引入一个辅助向量 $D \in \mathbb{R}^N$，预计算为：

$$
D\_i = \text{rowsum}(dO\_i \odot O\_i)
$$

这个向量在反向传播的 Softmax 梯度计算中反复使用，预计算避免了重复运算。

---

## 8. 性能分析与对比

### 8.1 A100 上的性能

| 📊 序列长度 | V1 TFLOPS | V2 TFLOPS | V2 利用率 |
|------------|-----------|-----------|-----------|
| 1024 | 124 | 196 | 63\% |
| 2048 | 136 | 218 | 70\% |
| 4096 | 141 | 227 | 73\% |
| 8192 | 138 | 222 | 71\% |

V2 相比 V1 实现了约 1.5-1.7x 的加速。

### 8.2 与其他实现的对比

在 A100-80GB 上，序列长度 2048，head dim 128 的基准测试：

| 📊 实现 | 前向速度 | 反向速度 |
|---------|---------|---------|
| PyTorch 标准 | 1.0x | 1.0x |
| FlashAttention V1 | 2.8x | 2.5x |
| FlashAttention V2 | 4.3x | 3.9x |
| xFormers (cutlass) | 3.1x | 2.8x |

### 8.3 长序列的显存优势

| 📊 序列长度 | 标准 Attention 显存 | FlashAttention V2 显存 |
|------------|-------------------|---------------------|
| 4K | 128 MB | 4 MB |
| 16K | 2 GB | 16 MB |
| 64K | 32 GB (OOM) | 64 MB |

---

## 📝 总结

FlashAttention V2 通过三个核心改进将 A100 上的效率从 25-40% 提升到 50-73%：

1. **调换循环顺序**：外循环遍历 Q 块（并行），内循环遍历 K/V 块——减少 O 的 HBM 读写并实现天然并行
2. **Warp 级优化**：每个 Warp 负责 Q 的不同行，避免 Warp 间同步和通信
3. **延迟归一化**：内循环中只做指数缩放不做除法，最终统一归一化减少非 GEMM 操作
4. **Causal 块跳过**：利用 mask 的三角结构跳过无效块，对 Causal Attention 额外减少约 50% 计算量

---

## 🎯 自我检验清单

- 能解释 V2 为什么要把外循环从 K/V 改为 Q
- 能说明 V2 中 O 的 HBM 读写次数从 $T\_c$ 降到 1 的原因
- 能描述 V2 的 Warp 分配策略及其相对于 V1 的优势
- 能推导延迟归一化的正确性（未归一化累积最终除以 $l$ 等价于每步归一化）
- 能画出 Causal Mask 下块级跳过的三种情况
- 能解释 logsumexp $L = m + \log(l)$ 如何简化反向传播
- 能对比 V1 和 V2 在 A100 上的 TFLOPS 利用率差异

---

## 📚 参考资料

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention 官方实现 - GitHub](https://github.com/Dao-AILab/flash-attention)
- [Tri Dao - FlashAttention-2 Blog Post](https://tridao.me/publications/flash2/flash2.html)
- [NVIDIA A100 Tensor Core GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)
