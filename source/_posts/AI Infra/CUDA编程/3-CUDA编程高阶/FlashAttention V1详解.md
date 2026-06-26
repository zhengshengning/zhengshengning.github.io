---
title: FlashAttention V1详解
date: 2026-06-08 10:00:00
mathjax: true
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程高阶]
tags: [FlashAttention, Attention, CUDA, GPU优化, IO-Awareness]
---

本文深入剖析 FlashAttention V1 的核心原理与实现细节，理解如何通过 Tiling + Online Softmax 将 Attention 的额外显存占用从 $O(N^2)$ 降至 $O(N)$，同时大幅减少 HBM 访问量，实现无精度损失的加速。

<!-- more -->

## 📑 目录

- [1. 标准 Attention 的性能瓶颈](#1-标准-attention-的性能瓶颈)
- [2. FlashAttention 核心思想](#2-flashattention-核心思想)
- [3. Online Softmax 算法](#3-online-softmax-算法)
- [4. Tiling 分块计算策略](#4-tiling-分块计算策略)
- [5. 前向传播完整流程](#5-前向传播完整流程)
- [6. 反向传播与重计算](#6-反向传播与重计算)
- [7. IO 复杂度分析](#7-io-复杂度分析)
- [8. 实现要点与代码示例](#8-实现要点与代码示例)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)
- [参考资料](#-参考资料)

---

## 1. 标准 Attention 的性能瓶颈

### 1.1 标准 Attention 计算流程

标准的 Scaled Dot-Product Attention 按以下步骤执行：

$$
S = QK^\top / \sqrt{d}
$$

$$
P = \text{softmax}(S)
$$

$$
O = PV
$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 是序列长度，$d$ 是 head dimension。

### 1.2 显存瓶颈分析

标准实现的关键问题在于中间矩阵 $S$ 和 $P$ 的大小为 $N \times N$：

| 📊 指标 | 📝 数值 | 💡 说明 |
|---------|---------|---------|
| 中间矩阵大小 | $O(N^2)$ | S 和 P 各需 $N \times N$ 存储 |
| HBM 读写量 | $O(N^2)$ | 需要反复读写 $N \times N$ 矩阵 |
| 计算量 | $O(N^2 d)$ | 两次矩阵乘法 |

当序列长度 $N = 4096$、$d = 128$ 时，单个 head 的 $S$ 矩阵就需要 $4096 \times 4096 \times 2 = 32$ MB（FP16）。对于多 head 和 batch 的场景，显存开销极为可观。

### 1.3 计算强度不足

标准 Attention 的算术强度（Arithmetic Intensity）为：

$$
\text{AI} = \frac{O(N^2 d)}{O(N^2 + Nd)} \approx O(d)
$$

由于 $d$ 通常只有 64 或 128，算术强度较低，运算瓶颈在于显存带宽而非算力。这意味着大量时间花在数据搬运上，GPU 的计算单元处于饥饿状态。

💡 **提示**：就像一个厨师做菜的速度很快，但食材从仓库搬到厨房的速度太慢，导致厨师大部分时间在等食材——标准 Attention 的核心瓶颈就是 HBM 和 SRAM 之间的数据搬运。

---

## 2. FlashAttention 核心思想

<img src="/images/flashattentionv1-0.png" alt="" style="max-width: 100%; display: block; margin: 0 auto;" />

### 2.1 IO-Awareness：关注数据搬运

FlashAttention 的核心洞察是：**在现代 GPU 上，Attention 是一个 Memory-Bound 操作，优化的关键不在于减少浮点运算，而在于减少 HBM 访问次数。**

传统优化思路专注于减少 FLOPs（如稀疏 Attention），但 FlashAttention 另辟蹊径——通过精巧的分块计算策略，让数据尽可能留在 SRAM（Shared Memory）中完成计算，避免中间结果落地到 HBM。

### 2.2 两个关键技术

FlashAttention V1 依赖两个核心技术的组合：

1. **Tiling（分块）**：将 Q、K、V 切成小块，每次只加载一小块到 SRAM 中计算
2. **Online Softmax**：在不知道全局最大值的情况下，边加载新块边修正 Softmax 结果

📌 **关键点**：这两个技术缺一不可——Tiling 解决了"如何不生成完整的 $N \times N$ 矩阵"的问题，Online Softmax 解决了"分块计算时 Softmax 依赖全局信息"的问题。

### 2.3 直觉类比

想象你在做一道需要全班考试成绩来计算每个人排名百分比的题目。传统方法是先收齐所有人的分数（全部写在黑板上），再统一计算。FlashAttention 的做法相当于：每收到一组学生的分数，就立刻更新当前的统计量（最大值、求和），并修正之前的计算结果——最终得到的答案和收齐后统一计算完全相同。

---

## 3. Online Softmax 算法

### 3.1 标准 Softmax 的三遍扫描

为保证数值稳定性，标准（safe）Softmax 需要三遍扫描输入向量 $x \in \mathbb{R}^N$：

**第一遍**：求全局最大值用于数值稳定性

$$
m = \max\_{i=1}^{N} x\_i
$$

**第二遍**：计算归一化分母（指数和）

$$
\ell = \sum\_{j=1}^{N} e^{x\_j - m}
$$

**第三遍**：计算每个元素的归一化输出

$$
\text{softmax}(x\_i) = \frac{e^{x\_i - m}}{\ell}
$$

第二、三遍都依赖第一遍得到的全局最大值 $m$，第三遍又依赖第二遍得到的 $\ell$，这要求我们必须先完整扫描一次才能开始下一遍，显然与分块策略矛盾。

> 📖 **延伸**：Online Softmax（NVIDIA, 2018）将上述三遍合并为两遍——边扫描边同步维护 $m$ 和 $\ell$；FlashAttention 在此基础上进一步把"乘 $V$ 求输出"也融合进来。

### 3.2 Online Softmax 的递推公式

Online Softmax 的关键创新在于：维护一个**可修正**的局部统计量，随着新数据块的到来逐步更新。

假设我们已经处理了前 $j-1$ 块数据，维护了局部最大值 $m^{(j-1)}$ 和局部指数和 $l^{(j-1)}$。当第 $j$ 块数据 $x^{(j)}$ 到来时：

**步骤 1**：更新最大值

$$
m^{(j)} = \max(m^{(j-1)}, \max(x^{(j)}))
$$

**步骤 2**：修正并更新指数和

$$
l^{(j)} = l^{(j-1)} \cdot e^{m^{(j-1)} - m^{(j)}} + \sum\_{i} e^{x^{(j)}\_i - m^{(j)}}
$$

**步骤 3**：修正已有输出

$$
O^{(j)} = O^{(j-1)} \cdot \frac{l^{(j-1)} \cdot e^{m^{(j-1)} - m^{(j)}}}{l^{(j)}} + \frac{\sum\_i e^{x^{(j)}\_i - m^{(j)}} \cdot V^{(j)}\_i}{l^{(j)}}
$$

### 3.3 正确性保证

Online Softmax 的数学正确性来源于一个简单的恒等式：

$$
e^{x\_i - m\_{\text{new}}} = e^{x\_i - m\_{\text{old}}} \cdot e^{m\_{\text{old}} - m\_{\text{new}}}
$$

每当全局最大值更新时，之前所有的指数项都可以通过乘以一个修正因子 $e^{m\_{\text{old}} - m\_{\text{new}}}$ 来得到正确结果。这个操作是精确的，**没有任何近似**。

⚠️ **注意**：Online Softmax 产生的结果与标准 Softmax 在数学上完全等价（除了浮点运算顺序不同可能带来的微小舍入差异），这不是一种近似算法。

---

## 4. Tiling 分块计算策略

### 4.1 分块方案

将 $Q$、$K$、$V$ 沿序列维度分成大小为 $B\_r$（Q 的块大小）和 $B\_c$（K/V 的块大小）的小块：

- $Q$ 被分成 $T\_r = \lceil N / B\_r \rceil$ 个块
- $K, V$ 被分成 $T\_c = \lceil N / B\_c \rceil$ 个块

<img src="/images/flashattentionv1-1.png" alt="Tiling 分块计算策略" style="max-width: 85%; display: block; margin: 0 auto;" />

块大小的选择取决于 SRAM 容量 $M$（单位：元素个数，可理解为 `M = SRAM字节数 / sizeof(dtype)`）和注意力头维度 $d$（即每个 token 的特征维度，例如 LLaMA-7B 中 $d = 128$）：

$$
B\_c = \left\lceil \frac{M}{4d} \right\rceil, \quad B\_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil, d\right)
$$

**为什么是 $\dfrac{M}{4d}$？** 

计算一个块时，SRAM 需要同时驻留 $Q\_i, K\_j, V\_j, O\_i$ 四块大小约为 $B \times d$ 的张量（参见 4.2 节），合计约 $4 B d$ 个元素。要求 $4 B d \le M$，即可解出每块最多放 $\dfrac{M}{4d}$ 行。这里忽略了相对较小的 $S\_{ij}$（$B\_r \times B\_c$）和统计量 $m\_i, l\_i$（$2B\_r$），因为在 $B \ll d$ 量级下它们不主导 SRAM 占用。

**为什么 $B\_r$ 还要再和 $d$ 取 $\min$？** 

Online Softmax 的修正系数 $e^{m^{(j-1)} - m^{(j)}}$ 是 **逐行** 标量（每行一个），原论文为简化分析约束 $B\_r \le d$，避免行数显著超过列维度时统计量管理变复杂；这一约束在 V2 中被放宽。

### 4.2 SRAM 使用规划

在计算过程中，SRAM 需要同时容纳：

| 📊 数据 | 大小 | 📝 用途 |
|---------|------|---------|
| $Q\_i$ 块 | $B\_r \times d$ | 当前 Query 块 |
| $K\_j$ 块 | $B\_c \times d$ | 当前 Key 块 |
| $V\_j$ 块 | $B\_c \times d$ | 当前 Value 块 |
| $S\_{ij}$ 块 | $B\_r \times B\_c$ | 局部注意力分数 |
| $O\_i$ 块 | $B\_r \times d$ | 累积输出块 |
| $m\_i, l\_i$ | $2 B\_r$ | 行最大值、行指数和（每行一个标量） |

总 SRAM 需求约为 $(2B\_r + 2B\_c) \times d + B\_r \times B\_c + 2B\_r$，主导项是 $(2B\_r + 2B\_c) \times d$，这也是块大小公式中分母出现 $4d$ 的来源——需确保不超过 GPU 的 Shared Memory 容量 $M$。

### 4.3 外循环与内循环

FlashAttention V1 采用**外循环遍历 K/V 块，内循环遍历 Q 块**的策略：

```
外循环: for j = 1 to T_c (遍历 K/V 块)
    从 HBM 加载 K_j, V_j 到 SRAM
    内循环: for i = 1 to T_r (遍历 Q 块)
        从 HBM 加载 Q_i, O_i, m_i, l_i 到 SRAM
        计算局部注意力分数 S_ij = Q_i * K_j^T
        更新统计量和输出
        将更新后的 O_i, m_i, l_i 写回 HBM
```

💡 **提示**：外循环遍历 K/V 的设计意味着每个 K/V 块只从 HBM 加载一次，而 Q 和 O 会被多次加载。这在 V1 中是一个可以改进的点（V2 将改为外循环遍历 Q）。

---

## 5. 前向传播完整流程

### 5.1 逐步推导

以处理第 $i$ 个 Q 块、第 $j$ 个 K/V 块为例：

**Step 1**：计算局部注意力分数

$$
S\_{ij} = Q\_i K\_j^\top / \sqrt{d} \in \mathbb{R}^{B\_r \times B\_c}
$$

**Step 2**：计算当前块的行最大值

$$
\tilde{m}\_{ij} = \text{rowmax}(S\_{ij}) \in \mathbb{R}^{B\_r}
$$

**Step 3**：更新全局行最大值

$$
m\_i^{\text{new}} = \max(m\_i, \tilde{m}\_{ij})
$$

**Step 4**：计算修正因子并更新指数和

$$
\alpha = e^{m\_i - m\_i^{\text{new}}}
$$

$$
l\_i^{\text{new}} = \alpha \cdot l\_i + \text{rowsum}\left(e^{S\_{ij} - m\_i^{\text{new}}}\right)
$$

修正因子 $\alpha$ 的作用是把旧的指数和 $l\_i$（基于旧最大值 $m\_i$）"重缩放"到新最大值 $m\_i^{\text{new}}$ 下。

**Step 5**：更新输出

$$
O\_i^{\text{new}} = \frac{\alpha \cdot l\_i}{l\_i^{\text{new}}} \cdot O\_i + \frac{1}{l\_i^{\text{new}}} \cdot e^{S\_{ij} - m\_i^{\text{new}}} \cdot V\_j
$$


### 5.2 算法伪代码

```python
# FlashAttention V1 前向传播伪代码
def flash_attention_forward(Q, K, V, B_r, B_c):
    N, d = Q.shape
    O = zeros(N, d)
    m = full(N, -inf)  # 行最大值
    l = zeros(N)        # 行指数和

    # 外循环：遍历 K/V 块
    for j in range(0, N, B_c):
        K_j = K[j:j+B_c]  # 从 HBM 加载
        V_j = V[j:j+B_c]  # 从 HBM 加载

        # 内循环：遍历 Q 块
        for i in range(0, N, B_r):
            Q_i = Q[i:i+B_r]          # 从 HBM 加载
            O_i = O[i:i+B_r]          # 从 HBM 加载
            m_i = m[i:i+B_r]          # 从 HBM 加载
            l_i = l[i:i+B_r]          # 从 HBM 加载

            # 计算局部注意力分数
            S_ij = Q_i @ K_j.T / sqrt(d)  # [B_r, B_c]

            # 更新统计量
            m_new = max(m_i, rowmax(S_ij))
            l_new = l_i * exp(m_i - m_new) + rowsum(exp(S_ij - m_new))

            # 更新输出
            O_new = O_i * (l_i * exp(m_i - m_new) / l_new).unsqueeze(1)
            O_new += (exp(S_ij - m_new) / l_new.unsqueeze(1)) @ V_j

            # 写回 HBM
            O[i:i+B_r] = O_new
            m[i:i+B_r] = m_new
            l[i:i+B_r] = l_new

    return O
```

### 5.3 数值稳定性

整个计算过程通过减去最大值来保证数值稳定性：

- 所有指数运算的输入都经过减去当前已知最大值的处理，确保 $e^x$ 的参数 $x \leq 0$
- 修正因子 $e^{m\_{\text{old}} - m\_{\text{new}}}$ 也满足 $m\_{\text{old}} \leq m\_{\text{new}}$，因此 $\leq 1$

这保证了整个计算过程不会出现数值溢出。

---

## 6. 反向传播与重计算

### 6.1 反向传播的挑战

标准 Attention 的反向传播需要中间矩阵 $S$ 和 $P$（大小为 $N \times N$），用来计算梯度。其中 $S = QK^\top / \sqrt{d}$ 是注意力分数（pre-softmax logits），$P = \text{softmax}(S)$ 是 softmax 之后的注意力权重矩阵，最终输出 $O = PV$：

$$
dV = P^\top dO
$$

$$
dP = dO \cdot V^\top
$$

$$
dS = P \odot \left(dP - \text{rowsum}(dP \odot P) \cdot \mathbf{1}^\top\right)
$$

其中 FlashAttention 论文利用恒等式 $\text{rowsum}(dP \odot P) = \text{rowsum}(dO \odot O)$（记作 $D\_i$），从而避免显式构造完整的 $dP$ 矩阵。

如果保存这些中间矩阵，前向传播省下的显存就白费了。

### 6.2 重计算策略

FlashAttention V1 的解决方案是**不保存 $S$ 和 $P$，反向传播时重新计算它们**：

- 前向传播只保存：$Q, K, V, O, m, l$（输入、输出和统计量）
- 反向传播时：利用保存的 $m$ 和 $l$，重新加载 $Q\_i, K\_j$ 计算 $S\_{ij}$，再恢复 $P\_{ij}$

$$
P\_{ij} = \text{diag}(l\_i)^{-1} \cdot e^{S\_{ij} - m\_i}
$$

### 6.3 重计算的代价

重计算引入了额外的 FLOPs（相当于多做一次前向中的矩阵乘），但由于减少了 HBM 访问：

| ✅ 收益 | ❌ 代价 |
|---------|---------|
| 显存从 $O(N^2)$ 降到 $O(N)$ | 额外的 FLOP 开销（约 50\% 前向计算量） |
| HBM 访问大幅减少 | SRAM 需要同时存放更多数据 |
| 可以训练更长序列 | 反向传播核函数更复杂 |

由于 Attention 是 Memory-Bound 的，额外的计算可以被 HBM 访问的减少所"隐藏"，实际 wall-clock 时间反而更短。

---

## 7. IO 复杂度分析

### 7.1 标准 Attention 的 IO 复杂度

$$
\text{HBM 访问量} = O(Nd + N^2) = O(N^2)
$$

其中 $Nd$ 来自读写 $Q, K, V, O$ 等大小为 $N \times d$ 的张量，$N^2$ 来自读写中间矩阵 $S$ 和 $P$（各 $N \times N$）。当 $N \gg d$ 时，$N^2$ 项主导，因此整体为 $O(N^2)$。

### 7.2 FlashAttention 的 IO 复杂度

FlashAttention 的 HBM 访问量为：

$$
\text{HBM 访问量} = O\left(\frac{N^2 d^2}{M}\right)
$$

这个量级可以这样拆解：

- 由 4.1 节的块大小约束 $B\_c = O(M/d)$，总共 $T\_c = N / B\_c = O(Nd/M)$ 次外循环
- 每次外循环要遍历完整的 $Q 和 O$（共 $O(Nd)$ 个元素），所以 $Q 和 O$ 的总访问量为 $T\_c \cdot Nd = O(N^2 d^2 / M)$
- $K, V$ 整体只读一次，访问量为 $O(Nd)$，相对前者可忽略

合计即 $O(N^2 d^2 / M)$。直观上：**SRAM 越大（$M$ 越大），每次能装下的块越大，外循环次数越少，对 $Q/O$ 的重复读写也越少。**

其中 $M$ 是 SRAM 大小（以元素为单位）。由于 $d^2 < M$（典型值 $d = 128$，$M$ 对应约 $192 \text{KB} / 2 = 98304$ 个 FP16 元素，而 $d^2 = 16384$），这比 $O(N^2)$ 小得多。

### 7.3 下界证明

论文还证明了对于任何精确 Attention 算法，HBM 访问量的下界为 $\Omega(N^2 d^2 / M)$，这意味着 FlashAttention 的 IO 复杂度在渐近意义上是最优的。

### 7.4 实际加速效果

在 A100 GPU 上的典型性能对比：

| 📊 序列长度 | 标准 Attention | FlashAttention V1 | 加速比 |
|------------|---------------|-------------------|--------|
| 1024 | 1.0x | 2.4x | 2.4x |
| 2048 | 1.0x | 2.8x | 2.8x |
| 4096 | 1.0x | 3.5x | 3.5x |
| 8192 | OOM | 可运行 | - |

---

## 8. 实现要点与代码示例

### 8.1 张量形状贯穿全过程

输入 $Q, K, V$ 都是 4 维张量，前两维是 batch 与 head（作为"批次"维度原样传递），最后两维做矩阵运算：

```bash
Q: [B, H, N, D]
K: [B, H, N, D]   →  K^T (最后两维转置): [B, H, D, N]
V: [B, H, N, D]

Step 1  注意力分数：
  S = Q @ K^T / sqrt(D)
    = [B, H, N, D] @ [B, H, D, N]
    = [B, H, N, N]

Step 2  Softmax（沿最后一维 N，逐行归一化）：
  P = softmax(S, dim=-1)
    形状: [B, H, N, N]

Step 3  输出：
  O = P @ V
    = [B, H, N, N] @ [B, H, N, D]
    = [B, H, N, D]   （与 Q 同形）
```

辅助张量（FlashAttention 专属）：

- `M: [B, H, N]` —— 每行的最大值（数值稳定 + Online Softmax 修正）
- `L: [B, H, N]` —— 每行的指数和（softmax 分母）

### 8.2 Kernel 设计要点

> 📌 **关于并行划分（grid 与 block 设计）**：
> - **grid（program 粒度）=（B, H）**：每个 program 负责一个 `(batch, head)` 切片下完整的注意力计算，program 之间完全独立、无通信。
> - **block 内部线程**：在一个 program 内部，Triton 会把 `tl.dot`、`tl.max`、`tl.sum` 等向量化操作自动拆分到一个 thread block 的若干 warp 上。块大小由 `BLOCK_M`（Q 的行数 $B\_r$）、`BLOCK_N`（K/V 的行数 $B\_c$）、`BLOCK_D`（D 维向量化宽度）共同决定。
> - **program 内部串行**：外循环遍历 K/V 块、内循环遍历 Q 块
>
> 由于不同 Q 块在不同外循环迭代之间共享统计量 $m\_i, l\_i$ 与累积输出 $O\_i$，单个 program 的寄存器装不下"所有 Q 块"的 m/l/O，必须借助 HBM 的辅助张量 `M, L` 在迭代之间持久化（这正是 V1 IO 较多的根本原因——V2 通过交换循环顺序，把 Q 提到 grid 上并行，让 m/l/O 全程留在寄存器中，从而消除了这部分 HBM 访问）。
>
> ⚠️ **V1 并行度的天然瓶颈**：grid 大小 $B \times H$ 在推理场景常常只有几十（如 batch=1、heads=32 → 仅 32 个 program），而现代 GPU 通常有 80~140 个 SM（A100=108，H100=132），SM 利用率只有约 30%。这是 V1 的主要短板，也是 V2 必然出现的动因。


1. **Thread Block 映射**：V1 中每个 Thread Block 对应外循环的一个 K/V 块（即一个 $j$），内部遍历所有 Q 块
2. **Shared Memory 分配**：为 $K\_j$、$V\_j$（外循环常驻）以及 $Q\_i$、$S\_{ij}$ 预分配 SRAM
3. **寄存器使用**：统计量 $m\_i$、$l\_i$ 和累积输出 $O\_i$ 尽量放在寄存器中
4. **循环结构**：外循环加载 K/V 块，内部完成 GEMM 和 Softmax 更新

### 8.3 Triton 简化实现

```python
import math
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_v1_kernel(
    Q, K, V, O,    # [B, H, N, D]
    M, L,          # 行最大值与行指数和
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mh,
    N,
    D: tl.constexpr,        # 注意力头维度（head dimension），即每个 token 的特征长度，用在 scale = 1/sqrt(D)
    BLOCK_M: tl.constexpr,  # Q 块行数 B_r
    BLOCK_N: tl.constexpr,  # K/V 块行数 B_c
    BLOCK_D: tl.constexpr,  # 头维度上加载/计算的块宽度，当D本身就是2的幂时，BLOCK_D==D；否则 BLOCK_D > D，多出来的位置用 mask 屏蔽掉
):
    # ========== V1 并行划分：grid = (batch, head) ==========
    # 每个 program 串行扫完整个 (batch, head) 的所有 K/V 块和 Q 块
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # 当前 (batch, head) 在各张量上的起始偏移
    qkvo_bh_q = batch_id * stride_qb + head_id * stride_qh
    qkvo_bh_k = batch_id * stride_kb + head_id * stride_kh
    qkvo_bh_v = batch_id * stride_vb + head_id * stride_vh
    qkvo_bh_o = batch_id * stride_ob + head_id * stride_oh
    ml_bh = batch_id * stride_mb + head_id * stride_mh

    d_range = tl.arange(0, BLOCK_D)
    scale = 1.0 / math.sqrt(D)

    # ============ 外循环：遍历 K/V 块（V1 关键设计） ============
    for kv_start in range(0, N, BLOCK_N):
        kv_range = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_range < N

        # K_j, V_j 一次加载，常驻寄存器/SRAM，被所有 Q 块复用
        k = tl.load(K + qkvo_bh_k + kv_range[:, None] * stride_kn + d_range[None, :] * stride_kd,
                    mask=kv_mask[:, None] & (d_range[None, :] < D), other=0.0)
        v = tl.load(V + qkvo_bh_v + kv_range[:, None] * stride_vn + d_range[None, :] * stride_vd,
                    mask=kv_mask[:, None] & (d_range[None, :] < D), other=0.0)

        # ============ 内循环：遍历 Q 块 ============
        for q_start in range(0, N, BLOCK_M):
            q_range = q_start + tl.arange(0, BLOCK_M)
            q_mask = q_range < N

            # 加载 Q_i, O_i, m_i, l_i —— V1 中每个外循环都要从 HBM 重新读
            q = tl.load(Q + qkvo_bh_q + q_range[:, None] * stride_qn + d_range[None, :] * stride_qd,
                        mask=q_mask[:, None] & (d_range[None, :] < D), other=0.0)
            o_i = tl.load(O + qkvo_bh_o + q_range[:, None] * stride_on + d_range[None, :] * stride_od,
                          mask=q_mask[:, None] & (d_range[None, :] < D), other=0.0).to(tl.float32)
            m_i = tl.load(M + ml_bh + q_range, mask=q_mask, other=float('-inf'))
            l_i = tl.load(L + ml_bh + q_range, mask=q_mask, other=0.0)

            # S_ij = Q_i @ K_j^T / sqrt(d)
            s = tl.dot(q, tl.trans(k)) * scale
            # 屏蔽 K/V 越界列，防止假 0 污染 softmax 分母
            s = tl.where(kv_mask[None, :], s, float('-inf'))

            # Online Softmax 更新
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p, axis=1)

            # 输出修正 + 累加（与 5.1 Step 5 公式一致）
            o_new = o_i * (alpha * l_i / l_new)[:, None] \
                    + tl.dot(p.to(v.dtype), v) / l_new[:, None]

            # 把更新后的 O_i, m_i, l_i 写回 HBM，供下次外循环使用
            tl.store(O + qkvo_bh_o + q_range[:, None] * stride_on + d_range[None, :] * stride_od,
                     o_new.to(O.dtype.element_ty),
                     mask=q_mask[:, None] & (d_range[None, :] < D))
            tl.store(M + ml_bh + q_range, m_new, mask=q_mask)
            tl.store(L + ml_bh + q_range, l_new, mask=q_mask)


def flash_attn_v1(Q, K, V, BLOCK_M=64, BLOCK_N=64):
    # Q, K, V: [B, H, N, D]
    B, H, N, D = Q.shape
    O = torch.zeros_like(Q)
    M = torch.full((B, H, N), float('-inf'), device=Q.device, dtype=torch.float32)
    L = torch.zeros((B, H, N), device=Q.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (B, H)  # V1 并行：仅在 batch×head 维度上
    flash_attn_v1_kernel[grid](
        Q, K, V, O, M, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        M.stride(0), M.stride(1),
        N, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return O
```

📌 **代码要点回顾**：

1. **V1 并行划分**：grid 为 `(batch, head)`，K/V 在外、Q 在内——每个 K/V 块只从 HBM 加载一次，但 Q/O/m/l 会被 $T\_c$ 次外循环反复读写
2. **辅助张量 M, L**：因为不同外循环迭代之间需要共享 Q 的统计量 $m\_i, l\_i$，必须在 HBM 中开辟 `[B, H, N]` 的张量持久化
3. **K/V 越界屏蔽**（`tl.where(kv_mask, s, -inf)`）是正确性的关键，遗漏会让 softmax 分母被假 0 污染
4. **`m, l` 初值** 分别为 `-inf` 和 `0`，配合 `alpha = exp(-inf - m_new) = 0`，让首次迭代自然退化为正常累加
5. **V1 的局限**：`(batch, head)` 通常远小于 GPU 上的 SM 数（典型 A100 有 108 个 SM，而 batch×head 在推理时常只有几十），SM 利用率不足；V2 把 Q 提到 grid 上并行，正是为了解决这一点

---

## 📝 总结

FlashAttention V1 通过 **IO-Awareness** 的视角重新审视 Attention 计算，提出了 Tiling + Online Softmax 的组合方案：

1. **不生成完整的 $N \times N$ 中间矩阵**，通过分块流式计算避免 $O(N^2)$ 的显存开销
2. **Online Softmax 保证精确性**，通过维护和修正局部统计量，实现与标准 Softmax 完全等价的结果
3. **重计算替代存储**，反向传播时重算 $S$ 和 $P$，用少量额外计算换取巨大的显存节省
4. **IO 复杂度渐近最优**，HBM 访问量从 $O(N^2)$ 降至 $O(N^2 d^2 / M)$

---

## 🎯 自我检验清单

- 能解释标准 Attention 为什么是 Memory-Bound 而非 Compute-Bound
- 能手动推导 Online Softmax 的递推更新公式
- 能说明 Tiling 块大小 $B\_r$、$B\_c$ 如何根据 SRAM 容量确定
- 能描述 FlashAttention 前向传播中外循环和内循环各遍历什么
- 能解释为什么重计算策略不会让 wall-clock 时间变长
- 能推导 FlashAttention 的 HBM IO 复杂度为 $O(N^2 d^2 / M)$
- 能用 Triton 或 CUDA 实现简化版 FlashAttention 前向 Kernel
- 能说明 FlashAttention V1 中外循环遍历 K/V 的局限性

---

## 📚 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867)
- [FlashAttention 官方实现 - GitHub](https://github.com/Dao-AILab/flash-attention)
- [Tri Dao - FlashAttention 论文解读](https://tridao.me/publications/flash2/flash2.html)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

