---
title: FlashAttention V4详解：Blackwell架构
date: 2026-06-26 11:30:00
mathjax: true
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程高阶]
tags: [FlashAttention, Attention, CUDA, Blackwell, 异步流水线, CuTe-DSL]
---

本文剖析 FlashAttention V4 如何应对 Blackwell 架构（B200/GB200）的"非对称扩展"——Tensor Core 算力翻倍而共享内存带宽、指数运算单元几乎原地踏步。FA4 通过全异步流水线重构、软件模拟指数函数、条件 Softmax 重缩放与 2-CTA 反向传播等手段，在 B200 上 BF16 达到 1613 TFLOPs/s（71% 利用率），相比 cuDNN 9.13 最高加速 1.3×，相比 Triton 最高加速 2.7×。

<!-- more -->

## 📑 目录

- [1. 从 V3 到 V4：硬件瓶颈又变了](#1-从-v3-到-v4硬件瓶颈又变了)
- [2. 前置：注意力的前向与反向](#2-前置注意力的前向与反向)
- [3. Blackwell 架构的关键变化](#3-blackwell-架构的关键变化)
- [4. Roofline 分析：瓶颈到底在哪](#4-roofline-分析瓶颈到底在哪)
- [5. 正向传播流水线重构](#5-正向传播流水线重构)
- [6. 软件模拟指数函数](#6-软件模拟指数函数)
- [7. 跳过在线 Softmax 重缩放](#7-跳过在线-softmax-重缩放)
- [8. 反向传播与 2-CTA MMA](#8-反向传播与-2-cta-mma)
- [9. 调度策略：LPT 与确定性模式](#9-调度策略lpt-与确定性模式)
- [10. CuTe-DSL：用 Python 写内核](#10-cute-dsl用-python-写内核)
- [11. 性能评估](#11-性能评估)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)
- [参考资料](#-参考资料)

---

## 1. 从 V3 到 V4：硬件瓶颈又变了

FlashAttention 每一代的核心命题，都是"硬件变了，算法怎么跟上"。V1 解决的是显存读写（IO-aware tiling），V2 解决的是 GPU 占用率（沿序列维度并行），V3 解决的是 Hopper 上 Tensor Core 与数据搬运的异步重叠。到了 V4，问题再次被硬件改写。

打个比方：过去几代芯片像是均衡升级的工厂，机器（Tensor Core）、传送带（显存/共享内存带宽）、检验台（指数运算单元）都按差不多的比例提速。但 Blackwell 这一代非常"偏科"——机器速度直接翻倍，而传送带和检验台几乎没变。结果就是机器越快，越容易卡在等料和等检验上。这种各功能单元增长速度不一致的现象，论文称之为**非对称扩展**（asymmetric scaling）。

🔑 **核心概念**：FlashAttention V4 的设计主线，不再是"把矩阵乘做得更快"，而是"在矩阵乘已经快到溢出的前提下，怎么把非矩阵乘运算（Softmax、共享内存搬运）藏到矩阵乘的影子里，并尽量减少它们的绝对量"。

为什么 V3 的方案不能直接搬过来？两个原因：

- **瓶颈位置变了**。V3 在 Hopper 上把 Softmax 与 GEMM 重叠就够用，因为那时 Tensor Core 与其他单元的差距还没那么大。Blackwell 上 Tensor Core 又翻了一倍，Softmax 和共享内存反而成了主导项。
- **指令不向前兼容**。Blackwell 的第五代 Tensor Core 写入的是全新的张量内存（TMEM），Hopper 的 MMA 指令在 Blackwell 上根本跑不起来，必须针对新硬件重写。

---

## 2. 前置：注意力的前向与反向

为了让后面的 roofline 分析和反向优化讲得清楚，这里先把注意力的数学定义对齐一遍。如果你已经熟悉 FlashAttention 系列，可以快速跳读。

设单个注意力头的查询、键、值为 $Q, K, V \in \mathbb{R}^{N \times d}$，其中 $N$ 是序列长度，$d$ 是头维度。**前向**计算分三步：

$$
S = \alpha Q K^\top \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S), \quad O = P V \in \mathbb{R}^{N \times d}
$$

其中缩放因子 $\alpha = 1/\sqrt{d}$，softmax 按行计算。为数值稳定，实践中会先减去每行的最大值再做指数——这正是后文"在线 softmax"和"条件重缩放"要处理的对象。

**反向**给定输出梯度 $dO \in \mathbb{R}^{N \times d}$，需要算出五组量：

$$
dV = P^\top dO, \quad dP = dO\, V^\top
$$

$$
dS = \text{dsoftmax}(dP), \quad dQ = \alpha\, dS\, K, \quad dK = \alpha\, dS^\top Q
$$

其中 softmax 的梯度按行为 $ds = (\text{diag}(p) - p p^\top)\, dp$，$p = \text{softmax}(s)$。

💡 **为什么反向比前向贵**：前向只有 2 次矩阵乘（$QK^\top$ 和 $PV$），而反向因为要重算 $S$（FlashAttention 不保存中间的 $N \times N$ 矩阵，靠重计算省显存）再加上 $dV, dP, dQ, dK$，一共是 5 次矩阵乘。这就是后文反向 FLOPs 大约是前向 2.5 倍的由来，也是反向更容易撞上共享内存瓶颈的根本原因。

---

## 3. Blackwell 架构的关键变化

要理解 FA4 的每一处设计，先得搞清楚 Blackwell（B200/GB200）相比 Hopper（H100）到底改了什么。在看变化之前，先把两个贯穿全文的硬件背景对齐：**内存层级**和**线程层级**。

**内存层级**，从外到内、容量递减而带宽递增：

- **全局内存（GMEM，即 HBM）**：片外 DRAM，所有 SM 可访问，最大但最慢；
- **L2 缓存**：GMEM 数据透明缓存于此；
- **共享内存（SMEM）**：每个 SM 内一块由程序员管理的高速缓存，CTA 内所有线程可直接访问；
- **寄存器文件（RMEM）**：最内层，每个线程最多 256 个私有寄存器；
- **张量内存（TMEM）**：Blackwell 新增，详见 §3.1。

**线程层级**，从细到粗：线程 → warp（32 线程）→ warpgroup（4 个连续 warp）→ 线程块（CTA）→ 线程块集群（cluster）→ 网格（grid）。同一 CTA 内的线程被共同调度到同一 SM，同一 cluster 内的多个 CTA 被共同调度到同一 GPC——这正是 §3.2 的 2-CTA 协同得以成立的硬件基础。

理清这两层之后，再看 Blackwell 相对 Hopper 的关键改动：

| 📊 特性 | Hopper (H100) | Blackwell (B200) |
|---------|---------------|------------------|
| FP16/BF16 Tensor Core | 1 PFLOPS | 2.25 PFLOPS（约 2×） |
| BF16 MMA 吞吐 | 4096 次/SM/周期 | 8192 次/SM/周期 |
| MMA tile 尺寸 | 64×N | 128×N（M 维 2×） |
| MMA 输出写入位置 | 寄存器 | 张量内存 TMEM（异步） |
| 指数运算单元 (MUFU) | 16 次/SM/周期 | 16 次/SM/周期（不变） |
| 共享内存读带宽 | 128 字节/SM/周期 | 128 字节/SM/周期（不变） |
| 跨 CTA 协同 MMA | 无 | 2-CTA MMA 模式 |

⚠️ **注意**：这张表里最值得盯着看的是后三行。Tensor Core 翻倍了，但**指数运算单元和共享内存带宽完全没动**。FA4 的几乎所有优化都是冲着这两个"没动"去的。

📌 **数字怎么来的**：8192 这个 BF16 MMA 吞吐可由理论峰值反推——2.25 PFLOPS ÷ 1850 MHz ÷ 148 SM ≈ 8192 次/SM/周期；指数单元（MUFU）16 次、SMEM 读带宽 128 字节均为微基准实测，与 Hopper 一致。补充一句：B300/GB300 已把指数吞吐翻倍到 32 次/SM/周期，但截至本文撰写尚未广泛上市，本文仍以 B200/GB200 为基准。

🔑 **瓶颈的转移**：Blackwell 体现的核心趋势是**Tensor Core 吞吐扩展得比其他单元快**——在相近的功耗/芯片面积约束下，厂商优先堆最关键的矩阵乘单元。其直接后果就是性能瓶颈从矩阵乘转移到了共享内存流量和 Softmax 这类非矩阵乘运算上。这正是后面 roofline 分析（§4）和全部优化的出发点。

### 3.1 张量内存 TMEM

Blackwell 给每个 SM 配了 256 KB 的张量内存。它和共享内存最大的区别在于：**Tensor Core 可以把 MMA 结果直接异步写进 TMEM，完全不占用寄存器**。

用一个类比理解：Hopper 时代，Tensor Core 算完一块结果必须立刻把它"塞进自己手里的寄存器"，手满了就得停下来等寄存器腾空——这就是 Hopper 内核饱受寄存器压力之苦的根源。Blackwell 给 Tensor Core 旁边放了一块专用储物柜（TMEM），算完直接扔进柜子，不占手，于是可以马不停蹄地继续算，也让更大的 tile 成为可能。

📌 **关键点**：TMEM 以 32 列（16 KB）为粒度分配，需要程序员显式管理分配、释放和数据搬移，是 warp 同步的。它解锁了 FA4 流水线设计的更大自由度——后面会看到，正向传播里 S、P、输出都被精心摆进 TMEM 的不同区域。

📌 **更高的异步性**：Blackwell 的 MMA 把输出**异步**写入 TMEM，不再像 Hopper 那样阻塞在寄存器回写上，计算与其他操作能更好地重叠。这种硬件级异步性是 **warp 专用化内核**的前提——一个 CTA 内的 warp 被划分为"生产者"（只搬数据）和"消费者"（只算）两类角色，这正是 §5 正向流水线 warp 角色划分的硬件依据。

⚠️ **别搞混：TMEM 不是输入操作数的中转站**。一个常见误解是"Tensor Core 计算前要把数据从全局内存加载到 TMEM"。实际数据流是分两侧的：

- **输入侧**（喂给 Tensor Core 的 A、B 操作数）：GMEM →（TMA 异步搬运）→ **SMEM** → Tensor Core。从全局内存新加载的数据始终先进**共享内存**，Tensor Core 从 SMEM 读操作数。这正是 §3.2 的 2-CTA 优化分摊 **SMEM**、§4.1 的 roofline 盯着 SMEM 带宽的原因。
- **输出侧**（MMA 的结果/累加器）：Tensor Core →（异步写入）→ **TMEM**。TMEM 装的是算出来的中间结果，不消耗寄存器。
- 唯一的例外是 **TS（张量-共享）操作**：当某次 MMA 的一个操作数恰好是**上一次 MMA 写进 TMEM 的输出**时，可直接留在 TMEM 复用，省去一次 SMEM 读取（§4.1 的 $PV$ 中 $P$ 即如此）。但这种"TMEM 作输入"只对前一步的产物成立，**不是**从 GMEM 加载的路径。


### 3.2 2-CTA MMA 模式

Blackwell 允许同一线程块集群（cluster）内的**一对 CTA 协同完成一次 MMA**。配对中一个 CTA 发起指令，另一个必须全程存活配合。

单 CTA MMA 把 M 维度限制在 128，而配对模式支持 M=128 或 256。它的妙处在于操作数分摊：在 M 维度上把累加器和 A 切到两个 CTA，在 N 维度上把 B 切到两个 CTA，于是**每个 CTA 只需在自己的共享内存里暂存 B 的一半**，硬件在做乘法时再合并出完整的 B。这直接把 B 操作数的共享内存占用和带宽消耗砍掉一半——正好打在 Blackwell 最稀缺的共享内存带宽上。

⚠️ **注意**：因为要跨 CTA 对访问张量内存，内核必须以固定配对方式启动 CTA，并在整个内核生命周期内对 TMEM 与 Tensor Core 操作保持一致的 2-CTA 模式。这个约束在 §8 反向传播里会再次出现。

---

## 4. Roofline 分析：瓶颈到底在哪

FA4 的设计不是拍脑袋，而是基于 roofline 分析。我们只看三种资源：Tensor Core（MMA）、共享内存（SMEM）、指数运算单元，分析在一个 tile 上各自需要多少周期。

⚠️ **注意**：这是一个简化模型，没有计入寄存器带宽、L2 带宽、普通浮点运算等资源，但已经足以定位主要瓶颈。读 roofline 表时记住一句话——**哪个资源的周期数最大，它就是瓶颈，其余资源只要能藏进它的影子里就不影响整体耗时**。

设 Q、K 沿序列维度的 tile 形状为 $M \times N$，头维度为 $d$。

### 4.1 正向传播的三笔账

正向每次迭代做两次 MMA：$S = QK^\top$ 和 $O = PV$，每次 MMA 是 $2MNd$ 次浮点运算。在 8192 FLOPs/周期下：

$$
T_{\text{MMA}} = \frac{4MNd}{8192} \quad \text{周期}
$$

共享内存这边，由于每条 MMA 指令处理 128×128 的 tile，当输出超过这个尺寸时操作数会被**重复读取**多次。算下来：

$$
T_{\text{smem}} = \frac{3MNd}{8192} \quad \text{周期}
$$

📌 **关键点**：这里有个容易忽略的细节。两次 MMA 中，$QK^\top$ 是"共享-共享"（SS）操作——两个操作数都从共享内存读；$PV$ 是"张量-共享"（TS）操作——$P$ 已经在张量内存里（前一次 MMA 写进去的），只需从共享内存读 $V$。TS 操作天然少读一个操作数，这也是 TMEM 帮上忙的地方之一。

指数运算单元要对 $M \times N$ 个分数做指数，在 16 次/周期下：

$$
T_{\text{exp}} = \frac{MN}{16} \quad \text{周期}
$$

把两种典型 tile 代进去：

| 📊 资源 | $128^3$ | $256\times128^2$ |
|---------|---------|------------------|
| MMA 计算 | 1024 | 2048 |
| 共享内存 | 768 | 1536 |
| 指数运算单元 | 1024 | 2048 |

💡 **结论**：正向传播里 **MMA 计算和指数运算单元并列为主要瓶颈**。这就解释了 FA4 为什么要专门优化指数函数——它跟矩阵乘一样贵。设计策略由此而来：用大 tile 最大化 MMA 与 Softmax 的重叠、想办法提升指数吞吐、砍掉不必要的非矩阵乘运算。

值得注意的是右列：把 $M$ 从 128 加到 256，MMA 和指数都翻倍（2048），但共享内存只到 1536——因为更大的 tile 摊薄了操作数的重复读取。这是 FA4 偏爱大 tile 的量化依据。

### 4.2 反向传播：共享内存才是大头

反向每次迭代要做五次 MMA（因为需要重算 S，再算 dV、dQ、dK、dP）：

$$
T_{\text{MMA}} = \frac{10MNd}{8192} \quad \text{周期}
$$

但反向还要把中间梯度 dS 写回共享内存、把 dQ 以 FP32 写出再读回做归约。把这些都算上，在 $M=N=d=128$ 时：

| 📊 资源 | 周期数（1-CTA, M=128） |
|---------|------------------------|
| MMA 计算 | 2560 |
| 共享内存（总计） | 3328 |
| 指数运算单元 | 1024 |

📌 **关键点**：反向传播里**共享内存流量才是瓶颈**，比 MMA 还高约 30%。这就是 FA4 反向引入 2-CTA 模式（砍 B 操作数的共享内存）的直接动机。

---

## 5. 正向传播流水线重构

<img src="/images/flashattentionv4-1.png" alt="" style="max-width: 100%; display: block; margin: 0 auto;" />

### 5.1 沿用 Ping-Pong，但解耦校正

FA4 正向沿用了 V3 的"乒乓"（ping-pong）思路：每个线程块负责两个输出 tile，当一个 tile 在跑 Tensor Core 时，另一个 tile 算 Softmax，让两个本应串行的阶段交替占满硬件。

但 Blackwell 的 TMEM 带来一个 V3 做不到的优化。V3 里累加器在寄存器，输出重缩放必须在关键路径上做；FA4 里 P 通过 TMEM 而非寄存器传递，于是可以把**输出重缩放解耦到一个独立的"校正"（correction）warpgroup**，让它彻底离开关键路径。

整体的 warp 角色划分如下：

{% mermaid graph LR %}
    subgraph Producer
        TMA["TMA + Tensor Core<br/>warpgroup"]
    end
    subgraph Compute
        SM1["Softmax<br/>warpgroup 1"]
        SM2["Softmax<br/>warpgroup 2"]
        COR["校正 warpgroup<br/>（输出重缩放）"]
    end
    TMA --> SM1
    TMA --> SM2
    SM1 --> COR
    SM2 --> COR
{% endmermaid %}

### 5.2 一个线程处理一整行

Blackwell 上单个累加器 tile 是 128×128（Hopper 是 64×128），更大了。FA4 选择让两个各含 128 线程的 warpgroup，**每个线程负责完整的一行**。好处是：算行最大值时不需要跨 warp 的 shuffle 归约，每个线程也不用维护多个统计寄存器。

每个 Softmax warpgroup 的处理流程是：

📥 加载整行到寄存器 → 求行最大值 → 减最大值、重缩放、指数化、转精度 → 📤 算行累加和

两个 Softmax warpgroup 之间显式同步，确保它们的指数计算关键段不重叠——这一点和 V3 一致。

### 5.3 TMEM 分区与寄存器压力

在头维度 128 时，TMEM 一半用来放两个 tile 的输出，剩下一半放 S 和 P。FA4 选择"两份 S 与 P 重叠存放"的方案，因为这样能立刻启动软件流水线、同时算两个 S tile，还给传给校正 warpgroup 的统计量留出空间。

为什么"另一半"够放？因为 S/P 是 BF16，而输出累加器是 FP32。同样的 TMEM 容量，一半放 FP32 输出，另一半的字节数足够塞下两份 S（BF16），或者四份 P。FA4 在"一份 S + 两份 P"和"两份 S 与 P 重叠"两种方案里选了后者：

{% mermaid graph LR %}
    subgraph TMEM["TMEM 256KB / SM"]
        OUT["输出累加器<br/>tile A + tile B（FP32）"]
        SP["S 与 P 重叠存放<br/>（BF16，含统计量）"]
    end
    OUT -.- SP
{% endmermaid %}

⚠️ **注意**：更大的 tile + "一线程一行"的分配带来一个副作用——必须把一整行 128 个元素留在寄存器里。为防止寄存器溢出，FA4 对存储 P 采用**分阶段策略**：先存前 3/4（并触发对应 MMA），最后 1/4 单独存。BF16 输入时大约需要 128 个寄存器存输入、64 个存输出。

---

## 6. 软件模拟指数函数

这是 FA4 最有"奇技淫巧"味道的一处优化，也是 roofline 分析逼出来的。

### 6.1 为什么指数运算是瓶颈

回顾一下数字：指数运算单元（MUFU）在 B200 上是 16 次/SM/周期，而 Tensor Core 是 8192 次/SM/周期，差了 **512 倍**。Softmax 里有大量指数运算，于是它直接成了和矩阵乘平起平坐的瓶颈（见 §4.1 的 roofline 表）。

🔑 **核心思路**：既然 MUFU 这条"专用通道"太窄，那就借用旁边闲着的浮点 FMA 单元，**用软件多项式去算指数**，让两条通道并行干活，从而提升总吞吐。

### 6.2 范围缩减 + 多项式近似

目标是算 $2^x$（softmax 内部用 2 为底更方便）。经典做法是把它拆成整数部分和小数部分：

$$
2^x = 2^{\lfloor x \rfloor} \cdot 2^{(x - \lfloor x \rfloor)}
$$

其中 $\lfloor x \rfloor$ 是整数部分，$x - \lfloor x \rfloor \in [0, 1)$ 是小数部分。

- **整数部分** $2^{\lfloor x \rfloor}$：直接利用 IEEE 754 浮点数的指数字段——因为指数位本来就表示 2 的幂，算它只相当于对指数位做移位加法，整数 ALU 就能搞定。
- **小数部分** $2^{x_{\text{frac}}}$：用一个低阶多项式近似：

$$
2^{x_{\text{frac}}} \approx \sum_{i=0}^{n} p_i \cdot x_{\text{frac}}^i
$$

其中 $p_0 = 1.0$，其余系数用 Sollya 工具在 $[0,1)$ 上最小化相对误差求得，求值时用霍纳法（Horner's method）配合 FMA 指令做到高吞吐。

完整流程（用向下舍入巧妙提取整数部分）：

1. 把 $x$ 钳到不小于 $-127$，避免下溢；
2. 给 $x$ 加上 $2^{23} + 2^{22}$ 逼小数位进尾数，再以向下舍入减回，得到 $\lfloor x \rfloor$；
3. 算小数部分 $x_{\text{frac}} = x - \lfloor x \rfloor$；
4. 多项式求出 $2^{x_{\text{frac}}}$；
5. 把 $\lfloor x \rfloor$ 移进指数字段、加上小数部分的尾数，合并出结果。

💡 **直观理解**：这个技巧的精髓是"分工"。整数部分 $2^{\lfloor x \rfloor}$ 在二进制浮点数里本来就是免费的（改改指数位即可），真正需要算的只有 $[0,1)$ 这一小段上的 $2^{x_{\text{frac}}}$；而一个低阶多项式在这么窄的区间上拟合得相当好，于是把一次昂贵的超越函数化简成了几条廉价的 FMA。

### 6.3 只模拟一部分

⚠️ **注意**：软件模拟不是免费的——它要更多寄存器存中间值和系数，吞吐高但延迟也比 MUFU 长。如果全量替换，反而会因寄存器溢出抵消收益。

所以 FA4 的做法是**部分模拟**：只对每行 10%–25% 的条目走软件多项式，其余仍用硬件 `MUFU.EX2`。具体比例按 tile 配置下 MMA 与指数吞吐的比值实验调优。这是一种"两条腿走路"的负载均衡。

### 6.4 精度够用吗

够。关键洞察是：**Softmax 输出最终要被舍入成 BF16**，而 BF16 本身的量化误差（约 $3.9\times10^{-3}$）远大于多项式近似误差。

| 📊 方法 | FP32 最大相对误差 | BF16 最大相对误差 |
|---------|-------------------|-------------------|
| 硬件 MUFU.EX2 | $1.41\times10^{-7}$ | $3.89\times10^{-3}$ |
| 三阶多项式 | $8.77\times10^{-5}$ | $3.90\times10^{-3}$ |
| 五阶多项式 | $1.44\times10^{-7}$ | $3.89\times10^{-3}$ |

💡 **结论**：FP32 层面三阶多项式比硬件差约 600 倍，但**舍成 BF16 后几乎不可区分**——三阶多项式在 99% 的输入上与硬件相差不超过 1 个 BF16 ULP。对 BF16 注意力来说三阶就够了。

---

## 7. 跳过在线 Softmax 重缩放

### 7.1 在线 Softmax 回顾

FlashAttention 按块计算 Softmax，每处理一块就更新运行统计量。设处理第 $j$ 块时分数为 $S_j$：

$$
m_j = \max(m_{j-1}, \text{rowmax}(S_j))
$$

$$
\ell_j = e^{m_{j-1} - m_j} \cdot \ell_{j-1} + \text{rowsum}(e^{S_j - m_j})
$$

每当遇到更大的最大值 $m_j$，就要用因子 $e^{m_{j-1} - m_j}$ 把之前累积的输出 $O_{j-1}$ 重新缩放一遍——这一步是一次向量乘法，属于非矩阵乘运算。

### 7.2 两个观察，省掉大部分重缩放

FA4 做了两个简单但有效的观察：

- ✅ 只有 $m_j > m_{j-1}$（真的发现更大值）时才需要重缩放。
- ✅ 重缩放可以有"宽容度"：只在 $m_j - m_{j-1} > \tau$ 时才做，$\tau$ 典型取 $\log_2(256) = 8.0$。只要持续追踪累计缩放量，最后仍能还原正确的归一化分母。

于是算法改为：

```
        ⎧ e^(mⱼ₋₁−mⱼ) Oⱼ₋₁ + e^(Sⱼ−mⱼ) Vⱼ    若 mⱼ − mⱼ₋₁ > τ
Oⱼ =    ⎨                                                       
        ⎩ Oⱼ₋₁ + e^(Sⱼ−mⱼ₋₁) Vⱼ              否则
```

当差值 $\le \tau$ 时直接跳过 $m$ 的更新、继续用 $m_{j-1}$。正确性靠的是**最终统一归一化**：

$$
\text{输出} = \frac{1}{\ell_{\text{final}}} \cdot O_{\text{final}}
$$

📌 **关键点**：跳过的中间重缩放引入的微小偏差，会被最后这一步用真实的 $m_{\text{final}}$ 和 $\ell_{\text{final}}$ 纠正回来。实践中为避免 warp 发散，只要 warp 内任一线程需要重缩放，就对整个 warp 一起做。

---

## 8. 反向传播与 2-CTA MMA

<img src="/images/flashattentionv4-2.png" alt="" style="max-width: 100%; display: block; margin: 0 auto;" />

反向的瓶颈在共享内存（§4.2），FA4 的解法是把 Blackwell 的 2-CTA 模式用足。

### 8.1 流水线：用上一轮的 MMA 藏住这一轮的 Softmax

V3 反向因为累加器全在寄存器，五次 MMA 被迫几乎串行：S → dP → dV → dQ → dK。FA4 借助 TMEM 把其中两个操作数常驻张量内存，从而能做更灵活的调度。

核心技巧：**用前一次迭代 dQ、dK 的 MMA，来掩盖当前迭代的 Softmax 计算延迟**。这需要在 load、MMA、计算、归约之间精细管理共享内存和 TMEM。一个硬约束是 TMEM 装不下五个累加器 tile（最多四个 128×128，而 dV 和 dK 都要跨迭代累加、不能复用空间），所以 FA4 让 S 和 P 共享一块 TMEM，dP、dS、dQ 共享另一块。

### 8.2 2-CTA：把 B 操作数砍一半

反向五个 GEMM 里仍有八个 BF16 操作数要从共享内存喂给 Tensor Core，多耗约 30% 周期。FA4 用 $M=256, N=K=128$ 的 2-CTA tile：两个 CTA 合当一个大 tile，每个只暂存 B 的一半、只保留自己那片累加器。

<img src="/images/flashattentionv4-3.png" alt="" style="max-width: 100%; display: block; margin: 0 auto;" />

但有个麻烦：dQ 的归约轴恰好是 N，而 2-CTA 只分区输出 tile、不分区归约轴。FA4 的解法是用**分布式共享内存（DSMEM）**在配对的两个 CTA 间交换一半 dS，把 dS 重新打包成沿非归约轴分区——每个 CTA 拿到自己的 $M/2$ 行、完整的 $2N$ 归约范围。

{% mermaid graph TD %}
    A["CTA 0: 持有 dS 的一半"] -->|DSMEM 交换| B["重新打包 dS"]
    C["CTA 1: 持有 dS 的另一半"] -->|DSMEM 交换| B
    B --> D["每个 CTA: (M/2 × 2N) 操作数<br/>双倍归约的 dQ MMA"]
{% endmermaid %}

效果对比：

| 📊 资源 | 1-CTA (M=128) | 2-CTA (M=256) |
|---------|---------------|---------------|
| MMA 计算 | 2560 | 2560 |
| 共享内存（MMA 操作数） | 2048 | 1536 |
| 共享内存（dS 写入） | 256 | 256 |
| 共享内存（dS DSMEM） | 0 | 384 |
| 共享内存（dQ 写+读） | 1024 | 512 |
| 共享内存总计 | 3328 | 2688 |

💡 **结论**：2-CTA 把共享内存总耗时从比 MMA 高 30% 压到只高约 5%，瓶颈基本被填平。

### 8.3 附带福利：原子加减半

dQ 的累加原本要在内层循环每次迭代做一次全局原子归约，既慢又引入非确定性。2-CTA 分解后**每个 CTA 只写 dQ tile 的一半**，全局原子归约次数也直接减半。

### 8.4 确定性模式

跨 CTA 的全局归约会让 dQ（GQA 下还有 dK/dV）的梯度变得不确定，对需要可复现的训练（如强化学习）是个问题。FA4 提供确定性模式：用**信号量锁**序列化全局归约，每个 CTA 按预定顺序拿锁、归约、递增计数器释放锁。

⚠️ **注意**：锁会带来停顿。FA4 通过在头/批次维度 swizzle 重排、对因果掩码采用"最短处理时间优先（SPT）"调度（降序启动 KV 块、升序遍历查询块、降序归约 dQ）来把停顿降到最低，确定性模式最高可达 1-CTA 非确定性速度的 75%。

---

## 9. 调度策略：LPT 与确定性模式

因果掩码和可变序列长度（varlen）会让各 SM 拿到的工作量天差地别——这是个经典的负载均衡问题。FA4 借用了并行调度里"最长处理时间优先"（LPT, Longest Processing Time）的思想来最小化 makespan（总完工时间）。

💡 **为什么是 LPT**：把每个工作 tile 想象成一项任务、每个 SM 想象成一个工人。如果先派出耗时最长的任务，短任务会自然填补到结尾的空隙里，整体收尾更整齐；反过来先做短任务，最后剩下的长任务会让一部分工人干等。这是并行调度里一个有理论保证的经典启发式，且**对所有 GPU 架构通用**——它在 Hopper 上的 FA3 也被验证有效。

### 9.1 因果掩码的 LPT

标准网格按 (mblocks, heads, batches) 从左到右算，但因果掩码下对角线以上被掩掉，于是 SM 会按"从短到长"的低效顺序处理。朴素 LPT 又会破坏 L2 缓存局部性。

✅ **FA4 的折中**：始终把 batch 放最外层、对 head 维度做 swizzle 重排——把 head 切成不超出 L2 容量的分段，调度器按"段内 head → 逆序 mblocks → 段 → batch"遍历。实测在 H200 上 MHA 提升 4%–8%、MQA 提升 7%–14%。

### 9.2 可变序列长度的 LPT

varlen 下不同 batch 的序列长度差异巨大（比如短 prefill 紧跟长 decode）。FA4 启动一个预处理内核，按每个 batch 最大的单 tile 执行时间排序，写出一个"虚拟索引 → 实际 batch 索引"的映射，注意力内核读回后按排序顺序遍历。这份元数据可缓存，排序本身不带性能损失。

---

## 10. CuTe-DSL：用 Python 写内核

FA4 一个工程上的大变化是：**完全用嵌入 Python 的 CuTe-DSL 编写，没有任何 CUDA C++**。编译器把 Python 降级为 PTX，再交给 ptxas 生成 SASS。

为什么这是大事？编译时间。过去 FA2/FA3 用 C++ 模板元编程，针对不同注意力变体常要预编译成百上千个内核，编译慢得离谱。

| 📊 方法 | 正向编译 | 反向编译 |
|---------|---------|---------|
| FlashAttention-3 (C++ 模板) | 55s | 45s |
| FlashAttention-4 (CuTe-DSL) | 2.5s | 1.4s |
| 加速比 | 22× | 32× |

💡 **价值**：CuTe-DSL 与 CUTLASS C++ 同构，保留底层完整表达力（还提供直接写 PTX 的"逃生通道"），同时把编译提速 20–30 倍。这把内核开发的门槛从"精通 C++ 模板元编程"降到了"会 Python 的工程师也能上手"，已有人在 FA4 之上实现了 FlexAttention 和块稀疏注意力变体。

---

## 11. 性能评估

### 11.1 对比基线与测试设置

FA4 在 B200 GPU 上与一系列开源/闭源方案做了对比：PyTorch 原生实现、FlashAttention-2、Triton（用了 B200 专用指令）、Gluon（比 Triton 更底层的 GPU 语言）以及 cuDNN（NVIDIA 官方厂商库）。

测试覆盖 BF16 输入、是否因果掩码、头维度 64/128/(192,128)（最后一组对应 DeepSeek V3 的配置），序列长度从 1k 到 32k，批次大小调整到总 token 数恒为 32k。FLOPs 按 $4 \times N^2 \times d \times \text{头数}$ 计算（因果掩码再除以 2），反向 FLOPs 取前向的 2.5 倍。

### 11.2 核心结果

FA4 在 B200 上的测试结果（BF16）：

- 相比 cuDNN 9.13 最高加速 **1.3×**（1.1–1.3× 区间）
- 相比 Triton 最高加速 **2.7×**（2.1–2.7× 区间）
- 峰值达 **1613 TFLOPs/s**，约为 B200 理论峰值的 **71\%**

✅ **正向传播**：对中长序列（4k 及以上），FA4 在各头维度和因果设置下持续领先。因果场景收益更大，归功于 LPT 调度。

✅ **反向传播**：长序列和因果场景下持续加速，验证了 2-CTA 反向方案的有效性。确定性反向最高可达非确定性的 75% 速度。

⚠️ **注意**：论文也坦言，自 FA4 发布后，新版 cuDNN（9.19）已经吸收了 FA4 的多项技术，性能与 FA4 接近——这恰恰说明 FA4 的方法被验证为正确的方向。

---

## 📝 总结

FlashAttention-4 针对非对称硬件扩展问题进行设计：张量核心速度极快，使主要瓶颈转移到共享内存流量和指数运算吞吐量上，这促使我们进行算法与内核的协同设计来缓解这些限制。我们围绕全异步 MMA 重新设计了流水线，使 softmax 能够与更大 tile 的矩阵乘法重叠，并引入软件模拟的指数运算和条件 softmax 重缩放来减少非矩阵乘法运算。我们利用张量内存和 2-CTA MMA 模式来降低共享内存流量。此外，2-CTA 模式还使我们能够重构全局原子累加方式，将全局原子加操作的数量减半。FlashAttention-4 完全以嵌入 Python 的 CuTe-DSL 实现，相比基于 C++ 模板的内核，编译速度提升了 20–30 倍，同时保留了底层控制能力。尽管该方案针对 Blackwell GPU 进行了优化，但其中部分算法思路也可以扩展到其他加速器上，因为算力持续超越非矩阵乘法单元发展速度这一趋势仍将延续。

FlashAttention V4 是一次典型的"硬件逼出算法"的协同设计。Blackwell 的非对称扩展把瓶颈从矩阵乘推到了 Softmax 和共享内存上，FA4 的每一招都精准对应：

| 📊 瓶颈 | FA4 的对策 |
|---------|-----------|
| 指数运算单元慢（512× 差距） | 软件多项式模拟 + 部分模拟分流 |
| Softmax 重缩放是非矩阵乘开销 | 条件重缩放（带阈值 $\tau$），最终统一归一化 |
| 反向共享内存带宽瓶颈 | 2-CTA MMA 砍半 B 操作数 + TMEM 常驻累加器 |
| 全局原子加慢且不确定 | 2-CTA 分解减半原子加 + 信号量锁确定性模式 |
| 负载不均衡 | LPT 调度（因果 swizzle / varlen 预排序） |
| C++ 模板编译慢 | CuTe-DSL 嵌入 Python，编译提速 20–30× |

更大的启示是：算力超越非矩阵乘单元的趋势短期内不会停，FA4 里"识别非 MMA 瓶颈并把它藏进 MMA 影子"的方法论，会延续到未来的加速器上。

## 🎯 自我检验清单

- 能写出注意力前向的三步（$S = \alpha QK^\top$、$P = \text{softmax}(S)$、$O = PV$）并说明反向为何需要 5 次矩阵乘
- 能解释"非对称扩展"是什么，以及它如何把注意力的瓶颈从矩阵乘转移到 Softmax 与共享内存
- 能说出 Blackwell 相比 Hopper 在 TMEM、MMA tile 尺寸、2-CTA 模式上的三个关键变化
- 能复述正向与反向 roofline 分析的结论（正向 MMA/指数并列瓶颈，反向共享内存瓶颈）
- 能解释软件模拟指数函数的原理（范围缩减 + 多项式），以及为什么只模拟 10%–25% 的条目
- 能说明条件 Softmax 重缩放为何在跳过中间步骤后仍能保证正确性
- 能描述 2-CTA MMA 如何同时降低共享内存流量和全局原子加次数
- 能解释 LPT 调度为何对因果掩码和 varlen 场景有效
- 能说出 CuTe-DSL 相比 C++ 模板在编译时间上的量级优势及其工程意义

## 📚 参考资料

- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-1: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA CUDA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS - GitHub](https://github.com/NVIDIA/cutlass)
- [Tri Dao - FlashAttention Series](https://tridao.me/)
