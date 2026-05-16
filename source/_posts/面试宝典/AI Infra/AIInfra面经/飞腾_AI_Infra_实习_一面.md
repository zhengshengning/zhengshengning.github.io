---
title: '飞腾 AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: Transformer架构与多头注意力机制细节（Decoder-only、QKV生成、位置编码嵌入时机）？

**Decoder-only 架构**（GPT/LLaMA 系列）：

每层由两个子模块组成：Masked Multi-Head Self-Attention + Feed-Forward Network（FFN），两个子模块都有残差连接和 LayerNorm/RMSNorm。

- **Causal Mask**：注意力矩阵的上三角被置为 -inf，确保每个 token 只能看到自己和之前的 token（自回归性质）。这是与 Encoder 架构的核心区别——Encoder 的注意力是双向的。
- **为什么 Decoder-only 成为主流**：(1) 统一了预训练和生成的计算模式（都是下一个 token 预测）；(2) KV Cache 机制只需追加不需重算；(3) 无需设计 cross-attention，架构更简洁高效。

**QKV 生成过程**：
- 输入 x [batch, seq_len, hidden_dim] 经过三个独立的线性投影（或一个大线性层 split 三份）：
  - Q = x @ W_Q [hidden_dim, n_heads * head_dim]
  - K = x @ W_K [hidden_dim, n_kv_heads * head_dim]（GQA 时 n_kv_heads < n_heads）
  - V = x @ W_V [hidden_dim, n_kv_heads * head_dim]
- 然后 reshape 为 [batch, n_heads, seq_len, head_dim] 进行多头注意力计算
- 注意力输出 concat 后经 W_O 投影回 hidden_dim

**位置编码嵌入时机**：
- **RoPE（Rotary Position Embedding）**：在 Q、K 线性投影**之后**、attention 点积**之前**施加旋转变换
- 具体操作：将 Q、K 的每个 head_dim 维度按相邻两个一组，施加与位置相关的旋转矩阵
- RoPE 的优势：(1) 相对位置信息自然编码在 Q*K^T 中；(2) 可外推到训练时未见过的长度（配合 NTK-aware scaling）；(3) 不增加参数量
- **对比**：绝对位置编码（GPT-2）在 embedding 层加入（投影之前）；ALiBi 在 attention score 上直接加距离偏置

### Q: RMSNorm公式、计算访存特性及优化方法？

**RMSNorm（Root Mean Square Layer Normalization）**：

是 LayerNorm 的简化版本，移除了均值中心化步骤，只保留均方根归一化：

**公式**：y_i = x_i / RMS(x) * gamma_i，其中 RMS(x) = sqrt(1/n * sum(x_i^2) + eps)

**与 LayerNorm 的对比**：
- LayerNorm：y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
- RMSNorm 省略了 mean 计算和 beta 参数，计算量减少约 15-20%
- 实验表明精度几乎无损（LLaMA/Qwen/Mistral 全部使用 RMSNorm）

**计算访存特性**：
- 对 hidden_dim（如 4096）做 reduction 操作（计算平方和）
- 算术强度极低：每个元素约 3-4 次浮点操作（平方+累加+除法+乘gamma），但需要读写整个 hidden 向量
- 典型的 **memory-bound** 操作：计算量 O(hidden_dim)，访存量 O(hidden_dim)，算术强度 < 1 FLOP/byte

**优化方法**：

1. **Kernel Fusion**：将 RMSNorm 与相邻算子融合（如 Residual Add + RMSNorm + Linear 的第一次读取），避免中间 tensor 的全局内存读写。这是最有效的优化（减少 2-3 次全局内存访问）。

2. **向量化加载**：使用 float4/int4 128-bit 加载指令，一次加载 4 个 float32，提升内存带宽利用率。

3. **Warp-level Reduction**：利用 `__shfl_down_sync` 在 warp 内做平方和归约，避免 shared memory 的读写和 `__syncthreads` 同步开销。

4. **指令替换**：用 `rsqrtf()` 直接计算 1/sqrt(x)（一条特殊函数指令），替代分别计算 sqrt 再做除法（两条指令）。精度损失可通过一轮 Newton-Raphson 迭代补偿。

5. **Double Buffer / Prefetch**：在计算当前行的 RMSNorm 时，预取下一行的数据到寄存器，隐藏访存延迟。

6. **与 Residual 融合**：Pre-Norm 架构中 `x = x + sublayer(RMSNorm(x))` 的 residual add 和 RMSNorm 可融合为单个 kernel。

### Q: Softmax数值稳定性处理与Online实现？

**数值稳定性问题与解决**：

标准 softmax：P_i = exp(x_i) / sum(exp(x_j))。当 x_i > 88（FP32）时 exp 溢出为 inf。

**Safe Softmax**：减去最大值保证 exp 输入 <= 0：
```
m = max(x_1, ..., x_n)
P_i = exp(x_i - m) / sum(exp(x_j - m))
```
数学等价性：分子分母同乘 exp(-m) 抵消。减去 max 后最大的 exp 值为 exp(0)=1，不会溢出。

**传统实现需要 3 遍遍历**：
1. 第一遍：求 max（reduction）
2. 第二遍：计算 exp(x_i - max) 并求 sum（reduction）
3. 第三遍：归一化 exp(x_i - max) / sum

**Online Softmax（2 遍实现）**：

核心思想：在一遍扫描中同时维护 running max 和 running sum，通过修正因子更新：

```
初始化：m_0 = -inf, d_0 = 0
扫描第 j 个元素：
  m_j = max(m_{j-1}, x_j)
  d_j = d_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
```

当发现新的 max 时，历史累积的 sum 乘以修正因子 exp(old_max - new_max) 进行缩放。最终 softmax(x_i) = exp(x_i - m_n) / d_n。

**与 FlashAttention 的关系**：

Online Softmax 是 FlashAttention 的数学基础。FlashAttention 将 attention 矩阵分块计算：
1. 每块在 SRAM 中计算局部 softmax（local max + local sum）
2. 块间用 Online Softmax 的修正因子合并：当处理新块时，如果发现更大的 max，则回头对之前块的输出乘以修正因子 exp(old_max - new_max)
3. 最终得到精确的 softmax 结果（非近似），同时避免了 N*N 的 attention 矩阵物化到 HBM

**性能影响**：3-pass -> 2-pass 减少一次全局内存遍历，对 memory-bound 的 softmax 操作直接节省 33% 的内存带宽消耗。

### Q: 矩阵乘与反量化融合算子的内存优化策略？

**问题背景**：W4A16/W8A16 量化方案中，权重以 INT4/INT8 格式存储，推理时需要反量化为 FP16 再做 GEMM。如果反量化和 GEMM 是两个独立 kernel：
- 额外 kernel launch 开销（~5us）
- 需要额外的 FP16 权重中间缓冲区（7B 模型 = 14GB！）
- 两次全局内存读写（写 FP16 权重 + 读 FP16 权重）

**融合策略**：将反量化操作嵌入 GEMM kernel 内部，在数据从 Global Memory 搬运到 Shared Memory 的过程中完成反量化：

**实现细节**：
1. GEMM kernel 的 K-dimension 循环中，每次加载一个权重 tile
2. 加载时从 Global Memory 读取 INT4/INT8 格式的压缩数据
3. 在寄存器或 Shared Memory 写入前，执行反量化：`fp16_val = (int_val - zero_point) * scale`
4. Scale/zero_point 按 group（如每 128 个元素一组）存储，加载开销极小
5. 反量化后的 FP16 值直接用于后续 Tensor Core 计算

**性能收益**：
- 省去独立反量化 kernel 的 launch 开销
- 节省一整份 FP16 权重的显存（7B 模型省 14GB -> 只需 3.5GB INT4 权重）
- 减少一次完整权重矩阵的 Global Memory 读写（带宽收益巨大）
- 实际加速：在 memory-bound 的 decode 阶段，融合后速度接近 INT4 的理论带宽节省（~4x 带宽减少）

**工程挑战**：
- INT4 解包（两个 INT4 打包在一个 byte 中）需要位操作，增加指令数
- Group quantization 的 scale/zp 加载和广播需要额外寄存器
- 需要保证 tile 边界与 quantization group 边界对齐

**典型实现**：GPTQ/AWQ 的推理 kernel（如 exllama/AutoGPTQ）、TensorRT-LLM 的 W4A16 kernel、Marlin kernel（4-bit GEMM，达到接近 FP16 GEMM 4x 加速）。

### Q: 稀疏矩阵SpMV的负载均衡与带宽优化？

**SpMV（Sparse Matrix-Vector Multiplication）**：y = A*x，其中 A 是稀疏矩阵。核心挑战是非零元素分布不均导致的负载不均衡和不规则访存。

**负载均衡问题**：

在 CSR（Compressed Sparse Row）格式下，自然的并行方式是每行分配一个线程/warp。但稀疏矩阵行长度差异巨大（power-law 分布常见），某些行有 10000 个非零元素而其他行可能只有 1-2 个。

**解决方案**：

1. **CSR-Segmented（按非零元素均分）**：
   - 将所有非零元素平均分配到各线程（每线程处理约 nnz/num_threads 个非零元素）
   - 一个线程可能处理多行（短行）或只处理一行的一部分（长行）
   - 跨行边界需要原子操作或 segmented reduction
   - 代表：CSR5 格式

2. **Merge-based SpMV（Merrill & Garland）**：
   - 将行指针数组和非零元素数组视为两条有序序列
   - 用 merge path 将两者统一切分为等大小的段
   - 每个线程处理一个 merge path 段，自然实现负载均衡
   - 优点：不依赖矩阵结构，通用性最好

3. **Warp-level 策略**：
   - 短行（<32 非零元素）：一个线程处理一行
   - 中行（32-1024）：一个 warp 处理一行
   - 长行（>1024）：一个 block 或多个 warp 处理一行
   - 按行长度分桶后分别处理

**带宽优化**：

1. **ELL（Ellpack）格式**：将所有行 padding 到相同长度，存储为规则二维数组。线程可以 coalesced 访问，但 padding 浪费严重。
2. **SELL-C-sigma**：按行长度排序分组，每组 C 行用 ELL 格式，组间长度不同。兼顾访存连续性和存储效率。
3. **向量化加载**：使用 float4/int4 128-bit 加载指令读取非零元素和列索引。
4. **x 向量缓存**：将 x 向量放入 texture cache 或 shared memory（x 被多行重复访问）。
5. **CSR5**：固定大小的 tile 存储非零元素，tile 内按列优先排列保证 coalesced 访问。

**性能参考**：稀疏矩阵的实际带宽利用率通常只有理论峰值的 30-60%（因为不规则访存和索引开销），远低于密集矩阵运算（可达 80-90%）。

### Q: IEEE浮点标准中FP16/32/64的位分配？

IEEE 754 浮点标准定义了不同精度的浮点数格式，位分配决定了数值范围和精度：

| 格式 | 符号位 | 指数位 | 尾数位 | 数值范围 | 精度（有效十进制位） | 最小正规数 |
|------|--------|--------|--------|----------|------|------------|
| FP16 | 1 | 5 | 10 | ±6.5×10^4 | ~3.3 位 | 6.1×10^-5 |
| BF16 | 1 | 8 | 7 | ±3.4×10^38 | ~2.4 位 | 1.2×10^-38 |
| FP32 | 1 | 8 | 23 | ±3.4×10^38 | ~7.2 位 | 1.2×10^-38 |
| FP64 | 1 | 11 | 52 | ±1.8×10^308 | ~15.9 位 | 2.2×10^-308 |

**关键解释**：
- **指数位决定范围**：指数位越多，能表示的数值范围越大（从极小到极大）
- **尾数位决定精度**：尾数位越多，相邻两个可表示数之间的间隔越小
- **BF16 的设计哲学**：保持与 FP32 相同的 8 位指数（相同范围，不会溢出），牺牲精度（7 vs 23 位尾数）。这样 FP32 -> BF16 只需截断尾数，无需重新缩放。

**在深度学习中的应用**：
- **FP32**：传统训练精度，优化器状态（Adam 的 m/v）必须用 FP32
- **FP16**：Tensor Core 混合精度训练（输入 FP16，累加 FP32），推理部署标准精度
- **BF16**：训练首选（范围同 FP32 不会 overflow，简化 loss scaling）。A100/H100/TPU 原生支持
- **FP8**（E4M3/E5M2）：H100 引入，E4M3 用于前向（精度优先），E5M2 用于反向（范围优先）
- **FP64**：科学计算、金融计算，深度学习中几乎不用

**精度对训练的影响**：FP16 训练需要 loss scaling（防止梯度下溢到 0）；BF16 通常无需 loss scaling（范围足够大）；FP8 需要更精细的 per-tensor scaling。

### Q: 快速排序步骤、堆性质、拓扑排序适用场景？

**快速排序（QuickSort）**：

**步骤**：
1. 选择 pivot（基准元素）：首元素/末元素/随机/三数取中
2. Partition：将数组分为 [<=pivot | pivot | >=pivot] 三部分
3. 递归对左右子数组执行同样操作

**复杂度分析**：
- 平均 O(N*logN)：每次 partition 近似二分
- 最坏 O(N^2)：已排序数组 + 固定选首元素作 pivot（每次只分出一个元素）
- 优化：随机化 pivot（期望 O(NlogN)）、三路划分（处理大量重复元素）、小数组切换插入排序（N<16）
- 空间：O(logN)（递归栈深度）

**实际性能**：尽管最坏情况不如归并排序，快排因为 cache 友好（in-place、顺序访问）和低常数因子，通常是最快的通用排序。STL 的 `std::sort` 使用 IntroSort（快排 + 堆排 + 插入排序的混合）。

---

**堆（Heap）的性质**：

- **结构性质**：完全二叉树（除最后一层外全满，最后一层从左到右填充）
- **堆序性质**：大根堆中父节点 >= 子节点；小根堆中父节点 <= 子节点
- **数组表示**：节点 i 的子节点在 2i+1 和 2i+2，父节点在 (i-1)/2

**操作复杂度**：
- 取极值 O(1)（堆顶）
- 插入 O(logN)（放末尾后 sift-up）
- 删除堆顶 O(logN)（末尾换到顶后 sift-down）
- 建堆 O(N)（自底向上 heapify，非 O(NlogN)）

**典型应用**：优先队列、Top-K 问题（维护 K 大小的堆）、堆排序、Dijkstra 最短路径。

---

**拓扑排序（Topological Sort）**：

**适用场景**：有向无环图（DAG）中的节点线性排序，使得对每条边 (u,v)，u 在排序中出现在 v 之前。

**典型应用**：
- 编译依赖顺序（文件 A include 文件 B，则 B 先编译）
- 任务调度（任务 A 依赖任务 B 完成）
- 课程先修关系
- 计算图的执行顺序（深度学习框架中 Op 的调度）

**算法**：
1. **Kahn's 算法（BFS）**：维护入度为 0 的队列，每次取出一个节点加入结果，并将其邻居入度减 1
2. **DFS 后序反转**：DFS 完成时间的逆序即为拓扑排序

**检测环**：如果图有环则无法完成拓扑排序（Kahn 算法中最终结果节点数 < 总节点数说明有环）。

### Q: 进程和线程的区别？Cache层级与替换策略？

**进程 vs 线程**：

| 特性 | 进程（Process） | 线程（Thread） |
|------|----------------|----------------|
| 本质 | 资源分配的基本单位 | CPU 调度的基本单位 |
| 地址空间 | 独立（互相隔离） | 共享进程的地址空间 |
| 创建开销 | 大（需分配页表、文件描述符表等） | 小（共享进程资源，只需分配栈和寄存器集） |
| 上下文切换 | 慢（需切换页表、flush TLB） | 快（只切换寄存器和栈指针） |
| 通信方式 | IPC（管道/共享内存/socket/信号） | 直接读写共享内存（需同步） |
| 崩溃影响 | 不影响其他进程 | 可能导致整个进程崩溃 |
| 典型应用 | Chrome 多进程隔离、微服务 | Web 服务器的请求处理、并行计算 |

**为什么 GPU 编程多用线程而非进程**：GPU kernel 内的"线程"共享全局内存和共享内存，需要高效的数据交换，这与 OS 线程的共享内存模型一致。GPU 不支持进程级隔离。

---

**Cache 层级架构**：

| 层级 | 典型容量 | 访问延迟 | 特点 |
|------|---------|---------|------|
| L1 I/D-Cache | 32-64KB per core | 1-4 周期（~1ns） | 每核私有，分指令/数据 |
| L2 Cache | 256KB-1MB per core | 10-14 周期（~5ns） | 每核私有或少数核共享 |
| L3 Cache | 4-64MB shared | 30-50 周期（~15ns） | 所有核共享，包容式/非包容式 |
| DRAM | GB 级 | 200-300 周期（~100ns） | — |

**替换策略**：
- **LRU（Least Recently Used）**：淘汰最久未使用的 cache line。实现精确 LRU 需要记录访问顺序（代价高，通常 4-8 way 可精确实现）
- **Pseudo-LRU（伪 LRU）**：用树状结构近似 LRU，硬件开销小，16-way 常用
- **Random**：随机替换，最简单，某些场景（如 streaming access）效果与 LRU 接近
- **RRIP（Re-Reference Interval Prediction）**：Intel 的改进策略，预测 re-reference 间隔而非简单的最近使用时间

**对 AI 计算的影响**：矩阵乘的 tiling 策略本质上就是让 tile 大小适配 L1/L2 容量，最大化 cache hit rate。GEMM 的峰值性能与 tile 匹配 cache 层级的程度直接相关。

### Q: Git分支操作：fetch+checkout vs pull？

**git fetch + checkout**：

- `git fetch origin`：从远程仓库下载所有更新（新分支、新 commit），但只更新远程追踪分支（origin/main 等），**不修改本地分支和工作目录**
- `git checkout branch_name`：切换到指定分支
- **组合使用场景**：先 fetch 看看远程有什么变化，确认安全后再操作。适合需要先 review 远程变更的协作场景

**git pull**：

- 等价于 `git fetch` + `git merge`（默认）或 `git fetch` + `git rebase`（配置 pull.rebase=true）
- 一步完成：拉取远程最新代码并合并到当前分支
- 可能产生 merge commit（如果本地和远程有分叉）

**对比与最佳实践**：

| 操作 | 安全性 | 自动合并 | 适用场景 |
|------|--------|---------|---------|
| fetch + checkout | 高（不自动修改） | 否 | 切换到远程新分支、review 后再合并 |
| fetch + merge | 中 | 是（显式） | 明确知道要合并时 |
| fetch + rebase | 中 | 是（线性历史） | 保持 commit 历史线性 |
| pull | 低（自动合并可能冲突） | 是 | 简单场景快速同步 |

**推荐工作流**：
1. `git fetch origin` 获取最新远程状态
2. `git log HEAD..origin/main` 查看远程新增的 commit
3. 确认无问题后 `git rebase origin/main`（保持线性历史）
4. 如有冲突，逐个解决后 `git rebase --continue`

这样比直接 `git pull` 更可控，避免意外的合并冲突或非预期的 merge commit。
