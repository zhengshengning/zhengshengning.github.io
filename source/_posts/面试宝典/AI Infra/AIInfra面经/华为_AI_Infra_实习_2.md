---
title: '华为 AI Infra 实习 (2)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 部署和推理一个模型分别用到的参数量如何估算？

**模型参数量与存储**：

以 LLaMA-2 7B 为例（hidden=4096, layers=32, heads=32, vocab=32K）：
- FP32 存储：7B * 4 bytes = 28 GB
- FP16/BF16 存储：7B * 2 bytes = 14 GB
- INT8 存储：7B * 1 byte = 7 GB
- INT4 存储：7B * 0.5 bytes = 3.5 GB

**推理显存组成**：

```
总推理显存 = 模型权重 + KV Cache + 激活值 + 运行时开销
```

1. **模型权重**：取决于量化精度（见上）
2. **KV Cache**（通常最大占比）：
   - 公式：2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
   - LLaMA-2 7B, FP16, batch=1, seq=4096：2 * 32 * 32 * 128 * 4096 * 2 = 2 GB
   - LLaMA-2 7B, FP16, batch=32, seq=4096：64 GB（比模型权重还大！）
3. **激活值**：前向计算中的临时 tensor，通常几百 MB（取决于 batch size 和 seq_len）
4. **运行时开销**：CUDA context (~500MB)、框架内存池、碎片等

**部署总显存估算**（实用经验公式）：
- W16 推理：~1.2x 模型参数大小（FP16 bytes）+ KV Cache
- W4 推理：~1.2x 模型参数大小（INT4 bytes）+ KV Cache + scale/zp 存储（~3%额外）
- 安全余量：实际分配 = 估算值 * 1.2（碎片和对齐开销）

**显卡选择参考**：
| 模型 | W4量化 | W16 | 推荐单卡（含 KV Cache） |
|------|--------|-----|------------------------|
| 7B | 3.5GB | 14GB | A10 24GB / RTX 4090 24GB |
| 13B | 6.5GB | 26GB | A100 40GB |
| 70B | 35GB | 140GB | 2-4x A100 80GB |

### Q: Prefill和Decoding的区别？KV Cache大小怎么算？

**Prefill 阶段**（"理解输入"）：
- 处理完整的输入 prompt（如 1024 个 token）一次性通过模型
- 计算类型：大矩阵乘 [batch, seq_len, hidden] @ [hidden, hidden]
- 特征：**Compute-bound**（Tensor Core 利用率高，计算密度大）
- 产出：填充完整的 KV Cache + 生成第一个输出 token
- 延迟指标：Time To First Token（TTFT）

**Decode 阶段**（"逐字生成"）：
- 每步只处理 1 个新 token（当前步的 Q 是 1 行向量）
- 计算类型：GEMV [batch, 1, hidden] @ [hidden, hidden]（矩阵-向量乘）
- 特征：**Memory-bound**（每步需读取全部模型权重和 KV Cache，但只做少量计算）
- 瓶颈：HBM 读取带宽。每生成 1 token 需读 ~14GB（7B FP16 模型权重）
- 延迟指标：Time Per Output Token（TPOT）

**性能差异的根源**：
- Prefill 的算术强度：(2 * seq_len * hidden) / (seq_len + hidden) ≈ hidden（当 seq_len >> 1）
- Decode 的算术强度：(2 * 1 * hidden) / (1 + hidden) ≈ 2（极低！）
- 这就是为什么 Prefill 充分利用算力而 Decode 受带宽限制

**KV Cache 大小计算**：

```
KV Cache = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
```

各参数含义：
- 2：K 和 V 各一份
- n_layers：模型层数（LLaMA-7B=32, 70B=80）
- n_kv_heads：KV 的 head 数（GQA 时 < n_heads。LLaMA-2 70B: n_heads=64, n_kv_heads=8）
- head_dim：每个 head 的维度（通常 128）
- seq_len：已生成的序列总长度（prompt + output so far）
- batch_size：并发请求数
- dtype_bytes：FP16=2, INT8=1

**数值示例**：
| 模型 | Batch | Seq_len | KV Cache 大小 |
|------|-------|---------|--------------|
| LLaMA-2 7B (FP16) | 1 | 2048 | 1 GB |
| LLaMA-2 7B (FP16) | 32 | 2048 | 32 GB |
| LLaMA-2 70B (FP16, GQA-8) | 1 | 4096 | 5 GB |
| LLaMA-2 70B (FP16, GQA-8) | 32 | 4096 | 160 GB |

**优化手段**：GQA 减少 kv_heads（70B 用 8 组 GQA 比 MHA 减少 8x KV Cache）、KV Cache 量化（FP16->INT8 减半）、PagedAttention（消除碎片，提高 batch 上限）。

### Q: DeepSpeed ZeRO 1/2/3的区别？分别优化了什么？

ZeRO 三级策略逐步消除数据并行中的显存冗余：

**背景**：Adam 优化器训练 7B FP16 模型时，每卡存储：
- 参数（FP16）：14 GB
- 梯度（FP16）：14 GB
- 优化器状态（FP32 参数副本 + momentum + variance）：7B * 12 = 84 GB
- 总计：~112 GB/卡，完全冗余存 N 份

**ZeRO Stage 1（切分优化器状态）**：
- 每卡只存 1/N 的 Adam states（FP32 参数 + momentum + variance）
- 前向/反向：正常执行（每卡有完整参数和梯度）
- 参数更新：AllReduce 梯度 -> 每卡只更新自己负责的 1/N 参数 -> AllGather 广播更新后的参数
- **显存节省**：8 卡时优化器从 84GB -> 10.5GB，总显存 14+14+10.5 = 38.5 GB（vs 112 GB）
- **通信量**：与标准 DDP 相同（一次 AllReduce = 2*model_size）

**ZeRO Stage 2（额外切分梯度）**：
- AllReduce 改为 ReduceScatter：梯度归约后只保留本卡负责的分片
- **显存节省**：梯度从 14GB -> 14/8 = 1.75 GB，总显存 14+1.75+10.5 = 26.25 GB
- **通信量**：ReduceScatter 量 = model_size（vs AllReduce 的 2*model_size），但实际差异很小因为 AllReduce = ReduceScatter + AllGather

**ZeRO Stage 3（额外切分参数）**：
- 每卡只存 1/N 的模型参数。需要某层参数时 AllGather 收集，用完释放
- **显存节省**：参数 14/8 + 梯度 14/8 + 优化器 84/8 = 1.75+1.75+10.5 = 14 GB（vs 原始 112 GB，8x 节省）
- **通信量**：每层前向 1 次 AllGather + 反向 1 次 AllGather + 1 次 ReduceScatter ≈ 3*model_size（约 DDP 的 1.5 倍）
- **计算效率权衡**：通信量增加但可通过计算-通信重叠部分隐藏

**选择建议**：
| 场景 | 推荐 Stage | 原因 |
|------|-----------|------|
| 模型装得下单卡 | Stage 1 | 通信最少，overhead 最低 |
| 优化器状态撑爆显存 | Stage 1/2 | 切分优化器/梯度足够 |
| 模型本身放不下单卡 | Stage 3 | 唯一选择（或配合 TP/PP） |
| 追求极致训练速度 | Stage 1 + 大 batch | 通信可完全重叠 |

### Q: DeepSpeed对矩阵A*B运算，4张卡如何分配参数并交互以节省显存？

以 ZeRO Stage 3 为例说明具体的参数切分和通信过程：

**初始状态**：矩阵 A 大小 [M, K]，参数沿 K 维度切分为 4 份：
- GPU 0 存 A[:, 0:K/4]
- GPU 1 存 A[:, K/4:K/2]
- GPU 2 存 A[:, K/2:3K/4]
- GPU 3 存 A[:, 3K/4:K]

**前向计算过程**：
1. **AllGather 收集完整 A**：4 卡协作，每卡广播自己持有的 1/4，所有卡拼出完整 A [M,K]
   - 通信量：每卡发 M*K/4，收 3*M*K/4
2. **本地计算**：每卡用完整 A 与自己的数据做 C = A * B
   - B 可能也是切分的（如按 batch 维度切分给各卡）
3. **释放非本地参数**：计算完成后，每卡释放不属于自己的 3/4 参数，只保留自己的 1/4
   - 显存峰值只在计算时短暂持有完整 A

**反向传播过程**：
1. **再次 AllGather A**（反向需要 A 计算梯度）
2. **计算梯度 dA**：dA = dC * B^T
3. **ReduceScatter dA**：
   - 所有卡的 dA AllReduce -> 各卡只保留自己负责的 1/4 梯度分片
   - GPU 0 保留 dA[:, 0:K/4]，GPU 1 保留 dA[:, K/4:K/2]，...
4. **参数更新**：每卡用自己的梯度分片更新自己的参数分片

**显存分析**（vs 不切分）：
- 不切分：每卡存完整 A（M*K 参数）+ 完整 dA（M*K 梯度）+ 完整优化器状态
- ZeRO-3：每卡存 1/4 A + 1/4 dA + 1/4 优化器状态 + 临时持有完整 A（计算时）
- 峰值显存减少 ~4x（4 卡情况）

**通信代价**：
- 前向 1 次 AllGather（数据量 M*K）
- 反向 1 次 AllGather + 1 次 ReduceScatter（数据量 2*M*K）
- 总通信：3*M*K（vs DDP 的 2*M*K，增加 50%）

**优化**：通信与计算重叠——在计算当前层时预取（AllGather）下一层的参数。

### Q: Temperature的数学原理？Temperature、Top-k、Top-p作用顺序？

**Temperature 的数学原理**：

Temperature 是 softmax 之前对 logits 的缩放参数：

```
P(x_i) = exp(z_i / T) / Σ_j exp(z_j / T)
```

**效果分析**：
- **T = 1**：标准 softmax，正常分布
- **T > 1（高温）**：logits 被压缩（除以大数），softmax 输出更接近均匀分布 -> 生成更随机/更有创意
- **T < 1（低温）**：logits 被放大（除以小数），softmax 输出更尖锐 -> 生成更确定/更保守
- **T -> 0**：退化为 argmax（确定性选择概率最高的 token）
- **T -> ∞**：退化为均匀分布（完全随机）

**数学直觉**：Temperature 控制 softmax 的"锐度"。logits 差异为 Δz 的两个 token，温度为 T 时概率比 = exp(Δz/T)。T 越大比值越接近 1（越平坦）。

---

**Top-k、Top-p 的作用**：

**Top-k**：只保留概率最高的 k 个 token，其余概率置 0，重新归一化。
- k=1 等同于 greedy decoding
- k=50 是常见默认值

**Top-p（Nucleus Sampling）**：保留累积概率 >= p 的最小 token 集合。
- 自适应：高置信时（一个 token 概率 >p）只从 1 个 token 中选；低置信时可能从数百个 token 中选
- p=0.9 是常见默认值

---

**作用顺序（从 logits 到最终采样）**：

```
Raw logits
  │
  ├── 1. Temperature scaling: logits = logits / T
  │
  ├── 2. Top-k filtering: 只保留概率最高的 k 个 token
  │
  ├── 3. Top-p (nucleus) filtering: 在 top-k 结果中再按累积概率截断
  │
  ├── 4. 重新归一化: 使保留 token 的概率和为 1
  │
  └── 5. 从归一化后的分布中采样
```

**为什么是这个顺序**：
- Temperature 必须在 softmax/filtering 之前，因为它改变的是 logits 而非概率
- Top-k 先粗筛（计算简单，直接排序取前 k），Top-p 再精筛（需要累积概率计算）
- 两者组合时的效果：Top-k 设定候选上限，Top-p 在此范围内动态调整实际候选数

**实践参数选择**：
- 事实性任务（问答/摘要）：T=0.1-0.3, top_p=0.9
- 创意生成（故事/诗歌）：T=0.7-1.0, top_p=0.95
- 代码生成：T=0.2-0.4, top_p=0.95（需要确定性但偶尔探索）

### Q: Softmax函数？

**Softmax** 将任意实数向量 z = [z_1, ..., z_n] 映射为概率分布 P = [p_1, ..., p_n]：

**公式**：
```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

**性质**：
- 所有输出 > 0（exp 保证正性）
- 输出和为 1（归一化保证概率性质）
- 保持相对顺序：z_i > z_j -> p_i > p_j
- 输入加常数不改变输出：softmax(z + c) = softmax(z)（利用此性质做数值稳定处理）

**数值稳定实现（Safe Softmax）**：
```
m = max(z)
softmax(z_i) = exp(z_i - m) / Σ_j exp(z_j - m)
```
减去 max 后最大的 exp 输入为 0（exp(0)=1），避免 FP32 中 exp(x>88) 溢出。

**梯度**：
- Jacobian 矩阵：∂p_i/∂z_j = p_i(δ_ij - p_j)
- 对角项（i=j）：p_i * (1 - p_i)
- 非对角项（i≠j）：-p_i * p_j
- 梯度向量：如果 loss 对 softmax 输出的梯度为 dp，则对 logits 的梯度为 p ⊙ (dp - <dp, p>)

**在深度学习中的应用**：
- 分类输出层：将 logits 转为类别概率，配合交叉熵 loss
- Attention 权重：softmax(QK^T/√d_k) 将 attention score 归一化为权重分布
- MoE 门控：决定每个 token 发送到哪些专家（expert routing）
- 温度采样：Temperature scaling 后的 softmax 控制生成随机性
- 知识蒸馏：soft target = softmax(logits/T) 中高温 softmax 提供更多信息

**计算特点**：需要全局 reduction（max 和 sum），是 memory-bound 操作。FlashAttention 用 Online Softmax 实现分块计算。

### Q: 了解过通信算子吗？

集合通信（Collective Communication）是分布式训练的基础设施，所有并行策略都依赖这些通信原语：

**核心通信原语**：

**1. AllReduce**：所有节点的数据归约（如求和），结果广播到所有节点。
- 输入：每卡有 tensor A_i
- 输出：每卡都有 sum(A_0, A_1, ..., A_{N-1})
- 用途：DDP 梯度同步（所有卡获得完整梯度均值）
- 通信量：2 * data_size * (N-1)/N（Ring AllReduce）

**2. AllGather**：收集所有节点的数据片段，每个节点获得完整数据。
- 输入：每卡有数据片段 A_i（大小 S/N）
- 输出：每卡都有完整数据 [A_0, A_1, ..., A_{N-1}]（大小 S）
- 用途：ZeRO-3 前向时收集完整参数、TP 中收集分片结果
- 通信量：data_size * (N-1)/N

**3. ReduceScatter**：归约后结果按分片分配到各节点。
- 输入：每卡有完整 tensor A_i（大小 S）
- 输出：每卡获得归约结果的 1/N 分片
- 用途：ZeRO-2/3 的梯度切分、TP 后的结果切分
- 通信量：data_size * (N-1)/N
- 关系：AllReduce = ReduceScatter + AllGather

**4. Broadcast**：一个节点的数据广播到所有节点。
- 用途：参数初始化同步、PP 中传递激活值
- 通信量：data_size

**5. All-to-All**：每个节点向每个其他节点发送不同数据。
- 用途：MoE 中 token 路由到不同专家
- 通信量：取决于具体数据分布

**实现拓扑**：
| 算法 | 特点 | 适用场景 |
|------|------|---------|
| Ring | 带宽最优（2*(N-1)/N * S），延迟 O(N) | N 较小（<16），数据量大 |
| Tree | 延迟最优 O(logN)，带宽非最优 | N 大，数据量小（如标量 reduce） |
| Recursive Halving-Doubling | 延迟和带宽都较优 | 中等规模（8-64 节点） |
| NCCL 自动选择 | 根据 N 和数据量选最优算法 | 生产环境 |

**NCCL（NVIDIA Collective Communication Library）**：
- GPU 专用的集合通信库，自动利用 NVLink/PCIe/InfiniBand
- 支持多节点多卡的高效通信
- 自动选择最优拓扑和算法
- 是 PyTorch DDP/DeepSpeed/Megatron-LM 的底层通信后端

**通信-计算重叠**：将 AllReduce 拆分为多个 bucket，每个 bucket 的 backward 完成后立即开始 AllReduce，与其他层的 backward 计算重叠。这是 DDP 高效的关键。
