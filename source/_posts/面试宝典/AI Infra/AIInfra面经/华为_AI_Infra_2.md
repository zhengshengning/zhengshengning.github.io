---
title: '华为 AI Infra (2)'
date: 2026-04-16 11:45:58
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 训练优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: LAMB优化器介绍？

**LAMB（Layer-wise Adaptive Moments optimizer for Batch training）** 是 Google 提出的专为大 batch 训练设计的优化器，解决了"Adam + 大 batch 训练不收敛"的问题：

**核心思想**：在 Adam 的基础上，对每层的更新量做基于权重范数的自适应缩放。

**更新公式**：
```
m_t = β1 * m_{t-1} + (1-β1) * g_t          # 一阶矩
v_t = β2 * v_{t-1} + (1-β2) * g_t^2        # 二阶矩
m̂_t = m_t / (1-β1^t)                        # 偏差修正
v̂_t = v_t / (1-β2^t)
r_t = m̂_t / (√v̂_t + ε) + λ * w_t          # Adam update + weight decay

# LAMB 特有的 layer-wise scaling
φ(||w_t||) = ||w_t||                         # 权重范数
ψ(||r_t||) = ||r_t||                         # 更新范数
trust_ratio = φ / ψ                          # 信任比

w_{t+1} = w_t - lr * trust_ratio * r_t
```

**为什么需要 Layer-wise Scaling**：
- 不同层的权重范数差异巨大（embedding 层可能 ||w||=100，深层可能 ||w||=0.1）
- 如果用统一的学习率，某些层的更新幅度相对于其权重来说太大（不稳定）或太小（学不动）
- Trust ratio = ||w|| / ||update||：保证每层的相对更新幅度（||update|| / ||w||）大致相同

**关键效果**：
- 允许使用极大 batch size（如 32K-64K）而不损失收敛性
- BERT 预训练：LAMB 用 64K batch size 在 76 分钟完成（vs Adam 用 256 batch size 需数天）
- 大 batch -> 高 GPU 利用率 -> 训练 wall-clock time 缩短

**与其他优化器对比**：
| 优化器 | Layer-wise | 大 Batch 支持 | 适用场景 |
|--------|-----------|---------------|---------|
| Adam | 否 | 差（>4K batch 不稳定） | 小/中 batch |
| LARS | 是（SGD 版本） | 好 | CV 模型大 batch |
| LAMB | 是（Adam 版本） | 好 | NLP/LLM 大 batch |
| AdaFactor | 否（但低显存） | 中 | 显存受限场景 |

---

### Q: 增大batch size时学习率如何调整？模型很大但batch不变呢？

**增大 Batch Size 时的学习率调整**：

**Linear Scaling Rule**（基本法则）：
- 当 batch size 翻 k 倍时，学习率也翻 k 倍
- 直觉：大 batch 的梯度估计更准确（方差减小 k 倍），可以步子迈更大
- 数学解释：k 步小 batch SGD 的参数更新 ≈ 1 步 k 倍 batch SGD 乘以 k 的更新（梯度求和 vs 求平均）

**配合策略**：
- **Warmup**：训练初始阶段 lr 从很小值线性增长到目标值（通常 1000-2000 步）。原因：训练初期参数变化剧烈，大 lr 容易发散
- **Gradual warmup**：lr 从 base_lr 线性增长到 k*base_lr（k 为 batch scale factor）
- **Square Root Scaling**：某些场景下 lr ∝ sqrt(batch_size) 比线性更稳定（如 Transformer 架构）

**实际限制**：
- batch size 太大时（>critical batch size），继续增大 batch 边际收益递减（梯度噪声已经很小，增大 batch 的"降噪"效果有限）
- Critical batch size 与模型大小和任务相关：GPT-3 175B 的 critical batch size 约 3.2M tokens

---

**模型很大但 Batch Size 不变**：

学习率**不需要因模型大小调整**——学习率是相对于梯度更新幅度的缩放因子，与模型参数量无直接关系。

但大模型训练需要的其他调整：
- **更长 Warmup**：大模型的 loss landscape 更复杂，初始化位置敏感，需要更保守的启动
- **更小初始 lr**：如 GPT-3 用 6×10^-5，远小于小模型常用的 1×10^-3
- **梯度累积**：如果单卡 batch size 太小（受显存限制），用梯度累积模拟大 batch：
  ```
  loss.backward()  # 多次
  if step % accumulation_steps == 0:
      optimizer.step()
      optimizer.zero_grad()
  ```
- **更稳定的优化器**：大模型更容易出现 loss spike，需要梯度裁剪（gradient clipping，通常 max_norm=1.0）

**经验值参考**：
- GPT-3 175B：lr=6e-5, batch=3.2M tokens, warmup=375步
- LLaMA-2 70B：lr=1.5e-4, batch=4M tokens, warmup=2000步
- 小模型（<1B）：lr=3e-4~1e-3, batch=256K-1M tokens

---

### Q: 你知道的并行切分策略？

分布式训练中，将计算和数据切分到多卡/多机是核心问题，现代系统通常组合使用多种并行策略：

**1. 数据并行（Data Parallelism）**：
- **原理**：每卡持有完整模型副本，切分训练数据。前向反向各自计算，梯度 AllReduce 同步
- **变体**：DDP（梯度同步）、FSDP/ZeRO（参数也切分，用时 AllGather）
- **通信量**：AllReduce 2*model_size，与卡数无关
- **适用**：模型能装下单卡时的基础并行方式

**2. 张量并行（Tensor Parallelism, TP）**：
- **原理**：切分单个算子的权重矩阵。如线性层 Y=XW，将 W 按列切分到 N 卡，各卡计算部分结果，AllReduce 合并
- **通信**：每层前向 1 次 AllReduce，反向 1 次 AllReduce。频繁但数据量小
- **约束**：需要高带宽互联（NVLink 600-900 GB/s），通常限制在 8 卡节点内
- **适用**：单层参数太大（如 hidden=12288 的大模型）

**3. 流水线并行（Pipeline Parallelism, PP）**：
- **原理**：按层切分模型，不同 stage 处理不同 micro-batch 形成流水线
- **调度**：1F1B（交替前向反向减少气泡）、Interleaved（虚拟 stage 进一步减少气泡）
- **通信**：只在 stage 边界传递激活值，通信量小
- **缺点**：Pipeline bubble 导致 GPU 利用率损失（气泡率 = (p-1)/(p-1+m)）
- **适用**：模型层数多、跨节点并行（通信量小适合低带宽网络）

**4. 序列并行（Sequence Parallelism, SP）**：
- **原理**：将 LayerNorm 和 Dropout 的序列维度切分（这些操作在 TP 中本来是冗余计算）
- **配合 TP**：TP 区域的 AllReduce 变为 ReduceScatter + AllGather，中间的非 TP 操作按序列切分
- **收益**：减少约 30% 的激活值显存占用

**5. 专家并行（Expert Parallelism, EP）**：
- **原理**：MoE 模型中不同专家分布在不同卡上
- **通信**：All-to-All（将 token 路由到对应专家所在卡）
- **挑战**：负载均衡（某些专家接收更多 token）、通信量与激活 token 数成正比

**6. Context Parallelism（CP）**：
- **原理**：切分长序列的 attention 计算。如 Ring Attention：Q 按序列分块，K/V 在设备间循环传递
- **适用**：超长上下文（128K-1M tokens）单卡 KV Cache 放不下的场景
- **通信**：K/V 块在环中传递，可与计算重叠

**3D 并行组合示例**（Megatron-LM 训练 175B）：TP=8（节点内 NVLink）× PP=8（跨节点，通信量小）× DP=16（跨节点，AllReduce）= 1024 卡。

---

### Q: ZeRO介绍？

**ZeRO（Zero Redundancy Optimizer）** 是 DeepSpeed 提出的内存优化技术，通过消除数据并行中的冗余存储实现大模型训练：

**问题背景**：标准 DDP 中每卡存储完整的模型参数 + 梯度 + 优化器状态。对 7B FP16 模型 + Adam 优化器：
- 参数：7B * 2 = 14GB
- 梯度：7B * 2 = 14GB
- 优化器状态（Adam）：FP32 参数 + momentum + variance = 7B * (4+4+4) = 84GB
- 总计：~112GB/卡（全部冗余存 N 份！）

**三级切分策略**：

**ZeRO Stage 1（切分优化器状态）**：
- 每卡只存 1/N 的 Adam 状态（momentum + variance + FP32 参数副本）
- 参数更新时：梯度 AllReduce 后，每卡只更新自己负责的参数分片，然后 AllGather 分发更新后的参数
- 显存节省：优化器状态从 84GB -> 84/N GB。8 卡时 84 -> 10.5 GB
- 通信量：与标准 DDP 相同（一次 AllReduce）

**ZeRO Stage 2（额外切分梯度）**：
- 梯度不再 AllReduce（每卡存完整梯度），改为 ReduceScatter（每卡只保留负责的梯度分片）
- 显存节省：梯度从 14GB -> 14/N GB
- 通信量：ReduceScatter（vs AllReduce，量相同但内存峰值更低）

**ZeRO Stage 3（额外切分参数）**：
- 前向/反向需要某层参数时，AllGather 从所有卡收集完整参数；用完后释放非本卡分片
- 显存节省：参数从 14GB -> 14/N GB。理论极限：总显存 = (14+14+84)/N = 112/8 = 14 GB/卡
- 通信量：每层前向 1 次 AllGather + 反向 1 次 AllGather + 1 次 ReduceScatter。通信量约为 DDP 的 1.5x
- 权衡：通信量增加但显存大幅减少，允许训练更大的模型

**实际使用建议**：
- 模型能用 Stage 1 训完就不用 Stage 2/3（通信量最少）
- Stage 3 适合单卡放不下模型的情况，但需要通信带宽足够（否则通信成为瓶颈）
- ZeRO-Offload/Infinity：将部分参数/优化器状态卸载到 CPU/NVMe，进一步扩大可训练模型规模

---

### Q: PP并行下每张卡的显存、计算量一样吗？激活值呢？

Pipeline Parallelism 中各 stage 的资源分配并非完全对称：

**计算量分析**：
- **理想情况**：将 N 层均分到 P 个 stage，每 stage N/P 层，计算量大致相同
- **实际偏差**：
  - 第一个 stage 包含 embedding 层（查表操作，计算量小但显存大）
  - 最后一个 stage 包含 LM head（一个大型线性层 [hidden, vocab_size]，vocab=128K 时计算量显著）
  - 某些层有不同结构（如 MoE 层 vs dense 层）
- **不均等导致**：最慢的 stage 决定整体速度（木桶效应），其他 stage 空等形成额外气泡

**显存分析**：
- **参数显存**：按层均分时各 stage 参数量接近。但 embedding 层参数 = vocab_size * hidden_dim，可能远大于单个 transformer 层
- **优化器状态**：与参数成正比，同样大致均匀
- **关键差异在激活值**

**激活值分析（重点）**：

在 1F1B（One Forward One Backward）调度下：
- 每个 stage 需要保存"已前向但未反向"的 micro-batch 的激活值
- **Stage 0（最前）**：最先完成前向，但最后才能做反向（需等所有 micro-batch 流完整个 pipeline）。在 steady state 前积累最多未释放的激活值
- **Stage P-1（最后）**：前向完成后立即可以反向释放激活。持有的激活值最少
- 具体：Stage k 同时持有约 (P-1-k) 个 micro-batch 的激活值

**数值示例**（4 stage, 8 micro-batch）：
- Stage 0：开始时最多同时持有 3+1=4 个 micro-batch 的激活值（在 steady state 前积累）
- Stage 3：始终只持有 1 个 micro-batch 的激活值

**缓解方法**：
- 激活重计算（Activation Checkpointing）：不保存中间激活，反向时重新前向计算。将激活显存降为 O(1)，但计算量增加 ~33%
- Interleaved PP：虚拟 stage 交错减少每 stage 同时持有的 micro-batch 数
- 不均匀切分：给前面 stage 分配更少的层来平衡激活显存

---

### Q: T5和GPT-2的差异？

T5 和 GPT-2 代表了 Transformer 的两种主要架构范式：

**架构对比**：

| 特性 | T5 (Text-to-Text) | GPT-2 (Generative) |
|------|-------------------|---------------------|
| 结构 | Encoder-Decoder | Decoder-only |
| 注意力 | Encoder 双向 + Decoder causal + Cross-attention | 纯 causal（单向） |
| 输入输出 | 输入经 encoder 编码，decoder 交叉注意力解码 | 自回归逐 token 生成 |
| 位置编码 | 相对位置编码（T5 Bias） | 可学习绝对位置编码 |
| 分词器 | SentencePiece (Unigram) | BPE |
| 词表大小 | 32K | 50K |
| 归一化 | Pre-Norm (RMSNorm) | Pre-Norm (LayerNorm) |
| 激活函数 | GEGLU (T5 1.1) | GELU |

**Encoder-Decoder 的工作方式**：
1. Encoder 对输入序列做双向 self-attention，得到上下文表示
2. Decoder 每步生成一个 token：causal self-attention + cross-attention（attend to encoder output）
3. Cross-attention：decoder 的 Q 与 encoder 的 K/V 做注意力，让生成过程参考输入信息

**为什么 Decoder-only 成为 LLM 主流**：
- 统一的预训练范式：所有任务都转化为"续写"（next token prediction），无需设计不同的 encoder/decoder 输入格式
- KV Cache 更简单：decoder-only 只需一路 KV Cache，encoder-decoder 需要两路（self-attention + cross-attention）
- 通用性更好：encoder-decoder 在特定任务（翻译/摘要）上有优势，但 decoder-only 在通用生成任务上更灵活
- 规模效应：decoder-only 架构在扩大参数量时效果提升更平滑

**T5 的独特设计**：
- Text-to-Text 统一框架：所有 NLP 任务转换为文本到文本格式（"translate English to German: ..." -> 德文输出）
- 相对位置编码：不绝对编码位置，而是编码 token 间的相对距离（更适合变长序列和外推）
- 多任务预训练：同时在多种任务上预训练，增强模型的指令跟随能力

---

### Q: Transformer结构？

Transformer 是现代深度学习的核心架构，由 Multi-Head Self-Attention 和 Feed-Forward Network 两大组件构成：

**单层 Transformer Block 的完整流程**（以 Pre-Norm、Decoder-only 为例）：

```
输入 x [batch, seq_len, hidden_dim]
│
├── 残差分支 ──────────────────────────────────┐
│                                               │
├── RMSNorm(x)                                  │
├── Multi-Head Self-Attention                   │
│   ├── Q = x @ W_Q, K = x @ W_K, V = x @ W_V │
│   ├── Q, K 施加 RoPE                          │
│   ├── Attn = softmax(Q @ K^T / √d_k) @ V     │
│   │   (with causal mask)                      │
│   └── Output = Concat(heads) @ W_O            │
│                                               │
├── x = x + Attention_output  ←─────────────────┘
│
├── 残差分支 ──────────────────────────────────┐
│                                               │
├── RMSNorm(x)                                  │
├── FFN (SwiGLU variant)                        │
│   ├── gate = x @ W_gate                       │
│   ├── up = x @ W_up                           │
│   └── output = (SiLU(gate) ⊙ up) @ W_down    │
│                                               │
└── x = x + FFN_output  ←──────────────────────┘
```

**关键组件详解**：
- **Multi-Head Attention**：将 hidden_dim 切分为 n_heads 个 head，每个 head 独立做 attention。多头允许模型关注不同位置的不同表示子空间
- **FFN（Feed-Forward Network）**：先升维（hidden -> 4*hidden），激活后降维（4*hidden -> hidden）。SwiGLU 变体使用门控机制
- **残差连接**：确保梯度可以直接流向浅层（缓解梯度消失）
- **Pre-Norm vs Post-Norm**：Pre-Norm（现代标准）在子层输入前归一化，训练更稳定

**参数量估算**（per layer, hidden=d, 假设 FFN 中间维度 = 4d）：
- Attention: W_Q + W_K + W_V + W_O = 4 * d^2
- FFN (SwiGLU): W_gate + W_up + W_down = 3 * d * 4d = 12d^2
- 总计每层 ≈ 12d^2（标准 FFN）或 16d^2（SwiGLU）
- LLaMA-7B: d=4096, 32 layers -> ~6.7B 参数

---

### Q: 残差连接的作用？

残差连接（Residual Connection）是深度网络可训练的基础保障，其作用超出"简单加一个 shortcut"的表面理解：

**1. 缓解梯度消失（最核心作用）**：
- 无残差时，梯度从第 L 层传到第 1 层需要经过 L 次矩阵乘法：dL/dx_1 = ∏(dL/dx_i)
- 如果每次乘法的范数 <1，梯度指数衰减；>1 则指数爆炸
- 有残差时：y = x + F(x)，梯度 = 1 + dF/dx。"1"项保证梯度至少有直通路径
- 即使 dF/dx 接近 0，梯度仍可通过 shortcut 完整传播到浅层

**2. 学习恒等映射（Identity Mapping）**：
- F(x) = y - x 学习的是"残差"（与恒等映射的差异）
- 如果某层最优解就是恒等映射（什么都不做），只需 F(x) -> 0 即可
- 而没有残差的情况下学习恒等映射（y = x）非常困难——需要 W 精确为单位矩阵
- 这保证了"加深网络至少不比浅网络差"

**3. 加速收敛**：
- 学习残差比学习完整映射更容易：残差通常比完整映射更小/更平滑
- 训练初期 F(x) ≈ 0，网络行为接近浅层网络（因为中间层"透明"）
- 随着训练进行，各层逐步学习有意义的特征变换
- 实验验证：ResNet 比同深度的 plain network 收敛快 2-5x

**4. 信息保留（Feature Reuse）**：
- 底层特征（如低频结构/位置信息）可以通过 shortcut 直接传到高层
- 高层不需要"重新发现"底层已有的信息，可以专注学习高级特征
- 在 Transformer 中：token 的原始表示（位置信息/基本语义）通过残差一路保持到最后一层

**5. 训练稳定性**：
- Pre-Norm + Residual 的组合使得每层的输出分布相对稳定
- 梯度范数在各层之间变化不大（不会层间剧烈波动）
- 允许使用更大的学习率而不发散

**深度与残差的关系**：没有残差连接的网络超过 20-30 层就难以训练（梯度消失/爆炸）。有残差后 100-1000 层都可以有效训练。Transformer 的 32-96 层完全依赖残差连接实现训练。

---

### Q: 3D并行相关？

3D 并行是将 Data Parallelism（DP）+ Tensor Parallelism（TP）+ Pipeline Parallelism（PP）组合使用的策略，是训练百亿/千亿参数模型的标准配置：

**为什么需要组合**：
- 单一并行策略各有局限：DP 无法切分单卡放不下的模型，TP 通信频繁不能跨节点，PP 有气泡且不减少单层显存
- 组合使用可以同时解决：显存（TP+PP 切分模型）+ 效率（DP 增加吞吐）+ 通信（按拓扑分配）

**典型配置原则**（以 8 GPU/节点为例）：

| 并行维度 | 组内通信特征 | 部署位置 | 原因 |
|---------|------------|---------|------|
| TP（tensor） | 每层 2 次 AllReduce，频繁且延迟敏感 | 节点内（NVLink 900 GB/s） | 高频通信需要最高带宽 |
| PP（pipeline） | 只传激活值（hidden_dim * batch），通信量小 | 跨节点（可用较低带宽） | 通信不频繁可容忍延迟 |
| DP（data） | 每步 1 次 AllReduce（可与 backward 重叠） | 跨节点（InfiniBand 400 Gbps） | 通信可完全与计算重叠 |

**配置搜索需考虑的因素**：

1. **通信拓扑**：节点内 NVLink 带宽 >> 节点间 InfiniBand。TP 必须在高带宽域内
2. **显存约束**：TP*PP 的组合必须使每卡的模型分片 + 激活 + KV Cache 能放入显存
3. **气泡率**：PP stages 越多气泡越大（但每 stage 模型更小）。micro-batch 数应 >> PP stages
4. **负载均衡**：各 stage 计算量需接近均匀，否则快 stage 等慢 stage

**实际案例**：
- **LLaMA-2 70B 训练（2048 卡）**：TP=8（节点内）× PP=4（跨 4 节点）× DP=64 = 2048 GPU
- **GPT-3 175B（1024 卡）**：TP=8 × PP=16 × DP=8 = 1024 GPU
- **Megatron-Turing 530B**：TP=8 × PP=35 × DP=未公开

**MFU（Model FLOPs Utilization）参考**：
- 好的 3D 并行配置可达 MFU 50-60%
- 主要损失来自：PP 气泡（~10-15%）+ TP 通信（~5-10%）+ 其他开销

---

### Q: DDP/DeepSpeed中的异步保存机制？

大规模训练中 Checkpoint 保存是关键的可靠性机制，但 naive 实现会阻塞训练：

**问题**：
- 7B 模型 FP32 checkpoint 约 28GB，70B 约 280GB
- 写入速度：NVMe SSD ~3 GB/s，网络存储 ~1 GB/s
- 同步保存阻塞训练：280GB / 3 GB/s ≈ 90 秒停顿
- 千卡训练 1 小时成本数千美元，90 秒停顿代价显著

**异步 Checkpoint 机制**：

**基本原理**：
1. 将 GPU 上的参数快速拷贝到 CPU pinned memory（GPU->CPU 带宽 ~32 GB/s，几秒完成）
2. 拷贝完成后 GPU 继续训练，CPU 后台线程异步将数据写入存储
3. 训练与 IO 完全重叠，零停顿

**PyTorch/DeepSpeed 实现**：
```python
# DeepSpeed 的异步保存
# 1. GPU 参数 snapshot 到 CPU（阻塞但快）
# 2. 后台线程写入存储（非阻塞）
engine.save_checkpoint(path, async_save=True)
```

**分布式 Checkpoint（DeepSpeed/FSDP）**：
- 每卡只保存自己负责的参数分片（ZeRO-3 下各卡持有 1/N 参数）
- 分布式写入消除单点 IO 瓶颈：N 卡并行写 N 个文件
- 总写入吞吐 = 单卡 IO 带宽 × N
- 加载时各卡读取自己的分片，AllGather 拼出完整参数

**Torch Distributed Checkpoint（PyTorch 2.0+）**：
- **Resharding 支持**：保存时用 64 卡（每卡存 1/64），加载时可用 128 卡（自动重新切分）
- 解决了"训练时并行度"和"加载时并行度"不同的实际需求
- 使用 DTensor 元数据描述切分方式，加载时自动进行必要的 AllGather/Scatter

**容错与一致性**：
- 所有卡在同一 step 做 snapshot（需要 barrier 同步确保一致性）
- 两阶段提交：先写数据文件，确认所有卡写完后再写 metadata 文件（标记 checkpoint 完整有效）
- 恢复时只加载 metadata 完整的最新 checkpoint

**实践建议**：
- 保存频率：每 100-500 步（平衡恢复点粒度和 IO 开销）
- 保留最近 3-5 个 checkpoint（防止最新的 checkpoint 损坏）
- 大模型用分布式 checkpoint + 异步保存：几乎零训练中断
