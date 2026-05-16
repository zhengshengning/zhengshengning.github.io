---
title: '荣耀 AI Infra 一面'
date: 2026-04-16 11:45:34
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 面经]
---


<!-- more -->

---

### Q: 模型训练优化有哪些方法？（算法层面和硬件层面）

训练优化的目标是在保证收敛质量的前提下，最大化训练吞吐（tokens/s）和资源利用率。

**算法层面优化**：

| 方法 | 原理 | 典型收益 | 适用场景 |
|---|---|---|---|
| **混合精度训练** | 计算用 BF16/FP16，主权重保持 FP32 | 训练速度 1.5-2x | 所有模型 |
| **梯度累积** | 小 batch 多步累积后才更新 | 模拟大 batch，不增显存 | 单卡显存不够 |
| **学习率调度** | Warmup + Cosine Decay | 训练稳定 + 收敛更好 | 所有大模型训练 |
| **梯度 Checkpoint** | 只保存部分层的激活，反向时重算 | 激活显存减 60-80% | 大模型/长序列 |
| **FlashAttention** | Tiling + Online Softmax | 速度 2-4x + 显存 O(N²)→O(N) | Attention 密集模型 |
| **知识蒸馏** | 大模型指导小模型训练 | 小模型质量提升 | 部署受限场景 |
| **课程学习** | 从简单样本逐步到困难样本 | 收敛更快更稳定 | 数据难度差异大 |

**混合精度训练的具体流程**：
```
每个训练步:
  1. FP32 主权重 → 转为 BF16 副本
  2. 前向: 用 BF16 权重和 BF16 输入计算（Tensor Core 加速）
  3. 计算 BF16 loss → loss scaling (×65536) 防止梯度下溢
  4. 反向: 计算 BF16 梯度
  5. 梯度 unscale (÷65536) → 检查 inf/nan → FP32 梯度
  6. FP32 梯度更新 FP32 主权重（精确累加）

为什么需要 FP32 主权重:
  权重更新量 = lr × gradient ≈ 3e-4 × 0.01 = 3e-6
  BF16 精度: 最小可表示的变化 ≈ 2^{-7} × 权重量级 ≈ 0.008
  3e-6 << 0.008 → BF16 无法累加如此小的更新!
  FP32 精度: 可以精确累加微小更新
```

---

**系统/硬件层面优化**：

| 方法 | 原理 | 典型配置 | 适用规模 |
|---|---|---|---|
| **数据并行 (DP)** | 模型副本 + 数据切分 + AllReduce 梯度 | N 卡 = N 倍数据 | 模型放得下单卡 |
| **张量并行 (TP)** | 切分权重矩阵到多卡 | 通常 TP=2-8 (节点内) | 单层太大 |
| **流水线并行 (PP)** | 切分层到多卡 + 微批次流水 | PP=2-8 (跨节点) | 总层数多 |
| **ZeRO** | 分片优化器/梯度/参数 | Stage 1/2/3 | 显存不足 |
| **Sequence Parallel** | 沿序列维度分片激活 | 通常与 TP 配合 | 长序列 |
| **计算通信重叠** | AllReduce 与反向计算并行 | bucket gradient | 多卡训练 |
| **高速互联** | NVLink (900 GB/s) / IB (400 Gbps) | 节点内 NVLink + 节点间 IB | 大规模集群 |
| **Offload** | 优化器状态/激活卸载到 CPU/NVMe | ZeRO-Offload | 显存极紧张 |

**3D 并行组合（大模型训练标配）**：
```
典型配置 (70B 模型, 128 GPU):
  TP = 8 (节点内, 需要 NVLink 高带宽)
  PP = 4 (跨节点, 通信量较小)
  DP = 4 (跨节点, AllReduce 梯度)
  
  总 GPU = TP × PP × DP = 8 × 4 × 4 = 128

显存分布:
  每张卡只放 1/8 层的 1/4 模型参数（TP 切分后的 PP 中的一个 stage）
  ZeRO-1 再将优化器状态分到 DP=4 的 4 张卡上
```

**前沿优化方向**：
- **FP8 训练**（H100）：计算吞吐再提 2x，需要处理精度问题
- **Zero-Bubble PP**：通过拆分 B/W 消除流水线空泡
- **异步 TP**：通信与计算重叠，减少 TP 同步等待
- **Selective Activation Recomputation**：只重算占显存大的层（如 Attention），保留小的层

---

### Q: 模型压缩的方法有哪些？

模型压缩旨在减小模型大小和计算量，同时尽可能保持精度。各方法可独立或组合使用：

**1. 量化（Quantization）——降低数值精度**：

| 方法 | 精度格式 | 压缩比 | 精度损失 | 硬件要求 |
|---|---|---|---|---|
| FP8 (E4M3) | 8-bit 浮点 | 2x | 极小 | H100+ |
| W8A8 (SmoothQuant) | INT8 | 2x | 小 | A100+ |
| W4A16 (AWQ/GPTQ) | INT4 权重 | 4x | 中 | 通用 |
| W4A8 | INT4 权重 + INT8 激活 | 4x+ | 中高 | 专用 kernel |
| NF4 (QLoRA) | 4-bit NormalFloat | 4x | 小(微调) | 通用 |

**2. 剪枝（Pruning）——移除不重要的参数**：

```
结构化剪枝（推理友好，可直接加速）:
  - 整个 attention head 剪除（如果某些 head 重要性低）
  - 整个 FFN 神经元剪除
  - 整层剪除（浅层压缩）
  → 直接减小模型维度，无需特殊硬件

非结构化剪枝（高压缩比，但需要稀疏硬件）:
  - 将不重要的权重置零（如 magnitude < threshold）
  - SparseGPT: LLM 的高效非结构化剪枝
  - 90% 稀疏度 → 理论 10x 压缩
  → 需要稀疏矩阵乘硬件（如 A100 的 2:4 Structured Sparsity）
```

**3. 知识蒸馏（Knowledge Distillation）——大模型教小模型**：
```
Teacher (大模型, 如 70B) → Student (小模型, 如 7B)

蒸馏方式:
  - 输出蒸馏: Student 学习 Teacher 的 logits 分布 (soft labels)
    L = α × CE(y, student_logits) + (1-α) × KL(teacher_logits, student_logits)
  
  - 特征蒸馏: Student 学习 Teacher 的中间层表示
  - 数据蒸馏: 用 Teacher 生成高质量合成数据训练 Student

实际案例:
  - Phi-3: 微软用 GPT-4 生成数据训练小模型
  - Gemma: Google 的小模型用大模型蒸馏
  - OpenHermes: 开源社区用 ChatGPT 蒸馏 7B/13B 模型
```

**4. 低秩分解（Low-Rank Decomposition）**：
```
原始权重: W [d × d] = d² 参数
低秩分解: W ≈ A × B, 其中 A [d × r], B [r × d], r << d
参数量: 2dr << d² (当 r << d/2)

代表方法:
  - LoRA: 冻结原权重 W，训练增量 ΔW = AB
  - 推理时: W' = W + AB → 合并后无额外推理开销
  - 适用: 微调场景（不是预训练压缩）
```

**5. 架构搜索 / 高效架构设计**：
- MoE：稀疏激活，同等计算量下更大参数量
- MobileLLM：针对端侧设计的高效 Transformer 变体
- Mamba/RWKV：线性复杂度替代 attention（推理无需 KV Cache）

**各方法适用场景总结**：
```
推理部署（精度优先）→ W8A8 量化
推理部署（速度优先）→ W4A16 量化
端侧/嵌入式 → W4A16 + 结构化剪枝 + 蒸馏
微调场景 → LoRA/QLoRA
训练预算有限 → 蒸馏小模型
```

---

### Q: 训练框架有哪些？各自特点？

**主流 LLM 训练框架对比**：

| 框架 | 开发方 | 核心特色 | 适用场景 |
|---|---|---|---|
| **PyTorch (FSDP/DDP)** | Meta | 灵活易用、动态图、生态最大 | 研究/中等规模训练 |
| **DeepSpeed** | 微软 | ZeRO 系列显存优化、易集成 | 大模型训练（学术友好） |
| **Megatron-LM** | NVIDIA | 极致 TP+PP+SP、最高吞吐 | 超大规模预训练 |
| **ColossalAI** | HPC-AI Tech | 多种并行策略整合、低门槛 | 快速搭建训练系统 |
| **veRL** | 字节/火山 | RLHF 训练 + Hybrid Engine | LLM 对齐训练 |
| **OpenRLHF** | 开源社区 | Ray 编排 + 多模块解耦 | RLHF/GRPO 训练 |
| **JAX/XLA** | Google | 函数式 + 编译器优化 | TPU 训练 |
| **Llama-Factory** | 开源 | 一站式微调（100+ 模型） | 微调/实验 |

**详细特点分析**：

**DeepSpeed**：
```
核心: ZeRO 分片（最知名的显存优化）
  ZeRO-1: 优化器状态分片 → 显存 ÷ N
  ZeRO-2: + 梯度分片
  ZeRO-3: + 参数分片 → 几乎无显存限制
  ZeRO-Offload: 状态卸载到 CPU
  ZeRO-Infinity: 卸载到 NVMe SSD

特点:
  - 与 HuggingFace Transformers 深度集成（几行代码启用）
  - 配置驱动（json 文件配置策略）
  - 适合学术团队快速训练大模型
  
局限:
  - TP 实现不如 Megatron 极致
  - 超大规模（千卡以上）通信效率不如 Megatron
```

**Megatron-LM**：
```
核心: 极致的 3D 并行 + 通信优化
  - TP: 优化的 column/row parallel（减少通信量）
  - PP: Interleaved 1F1B（减少 bubble）
  - SP: Sequence Parallel（activation 分片）
  - 计算通信重叠（TP All-Reduce 与计算 overlap）

特点:
  - 训练吞吐最高（NVIDIA 内部使用）
  - 经过千卡级别验证
  - 代码复杂，定制性强

局限:
  - 学习曲线陡峭
  - 支持模型不如 HuggingFace 多
  - 与 HF 生态整合需要额外工作
```

**选择建议**：
```
研究实验 + 7B 模型 → PyTorch DDP + DeepSpeed ZeRO-2
快速微调 → Llama-Factory 或 HuggingFace + PEFT
70B 预训练 → Megatron-LM（追求极致吞吐）
RLHF 训练 → veRL（单集群）或 OpenRLHF（多集群）
Google 生态/TPU → JAX + T5X/MaxText
```

---

### Q: 模型调优的经验和方法？

模型调优（训练调优）是一个系统化的实验和分析过程，需要对训练动态有深入理解。

**超参数搜索（最核心）**：

| 超参数 | 推荐范围 | 影响 | 搜索策略 |
|---|---|---|---|
| **学习率 (lr)** | 1e-5 ~ 5e-4 | 最关键，决定收敛速度和最终质量 | 先粗搜（10x 间隔）再细调 |
| **Batch size** | 256K~4M tokens | 影响收敛速度和泛化 | 通常固定（受硬件限制） |
| **Warmup steps** | 总步数的 1-5% | 训练初期稳定性 | 固定为 2000-5000 步 |
| **Weight decay** | 0.01-0.1 | 正则化强度 | 通常 0.1 for LLM |
| **lr schedule** | Cosine / WSD | 后期学习率衰减方式 | Cosine 最常用 |
| **Adam β₁/β₂** | 0.9 / 0.95 | 动量和二阶矩的平滑度 | 通常不需要调 |
| **Gradient clip** | 1.0 | 防止梯度爆炸 | 固定为 1.0 |

**学习率的关键地位**：
```
lr 太大: loss spike → NaN → 训练崩溃
lr 太小: 收敛极慢，训练 tokens 浪费
最优 lr: 在"刚好不崩溃"的边界附近

搜索方法:
  1. 短程实验（训练 1-5% 的 tokens），测试多个 lr
  2. 观察 loss 曲线：平稳下降 > 剧烈波动
  3. 选择 loss 下降最快且稳定的 lr
  4. 对于不同模型规模，lr 需要相应调整（μP 理论）
```

**数据质量与配比**：

| 维度 | 影响 | 实践建议 |
|---|---|---|
| **去重** | 重复数据降低训练效率 | MinHash / exact dedup |
| **质量过滤** | 低质量数据（乱码、广告）污染模型 | 用分类器过滤 + 规则过滤 |
| **领域配比** | 不同领域数据比例影响模型能力分布 | 代码:网页:书籍 ≈ 1:4:1 |
| **多语言配比** | 影响多语言能力 | 英:中:其他 ≈ 6:2:2 |
| **数据课程** | 训练顺序影响收敛 | 先通用后专业，先简单后困难 |

**训练监控和 Debug**：

```
必监控指标:
  - Loss 曲线（应平滑下降，突刺需分析）
  - 梯度范数（应稳定，不应持续增长）
  - 学习率曲线（确认 schedule 正确）
  - GPU 利用率/吞吐（确认系统正常）

Loss Spike 处理:
  1. 跳过该 batch（skip bad batch）
  2. 降低 lr 重新开始
  3. 分析数据：spike 时的 batch 是否异常
  4. 增大 gradient clip 阈值

梯度分析:
  - 分层梯度范数：看是否某些层梯度异常
  - 梯度方向变化：cosine similarity between consecutive gradients
  - 权重范数：应缓慢增长，暴增说明有问题
```

**评估策略**：
```
训练中评估:
  每 1000-5000 步在验证集评估 loss
  每 10000 步跑轻量 benchmark (如 HellaSwag, ARC)
  
关键判断:
  - 训练 loss 下降但 val loss 上升 → 过拟合（减少训练步数或加正则）
  - 训练 loss 平台期 → 可能需要更大 lr 或数据问题
  - 某个 benchmark 突然下降 → 检查最近数据变化
```
