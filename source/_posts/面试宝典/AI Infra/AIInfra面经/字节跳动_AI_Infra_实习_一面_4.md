---
title: '字节跳动 AI Infra 实习 一面 (4)'
date: 2026-04-16 11:43:27
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 训练优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 讲讲对DeepSpeed的理解？

DeepSpeed是微软开发的大模型训练优化库，其核心创新在于ZeRO系列技术——通过分片消除数据并行中的内存冗余：

**ZeRO的核心洞察**：标准DDP中每张卡保存完整的参数+梯度+优化器状态，N卡中有N-1份是冗余的。ZeRO逐步分片这些冗余：

| Stage | 分片内容 | 每卡显存(模型Φ参数) | 通信量变化 |
|-------|---------|-------------------|----------|
| DDP | 无 | 16Φ字节 | 2Φ(AllReduce) |
| ZeRO-1 | 优化器状态(m+v+fp32_param) | 4Φ + 12Φ/N | 不变(2Φ) |
| ZeRO-2 | +梯度 | 2Φ + 14Φ/N | 不变(2Φ) |
| ZeRO-3 | +参数 | 16Φ/N | 1.5x(3Φ) |

**ZeRO-Offload/Infinity**：
- CPU Offload：优化器状态和参数放CPU RAM，GPU只做前向/反向计算。
- NVMe Offload：进一步将不活跃数据放SSD（利用NVMe的4GB/s+带宽）。
- 适用：单卡/少卡要训大模型的资源受限场景。
- 代价：PCIe带宽成为瓶颈（Gen4: 32GB/s双向），训练速度降低30-50%。

**其他关键功能**：
- **混合精度训练**：自动管理FP16前向反向 + FP32主权重 + 动态Loss Scaling。
- **Pipeline Parallelism**：支持1F1B和interleaved调度，与ZeRO组合使用。
- **Activation Checkpointing**：集成深度选择性checkpoint（可按层设置）。
- **通信优化**：梯度压缩、分桶通信、compute-communication overlap。
- **Sparse Attention**：长序列稀疏注意力训练支持。

**使用方式（PyTorch集成）**：
```python
import deepspeed
model, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    config={
        "train_batch_size": 64,
        "fp16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "gradient_accumulation_steps": 4
    }
)
# 训练循环中直接使用model前向反向，deepspeed自动处理通信和显存管理
```

### Q: 为什么用ZeRO-3？

**适用场景——模型参数放不下单卡时的选择**：

| 模型大小 | 单卡80GB能否放下参数(FP16) | 推荐方案 |
|---------|--------------------------|---------|
| <40B | 能(80GB > 40B×2=80GB边界) | ZeRO-1/2足够 |
| 40-200B | 不能 | ZeRO-3 或 TP |
| >200B | 远不能 | ZeRO-3 + TP + PP组合 |

**ZeRO-3 vs 张量并行(TP)的选择**：

| 维度 | ZeRO-3 | TP |
|------|--------|-----|
| 互联要求 | 中（跨节点IB可接受） | 高（必须NVLink） |
| 通信模式 | AllGather(前向) + ReduceScatter(反向) | AllReduce/AllGather(每层2次) |
| 通信频率 | 每层前向+反向各1次 | 每层2次（频率更高） |
| 编程复杂度 | 低（wrap模型即可） | 高（需修改模型代码） |
| 单步延迟 | 较高（每层都有通信） | 较低（NVLink下通信快） |
| 灵活性 | 任意模型结构 | 需要模型支持切分 |

**决策建议**：
- 节点内8卡NVLink：**用TP**（通信快，延迟低）。
- 跨节点扩展（IB互联）：**用ZeRO-3或ZeRO-2+梯度累积**（TP跨节点太慢）。
- 最常见组合：TP=8(节点内) + ZeRO-1(跨节点DP)，兼顾效率和扩展性。

**ZeRO-3的通信开销详解**：
```
前向 Layer i: AllGather参数（从各卡收集完整参数）→ 计算 → 释放参数
反向 Layer i: AllGather参数 → 计算梯度 → ReduceScatter梯度
总通信: 前向N层×AllGather(2Φ/N×N=2Φ) + 反向N层×(AllGather+ReduceScatter)
      = 2Φ + 2Φ + 2Φ = 6Φ（约DDP的3倍通信量，但显存节省N倍）
```

### Q: 还用过哪些加速框架？区别是什么？推导显存节省公式。

**主流框架对比**：

| 框架 | 核心技术 | 优势 | 劣势 | 典型用户 |
|------|---------|------|------|---------|
| DeepSpeed | ZeRO分片 | 灵活、易用 | 大规模效率略逊Megatron | 中小团队 |
| FSDP | 类ZeRO-3(PyTorch原生) | 生态集成好 | 功能比DeepSpeed少 | PyTorch用户 |
| Megatron-LM | 3D并行 | 大规模极致效率 | 需要改模型代码 | 大厂训练 |
| ColossalAI | Gemini+并行策略 | API友好 | 社区小 | 入门级 |

**显存公式推导（混合精度Adam）**：

假设模型参数量Φ（每个参数FP16占2字节），数据并行度为N：

**DDP（无分片）每卡显存**：
```
参数(FP16):    2Φ   (前向反向计算用)
梯度(FP16):    2Φ   (反向计算后)
FP32主权重:    4Φ   (Adam更新用)
Adam动量m:     4Φ   (一阶矩)
Adam方差v:     4Φ   (二阶矩)
─────────────────────
总计:          16Φ 字节/卡
```

**ZeRO-1（分片优化器状态）**：
```
参数(FP16):    2Φ    (每卡完整)
梯度(FP16):    2Φ    (每卡完整，AllReduce后)
FP32主权重:    4Φ/N  (只存1/N)
Adam m:        4Φ/N
Adam v:        4Φ/N
─────────────────────
总计:          4Φ + 12Φ/N 字节/卡

N=8时: 4Φ + 1.5Φ = 5.5Φ（vs DDP的16Φ，节省约3倍）
```

**ZeRO-2（分片优化器+梯度）**：
```
参数(FP16):    2Φ    (每卡完整)
梯度(FP16):    2Φ/N  (只存1/N，ReduceScatter后)
FP32主权重:    4Φ/N
Adam m:        4Φ/N
Adam v:        4Φ/N
─────────────────────
总计:          2Φ + 14Φ/N 字节/卡

N=8时: 2Φ + 1.75Φ = 3.75Φ（节省约4.3倍）
```

**ZeRO-3（全分片）**：
```
参数(FP16):    2Φ/N  (只存1/N，需要时AllGather)
梯度(FP16):    2Φ/N
FP32主权重:    4Φ/N
Adam m:        4Φ/N
Adam v:        4Φ/N
─────────────────────
总计:          16Φ/N 字节/卡

N=8时: 2Φ（节省8倍！）
N=64时: 0.25Φ（几乎可以训任何模型）
```

### Q: Q-Former是什么？

Q-Former（Querying Transformer）是BLIP-2中连接视觉和语言的核心桥接模块：

**设计动机**：
- 视觉编码器（ViT）输出大量patch tokens（如256个），直接全部输入LLM太长且有噪声。
- 需要一个"信息瓶颈"提取最相关的视觉信息，压缩为少量token供LLM消费。

**结构详解**：
```
                 ┌─── Cross-Attention ───┐
                 ↓                       ↑
Learnable Queries (32个) → Transformer Encoder → Visual Embeddings (32个)
                                ↑
                      Frozen ViT Features (256个patch tokens)
```

- 基于BERT架构的Transformer（有Self-Attention + Cross-Attention + FFN）。
- 包含32个可学习的Query Tokens（随机初始化，通过训练学习"问什么问题"）。
- Cross-Attention层让Query Tokens关注ViT的视觉特征，提取关键信息。
- 输出：32个固定维度的向量，作为LLM的soft visual prefix（类似prompt）。

**训练流程（两阶段）**：
- **Stage 1**：Q-Former + ViT联合训练（三种任务：Image-Text Matching + Contrastive + Image-grounded Text Generation）。目标：让Queries学会提取文本相关的视觉信息。
- **Stage 2**：冻结ViT + Q-Former，训练线性投影层将Q-Former输出映射到LLM的embedding空间。LLM也保持冻结。

**关键设计选择**：
- 为什么32个Query而不是更多？足够表达图像语义，且保持LLM输入长度合理。
- 为什么用Cross-Attention而不是直接拼接？Cross-Attention是信息瓶颈，迫使学习最重要的信息。

### Q: BLIP和BLIP2的区别？现在主流模型用什么？

**BLIP vs BLIP-2对比**：

| 维度 | BLIP | BLIP-2 |
|------|------|--------|
| 视觉编码器 | 端到端联合训练 | Frozen（不训练） |
| 语言模型 | 自带Decoder(较小) | 接入Frozen大LLM |
| 视觉-语言连接 | 直接用视觉特征 | Q-Former桥接 |
| 训练成本 | 需要大规模端到端训练 | 只训Q-Former+投影层(极低成本) |
| 视觉token数输入LLM | ~196（全部patch） | ~32（Q-Former压缩） |
| 代表能力 | 图文匹配、描述 | 视觉问答、图文对话 |

**现在的主流方案（2024-2025）**：

Q-Former的复杂设计逐渐被更简单的方案取代：

| 模型 | 连接方式 | 为什么更好 |
|------|---------|-----------|
| **LLaVA** | 线性投影/MLP | 简单高效，数据规模弥补结构 |
| **Qwen-VL** | Resampler(类似Q-Former但更简) | 压缩token数同时保持信息 |
| **InternVL** | 动态分辨率ViT + MLP | 支持高分辨率 |
| **GPT-4V** | 未公开(推测直接patch token) | 算力换结构 |

**趋势总结**：
- 从复杂连接器（Q-Former）→ 简单连接器（MLP/线性层）。
- 从冻结ViT → 联合训练或fine-tune ViT。
- 关键不在连接器复杂度，在于：更多高质量图文数据 + 更大的ViT + 更长的context支持高分辨率。

### Q: 手撕：sqrt(x)（LeetCode 69），有优化方法吗？

（编程题）
