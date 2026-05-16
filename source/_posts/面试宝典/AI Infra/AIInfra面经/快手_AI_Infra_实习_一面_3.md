---
title: '快手 AI Infra 实习 一面 (3)'
date: 2026-04-16 12:15:35
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: KV Cache Compression 有哪些形式？

KV Cache 是 LLM 推理的显存瓶颈（长序列/大 batch 时可占总显存的 50%+），压缩方案分为多个层次：

**1. 架构级压缩（训练时确定）**：
- **MQA（Multi-Query Attention）**：所有 Q head 共享 1 组 KV，压缩比 = num_heads。如 Falcon-40B
- **GQA（Grouped Query Attention）**：每 G 个 Q head 共享 1 组 KV，压缩比 = G。如 LLaMA-2/3
- **MLA（Multi-head Latent Attention）**：将 KV 投影到低秩潜空间存储 latent vector，压缩比可达 GQA 的 5-10x。如 DeepSeek-V2/V3

**2. 量化压缩（推理时应用，训练无关）**：
- KV Cache FP8 量化：减少 50% 存储，PPL 增加约 0.1-0.2
- KV Cache INT4 量化：减少 75% 存储，但精度损失较大，需要 per-head/per-token scale
- 渐进精度：最近的 token 用高精度，远处的 token 用低精度

**3. 稀疏/驱逐策略（动态选择保留哪些 token）**：
- **H2O（Heavy Hitter Oracle）**：统计每个 token 的累积 attention score，保留 "heavy hitter"（高频被注意的 token）+ 最近窗口
- **StreamingLLM**：保留 attention sink（前 4 个 token，它们累积了大量 attention）+ 固定大小滑动窗口。支持无限长度推理
- **Scissorhands**：观察到重要 token 在不同层之间高度一致，用一次判断决定全层驱逐

**4. Token 合并/压缩**：
- 将语义相似的 token KV 合并为一个（平均/加权）
- 适合重复性高的文档（如代码）

**5. 分层压缩**：
- 不同层对 KV 精度的敏感度不同
- 浅层保留更多 token（捕获局部模式），深层可以更激进压缩（已经是高层语义）

### Q: MHA、MQA、GQA 的概念？MLA 与 GQA 的关系？

**三种注意力机制的演进**：

| 机制 | KV Head 数 | KV Cache 大小 | 精度 | 代表模型 |
|---|---|---|---|---|
| MHA | = Q Head 数 | 2 × H × d × seq | 最高 | GPT-3, LLaMA-1 |
| MQA | 1 | 2 × 1 × d × seq | 较低 | Falcon, PaLM |
| GQA | H/G | 2 × (H/G) × d × seq | 折中 | LLaMA-2/3, Qwen |

- **MHA**：每个 head 各有独立的 Q/K/V 投影。精度最高但 KV Cache 开销与 head 数线性增长
- **MQA**：所有 Q head **共享一组 K/V**。KV Cache 减少到 MHA 的 1/H（如 H=32 则减少 32x），但精度可能下降明显
- **GQA**：折中方案——每 G 个 Q head 共享一组 K/V。G=1 退化为 MHA，G=H 退化为 MQA。通常 G=4 或 G=8

**MLA（Multi-head Latent Attention）—— DeepSeek 的创新**：

核心思想：不存储完整的 K/V，而是存储**低维 latent vector**（压缩表示）。

```
标准 GQA：cache 尺寸 = num_kv_heads × head_dim × seq_len（如 8×128=1024 dim/token）
MLA：    cache 尺寸 = latent_dim × seq_len（如 512 dim/token）
```

- 训练时学习 down-projection（压缩）和 up-projection（恢复）矩阵
- 推理时只缓存压缩后的 latent，计算 Attention 时通过 up-projection 恢复完整 KV
- **权重吸收**优化：将 up-projection 矩阵吸收进 Q_proj 和 O_proj，推理时无需显式恢复

**MLA vs GQA 的等效关系**：
- GQA 的 KV Cache = num_kv_heads × head_dim（如 8×128 = 1024 维/token）
- MLA 的 KV Cache = latent_dim（如 512 维/token）
- MLA 等效压缩比 = 1024/512 = 2x（相对于已经很高效的 GQA！）
- 相对于 MHA（128×128=16384 维），MLA 压缩比高达 32x

### Q: DeepSpeed ZeRO 1/2/3 分别做了什么？ZeRO-1 在 N 参数 P 卡下如何分配？

**核心问题**：DDP 中每张卡冗余存储完整的优化器状态 + 梯度 + 参数，8B 模型 Adam 训练每卡需要 ~100+ GB 显存。

**ZeRO 三级优化——逐步分片**：

| 级别 | 分片内容 | 每卡存储（混合精度 Adam） | 通信量 |
|---|---|---|---|
| Baseline（DDP） | 无 | 4N + 12N = 16N bytes | 2N（AllReduce） |
| ZeRO-1 | 优化器状态 | 4N + 12N/P bytes | 2N（同 DDP） |
| ZeRO-2 | 优化器状态 + 梯度 | 2N + (2N+12N)/P bytes | 2N（同 DDP） |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 16N/P bytes | 3N（多 1.5x） |

**ZeRO-1 详细显存计算**（Adam + FP16 混合精度，N 参数，P 卡）：

每卡存储：
- 模型参数（FP16）：2N bytes/卡 —— **每卡全量**（因为前向反向需要完整参数）
- 梯度（FP16）：2N bytes/卡 —— **每卡全量**
- 优化器状态（只存 1/P）：
  - FP32 参数副本：4N/P bytes
  - FP32 momentum：4N/P bytes
  - FP32 variance：4N/P bytes
  - 小计：12N/P bytes/卡

**总计每卡**：`4N + 12N/P` bytes

**示例**（8B 模型，8 张 A100）：
- DDP：4×8 + 12×8 = 128 GB/卡（放不下 80GB）
- ZeRO-1：4×8 + 12×8/8 = 32 + 12 = 44 GB/卡（可以放下！）

**通信模式**：ZeRO-1/2 通信量与 DDP 相同（梯度 AllReduce），因为参数更新时各卡只更新自己负责的 1/P 优化器状态，然后通过 AllGather 广播更新后的参数。

### Q: SmoothQuant 的原理？为什么要 Smooth？如何判断模型是否适合？

**问题背景**：W8A8 量化（权重和激活都 INT8）是推理加速的理想方案（INT8 Tensor Core 加速），但激活值中存在**离群值（Outlier）**使得激活量化困难。

**Outlier 问题**：
- LLM 的激活在某些 channel（input channel 维度）上有极大值（如绝对值 >100，其余 <1）
- 这些 outlier channel 是固定的（与输入无关，由权重决定）
- 直接 per-tensor 量化：scale 被 outlier 支配，大部分正常值被量化到很少的 bin 中，精度崩溃

**SmoothQuant 核心思想**——数学等价变换：

```
Y = X · W = (X · diag(s)^{-1}) · (diag(s) · W) = X̂ · Ŵ
```

- `s` 是 per-channel 的平滑因子（与 input channel 维度对应）
- 将 X 的 outlier channel 除以 s（变平滑），同时将 W 对应 channel 乘以 s（吸收难度）
- 效果：X̂ 和 Ŵ 的量化友好度都大幅提升

**平滑因子 s 的计算**：
```
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
```
- α 控制难度迁移程度（通常 α=0.5，平均分配）
- α→1：更多难度留在 X（激活），W 更好量化
- α→0：更多难度转给 W（权重），X 更好量化

**判断模型是否适合 SmoothQuant**：
1. 观察激活的 per-channel 分布：如果存在**固定的 outlier channel**（跨不同输入一致），则适合
2. Outlier 应该是 channel-wise 的（而非 token-wise），因为 smooth 是 per-channel 操作
3. 适合的模型：OPT、BLOOM、LLaMA 等主流 LLM（它们都有明显的 channel-wise outlier）
4. 不适合的情况：outlier 是 token-wise 的（每个 token 不同位置出现 outlier）

### Q: AWQ 和 GPTQ 的原理和区别？

**AWQ（Activation-aware Weight Quantization）**：

核心观察：不是所有权重 channel 同等重要——与**显著激活值**对应的权重 channel 更重要（因为误差会被大激活放大）。

**方法**：
1. 用校准数据找到每个 input channel 的激活幅度 `s_j = mean(|X_j|)`
2. 对重要 channel（s 大的）的权重做 per-channel scale 保护：`W_j_protected = W_j × s_j`
3. 量化保护后的权重（重要 channel 精度更高）
4. 推理时需要补偿：`Y = X · (W/s)_quantized × s`（s 可被吸收进前一层）

**特点**：不需要反向传播，仅需少量校准数据（128 条），量化速度快（分钟级），精度好

---

**GPTQ（GPT Quantization）**：

基于 OBQ（Optimal Brain Quantization）的**逐列量化 + 误差补偿**：

1. 计算校准数据的 Hessian 矩阵 `H = 2X^TX`（反映每列权重的重要性）
2. 按列顺序逐列量化：
   - 量化第 j 列：`W_j_quant = round(W_j / scale) × scale`
   - 计算量化误差：`δ = W_j - W_j_quant`
   - **误差补偿**：将误差分摊到尚未量化的列——`W_{j+1:} += δ × H_{j,j+1:} / H_{j,j}`
3. 这种补偿使得整体输出误差最小化（最优 Hessian-based 更新）

**特点**：需要 Hessian 计算（校准耗时较长，小时级），但精度更高（尤其 INT3/INT4 低比特场景）

---

**对比总结**：

| 维度 | AWQ | GPTQ |
|---|---|---|
| 核心思想 | 保护重要权重 channel | 逐列量化+误差补偿 |
| 校准数据需求 | 少（128条） | 中（128-512条） |
| 量化速度 | 快（分钟级） | 慢（小时级，需要 Hessian） |
| 精度（W4） | 好 | 略优（尤其极低比特） |
| 适用场景 | 快速部署，工程友好 | 对精度要求极高的场景 |
| 代表工具 | AutoAWQ | AutoGPTQ |

### Q: 分布式 GPU 通信原语有哪些？

集合通信是分布式训练的基础，NCCL 提供以下核心原语：

| 原语 | 功能 | 输入/输出 | 典型应用 |
|---|---|---|---|
| AllReduce | 所有节点归约+广播 | 每节点 N → 每节点 N | DDP 梯度同步 |
| AllGather | 收集所有节点数据 | 每节点 N/P → 每节点 N | ZeRO-3 参数获取 |
| ReduceScatter | 归约后分片分发 | 每节点 N → 每节点 N/P | ZeRO-2 梯度处理 |
| All-to-All | 全交换（每个发不同） | 每节点 N → 每节点 N | MoE Expert Parallel |
| Broadcast | 一对多广播 | 1 节点 N → 所有节点 N | 参数初始化 |
| Reduce | 多对一归约 | 所有节点 N → 1 节点 N | 聚合到 master |

**通信量分析（P 个节点，每节点数据量 N）**：
- AllReduce = ReduceScatter + AllGather，总通信量 = 2N×(P-1)/P ≈ 2N
- AllGather：通信量 = N×(P-1)/P ≈ N
- ReduceScatter：通信量 = N×(P-1)/P ≈ N

**实现方式**：
- **Ring AllReduce**：延迟 = 2(P-1)×α + 2(P-1)/P×N/β，对大消息带宽最优
- **Tree AllReduce**：延迟 = 2×log₂P×α + 2×log₂P×N/β，对小消息延迟更低
- NCCL 根据消息大小自动选择最优算法

**在训练中的使用场景**：
- **DP**：AllReduce 同步梯度
- **TP**：AllReduce/ReduceScatter+AllGather（Attention 和 MLP 后）
- **PP**：P2P Send/Recv（层间传递激活/梯度）
- **ZeRO-3**：AllGather 获取参数 + ReduceScatter 分散梯度
- **MoE EP**：All-to-All dispatch token 到对应 Expert
