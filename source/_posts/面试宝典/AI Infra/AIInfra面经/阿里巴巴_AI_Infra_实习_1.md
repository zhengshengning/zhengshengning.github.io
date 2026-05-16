---
title: '阿里巴巴 AI Infra 实习 (1)'
date: 2026-04-16 12:14:04
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: MHA、GQA与MLA的区别和各自特点？

三种注意力机制的核心差异在于**KV head的共享程度**，直接影响KV Cache大小和推理效率：

**MHA（Multi-Head Attention）— 原始Transformer**：
- 每个head拥有独立的Q、K、V投影
- KV Cache大小 = `2 * num_heads * head_dim * seq_len * num_layers * dtype_size`
- 例如GPT-3 175B（96层，96头，head_dim=128，FP16）：
  - 单条2048序列：2 * 96 * 96 * 128 * 2048 * 2B ≈ 9.2GB
- 优点：每个head有完全独立的KV表示，模型表达力最强
- 缺点：KV Cache巨大，decode阶段带宽瓶颈严重

**GQA（Grouped Query Attention）— LLaMA-2/3, Qwen**：
- 将query head分组，每组共享一套KV head
- 如LLaMA-2 70B：32个query head分为8组，只有8个KV head
- KV Cache缩小为MHA的 `num_kv_heads / num_heads`（如1/4）
- **关键洞察**：研究表明KV的冗余度远高于Q——不同query head经常关注相似的key-value模式

| 配置 | num_q_heads | num_kv_heads | KV Cache缩减 | 模型 |
|------|------------|-------------|-------------|------|
| MHA | 32 | 32 | 1x (基线) | GPT-3 |
| GQA-8 | 32 | 8 | 4x | LLaMA-2 70B |
| GQA-2 | 32 | 2 | 16x | - |
| MQA | 32 | 1 | 32x | PaLM, StarCoder |

**MLA（Multi-head Latent Attention）— DeepSeek-V2/V3**：
- **核心思想**：将KV投影到低维latent空间再存储，推理时从latent解压出KV
- 训练时：`K = Up_K @ (Down_K @ X)`，存储中间的latent `c = Down_K @ X`（维度远小于K）
- KV Cache大小 = `latent_dim * seq_len`（如512维 vs GQA的 8*128=1024维）
- **进一步优化**：latent与RoPE解耦——将position-dependent的部分单独存储，允许更激进的latent压缩

**三者对比**：

| 维度 | MHA | GQA | MLA |
|------|-----|-----|-----|
| KV Cache（相对） | 100% | 12.5-50% | 5-15% |
| 模型质量 | 最高 | 接近MHA | 接近MHA |
| 计算效率 | 基线 | decode加速显著 | decode加速最显著 |
| 实现复杂度 | 简单 | 简单 | 较复杂（需要latent编解码） |
| 代表模型 | GPT-3/4 | LLaMA-2/3, Qwen | DeepSeek-V2/V3 |

**选择建议**：目前GQA是最主流的选择（简单有效），MLA在极端KV Cache压缩需求下更有优势（如超长序列或大批量推理）。

### Q: MoE模型和多模态模型训练还有哪些优化点？

**MoE训练优化**：

**1. AllToAll通信优化**
- **通信-计算Overlap**：将AllToAll分解为多步P2P通信，与专家计算流水线化
- **分层AllToAll**：先节点内NVLink通信，再节点间InfiniBand通信（减少跨节点数据量）
- **压缩通信**：对dispatch的token activation做FP8量化传输（DeepSeek-V3的做法）
- 典型开销：MoE层中AllToAll可占30-50%时间，overlap后可隐藏大部分

**2. 专家负载均衡**
- Auxiliary loss惩罚路由偏斜（α=0.01-0.1的auxiliary loss系数需仔细调节）
- Expert Choice反向路由 / Auxiliary-loss-free bias方法
- 动态容量因子：根据训练进度调整专家容量上限

**3. Group Gemm优化**
- 多个专家的输入batch大小不同→ 不能直接做batched GEMM
- 使用**Grouped GEMM**（CUTLASS支持）：一次kernel call处理多个不同大小的矩阵乘
- 或者padding到相同大小（浪费计算但实现简单）

**4. 专家并行策略**
- **Expert Parallelism（EP）**：不同专家分布在不同GPU
- **EP + TP 混合**：每个专家内部再做tensor parallelism（超大专家时需要）
- **EP + DP**：同一专家在多GPU上有replica（增加吞吐）

**5. 专家容量动态调整**
- 训练初期路由不稳定→用较大capacity factor（1.5-2.0）
- 训练后期路由稳定→减小capacity factor（1.0-1.25）节省计算

**多模态模型训练优化**：

**1. 视觉编码器与LLM异构并行**
- 视觉编码器（如ViT-L）远小于LLM→二者对GPU资源的需求不对称
- 方案：将视觉编码器单独部署在少量GPU上，LLM做3D并行
- 异步流水线：视觉编码与LLM前向计算overlap

**2. 动态分辨率处理**
- 不同图片分辨率不同→固定分辨率会浪费计算或丢失信息
- 动态分辨率：按照图片原始比例选择最接近的分辨率切分方案
- NaViT/Qwen-VL的做法：将不同大小的图片token packing到同一序列，减少padding浪费

**3. 视觉token压缩/下采样**
- 问题：一张高分辨率图片可能产生1000+个vision token，占大量context window
- 解决：Perceiver Resampler（固定数量的可学习query）、Q-Former、C-Abstractor等压缩模块
- 或简单的2x2 pooling将token数减为1/4

**4. 多模态数据混合比例调优**
- 文本:图文:视频数据的比例需要仔细调节
- 通常先text-only预训练→再逐步引入多模态数据
- 不同阶段的最优比例不同（早期更多文本打基础，后期增加多模态）

**5. 视觉特征缓存**
- 同一张图片在不同对话轮次中重复出现→无需每次重新编码
- 缓存视觉编码器的输出特征，按图片hash查找复用

### Q: 大模型预训练和强化学习训练在工程上有哪些可优化的点？

**预训练工程优化**：

**1. 3D并行策略调优**
- DP（数据并行）× TP（张量并行）× PP（流水线并行）的最优组合与模型大小、集群拓扑强相关
- 经验法则：TP在NVLink节点内（如TP=8），PP跨节点，DP覆盖剩余GPU
- 对于MoE还需要EP（专家并行），四维组合更复杂

**2. 通信与计算Overlap**
- PP：1F1B调度使micro-batch的前向和反向交替执行，隐藏bubble
- DP：gradient AllReduce与反向计算overlap（PyTorch DDP默认行为）
- TP（Sequence Parallelism）：将AllReduce分解为RS+AG，分别与不同计算overlap
- Zero Bubble Pipeline：进一步优化PP调度减少idle时间

**3. 数据加载Pipeline优化**
- 多进程预取 + GPU异步预处理
- Tokenized数据预处理离线完成（不在训练时做tokenization）
- 大规模分布式文件系统优化（数据locality、prefetch）
- 数据Packing：多个短序列拼接到max_seq_len，减少padding浪费（提升30%+效率）

**4. 混合精度/FP8训练**
- FP16/BF16混合精度是标配
- FP8训练（DeepSeek-V3）：前向FP8/反向FP8，进一步降低通信量和显存
- 需要per-tensor/per-channel动态scaling + delayed scaling策略保证精度

**5. 长序列训练（Context Parallelism）**
- Ring Attention：将长序列分到多GPU，每GPU处理一段，通过环形通信传递KV
- DeepSpeed Ulysses：AllToAll重新分布attention的head维度
- 与TP的区别：CP切分序列维度，TP切分head维度

**6. 故障恢复和弹性训练**
- 大规模训练中GPU故障率 ~0.1%/天/GPU → 1000GPU集群每天~1次故障
- 频繁Checkpoint（每1-2小时）+ 异步checkpoint不阻塞训练
- 弹性训练：故障GPU自动剔除，剩余GPU重新组网继续训练
- Redundant Computation：对关键梯度做冗余计算（少量性能换可靠性）

**RL训练工程优化**：

**1. 采样与训练解耦**
- PPO/GRPO需要大量在线采样（推理模式）
- 分离推理集群（生成回答）和训练集群（更新参数）
- 异步Pipeline：生成集群持续产生数据，训练集群消费数据更新→参数同步回生成集群
- vLLM/SGLang作为高效采样引擎

**2. 经验回放缓冲管理**
- 大batch采样（如一次生成数千条回答）的高效存储和索引
- 支持按reward/advantage值排序采样（优先经验回放）
- GPU-CPU间数据传输优化（pinned memory + async transfer）

**3. PPO/GRPO的多阶段Pipeline**
- PPO一个iteration涉及：采样→奖励打分→GAE计算→多epoch策略更新
- 各阶段可以pipeline化执行（上一batch的训练与下一batch的采样overlap）
- GRPO简化了pipeline（无Value model），但组内采样量大

**4. 奖励模型推理加速**
- 奖励模型推理是RL训练的瓶颈之一（每条生成都需要打分）
- 优化：奖励模型量化（INT8）、batch推理、TP并行
- 某些场景可用规则奖励替代模型奖励（如数学题的答案验证）

**5. Reference模型共享/Offload**
- PPO需要frozen的reference model计算KL惩罚
- 占用额外一份模型显存→ offload到CPU / 与training model共享权重（定期snapshot）
- GRPO可以用当前batch的旧策略概率近似，减少reference model的需求

### Q: 手撕：二叉树的层序遍历？

（编程题）

### Q: 手撕：实现Multi-Head Attention？

（编程题）
