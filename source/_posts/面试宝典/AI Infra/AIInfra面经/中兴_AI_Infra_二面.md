---
title: '中兴 AI Infra 二面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: 数据并行、模型并行、流水线并行分别是什么？

三种并行策略从不同维度切分计算任务，解决不同瓶颈：

**数据并行（Data Parallelism, DP/DDP）**：

```
GPU 0: 完整模型 + 数据batch 0 → 梯度0 ─┐
GPU 1: 完整模型 + 数据batch 1 → 梯度1 ─┼→ AllReduce → 统一梯度 → 各卡更新
GPU 2: 完整模型 + 数据batch 2 → 梯度2 ─┘
```

- **核心思想**：模型复制N份，数据切分N份，梯度AllReduce同步后各卡独立更新。
- **通信量**：每步AllReduce 2×模型参数量的数据（Ring AllReduce下），与batch size无关。
- **扩展性**：理论上线性加速，实际受通信带宽限制（大模型时通信占比高）。
- **适用条件**：单卡能放下完整模型。ZeRO系列在此基础上分片优化器/梯度/参数以减少显存冗余。

**张量并行（Tensor Parallelism, TP）**：

```
            Input X
           /       \
GPU 0: X×W1[:, :N/2]   GPU 1: X×W1[:, N/2:]    ← 列切分
           \       /
         AllReduce / AllGather
           /       \
GPU 0: Y[:, :N/2]×W2[:N/2, :]   GPU 1: Y[:, N/2:]×W2[N/2:, :]  ← 行切分
           \       /
         AllReduce
            Output
```

- **核心思想**：将单层的权重矩阵切分到多卡，每卡计算部分结果后通信合并。
- **通信频率**：每个Transformer层需要2次AllReduce（attention + FFN各一次），通信极频繁。
- **通信量**：每次AllReduce的数据量 = batch_size × seq_len × hidden_dim × 2bytes(FP16)。
- **适用条件**：节点内NVLink/NVSwitch互联（A100 8卡NVLink带宽600GB/s）。跨节点TP通常太慢。
- **优势**：降低单层的计算延迟（时间 ÷ TP度），适合延迟敏感场景。

**流水线并行（Pipeline Parallelism, PP）**：

```
Stage 0 (GPU 0-1): Layer 0-15    Stage 1 (GPU 2-3): Layer 16-31
     ↓ 激活值传递(仅一次)              ↓ 激活值传递
micro-batch 0: [→计算→]    [      ]    [→计算→]    [      ]
micro-batch 1: [      ]    [→计算→]    [      ]    [→计算→]
...（1F1B调度减少bubble）
```

- **核心思想**：将模型按层分为多个Stage，每Stage放在不同设备，以micro-batch流水执行。
- **通信量**：仅相邻Stage间传递一次激活值（batch_size × seq_len × hidden_dim），通信量小。
- **Pipeline Bubble**：调度不完美导致的空闲时间。1F1B调度下bubble比例 ≈ (pp_degree-1) / (num_microbatch + pp_degree - 1)。
- **适用条件**：跨节点通信（IB带宽足够传激活值），模型层数可均匀划分。
- **劣势**：bubble导致硬件利用率<100%；需要缓存micro-batch的中间激活（显存开销）。

**3D并行组合（大模型训练标配）**：

| 并行维度 | 切分对象 | 通信特征 | 典型范围 |
|---------|---------|---------|---------|
| DP | 数据 | AllReduce，每步一次 | 跨节点 |
| TP | 层内权重 | AllReduce，每层两次 | 节点内(NVLink) |
| PP | 层间 | P2P，相邻stage | 跨节点(IB) |

典型配置（LLaMA-70B训练）：TP=8（节点内8卡NVLink），PP=4（4个节点间），DP=16（16组副本），总GPU=8×4×16=512。

---

### Q: DeepSpeed各stage的特点？

DeepSpeed ZeRO（Zero Redundancy Optimizer）通过逐步分片消除数据并行中的显存冗余：

**显存组成分析（以混合精度训练为例，模型参数量Φ）**：

| 存储项 | 内容 | 每参数字节数 | 示例(7B模型) |
|-------|------|------------|-------------|
| FP16参数 | 前向/反向用 | 2 | 14GB |
| FP16梯度 | 反向计算 | 2 | 14GB |
| FP32主权重 | 优化器更新用 | 4 | 28GB |
| FP32动量(m) | Adam一阶矩 | 4 | 28GB |
| FP32方差(v) | Adam二阶矩 | 4 | 28GB |
| **总计** | | **16** | **112GB** |

**各Stage对比**：

| Stage | 分片内容 | 每卡显存(N卡) | 通信量(vs DDP) | 通信模式 |
|-------|---------|-------------|--------------|---------|
| DDP baseline | 无分片 | 16Φ | 2Φ (AllReduce) | AllReduce |
| **ZeRO-1** | 优化器状态 | 4Φ + 12Φ/N | 2Φ (不变) | AllReduce |
| **ZeRO-2** | 优化器+梯度 | 2Φ + 14Φ/N | 2Φ (不变) | Reduce-Scatter + AllGather |
| **ZeRO-3** | 优化器+梯度+参数 | 16Φ/N | 3Φ (1.5x) | AllGather(前向) + Reduce-Scatter(反向) + AllGather(更新) |

**ZeRO-1详解**：
- 每卡只保存1/N的优化器状态（Adam的m和v以及FP32参数副本）。
- AllReduce梯度后，各卡只更新自己负责的那1/N参数的优化器状态。
- 更新完成后AllGather将更新后的参数广播给所有卡。
- 显存节省：从16Φ降到约4Φ+12Φ/N（8卡时≈5.5Φ，约3倍节省）。

**ZeRO-2详解**：
- 在ZeRO-1基础上，梯度也分片——每卡只保存自己负责的那1/N参数的梯度。
- 反向传播时用Reduce-Scatter（各卡得到不同部分的已规约梯度），而非AllReduce（每卡得到全部梯度）。
- 显存节省：8卡时约2Φ+14Φ/8≈3.75Φ。

**ZeRO-3详解**：
- 连模型参数也分片——每卡只存1/N的模型参数。
- **前向传播**：需要某层参数时AllGather收集完整参数，用完立即释放。
- **反向传播**：同样AllGather参数计算梯度，然后Reduce-Scatter梯度。
- 显存节省：16Φ/N（8卡时仅2Φ，8倍节省）。
- 代价：通信量增加50%（额外的前向AllGather），且前向路径也有通信延迟。

**ZeRO-Offload/Infinity**：
- 将优化器状态和参数卸载到CPU RAM或NVMe SSD。
- CPU Adam：优化器更新在CPU上执行（足够快因为是一阶操作）。
- PCIe带宽（Gen4: 32GB/s）成为瓶颈，适合资源受限场景。

**选择建议**：

| 场景 | 推荐 | 原因 |
|------|------|------|
| 模型能放单卡 | DDP或ZeRO-1 | 简单高效 |
| 优化器状态放不下 | ZeRO-1/2 | 通信量不增 |
| 模型参数放不下 | ZeRO-3或TP | 选择取决于互联带宽 |
| GPU极有限 | ZeRO-Offload | 用CPU/NVMe扩展容量 |

---

### Q: 给定200B模型，16张GPU如何设计分布式训练？

**问题分析**：

200B参数的显存需求估算：
- FP16参数：200B × 2 = 400GB
- 梯度（FP16）：400GB
- 优化器状态（FP32 param + m + v）：200B × 12 = 2400GB
- 总：3200GB，16张80GB GPU总共1280GB（不够）

结论：**必须使用分片策略**，且单层参数也可能超单卡（取决于hidden_size）。

**推荐设计方案**：

假设硬件拓扑：2个节点，每节点8卡A100-80GB，节点内NVLink（600GB/s），节点间4×IB HDR（800Gbps ≈ 100GB/s）。

```
节点0 (8×A100, NVLink互联)     节点1 (8×A100, NVLink互联)
┌─────────────────────┐      ┌─────────────────────┐
│  TP=8 (一个TP group)  │      │  TP=8 (一个TP group)  │
│  Stage 0: Layer 0-39 │◄─IB─►│  Stage 1: Layer 40-79 │
└─────────────────────┘      └─────────────────────┘
        PP=2 (跨节点流水线并行)
```

**参数选择**：
- **TP=8**：节点内8卡NVLink互联，TP通信频繁但带宽充足。每层参数200B/80层≈2.5B/层，切8份后每卡约312M/层。
- **PP=2**：跨2节点，各负责40层。PP通信少（只传激活值），IB带宽足够。
- **DP=1**：16卡全用于模型并行（TP×PP=16），无数据并行。

**显存估算（每卡）**：
- 参数：400GB / 16 = 25GB（ZeRO-3全分片）
- 不用ZeRO-3（纯TP+PP）：400GB / 16 = 25GB参数 + 梯度25GB + 优化器150GB = 200GB（超出80GB！）
- **必须配合ZeRO-1**：优化器分片后每卡约25+25+150/8≈69GB，接近但可行。

**激活显存管理**：
- 激活检查点（Activation Checkpointing）：只保存每N层的输入激活，反向时重计算中间激活。
- 选择性checkpointing：只对显存大的attention层做checkpoint，小的FFN保留。
- 预计激活显存（checkpointing后）：约5-10GB/卡（取决于micro-batch size和序列长度）。

**训练配置**：
- Micro-batch size：1-2（受显存限制）
- Gradient accumulation：16-32步（凑成合理的global batch）
- Global batch：1×16×32 = 512 samples（或等效token数）
- Sequence length：2048-4096
- 混合精度：BF16前向反向 + FP32优化器

**性能预估**：
- 有效MFU（Model FLOPs Utilization）：约40-50%（受PP bubble和通信影响）
- PP bubble比例：(2-1)/(32+2-1) ≈ 3%（micro-batch数足够时可忽略）
- TP通信占比：约10-15%（NVLink下较小）

---

### Q: 混合精度训练的原理？

混合精度训练（Mixed Precision Training）通过在不同计算阶段使用不同精度，兼得速度和精度：

**为什么不能纯FP16训练**：

| 问题 | 原因 | 后果 |
|------|------|------|
| 梯度下溢 | FP16最小正数≈6e-8，梯度小于此则变0 | 权重不更新，训练停滞 |
| 权重更新消失 | FP16精度仅1e-3，lr×grad < 1e-3时更新被舍入吞掉 | 相当于学习率实际降低 |
| 累加误差 | 大量FP16数累加精度快速下降 | Reduce/BN/LayerNorm结果不准 |

**混合精度训练流程**：

```
FP32主权重 W_fp32 (存储+更新)
    ↓ cast
FP16参数 W_fp16 (前向计算)
    ↓ 前向
FP16激活 (节省显存)
    ↓ 反向
FP16梯度 g_fp16
    ↓ unscale (÷ loss_scale)
FP32梯度 g_fp32
    ↓ Adam更新
FP32主权重 W_fp32 (更新后)
```

**Loss Scaling机制详解**：

问题：FP16梯度可表示的最小正值为~6e-8，训练后期很多梯度小于此值而下溢为0。

解决：将loss乘以一个大的scale factor（如2^16=65536），等效于将梯度放大到FP16可表示范围内。优化器更新前再除以相同factor恢复原始值。

```python
scaler = torch.cuda.amp.GradScaler(init_scale=2**16)
with torch.cuda.amp.autocast(dtype=torch.float16):
    loss = model(input)
scaled_loss = scaler.scale(loss)  # loss × scale
scaled_loss.backward()            # 梯度也被放大
scaler.unscale_(optimizer)        # 梯度 ÷ scale
scaler.step(optimizer)            # 检查inf/nan，有则跳过
scaler.update()                   # 动态调整scale
```

**动态Loss Scaling策略**：
- 连续N步（如2000步）无overflow → scale ×2（增大scale提高梯度精度）。
- 出现overflow（梯度中有inf/nan）→ scale ÷2，跳过本步更新。
- 实践中scale最终稳定在某个值附近（通常2^10-2^20之间）。

**FP16 vs BF16**：

| 特性 | FP16 | BF16 |
|------|------|------|
| 指数位/尾数位 | 5/10 | 8/7 |
| 数值范围 | ±65504 | ±3.4e38（≈FP32） |
| 精度 | 较高（~3位有效十进制） | 较低（~2位） |
| 是否需要Loss Scaling | 必须 | 通常不需要（范围够大） |
| 硬件支持 | 所有GPU | Ampere+(A100/H100) |

BF16优势：与FP32相同的数值范围，梯度几乎不会下溢，简化训练流程。LLM训练中BF16已成标配。

**显存节省分析**（模型参数量Φ）：

| 存储项 | 纯FP32 | 混合精度 | 节省 |
|-------|--------|---------|------|
| 参数 | 4Φ | 2Φ(FP16) + 4Φ(FP32主权重) | -2Φ |
| 梯度 | 4Φ | 2Φ(FP16) | 2Φ |
| 激活 | 4Φ_act | 2Φ_act | 2Φ_act |
| 优化器 | 8Φ | 8Φ(始终FP32) | 0 |
| **总** | **16Φ+4Φ_act** | **16Φ+2Φ_act** | **激活减半** |

实际节省主要在激活值（占比大），总显存约为纯FP32的60-70%。

---

### Q: 如何加速大模型推理？

大模型推理面临的核心矛盾：Decode阶段每步只生成1个token但需读取全部模型权重，导致GPU算力严重浪费（计算利用率<5%）。加速技术从多个层面解决：

**1. KV Cache（消除重复计算）**：

- 原理：缓存所有历史token的Key和Value，每步decode只计算新token的QKV。
- 复杂度：从O(n^2)降为O(n)每步（但总KV Cache大小仍O(n)增长）。
- 显存公式：`KV_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × 2bytes(FP16)`
- 示例（LLaMA-70B, seq=4K）：2×80×8×128×4096×2 = 13.1GB/请求。

**2. 量化（减少带宽需求）**：

| 方案 | 加速比(vs FP16) | 精度损失 | 实现 |
|------|---------------|---------|------|
| W8A8 | 1.5-2x | <1% perplexity | SmoothQuant + INT8 Tensor Core |
| W4A16 | 1.5-2x | 0.5-2% | GPTQ/AWQ + FP16计算 |
| W4A8 | 2-3x | 1-3% | 组合方案 |
| FP8 | 1.5-2x | <0.5% | H100原生支持 |

Decode阶段是memory-bound，权重量化直接减少带宽需求从而加速。

**3. FlashAttention（减少HBM访问）**：

- Prefill加速：分块计算避免N×N矩阵存HBM，I/O从O(N^2)降为O(N^2/SRAM_size)。
- A100实测：Prefill加速2-4x，Decode阶段因seq已缓存在KV Cache，加速较小。
- FlashDecoding：针对Decode阶段的优化变体，将KV Cache在序列维度split给多个thread block并行。

**4. PagedAttention（内存效率）**：

- 传统方式：按最大序列长度预分配KV Cache，浪费50-80%显存。
- PagedAttention：按需分块分配，利用率接近100%。
- 效果：同样80GB显存，并发请求数从5→20+（4倍吞吐提升）。

**5. Continuous Batching（调度效率）**：

- 传统静态Batch：所有请求等最长的完成，短请求GPU时间被浪费。
- Continuous Batching：每步decode后移出完成的请求，立即填入新请求。
- 效果：GPU利用率从30-50%提升到80%+，吞吐提升2-3倍。

**6. 投机解码（Speculative Decoding）**：

- 原理：小模型（draft model, ~1B参数）快速生成K个候选token，大模型一次性验证（并行检查K个位置）。
- 加速原理：大模型验证K个token的时间≈生成1个token（同样的权重读取），接受率70-90%时等效加速2-3倍。
- 变体：Medusa（多MLP头预测）、EAGLE（自回归draft）、MTP（额外预测头）。

**7. 张量并行（降低延迟）**：

- 2-8卡TP：单次decode延迟线性降低（每卡读取的权重量减少）。
- 适用：对延迟敏感的实时应用（如对话）。
- 代价：多卡通信（但NVLink延迟<5us，可接受）。

**8. 算子融合（减少overhead）**：

- 典型融合：QKV投影三合一、Add+LayerNorm、GEMM+Bias+Activation。
- 效果：减少kernel launch开销（每次~5us）、消除中间tensor的HBM写入/读取。
- Decode阶段小tensor特别受益（kernel launch开销占比高）。

**9. PD分离（Prefill-Decode Disaggregation）**：

- 问题：Prefill是compute-bound（高算力需求），Decode是memory-bound（高带宽需求），混合调度两者都不优。
- 解决：用不同配置的GPU分别服务Prefill和Decode。
- Prefill节点：少量高算力GPU，大batch处理长prompt。
- Decode节点：多个高带宽GPU，服务大量并发decode请求。
- 效果：整体资源利用率提升30-50%。

---

### Q: KV Cache如何优化？

KV Cache是LLM推理的最大显存消费者，直接决定了系统能服务的并发数和最大序列长度：

**显存占用分析**（LLaMA-70B, FP16, 每请求）：

```
KV Cache = 2(K+V) × 80层 × 8个KV头 × 128头维度 × seq_len × 2字节
         = 2 × 80 × 8 × 128 × 2 × seq_len bytes
         = 327,680 × seq_len bytes
         ≈ 0.31MB per token
```

4K序列：1.25GB/请求；128K序列：40GB/请求。80GB GPU在70B模型下（参数占~35GB），KV Cache仅能支持~35个4K请求并发。

**优化技术详解**：

**1. PagedAttention（内存管理层）**：
- 将KV Cache按fixed-size block分配（如每block 16 tokens = 16×128×2×2×head_num bytes）。
- Block Table维护逻辑位置→物理block映射，类似OS页表。
- 效果：显存碎片率从50-80%降到<5%，有效容量翻倍。
- Copy-on-Write：多请求共享前缀block，仅在修改时复制（如beam search的多beam共享前缀）。

**2. GQA/MQA（架构层减少KV量）**：

| 方案 | KV头数 | KV Cache大小(vs MHA) | 代表模型 |
|------|--------|---------------------|---------|
| MHA | 等于Q头数(如64) | 100% | GPT-3 |
| GQA | Q头数的1/4-1/8(如8) | 12.5% | LLaMA-2/3, Qwen2 |
| MQA | 仅1个KV头 | 1.5% | PaLM, Falcon |
| MLA | 压缩到低维潜在空间 | ~5% | DeepSeek-V2/V3 |

GQA实测：64Q头配8KV头，KV Cache减少8倍，精度损失<0.5%。

**3. KV Cache量化**：
- INT8量化：KV值范围有限（通常[-3,3]），INT8量化几乎无损。显存减半。
- FP8（E4M3）：H100+支持，比INT8保留更多动态范围。
- 实现：存储时量化（per-head或per-token scale），attention计算前反量化。
- 效果：同样显存下并发翻倍，perplexity增加<0.1。

**4. 滑动窗口注意力（架构层）**：
- 只保留最近W个token的KV Cache（如Mistral W=4096）。
- KV Cache大小恒定，不随序列增长。
- 需要训练时就使用窗口注意力（推理时不能直接应用于全注意力模型）。
- 适合：流式对话场景，不需要回看很远的历史。

**5. Prefix Caching（复用层）**：
- 对相同system prompt的多请求，KV Cache计算一次后缓存。
- 实现：对prefix文本hash，命中则直接引用已有block。
- 效果：1000个使用相同2K system prompt的请求，省去1000×2K次prefill计算。
- vLLM的APC（Automatic Prefix Caching）自动检测和缓存。

**6. KV Cache压缩/稀疏化**：
- **H2O（Heavy Hitter Oracle）**：保留注意力权重最大的top-K个token的KV + 最近N个token。
- **Scissorhands**：类似思路，丢弃低注意力token。
- **StreamingLLM**：保留开头的attention sink token（4个）+ 最近窗口。
- 效果：KV Cache可压缩到20-50%而精度损失可控。
- 注意：这些方法可能影响长距离依赖的任务（如多跳推理）。

**7. Offloading（容量扩展）**：
- 不活跃请求的KV Cache卸载到CPU RAM（PCIe Gen4: 32GB/s）。
- 请求重新激活时加载回GPU。
- 适合：请求间隔较长的交互式场景。
- 局限：加载延迟约1-5ms，不适合低延迟要求。

---

### Q: 算子融合是什么？有什么好处？

算子融合（Operator Fusion / Kernel Fusion）是推理和训练优化中最有效的技术之一，将多个独立kernel合并为一个kernel执行：

**为什么融合有效——根本原因分析**：

```
未融合: GEMM(A,B) → [写HBM] → [读HBM] → BiasAdd → [写HBM] → [读HBM] → ReLU → [写HBM]
融合后: FusedGEMM_Bias_ReLU(A,B) → [写HBM]（中间结果全在寄存器/shared memory中）
```

每次HBM读写在A100上耗费：数据量 / 2TB/s。对于一个[4096,4096]的FP16 tensor，单次读写约16us。未融合时3个算子需6次HBM访问（3读+3写），融合后仅2次（读输入+写最终输出），**节省67%的内存带宽**。

**量化好处**：

| 优化项 | 未融合开销 | 融合后开销 | 节省 |
|-------|-----------|-----------|------|
| Kernel Launch | N × 5us | 1 × 5us | (N-1) × 5us |
| 中间Tensor HBM读写 | (N-1) × 2 × data_size/BW | 0 | ~毫秒级(大tensor) |
| 内存分配 | (N-1) × alloc overhead | 0 | 减少显存碎片 |
| GPU idle gap | N-1 个gap(~1-3us) | 0 | 消除空闲 |

**典型融合模式**：

**1. GEMM Epilogue融合**（最常见、收益最大）：
```
GEMM → Bias → Activation(ReLU/GeLU/SiLU)
```
CUTLASS/cuBLAS的epilogue机制：GEMM计算完每个输出tile后，在写回HBM前直接在寄存器中执行bias add和activation。零额外访存成本。

**2. Attention融合（FlashAttention）**：
```
Q×K^T → Scale → Mask → Softmax → ×V
```
5个独立操作融合为一个kernel，避免N×N的注意力矩阵写入HBM。这是FlashAttention的核心思想。

**3. Normalization融合**：
```
Residual Add → LayerNorm → (下一层输入)
```
残差加法和LayerNorm的reduce操作可在一个kernel中完成，avoid中间tensor的HBM写入。

**4. QKV投影融合**：
```
Q = X × W_Q,  K = X × W_K,  V = X × W_V
→ [Q, K, V] = X × [W_Q; W_K; W_V]  (单次大GEMM)
```
三次GEMM合并为一次（batch GEMM），提高compute intensity和Tensor Core利用率。

**5. SwiGLU融合**：
```
FFN: SiLU(X×W_gate) × (X×W_up)
→ Fused: 两个GEMM + SiLU + 逐元素乘 在一个kernel中
```

**融合可能失败的场景**：
- 融合后寄存器压力过大→occupancy暴跌→延迟隐藏不足→反而更慢。
- 两个op的最优block size/tiling不兼容。
- 数据依赖模式不允许（如op2需要op1的全局结果才能开始）。
- 实践中需要benchmark验证融合确实有收益。
