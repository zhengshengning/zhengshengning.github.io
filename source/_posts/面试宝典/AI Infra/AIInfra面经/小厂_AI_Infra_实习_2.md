---
title: '小厂 AI Infra 实习 (2)'
date: 2026-04-16 12:07:24
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 大模型RL全流程？涉及哪些模型？PPO和GRPO的区别？

**RLHF/PPO全流程：**

```
┌──────────────────────────────────────────────────────────────┐
│                      PPO训练循环                               │
│                                                              │
│  ┌──────────┐   生成回复    ┌───────────┐   打分     ┌─────────┐
│  │ Policy   │ ──────────→ │ Reward    │ ────────→ │ PPO     │
│  │ Model    │              │ Model     │           │ Update  │
│  └──────────┘              └───────────┘           └────┬────┘
│       ↑                                                 │
│       │ 更新参数                                         │
│       └─────────────────────────────────────────────────┘
│                                                              │
│  ┌──────────┐   KL散度约束                                    │
│  │Reference │ ← 冻结的初始模型,防止policy偏离太远              │
│  │ Model    │                                                │
│  └──────────┘                                                │
│                                                              │
│  ┌──────────┐   估计Advantage                                │
│  │ Value    │ ← Critic网络,估计每个状态的价值                  │
│  │ Model    │   Advantage = Reward - Value (GAE)             │
│  └──────────┘                                                │
└──────────────────────────────────────────────────────────────┘
```

**4个模型的作用和规模：**

| 模型 | 作用 | 规模 | 训练状态 |
|------|------|------|---------|
| Policy Model | 生成回复(被优化的对象) | 与SFT模型相同(如70B) | 可训练 |
| Reference Model | 计算KL散度约束 | 同Policy(冻结) | 冻结 |
| Reward Model | 对回复质量打分 | 通常较小(如7B) | 冻结 |
| Value Model | 估计状态价值(Critic) | 同Policy或更小 | 可训练 |

**显存需求：** 4个模型 + 优化器状态 → 极其消耗资源。70B Policy需要至少32-64张A100。

**GRPO（Group Relative Policy Optimization）的简化：**

```
PPO:
  Advantage_i = Reward_i - V(state_i)  (需要Value Model估计V)
  
GRPO:
  对同一prompt生成N个回复: {r_1, r_2, ..., r_N}
  对每个回复计算reward: {R_1, R_2, ..., R_N}
  Advantage_i = (R_i - mean(R)) / std(R)  ← 组内相对排名!
  
  不需要Value Model! 用组内比较替代价值估计
```

**PPO vs GRPO对比：**

| 维度 | PPO | GRPO |
|------|-----|------|
| 模型数量 | 4个(Policy+Ref+Reward+Value) | 3个(Policy+Ref+Reward) |
| 显存 | 极大(Value Model额外占用) | 减少~25%(省掉Value) |
| Advantage估计 | Value Model + GAE | 组内相对排名 |
| 训练稳定性 | 需要仔细调参(clip_range等) | 相对稳定(归一化) |
| 样本效率 | 每prompt生成1个回复 | 每prompt生成N个(更多显存换效率) |
| 典型使用 | OpenAI InstructGPT | DeepSeek-R1 |

---

### Q: RL中Rollout耗时占比？Policy MFU和计算公式？6Nd公式是什么？

**Rollout(生成阶段)是RL训练的瓶颈：**

```
RL训练一步的时间分解:
  ┌────────────────────────────────────────────────┐
  │ Rollout(自回归生成)                             │ 60-80%
  │ [tok1][tok2][tok3]...[tokN]  每步都是完整前向   │
  │ 无法并行(自回归), batch内可并行                  │
  ├────────────────────────────────────────────────┤
  │ Reward计算(Reward Model前向)                    │ 5-10%
  ├────────────────────────────────────────────────┤
  │ PPO Update(前向+反向+优化器)                    │ 15-30%
  └────────────────────────────────────────────────┘

为什么Rollout慢?
  生成100个token = 100次独立的前向传播
  每次前向: batch=1(单token), 极度memory-bound
  vs 训练: 一次前向处理整个序列, compute-bound, 效率高
```

**MFU(Model FLOPs Utilization)计算：**

```
MFU = 实际模型计算吞吐 / 硬件峰值FLOPs

计算公式:
  MFU = (tokens_processed_per_second × FLOPs_per_token) / (num_GPUs × peak_FLOPS)
  
  其中:
  FLOPs_per_token ≈ 6N (训练, 含前向+反向)
                  ≈ 2N (纯推理/Rollout, 只有前向)
  N = 模型参数量
  
示例:
  70B模型, 8×A100(每卡312 TFLOPS FP16), 训练时每秒处理1000 tokens
  MFU = (1000 × 6 × 70B) / (8 × 312T) = 420T / 2496T ≈ 16.8%
  
  好的MFU: 训练40-60%, Rollout 5-15%(因为memory-bound)
```

**6Nd公式的推导：**

```
模型参数N, 输入d个token
一次训练step的计算量:

前向传播: 每层Transformer的GEMM
  - QKV投影: 3 × 2Nd_model × d_model × d (3个矩阵)
  - Attention: 2 × d × d × d_model (QK^T + PV)
  - FFN: 2 × 2 × 4d_model × d_model × d (两个FFN矩阵)
  简化: 每层 ≈ 24 × d_model² × d
  L层总计: 24L × d_model² × d ≈ 2Nd (N ≈ 12L × d_model²)
  
反向传播: 约为前向的2倍(需要计算dL/dW和dL/dX)
  反向: 4Nd

总计: 前向(2Nd) + 反向(4Nd) = 6Nd FLOPs

注意:
  6Nd不含attention的QK^T和softmax(对大模型占比<10%)
  实际稍大于6Nd,但作为估算足够准确
```

---

### Q: RL中Rollout有哪些优化点？

**Rollout优化方案全景：**

| 优化方向 | 方法 | 加速比 | 复杂度 |
|---------|------|--------|--------|
| 推理加速 | Rollout量化(INT8/FP8) | 1.5-2x | 低 |
| 并行化 | 多机并行Rollout | 线性扩展 | 中 |
| 调度优化 | 异步Rollout+训练Pipeline | 1.5-2x | 高 |
| 推理引擎 | vLLM/SGLang做Rollout | 2-3x | 中 |
| 算法 | 投机解码 | 1.5-3x | 中 |
| KV复用 | 同Prompt多次采样复用prefix | 1.3-1.5x | 低 |

**详细方案：**

**1. Rollout量化：**
```
训练: Policy Model使用BF16/FP32 (需要精确梯度)
Rollout: 用INT8/FP8的量化Policy做生成 (只需要logits,精度要求低)

实现: 维护两份权重
  - FP16 master weight (用于训练更新)
  - INT8 inference weight (用于Rollout,定期从FP16同步)
  
加速: Decode是memory-bound, INT8权重=一半带宽→接近2x加速
```

**2. 异步Rollout (Pipeline化)：**
```
同步方式:
  [Rollout batch 0] → [Train batch 0] → [Rollout batch 1] → [Train batch 1]
  GPU在训练时空闲于生成, 生成时空闲于训练

异步Pipeline:
  Rollout GPU:  [Gen batch 1] [Gen batch 2] [Gen batch 3] ...
  Training GPU: [         ] [Train batch 0] [Train batch 1] ...
  
  生成下一批样本的同时训练当前批 → 接近2x利用率
  On-policy程度稍降(用旧权重生成), 但实践中可接受
```

**3. vLLM/SGLang集成(如OpenRLHF/veRL)：**
```python
# 传统: 用HuggingFace generate()做Rollout
outputs = model.generate(prompts, max_length=512)  # 慢!无batching优化

# 优化: 用vLLM做Rollout
from vllm import LLM
rollout_engine = LLM(model_path, tensor_parallel_size=4)
outputs = rollout_engine.generate(prompts)  # Continuous batching, PagedAttention
# 吞吐提升3-5x!
```

**4. Prefix KV-Cache复用(GRPO场景)：**
```
GRPO: 对同一prompt生成N=16个回复
传统: 16次独立生成, 每次都从头prefill prompt
优化: Prefill prompt一次, KV-Cache复用给16次decode
  → Prefill开销从 16×cost 降为 1×cost
  → 如果prompt很长(如RAG场景), 节省显著
```

---

### Q: RL中如何把预训练权重同步到推理引擎？

**权重同步的四种方案：**

**方案1: 共享显存(同GPU, 零拷贝)**

```
条件: 训练和推理在同一张GPU上
实现: 推理引擎直接引用训练引擎的weight tensor
  training_model.parameters() → vllm直接加载同一地址
  
优点: 零通信开销, 权重始终最新
缺点: 显存竞争(训练+推理同卡, 显存压力大)
适用: 小模型或显存充裕时
```

**方案2: NCCL传输(跨GPU)**

```
训练GPU (TP=4): 模型分布在4张卡上
推理GPU (TP=2): 推理引擎在另外2张卡上

同步流程:
  1. 训练完一步后, 触发权重同步
  2. All-Gather将分片权重聚合
  3. 通过NCCL Send/Recv传输到推理GPU
  4. 推理GPU resharding为自己的TP分片
  
延迟: 70B模型 × FP16 = 140GB
  NVLink 900GB/s → ~160ms
  InfiniBand 400Gb/s → ~2.8s (跨节点)
  
优化: 分层传输(先传前几层,推理引擎可以开始prefill)
```

**方案3: 异步更新(降低同步频率)**

```
每K步同步一次(如K=4):
  Step 0: 同步权重 → Rollout用最新权重
  Step 1: 训练更新 → Rollout用旧权重(off-policy一点)
  Step 2: 训练更新 → Rollout用旧权重
  Step 3: 训练更新 → Rollout用旧权重
  Step 4: 同步权重 → ...
  
Trade-off:
  K越大: 通信开销分摊更多, 但on-policy程度降低
  实验表明: K=2-8通常对训练质量影响不大
```

**方案4: Checkpoint + Reload(分离部署)**

```
训练集群: 定期保存checkpoint到共享存储(NFS/HDFS)
推理集群: 监听checkpoint更新, 热重载模型权重

优点: 训练和推理完全解耦, 容错性好
缺点: 延迟高(秒-分钟级), 需要大量存储IO
适用: 大规模分布式RL(如数千GPU的训练集群)
```

---

### Q: Megatron中TP是怎么切分的？MLP中两个矩阵分别是行切还是列切？通信算子分别是什么？

**Megatron TP切分Transformer层的核心设计：**

```
设计原则: 最小化通信次数
  每个Transformer层 = Attention + MLP
  目标: 前向+反向总共只需2次AllReduce/layer

MLP切分 (FFN: h→4h→h):
┌─────────────────────────────────────────────────────┐
│ 输入X: [batch, seq, h] (每个rank都有完整输入)        │
│                                                      │
│ 第一层 W1: [h, 4h] → 列切为 [h, 4h/tp]             │
│   Device 0: X × W1_0 = Y_0 [batch, seq, 4h/tp]     │
│   Device 1: X × W1_1 = Y_1 [batch, seq, 4h/tp]     │
│   → 各设备独立计算, 无需通信! (列切=输出分片)         │
│                                                      │
│ GeLU激活: 逐元素, 各设备独立                          │
│                                                      │
│ 第二层 W2: [4h, h] → 行切为 [4h/tp, h]             │
│   Device 0: Y_0 × W2_0 = Z_0 [batch, seq, h]       │
│   Device 1: Y_1 × W2_1 = Z_1 [batch, seq, h]       │
│   → Z = Z_0 + Z_1 → AllReduce!                     │
│                                                      │
│ 通信: 前向1次AllReduce (MLP出口)                     │
└─────────────────────────────────────────────────────┘
```

```
Attention切分:
┌─────────────────────────────────────────────────────┐
│ QKV投影 W_qkv: [h, 3h] → 列切 [h, 3h/tp]          │
│   每个设备得到部分head的Q,K,V                         │
│   独立计算attention(各head完全独立) → 无通信!          │
│                                                      │
│ 输出投影 W_o: [h, h] → 行切 [h/tp, h]              │
│   各设备得到部分结果 → AllReduce求和                   │
│                                                      │
│ 通信: 前向1次AllReduce (Attention出口)               │
└─────────────────────────────────────────────────────┘
```

**通信总结：**

| 位置 | 前向通信 | 反向通信 | 算子 |
|------|---------|---------|------|
| MLP W2输出 | AllReduce | AllReduce(梯度) | NCCL AllReduce |
| Attention W_o输出 | AllReduce | AllReduce(梯度) | NCCL AllReduce |
| **每层总计** | **2次AllReduce** | **2次AllReduce** | - |

**为什么这样切？**

```
列切第一层(W1)的好处:
  X × W1_col = 输出被分片 → 不需要通信
  如果行切W1: 需要先Scatter输入X → 额外通信!

行切第二层(W2)的好处:
  Y_part × W2_row = 部分和 → AllReduce得到完整输出
  如果列切W2: 需要先AllGather上一层的输出 → 额外通信!

关键insight: 列切→行切 的组合只需要一次AllReduce
  反之任何其他组合都需要更多通信
```

---

### Q: 预训练和SFT的loss、数据集有什么区别？

**对比：**

| 维度 | 预训练(Pretrain) | SFT(Supervised Fine-tuning) |
|------|-----------------|---------------------------|
| Loss | 所有token的NTP交叉熵 | 只有assistant token的NTP交叉熵 |
| 数据规模 | TB级(万亿token) | GB级(百万-千万样本) |
| 数据质量 | 混合质量(web crawl) | 高质量(人工标注/筛选) |
| 数据格式 | 纯文本(连续拼接) | 结构化(instruction-response对) |
| 训练目标 | 学习语言+世界知识 | 学习遵循指令+对齐行为 |
| Epoch数 | 1-2 epoch | 2-5 epochs |
| 学习率 | 较大(1e-4~3e-4) | 较小(1e-5~2e-5) |

**Loss Masking的关键区别：**

```python
# 预训练 — 所有token都参与loss:
text = "The cat sat on the mat"
labels = ["cat", "sat", "on", "the", "mat", "<eos>"]  # 全部计算loss
loss = cross_entropy(logits, labels)  # 无mask

# SFT — 只有assistant回复计算loss:
conversation = [
    {"role": "system", "content": "You are helpful."},    # mask掉
    {"role": "user", "content": "What is GPU?"},          # mask掉
    {"role": "assistant", "content": "GPU is a..."},       # 计算loss!
]
# loss_mask = [0,0,0,...,0, 1,1,1,...,1]
#              ^^^system+user^^^ ^^^assistant^^^
loss = (cross_entropy(logits, labels) * loss_mask).sum() / loss_mask.sum()
```

**数据集特点：**

```
预训练数据(如RedPajama/The Pile):
  - Web crawl(CommonCrawl): ~60%
  - 书籍/学术论文: ~15%
  - 代码(GitHub): ~15%
  - 百科/高质量文本: ~10%
  - 清洗: 去重, 有害内容过滤, 质量评分
  
SFT数据(如OpenAssistant/ShareGPT):
  - 对话数据: 人类提问+高质量回答
  - 指令数据: 多样化任务指令
  - 安全数据: 拒绝有害请求的样本
  - 格式: chat template(system/user/assistant结构)
```

---

### Q: 流水线并行怎么做？1F1B和DualPipe的区别？

**流水线并行基础：**

```
模型按层分到P个stage:
  Stage 0: Layer 0-9    (Device 0)
  Stage 1: Layer 10-19  (Device 1)
  Stage 2: Layer 20-29  (Device 2)
  Stage 3: Layer 30-39  (Device 3)

训练一个batch分成M个micro-batch:
  总batch → [micro_0, micro_1, ..., micro_{M-1}]
```

**1F1B (One Forward One Backward)：**

```
M=4 micro-batches, P=4 stages:

Stage 0: [F0][F1][F2][F3][B0][B1][B2][B3]
Stage 1:    [F0][F1][F2][F3][B0][B1][B2][B3]
Stage 2:       [F0][F1][F2][B0][F3][B1][B2][B3]
Stage 3:          [F0][F1][B0][F2][B1][F3][B2][B3]

改进的1F1B(交替前向反向):
Stage 0: [F0][F1][F2][F3][B0][B1][B2][B3]
Stage 1:    [F0][F1][F2][B0][F3][B1][B2][B3]
Stage 2:       [F0][F1][B0][F2][B1][F3][B2][B3]  
Stage 3:          [F0][B0][F1][B1][F2][B2][F3][B3]

关键: Stage 3在完成F0后立即开始B0(不等所有F完成)
好处: 减少内存峰值(不需要同时存所有micro-batch的激活)

Bubble ratio = (P-1) / (P-1+M)
P=4, M=16: bubble = 3/19 ≈ 16%
```

**DualPipe (DeepSeek V3)：**

```
核心idea: 前向和反向双向流水线 + 计算通信重叠

将每个micro-step分为:
  Computation chunk: 纯计算(GEMM, activation)
  Communication chunk: AllReduce/Send-Recv

双向调度:
  正向: F0→F1→F2→F3 (micro-batch从前往后)
  反向: B3→B2→B1→B0 (梯度从后往前)
  
重叠:
  当Stage i做正向计算时, 同时做反向的通信
  当Stage i做反向计算时, 同时做正向的通信
  → 通信被计算完全覆盖(如果计算≥通信时间)

效果:
  - Bubble比1F1B更小
  - 通信开销被隐藏
  - 更高的GPU利用率
  
限制:
  - 实现复杂(需要精确的计算-通信切分)
  - 要求计算和通信可分离
```

---

### Q: DeepSeek论文中FP8训练的关键点？

**FP8训练的核心技术细节：**

**1. Tile-wise Quantization (vs Per-tensor):**

```
Per-tensor: 整个weight tensor用一个scale
  scale = max(|W|) / FP8_MAX
  问题: 如果某个元素特别大, 其他元素的精度全部受损

Tile-wise: 将tensor切成128×128的tile, 每个tile独立scale
  对W[M,N], 共有 (M/128) × (N/128) 个tile
  每个tile: scale_ij = max(|W_tile_ij|) / FP8_MAX
  
  精度提升: tile内max值更接近大多数元素 → 精度损失小
  额外存储: (M/128)×(N/128)个FP32 scale值 → 可忽略
```

**2. 只在GEMM中使用FP8:**

```
FP8使用位置:
  ✓ 线性层(GEMM): W×X → FP8_W × FP8_X → FP32累加
  ✗ LayerNorm: 保持BF16(对数值稳定性敏感)
  ✗ Softmax: 保持FP32(指数函数对精度极敏感)
  ✗ Residual Add: 保持BF16(累积误差)
  ✗ Embedding: 保持BF16
  
原因: GEMM占训练计算量>90%, 只加速GEMM就够了
     非GEMM操作对精度敏感但计算量小, 保持高精度
```

**3. 延迟缩放(Delayed Scaling):**

```
理想: scale = FP8_MAX / max(|current_tensor|)
  问题: 计算max需要全tensor reduction → 额外kernel开销

延迟缩放:
  scale_t = FP8_MAX / amax_{t-1}  (用上一步的max)
  amax_t = max(|tensor_t|)         (同时记录当前max给下一步用)
  
  假设: 连续两步的tensor分布变化不大 → 上一步max是好的近似
  实验: 精度影响 < 0.01 PPL
  
  好处: 
  - amax计算可以与后续计算overlap(异步)
  - 不阻塞当前step的前向传播
```

**4. 关键层保持高精度:**

```
始终使用BF16/FP32的层:
  - Embedding layer: 第一层,误差会传播到所有后续层
  - Final LM head: 输出logits精度直接影响loss计算
  - Attention score: softmax对输入精度敏感
  
实验发现: 这3个地方用FP8会导致训练不稳定/发散
```

**5. 整体训练效率:**

| 指标 | BF16训练 | FP8训练 | 提升 |
|------|---------|---------|------|
| 训练速度(tokens/s) | 基准 | +40-50% | FP8 TC吞吐翻倍(部分) |
| 显存(weight+activation) | 基准 | -25-35% | FP8存储更小 |
| 模型质量 | 基准 | <0.1 PPL差异 | 几乎无损 |
| 额外工程量 | 无 | scale管理/精度监控 | 中等 |

---

### Q: vLLM/SGLang中Continuous Batching是怎么工作的？

**Continuous Batching的完整工作机制：**

```
传统Static Batching:
  Batch 1: [A(20tok), B(50tok), C(10tok), D(30tok)]
  所有请求pad到max_len=50, 直到B完成才处理新请求
  C和D完成后: GPU idle等B → 利用率低

Continuous Batching(Iteration-level scheduling):
  Step 1: batch=[A,B,C,D], 各生成1个token
  Step 10: C完成(10tok)→ 移出, E加入 → batch=[A,B,D,E]
  Step 20: A完成(20tok)→ 移出, F加入 → batch=[B,D,E,F]
  Step 30: D完成(30tok)→ 移出, G加入 → batch=[B,E,F,G]
  ...GPU始终有满载的batch
```

**实现细节：**

```python
# 每个iteration的调度逻辑:
def iteration_step():
    # 1. 收集当前running的decode请求
    decode_reqs = [r for r in running if r.state == DECODE]
    
    # 2. 检查已完成的请求
    finished = [r for r in decode_reqs if r.is_finished()]
    for r in finished:
        running.remove(r)
        free_kv_blocks(r)
        output_queue.put(r.result)
    
    # 3. 检查新请求是否可以加入(资源允许)
    while waiting and has_resources():
        new_req = waiting.pop(0)
        allocate_kv_blocks(new_req)
        running.add(new_req)
        new_req.state = PREFILL  # 新请求先prefill
    
    # 4. 组装混合batch
    prefill_tokens = collect_prefill_tokens()  # 新请求的全部input
    decode_tokens = collect_decode_tokens()    # 旧请求各1个token
    
    # 5. 执行前向(混合prefill+decode)
    logits = model.forward(prefill_tokens + decode_tokens)
    
    # 6. 采样下一个token
    for req, logit in zip(running, logits):
        next_token = sample(logit)
        req.append_token(next_token)
        if req.state == PREFILL:
            req.state = DECODE  # prefill完成转decode
```

**Continuous Batching的性能优势量化：**

| 场景 | Static Batching | Continuous Batching | 提升 |
|------|----------------|--------------------| -----|
| 输出长度均匀 | 基准 | +20-50% | 减少padding浪费 |
| 输出长度差异大(10x) | 基准 | +200-400% | 消除短请求等待 |
| 高并发(>100 QPS) | 可能OOM | 稳定运行 | 动态显存管理 |
| P99延迟 | 差(等最长请求) | 好(各自独立完成) | 显著改善 |

**与Chunked Prefill的配合：**
- 长Prefill分块: 避免一个大prefill独占整个step
- 混合执行: 每step同时处理一个prefill chunk + 多个decode
- 结果: decode延迟保持稳定,prefill不阻塞在线生成
