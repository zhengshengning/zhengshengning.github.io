---
title: '米哈游 AI Infra 实习 二面'
date: 2026-04-16 11:45:50
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 面经]
---


<!-- more -->

---

### Q: 如何评估大模型的性能？有哪些常用的评估指标？

大模型评估分为**系统性能指标**（Infra 关注）和**模型质量指标**（算法关注）两大维度：

**一、系统性能指标**：

| 指标 | 含义 | 典型值 | 优化手段 |
|---|---|---|---|
| **TTFT** (Time to First Token) | 从请求到达到输出第一个 token 的延迟 | 100ms-2s | 优化 Prefill、Prefix Caching |
| **TPOT** (Time Per Output Token) | 生成阶段每个 token 的平均延迟 | 20-100ms | 量化、Continuous Batching |
| **ITL** (Inter-Token Latency) | 相邻 token 间的实际延迟（P99） | 30-150ms | PD 分离、减少调度抖动 |
| **Throughput** | 系统级吞吐 | 1000-10000 tokens/s | 大 batch、量化、高效调度 |
| **QPS** (Queries Per Second) | 每秒处理的请求数 | 10-100 req/s | Continuous Batching |
| **GPU 利用率** | SM active 时间占比 | 目标 > 70% | 减少 bubble、overlap 通信 |
| **显存利用率** | KV Cache + 模型权重 / 总显存 | 目标 > 80% | PagedAttention |

**TTFT 和 TPOT 的关系**：
```
用户感知的总延迟 = TTFT + TPOT × (output_length - 1)

对话场景 (短输出 ~50 tokens): TTFT 主导体验
长文本生成 (长输出 ~2000 tokens): TPOT 主导体验

SLA 示例: TTFT < 500ms (P99), TPOT < 50ms (P99)
```

**二、模型质量指标**：

| 类别 | 指标 | 评估内容 | 评分方式 |
|---|---|---|---|
| 语言建模 | **PPL** (Perplexity) | 预测能力 | 越低越好，FP16 基准对比 |
| 知识 | **MMLU** | 57 学科选择题 | 准确率 (%) |
| 数学 | **GSM8K** / MATH | 数学推理 | 准确率 (%) |
| 代码 | **HumanEval** / MBPP | 代码生成 | pass@k |
| 对话 | **MT-Bench** | 多轮对话质量 | GPT-4 打分 1-10 |
| 长文本 | **RULER** / Needle-in-Haystack | 长上下文利用 | 准确率 (%) |
| 安全 | **TruthfulQA** | 事实性 | 准确率 (%) |
| 综合 | **OpenCompass** | 多维度综合 | 加权分数 |

**量化后的评估策略**：
- 关注 PPL 增加量（Δ PPL < 0.3 通常可接受）
- 关注 benchmark 下降幅度（< 2% 通常可接受）
- 特别关注数学和代码任务（对精度最敏感）

---

### Q: 如何对大模型进行优化以提高性能和效率？

大模型优化是一个多层次的系统工程，需要从模型、系统、算子三个维度协同：

**模型层面优化**：

| 方法 | 原理 | 典型加速 | 精度影响 |
|---|---|---|---|
| W4A16 量化 | 权重 INT4，减少 4x 读取量 | Decode 2-3x | PPL +0.3-1.0 |
| W8A8 量化 | 权重+激活 INT8，INT8 TC 加速 | Prefill 1.5-2x | PPL +0.1-0.3 |
| FP8 量化 | Hopper 原生支持 | 训练+推理 1.5-2x | PPL +0.05-0.15 |
| GQA/MLA | 减少 KV head / 低秩压缩 | KV Cache 减少 4-50x | 训练时决定 |
| MoE | 稀疏激活 | 同算力下参数量 ×5-10 | 需要 load balance |
| 知识蒸馏 | 大模型教小模型 | 小模型即可达标 | 取决于蒸馏质量 |

**系统层面优化**：

```
推理系统优化栈:

调度层:
  ├─ Continuous Batching (动态 batch，不等最长序列)
  ├─ PD 分离 (Prefill/Decode 部署到不同 GPU)
  ├─ Prefix Caching (共享前缀避免重复计算)
  └─ 投机解码 (小模型 draft + 大模型 verify)

显存管理层:
  ├─ PagedAttention (KV Cache 分页，消除碎片)
  ├─ KV Cache 量化 (FP8/INT4 减少存储)
  └─ KV Cache Offloading (CPU/SSD 卸载)

计算层:
  ├─ FlashAttention (IO-aware Attention，减少 HBM 读写)
  ├─ 算子融合 (减少 kernel launch + 中间 tensor 读写)
  └─ Tensor Parallel (多卡并行加速单请求)
```

**算子层面优化**：
- **GEMM 优化**：Tiling + Shared Memory + Tensor Core + 向量化
- **Attention 优化**：FlashAttention（tiling + online softmax）
- **Elementwise 融合**：LayerNorm + Residual + Linear 融合为单 kernel
- **Communication overlap**：All-Reduce 与 compute 重叠

**端到端优化决策树**：
```
目标: 降低 TTFT
  → Prefix Caching（减少重复 Prefill）
  → 更大 TP 度（单请求并行更多 GPU）
  → FP8 Prefill（加速矩阵乘）

目标: 降低 TPOT
  → W4A16 量化（减少权重读取）
  → 更大 batch（提高带宽利用率）
  → 投机解码（减少 decode 步数）

目标: 提高吞吐
  → Continuous Batching + PagedAttention
  → PD 分离（各自优化利用率）
  → 量化 + 大 batch
```

---

### Q: 大模型中注意力机制是如何工作的？起什么作用？

**注意力机制的完整工作流程**（Multi-Head Self-Attention）：

```
输入 X [batch, seq_len, d_model]
    ↓
线性投影: Q = X × W_Q, K = X × W_K, V = X × W_V
    ↓ reshape 为 [batch, num_heads, seq_len, head_dim]
    ↓
Scaled Dot-Product Attention (每个 head 独立):
    Score = Q × K^T / √d_k     [batch, heads, seq, seq]
        ↓
    + Causal Mask (下三角)       (Decoder: -inf 屏蔽未来 token)
        ↓
    Attn_weights = softmax(Score)  [batch, heads, seq, seq]
        ↓
    Output = Attn_weights × V      [batch, heads, seq, head_dim]
    ↓
reshape + 输出投影: O = concat(heads) × W_O  [batch, seq, d_model]
```

**为什么需要 √d_k 缩放**：
- Q 和 K 的每个元素独立标准正态分布时，Q×K^T 的方差 = d_k
- 不缩放 → Score 方差大 → softmax 趋向 one-hot（梯度消失）
- 除以 √d_k → Score 方差 = 1 → softmax 输出分布合理

**注意力的核心作用**：

1. **动态路由（最核心）**：
   - MLP 的权重是固定的（对所有输入相同）
   - Attention 的权重是**输入依赖的**（每个 token 根据内容决定关注谁）
   - 这使得模型能动态选择性地聚合信息

2. **全局依赖建模**：
   - 第 1 个 token 和第 1000 个 token 通过一次 attention 直接交互
   - 对比 CNN（局部感受野）和 RNN（信息逐步传递衰减）

3. **多头并行捕获不同模式**：
   ```
   Head 0: 可能学习语法关系（主语-谓语）
   Head 1: 可能学习共指关系（代词-实体）
   Head 5: 可能学习位置关系（相邻 token）
   Head 12: 可能学习语义相似性
   ```

4. **信息聚合与传递**：
   - 每一层的 attention 输出是 V 的加权组合
   - 经过多层后，每个位置的表示融合了全序列的信息
   - 残差连接确保原始信息不丢失

---

### Q: 大模型中的优化算法有哪些？各有什么优缺点？

**训练优化器对比**（以 LLM 训练为场景）：

| 优化器 | 核心思想 | 显存开销/参数 | 优缺点 |
|---|---|---|---|
| **SGD + Momentum** | 梯度 + 动量平滑 | +4 bytes (momentum) | 简单稳定，但收敛慢、需精细调 lr |
| **Adam** | 一阶矩(m) + 二阶矩(v) 自适应 | +8 bytes (m + v) | 收敛快，LLM 标配 |
| **AdamW** | Adam + 解耦 weight decay | +8 bytes | 修正 Adam 中 WD 与自适应 lr 的耦合 |
| **Adafactor** | 将 m/v 分解为行列向量 | +4~5 bytes | 显存省 ~30%，大模型可用 |
| **Lion** | 只用 sign(momentum) 更新 | +4 bytes | 显存比 Adam 小 50%，部分任务效果接近 |
| **LAMB/LARS** | 层级自适应学习率 | +8 bytes | 支持极大 batch (>32K) 训练 |
| **8-bit Adam** | 量化 m 和 v 为 INT8 | +2 bytes | 显存减 75%，精度几乎无损 |

**AdamW 详解（LLM 训练标配）**：
```
更新规则:
  m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        (一阶矩估计)
  v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²       (二阶矩估计)
  m̂_t = m_t / (1 - β₁^t)                      (偏差修正)
  v̂_t = v_t / (1 - β₂^t)
  θ_t = θ_{t-1} - lr × (m̂_t / (√v̂_t + ε) + λ × θ_{t-1})
                                                ↑ 解耦的 weight decay

典型超参: β₁=0.9, β₂=0.95, ε=1e-8, lr=3e-4, λ=0.1
```

**显存占用分析**（以 7B 参数 FP16 模型为例）：
```
模型参数:     7B × 2 bytes = 14 GB
梯度:         7B × 2 bytes = 14 GB (FP16)
Adam m:       7B × 4 bytes = 28 GB (FP32)
Adam v:       7B × 4 bytes = 28 GB (FP32)
FP32 主权重:  7B × 4 bytes = 28 GB
─────────────────────────────────
总计: 112 GB（单卡放不下，需要 ZeRO）
```

**选择建议**：
- 标准 LLM 预训练 → **AdamW**（成熟、稳定）
- 显存极度紧张 → **Adafactor** 或 **8-bit Adam**
- 超大 batch 分布式 → **LAMB**（自动调节层级 lr）
- 实验探索 → **Lion**（更新更简单，可能在某些任务优于 Adam）

---

### Q: 如何处理大模型训练中的梯度消失或梯度爆炸问题？

**梯度消失**（深层网络中梯度逐层指数衰减，底层几乎不更新）：

**根本原因**：反向传播中梯度 = 链式法则的连乘。如果每层的梯度乘子 < 1，经过 N 层后梯度 → 0。

| 解决方法 | 原理 | 效果 |
|---|---|---|
| **残差连接** | ∂L/∂x = ∂L/∂(x+F(x)) × (1 + ∂F/∂x)，至少有常数 1 项 | 最关键，使 100+ 层网络可训练 |
| **Pre-Norm** | Norm 在子层之前，梯度不经过 Norm 的缩放 | 比 Post-Norm 梯度流更稳定 |
| **ReLU/GELU/SiLU** | 正区间梯度 ≈ 1（vs Sigmoid 最大梯度 0.25） | 避免饱和区梯度为 0 |
| **Xavier/Kaiming 初始化** | 保持每层输出方差 = 1 | 防止信号逐层衰减 |
| **RMSNorm** | 归一化每层输出 | 保持每层输入分布稳定 |

**残差连接为什么有效的直觉**：
```
无残差: ∂L/∂x₁ = ∂L/∂x_N × ∂x_N/∂x_{N-1} × ... × ∂x₂/∂x₁
        = ∂L/∂x_N × ∏(各层 Jacobian)  → 连乘趋向 0 或 ∞

有残差: ∂L/∂x₁ = ∂L/∂x_N × (I + ∂F_N/∂x_{N-1}) × ... × (I + ∂F₂/∂x₁)
        展开后至少包含 ∂L/∂x_N 这一项（直接路径）
        → 梯度不会完全消失
```

---

**梯度爆炸**（梯度逐层指数增长，参数更新过大导致训练不稳定）：

**根本原因**：梯度链式乘积中每层乘子 > 1，经过多层后梯度 → ∞。表现为 loss spike 或 NaN。

| 解决方法 | 原理 | 典型配置 |
|---|---|---|
| **梯度裁剪 (Gradient Clipping)** | 限制梯度范数不超过阈值 | `max_norm=1.0` |
| **学习率 Warmup** | 初期小 lr，梯度大也不会造成大跳跃 | 前 1-5% steps 线性增长 |
| **Weight Decay** | 限制权重范数增长 | λ = 0.01-0.1 |
| **Loss Scaling** | 混合精度中放大 loss 防止 FP16 梯度下溢 | dynamic scaling |
| **RMSNorm** | 约束中间激活的范数 | 标准做法 |
| **μP (Maximal Update)** | 不同宽度层使用不同 lr 缩放 | 前沿研究 |

**梯度裁剪的两种方式**：
```python
# 方式 1: 按范数裁剪（更常用）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 将全局梯度缩放为: grad = grad × min(1, max_norm / ||grad||)

# 方式 2: 按值裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
# 逐元素裁剪到 [-0.5, 0.5]
```

**LLM 训练中的实践**：
- 几乎所有 LLM 训练都同时使用 Pre-Norm + 残差连接 + 梯度裁剪 + Warmup
- Loss spike 是常见问题：通常通过 skip batch（丢弃异常梯度的 step）处理
- DeepSeek 等报告中提到每 ~100B tokens 训练可能出现 1-3 次 loss spike

---

### Q: 如何权衡模型复杂度和性能？

这是 LLM 设计的核心决策，需要在**参数效率**、**计算效率**和**最终效果**之间找到最优点。

**Scaling Law 指导决策**：

Chinchilla Scaling Law：给定计算预算 C，最优配置满足：
```
最优模型参数 N ∝ C^0.5
最优训练 tokens D ∝ C^0.5
即 N ≈ D/20（每个参数应该看 ~20 tokens）

实际含义:
  预算允许训练 1T tokens → 最优模型约 50B 参数
  预算允许训练 15T tokens → 最优模型约 750B 参数
```

但实践中（DeepSeek/LLaMA-3）选择训练更多 token、更小模型——因为推理成本比训练成本更重要。

**关键权衡策略**：

| 策略 | 原理 | 适用场景 |
|---|---|---|
| **MoE (混合专家)** | 总参数 671B 但每 token 只激活 37B | 需要大容量但计算受限 |
| **知识蒸馏** | 大模型 teacher → 小模型 student | 部署预算固定 |
| **Early Exit** | 简单样本在浅层即可输出 | 输入难度差异大 |
| **LoRA/QLoRA** | 冻结主模型，只训练低秩增量 | 微调场景、显存受限 |
| **混合精度层** | 关键层 FP16，非关键层 INT8/INT4 | 精度敏感层不一致 |
| **Dynamic Computation** | 根据输入复杂度调整计算量 | 需要自适应推理 |

**MoE 的具体权衡**（以 DeepSeek-V3 为例）：
```
Dense 模型: 37B 参数, 每 token 计算 37B params 的 FLOPs
MoE 模型:   671B 参数 (256 expert + shared), 每 token 只激活 37B

优势:
  - 知识容量 ×18（更多参数 = 更多知识存储）
  - 计算量不变（每 token 同等 FLOPs）
  - 同等 FLOPs 下效果显著优于 Dense

代价:
  - 显存：671B × 0.5 bytes(INT4) = 84 GB（仍需多卡）
  - 通信：Expert Parallel 需要 All-to-All
  - 负载均衡：router 可能导致部分 expert 过载
```

---

### Q: 面对大模型训练和推理的庞大计算资源，有什么解决建议？

**训练阶段——资源优化路径**：

```
单卡放不下模型:
  ├─ ZeRO-1: 分片优化器状态 → 显存 ÷ N
  ├─ ZeRO-2: + 分片梯度 → 显存 ÷ N
  ├─ ZeRO-3: + 分片参数 → 显存 ÷ N (几乎无限制)
  └─ 梯度 Checkpoint: 时间换空间，激活显存 ÷ √N

单卡太慢:
  ├─ Tensor Parallel: 切分矩阵到多 GPU (通信密集，需 NVLink)
  ├─ Pipeline Parallel: 切分层到多 GPU (有 bubble，需调度)
  ├─ Data Parallel: 相同模型多 GPU 数据并行 (通信 AllReduce)
  └─ Sequence Parallel: 长序列切分到多 GPU

精度与速度:
  ├─ BF16 混合精度: 2x 加速，精度无损
  ├─ FP8 训练 (H100): 4x 计算吞吐
  └─ Gradient Compression: 减少通信量
```

**推理阶段——资源优化路径**：

```
单卡推理太慢:
  ├─ 模型量化: W4A16 (权重减 4x, decode 加速 2-3x)
  ├─ 投机解码: 小模型 draft + 大模型 verify (2-3x 加速)
  └─ TP 多卡: 拆分到 2-8 GPU 并行 (线性加速)

系统吞吐不够:
  ├─ Continuous Batching: 动态 batch，GPU 不空闲
  ├─ PagedAttention: KV Cache 利用率 ≈ 100%
  ├─ PD 分离: 各阶段最优硬件配置
  └─ Prefix Caching: 共享前缀避免重复计算

成本优化:
  ├─ 模型蒸馏: 训练小模型替代大模型
  ├─ 异构部署: 不同请求类型使用不同大小模型
  ├─ Batch 调度优化: 最大化 GPU 利用率
  └─ Spot Instance: 使用竞价实例降低成本
```

**实际案例——DeepSeek-V3 的训练优化**：
- 2048 GPU × H800，训练 14.8T tokens
- FP8 混合精度训练（节省 50% 通信量）
- DualPipe（双向流水线）减少 PP bubble
- DeepEP 自定义 MoE 通信库
- 总训练成本仅 $5.6M（远低于同等规模的其他模型）

---

### Q: Reward Model 分哪几类？如何训练？

**Reward Model 的三大分类**：

| 类型 | 代表 | 输出 | 优缺点 |
|---|---|---|---|
| **Generative RM** | GPT-4-as-Judge, Prometheus | 自然语言评分+理由 | 灵活可解释，但慢+贵+不稳定 |
| **Discriminative RM** | InternLM-Reward, ArmoRM | 标量分数 (scalar head) | 快、确定性强，但需训练数据 |
| **Implicit RM** | DPO 训练后的模型 | log_prob 差值即 reward | 无需单独训练 RM，但信号较弱 |

**Discriminative RM 训练流程**：

```
数据准备:
  (prompt, response_chosen, response_rejected) × 100K-1M 对

模型架构:
  LLM backbone + 最后一层后接 scalar head (Linear → 1 维标量)
  
  输入: [prompt; response] → LLM → last_token_hidden → Linear → r (标量 reward)

损失函数 (Bradley-Terry Model):
  L = -log σ(r(prompt, y_chosen) - r(prompt, y_rejected))
  
  含义: 让 chosen 的 reward 高于 rejected
  σ 是 sigmoid: 差值越大 → loss 越小
```

**训练技巧**：
- 数据质量远比数据量重要（高质量人类标注 > 大量 AI 标注）
- 通常从 SFT 模型初始化 backbone（而非 base 模型）
- 需要 margin loss 或 length penalty 防止模型偏好长回复
- 多维度 reward（正确性、安全性、格式）分别训练或联合训练

**Generative RM 的使用方式**：
```
Prompt to Judge LLM:
  "请对以下回答评分 (1-10)，并给出理由:
   问题: {question}
   回答: {response}
   评分标准: 正确性、完整性、简洁性..."

输出: "评分: 8/10。理由: 回答正确且全面，但表述略冗余..."
```

**Implicit RM（DPO 中的隐式 reward）**：
```
DPO 训练后的模型 π_θ 隐含了 reward:
  r(x, y) = β × log(π_θ(y|x) / π_ref(y|x)) + const

即: 模型相比 reference 更倾向生成的回答 = 高 reward
可以直接用 log_prob ratio 作为 reward signal
```

---

### Q: DPO 的损失函数和训练目标？DPO 如何改进？

**DPO（Direct Preference Optimization）的核心思想**：

将 RLHF 的两步（1. 训练 Reward Model；2. 用 PPO 优化 policy）合并为一步——直接用偏好数据优化 policy。

**数学推导直觉**：
```
RLHF 目标: max E[r(x,y)] - β × KL(π_θ || π_ref)

最优 policy 的解析解:
  π*(y|x) = π_ref(y|x) × exp(r(x,y) / β) / Z(x)

反解 reward:
  r(x,y) = β × log(π*(y|x) / π_ref(y|x)) + β × log Z(x)

代入 Bradley-Terry 偏好模型:
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))
                    = σ(β × [log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)])
```

**DPO 损失函数**：
```
L_DPO = -E_{(x, y_w, y_l)} [log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

其中:
  x: prompt
  y_w: preferred (chosen) response
  y_l: dispreferred (rejected) response
  π_θ: 当前 policy (训练中的模型)
  π_ref: reference policy (通常是 SFT 后的模型, 训练中冻结)
  β: 温度参数 (控制偏离 reference 的程度)
```

**训练过程**：
```
每个 batch:
  1. 对 (prompt, chosen, rejected) 三元组
  2. 分别用 π_θ 和 π_ref 计算 chosen/rejected 的 log_prob
  3. 计算 log_ratio_w = log π_θ(y_w|x) - log π_ref(y_w|x)
  4. 计算 log_ratio_l = log π_θ(y_l|x) - log π_ref(y_l|x)
  5. loss = -log σ(β × (log_ratio_w - log_ratio_l))
  6. 梯度更新 π_θ（π_ref 冻结不更新）
```

**DPO 的优缺点**：
- 优点：无需训练 RM、无需 PPO 的复杂训练循环、显存友好（不需要同时加载 4 个模型）
- 缺点：off-policy（偏好数据是旧模型生成的）、对数据质量敏感、may mode collapse

**DPO 改进方向**：

| 方法 | 改进点 | 核心变化 |
|---|---|---|
| **IPO** | 用回归损失替代 log-sigmoid | `(log_ratio_w - log_ratio_l - 1/(2β))²`，避免过拟合 |
| **KTO** | 不需要偏好对 | 只需 (x, y, good/bad label)，更容易收集数据 |
| **ORPO** | 不需要 reference model | 将 reference 合并进 SFT loss |
| **SimPO** | 用序列平均 log prob | 消除长度偏差，不需要 reference model |
| **Online DPO** | 在线生成偏好对 | 当前 policy 生成 → 打分 → 训练，解决 off-policy |
| **Self-Play** | 自我对弈生成数据 | 模型自己生成 chosen/rejected |
| **DPOP** | 防止 chosen prob 下降 | 添加 penalty 防止 chosen 的 log_prob 降低 |

**GRPO vs DPO 的关键区别**：
```
DPO: 使用固定的偏好数据集（离线）
  → 数据可能过时（off-policy 问题）
  → 只有 pairwise 信号

GRPO: 在线生成一组 response，用组内相对 reward 作为 advantage
  → On-policy（当前模型生成）
  → 多个 response 提供更丰富的比较信号
  → 不需要 reference model（advantage = normalized group reward）
```
