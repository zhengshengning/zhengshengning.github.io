---
title: '腾讯 AI Infra 实习 一面 (2)'
date: 2026-04-16 12:15:39
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: MoE为什么能在参数量很大的情况下把训练和推理成本压住？真正难点在哪？

**MoE的核心机制——稀疏激活：**

```
Dense Model (7B):  每个token经过所有7B参数 → FLOPs = 6×7B×token
MoE Model (47B参数, 8选2):  每个token只经过2个expert(各~5B) → FLOPs ≈ 6×10B×token
```

总参数量47B（容量大），但单次计算量只有10B的Dense模型水平。这就是MoE"**用少量计算获得大容量模型能力**"的核心。

**为什么成本可控？**
- 训练：计算量只与激活参数有关，47B MoE训练成本≈10B Dense
- 推理Prefill(compute-bound)：同理，计算量低
- 推理Decode(memory-bound)：**问题！** 所有expert权重需常驻显存（47B × 2bytes = 94GB）

**真正的难点：**

**1. 路由与负载均衡（训练稳定性的核心挑战）：**
```
理想情况: 8个expert各处理12.5%的token
实际情况: 热门expert处理60%token，冷门expert处理<1%
           → 热门expert被过度训练（退化为全能expert）
           → 冷门expert几乎没梯度（废弃）
```

**2. All-to-All通信开销（分布式训练/推理的瓶颈）：**
- Expert Parallelism中，token需要被路由到不同设备上的expert
- Forward: All-to-All dispatch token → Expert计算 → All-to-All combine结果
- 通信量 = batch_size × seq_len × hidden_size × 2 (dispatch和combine各一次)
- 是TP/DP中不存在的额外通信

**3. 推理显存压力：**
- 所有expert权重需常驻显存（即使每次只激活2个）
- 47B MoE模型的权重显存 ≈ Dense 47B模型（无法通过稀疏性节省）
- 解决方案：Expert offloading到CPU/NVMe，按需加载（增加延迟）

---

### Q: MoE里的负载均衡怎么做？为什么loss正常但expert可能已经废了？

**负载均衡方法：**

**1. 辅助负载均衡损失(Auxiliary Load Balancing Loss)：**
```python
# Switch Transformer的实现
# f_i = 分配给expert i的token比例
# P_i = router给expert i的平均概率
aux_loss = N * Σ(f_i * P_i)  # 惩罚不均匀分配
total_loss = task_loss + α * aux_loss  # α通常0.01-0.1
```

**2. 容量因子(Capacity Factor)：**
```
每个expert最多处理 capacity = (tokens / num_experts) × capacity_factor 个token
超出的token被丢弃(drop)或路由到次优expert
典型capacity_factor = 1.25（允许25%的不均衡）
```

**3. Expert Choice Routing（DeepSeek V3方案）：**
```
传统: Token选Expert (top-k experts per token)
Expert Choice: Expert选Token (top-k tokens per expert)
→ 天然保证每个expert处理相同数量的token
```

**4. DeepSeek V3的无辅助Loss方案——Bias-based Routing：**
```python
# 给每个expert加一个可学习的bias
router_logits = x @ W_router + bias  # bias动态调整
# bias不参与梯度计算，而是根据负载统计动态更新
# 负载过高的expert降低bias，负载过低的增加bias
```

**为什么loss正常但expert可能已经废了？**

主任务loss只反映**模型整体输出质量**，不反映**各expert的利用率**：
- 如果2个expert学会了处理所有类型的token，其他6个expert废弃
- 模型输出仍然正确（loss正常），但参数利用率极低（8个expert的容量只用了2个）
- 类似于：一个8人团队只有2人干活，项目完成了但效率极低

**需要监控的指标：**
- 每个expert的token占比（应接近1/N）
- Router entropy（高熵=均匀分配，低熵=集中到少数expert）
- Expert utilization（每个expert平均处理的token数/理论值）
- 各expert的梯度范数（废弃expert梯度接近0）

---

### Q: GQA、MQA和标准MHA的区别？为什么线上推理更关心GQA？

**三种Attention变体的KV头数比较：**

```
MHA:  Q heads: 32    K heads: 32    V heads: 32
      每个query head有独立的K/V

GQA:  Q heads: 32    K heads: 8     V heads: 8   (4个Q共享1个KV)
      每组4个query head共享一套K/V

MQA:  Q heads: 32    K heads: 1     V heads: 1   (所有Q共享1个KV)
      所有query head共享一套K/V
```

**KV-Cache显存对比（LLaMA-70B, seq_len=4096, FP16）：**

| 方案 | KV头数 | KV-Cache/token | 4K context/请求 |
|------|--------|---------------|----------------|
| MHA (32 heads) | 32 | 2×32×128×2B=16KB | 64MB |
| GQA (8 groups) | 8 | 2×8×128×2B=4KB | 16MB |
| MQA (1 head) | 1 | 2×1×128×2B=0.5KB | 2MB |

**为什么线上推理更关心GQA？**

1. **Decode阶段是memory-bound**：每步需要读取完整KV-Cache
   - MHA: 读64MB → 带宽限制（A100 2TB/s理论极限下也需要32us）
   - GQA: 读16MB → 带宽需求降低4倍 → 吞吐提升4倍

2. **更多请求可以同时服务**：
   - 80GB显存, MHA: 64MB/请求 → 最多~1000个并发长序列
   - 80GB显存, GQA: 16MB/请求 → 最多~4000个并发长序列

3. **精度损失可接受**：
   - GQA在效果上接近MHA（LLaMA-2 70B验证）
   - MQA精度损失较明显（极端压缩）
   - GQA是精度与效率的最优平衡点

---

### Q: RoPE为什么能做位置编码？长上下文外推为什么经常失真？

**RoPE的数学原理：**

RoPE将位置m编码为旋转矩阵，作用于Q和K的每对相邻维度：
```
对于第i对维度(2i, 2i+1)，位置m的旋转角度θ_i = m × base^(-2i/d)
RoPE(x, m) = x × cos(m·θ) + rotate(x) × sin(m·θ)
```

**核心性质——相对位置编码：**
```
<q_m, k_n> = <RoPE(q, m), RoPE(k, n)> 
           = f(q, k, m-n)  ← 只依赖相对位置差(m-n)！
```
内积结果只与位置差有关，自然编码了相对位置信息，不需要显式的相对位置embedding。

**外推失真的原因：**

训练时位置范围[0, L_train]（如4K），推理时需要[0, L_test]（如128K）：
- 旋转角度θ_i × m中，高频维度(i小)的θ大 → 位置m增大时角度快速旋转
- 训练时模型从未见过θ_i × m超过一定范围的值
- 推理时长位置的高频分量超出训练分布 → 注意力模式崩塌

**类比理解：** 就像一个只在0-360度范围训练的sin/cos预测模型，突然要预测3600度位置的值——虽然数学上是周期函数，但模型可能没有学到周期性。

**解决长上下文外推的方法：**

| 方法 | 原理 | 效果 |
|------|------|------|
| Position Interpolation (PI) | 将位置索引等比缩小到训练范围内 | 简单有效，需少量微调 |
| NTK-aware Scaling | 修改RoPE的base（提高低频不动高频） | 免训练或少量微调 |
| YaRN | 分频率段做不同缩放+注意力温度修正 | 效果最好，需微调 |
| Continual Pretraining | 在长序列上继续预训练 | 最可靠但成本高 |
| ALiBi | 不用RoPE，用线性偏置做位置 | 外推性好但效果可能略差 |

---

### Q: FlashAttention为什么快？优化的是算力还是访存？

**FlashAttention优化的是HBM访存（IO），而非计算量。** 计算量甚至略有增加。

**传统Attention的IO问题：**
```
标准实现 (seq_len=N, head_dim=d):
1. S = Q @ K^T     → 写 N×N 矩阵到HBM     [写: N²]
2. P = softmax(S)  → 从HBM读S，写P到HBM   [读: N², 写: N²]
3. O = P @ V       → 从HBM读P             [读: N²]
总HBM IO: O(N²) bytes (与d无关)
```

**FlashAttention的IO优化：**
```
分块处理 (block_size = B_r × B_c):
- Q分成N/B_r块，K/V分成N/B_c块
- 对每对(Q_block, KV_block):
  1. 加载Q_block和KV_block到SRAM（一次从HBM读取）
  2. 在SRAM中完成 QK^T + softmax + 乘V（全程不写中间结果到HBM）
  3. 利用online softmax增量更新输出

总HBM IO: O(N² × d / M)  (M = SRAM大小 ≈ 192KB on A100)
```

**为什么IO减少了？**
- SRAM(共享内存)容量有限，但带宽约19TB/s（HBM的10倍）
- 通过分块让小块数据在SRAM中完成所有计算，避免中间大矩阵落地到HBM
- N×N的attention矩阵从未完整存在于HBM中

**实际性能提升：**
- 速度：提升2-4倍（减少HBM IO）
- 显存：从O(N²)降到O(N)（不存储完整attention矩阵）
- 计算量：略有增加（online softmax需要额外的rescaling操作）
- A100上典型数据：seq_len=2K时提升2倍，seq_len=16K时提升4倍

**FlashAttention-2的进一步优化：**
- 外层循环改为遍历Q块（而非KV块），提高并行度
- 更好的warp级别work分配
- 减少非矩阵乘的计算（softmax rescaling）
