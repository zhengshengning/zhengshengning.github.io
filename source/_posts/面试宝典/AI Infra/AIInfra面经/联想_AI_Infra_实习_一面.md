---
title: '联想 AI Infra 实习 一面'
date: 2026-04-16 11:45:54
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 面经]
---


<!-- more -->

---

### Q: 有哪些 KV Cache 优化方法？

KV Cache 是 LLM 推理的核心显存瓶颈。以 LLaMA-70B 为例，batch=16, seq=4096 时 KV Cache 约 16 GB（FP16）。优化方法分为多个层次：

**1. 架构级优化（训练时确定，最根本）**：
- **MQA/GQA**：减少 KV head 数量。GQA 将 KV Cache 减少 G 倍（如 LLaMA-2 的 G=4/8）
- **MLA（Multi-head Latent Attention）**：DeepSeek 的低秩压缩，将 KV 投影到低维 latent 存储，压缩比可达 GQA 的 5-10x

**2. 量化优化（推理时应用，训练无关）**：
- FP8 KV Cache：减少 50% 存储，PPL 增加 ~0.1-0.2
- INT4 KV Cache：减少 75%，但精度损失较大，需 per-head scale
- 渐进精度：最近 token 高精度，远处 token 低精度

**3. 稀疏/驱逐策略（动态决定保留哪些 token）**：
- **H2O（Heavy Hitter Oracle）**：统计累积 attention score，保留 top-k "heavy hitter" token + 最近滑动窗口。典型保留 20% token 即可维持 95%+ 精度
- **StreamingLLM**：保留 attention sink（前 4 个 token，它们吸收了大量 attention） + 固定窗口。支持无限长度推理
- **Scissorhands**：观察到重要 token 跨层一致，一次判断全局驱逐

**4. 存储管理优化**：
- **PagedAttention**：分页管理消除碎片，利用率从 20-40% 提升到 ~100%
- **Prefix Caching**：相同 system prompt 的请求共享 KV Cache block

**5. 卸载（Offloading）**：
- 将不活跃请求的 KV Cache swap 到 CPU/SSD
- 请求重新激活时加载回 GPU
- 需要平衡加载延迟和显存利用率

---

### Q: DeepSeek R1 有什么注意力优化？

DeepSeek-V2/V3/R1 采用 **MLA（Multi-head Latent Attention）**，这是目前 KV Cache 压缩比最高的架构设计：

**核心原理**：
- 不存储完整的 K 和 V（维度 = num_kv_heads × head_dim），而是存储经过**下投影**后的低维 latent vector
- `c_kv = W_down × [K; V]`，其中 c_kv 的维度 << num_heads × head_dim
- 推理时通过 up-projection 恢复：`K' = W_up_K × c_kv`，`V' = W_up_V × c_kv`

**权重吸收技巧**：
- 核心观察：`Attention(Q, K', V') = Attention(Q, W_up_K × c_kv, W_up_V × c_kv)`
- 可以将 W_up_K 吸收进 Q_proj：`Q_new = Q × W_up_K^T`
- 将 W_up_V 吸收进 O_proj：`O_new = Attention × W_up_V × W_O`
- 效果：推理时**无需显式恢复完整 KV**，直接对 latent 做 attention

**RoPE 兼容性处理**：
- RoPE 需要对 K 施加位置旋转，但 latent 是压缩后的混合表示
- 解决方案：将 K 分为两部分：
  - 少量维度（如 64-d）不压缩，直接用于施加 RoPE（这部分需要正常缓存）
  - 其余维度压缩为 latent（不需要 RoPE，因为 RoPE 信息已在未压缩部分）

**压缩效果**：
- 标准 MHA：缓存 128×128 = 16384 维/token（128 heads × 128 dim）
- GQA (8 KV heads)：缓存 8×128 = 1024 维/token
- MLA：缓存 latent_dim + rope_dim ≈ 512+64 = 576 维/token
- 相比 GQA 还能再压缩 ~2x

---

### Q: Transformer 结构介绍？

**现代 Decoder-only Transformer（LLM 主流架构）每层结构**：

```
输入 x
  ↓
[Pre-RMSNorm] ─── 残差连接 ───┐
  ↓                           │
[Multi-Head Self-Attention]    │
  ├─ Q = x × W_Q             │
  ├─ K = x × W_K + RoPE      │
  ├─ V = x × W_V             │
  ├─ Score = QK^T / √d       │
  ├─ Attn = softmax(Score + CausalMask) │
  └─ O = Attn × V × W_O     │
  ↓                           │
x + O  ←─────────────────────┘
  ↓
[Pre-RMSNorm] ─── 残差连接 ───┐
  ↓                           │
[Feed-Forward Network (SwiGLU)] │
  ├─ gate = x × W_gate       │
  ├─ up = x × W_up           │
  ├─ hidden = SiLU(gate) ⊙ up │
  └─ down = hidden × W_down  │
  ↓                           │
x + down ←────────────────────┘
  ↓
输出
```

**关键组件**：
- **RMSNorm**（替代 LayerNorm）：去掉均值中心化，只保留缩放，减少计算量 ~15%
- **RoPE**（旋转位置编码）：在 Q/K 上施加与位置相关的旋转矩阵，天然支持长度外推
- **SwiGLU**（替代 ReLU/GELU）：带门控的激活 `SiLU(xW_g) ⊙ xW_u`，性能更好
- **因果 Mask**：下三角 mask 确保 token 只能看到之前的 token（自回归约束）
- **Pre-Norm**（替代 Post-Norm）：Norm 在子层之前，训练更稳定

---

### Q: 现在主流大模型架构有什么变化？

相比 2017 年原始 Transformer，现代 LLM 的关键架构改进及其原因：

| 组件 | 原始 | 现代 | 改进原因 |
|---|---|---|---|
| Normalization | Post-LayerNorm | Pre-RMSNorm | 训练更稳定 + 计算更快 |
| 位置编码 | 绝对/正弦 | RoPE | 支持长度外推，相对位置建模更好 |
| 激活函数 | ReLU | SwiGLU | 实验证明 SwiGLU 在同等参数下效果更好 |
| 注意力 | MHA | GQA/MLA | 大幅减少 KV Cache，推理友好 |
| 架构 | Dense | MoE | 稀疏激活，同等计算量下参数更多 |
| Norm 位置 | 子层之后 | 子层之前 | Pre-Norm 梯度流更稳定 |
| Bias | 有 | 去掉 | 减参数 + 简化量化（无 zero_point） |
| FFN dim | 4×hidden | 8/3×hidden (SwiGLU) | SwiGLU 有 gate 分支，等效参数量对齐 |

---

### Q: GRPO 的改进方法有哪些？了解 GSPO 吗？

**GRPO 改进方向**：

1. **奖励设计优化**：
   - Process Reward Model（PRM）：对推理过程的每一步给分（vs 只给最终结果打分）
   - 多维度 reward：正确性 + 格式 + 安全性 + 简洁性，不同维度独立评分再加权
   - 规则 reward（如数学验证器）比模型 reward 更可靠且不需要训练

2. **采样策略优化**：
   - 组大小选择：组越大（如 16-64）基线估计越稳定，但采样成本越高
   - 温度调节：高温采样增加多样性（组内对比更有信息量）
   - 拒绝采样：过滤掉全错/全对的组（它们梯度为 0）

3. **训练稳定性**：
   - KL 正则化：限制策略偏离 reference model 的程度
   - 多阶段训练：先简单 reward（格式），再复杂 reward（正确性）
   - 与 SFT 数据联合训练（replay buffer）

**GSPO（Group Scoring Policy Optimization）**：
- 在 GRPO 基础上改进**打分和排序机制**
- GRPO 使用 `(r_i - mean) / std` 作为优势估计——对组内奖励的绝对差异不敏感
- GSPO 使用更细粒度的组内排序/评分方式（如 pairwise comparison、Elo rating 风格的评分）
- 目的：当组内奖励差异很小时（如都对但质量有差异），提供更有区分度的梯度信号

---

### Q: 手撕：实现 Multi-Head Attention，并说明在哪里加 Mask 矩阵？

（编程题）
