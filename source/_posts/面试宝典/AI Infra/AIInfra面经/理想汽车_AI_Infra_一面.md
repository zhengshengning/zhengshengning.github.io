---
title: '理想汽车 AI Infra 一面'
date: 2026-04-16 12:16:15
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 投机解码（Speculative Decoding）的原理？大小模型怎么选择？

**核心原理**——用小模型的速度换大模型的质量：

```
传统自回归：大模型逐 token 生成，每步 1 次 forward
投机解码：  小模型快速生成 K 个 token → 大模型 1 次 forward 验证 K 个 token
```

**详细流程**：
1. **Draft Phase**：小模型（draft model）自回归生成 K 个候选 token：`t₁, t₂, ..., t_K`
2. **Verify Phase**：大模型（target model）对这 K 个 token 做**一次并行 forward**（不是 K 次！），得到每个位置的概率分布
3. **Accept/Reject**：从左到右逐个验证：
   - 用修正的接受概率 `min(1, p_target(t_i) / p_draft(t_i))` 决定是否接受
   - 接受则继续验证下一个
   - 拒绝则从大模型分布重新采样当前位置，后续 token 全部丢弃
4. **保证无损**：数学上证明输出分布与纯大模型生成**完全一致**

**加速原理**：
- 如果 draft model 接受率 α = 80%，K=5，则每次验证平均接受 4 个 token
- 1 次大模型 forward 产生 4 个 token（vs 原来 1 次只产 1 个）
- 实际加速比 ≈ α×K / (1 + cost_draft/cost_target)

**大小模型选择原则**：

| 维度 | 要求 | 原因 |
|---|---|---|
| 分布接近度 | draft 与 target 分布越接近越好 | 接受率 α 直接决定加速比 |
| 速度比 | draft 应比 target 快 10-20x | 否则 draft 开销抵消收益 |
| 词表一致 | 必须相同 tokenizer | 否则无法对齐验证 |

**常见选择**：
- 同系列小模型：LLaMA-7B (draft) + LLaMA-70B (target)
- 同模型早期退出：Self-speculation（前 N 层预测）
- 专用预测头：Medusa（额外 MLP head 预测多个未来 token）、EAGLE（特征级预测）

**EAGLE 的优势**：不直接预测 token，而是预测下一个 token 的**隐藏状态特征**，然后用 target model 的 LM head 得到 token，接受率更高（~90%+）。

### Q: Dense 模型和 MoE 模型推理实现有什么区别？Expert Parallel 怎么做？

**本质区别——FFN 层的计算路径**：

| 维度 | Dense | MoE |
|---|---|---|
| FFN 计算 | 每个 token 经过相同 FFN | 每个 token 经 Router 选 Top-K Expert |
| 计算量/token | 固定（hidden×4hidden×2） | 减少（只激活 K/N 的参数） |
| 总参数量 | 固定 | 大（所有 Expert 参数都存储） |
| 调度复杂性 | 无 | 需要 token dispatch/combine |

**MoE 推理挑战**：

1. **Expert 负载不均衡**：
   - 某些 Expert 被高频选中（热门 Expert），导致计算集中
   - 解决：Capacity Factor 限制每个 Expert 最多处理的 token 数；auxiliary loss 训练时鼓励均衡

2. **Token Dispatch/Combine 开销**：
   - Router 判断每个 token 去哪个 Expert → 需要 scatter（分发）
   - Expert 计算完成 → 需要 gather（收集）+ 加权求和
   - 这两步是额外的访存操作

3. **内存占用大**：
   - 虽然每 token 只激活部分 Expert，但**所有 Expert 参数**都需要驻留显存
   - 如 Mixtral-8x7B：总参数 47B（但每 token 只激活 12.9B）

**Expert Parallel（EP）实现**：

```
GPU 0: Expert 0, 1          GPU 1: Expert 2, 3
GPU 2: Expert 4, 5          GPU 3: Expert 6, 7

Token Routing 后：
1. All-to-All: 每个 GPU 将本地 token 发送到对应 Expert 所在的 GPU
2. 各 GPU 执行本地 Expert 计算
3. All-to-All: 将计算结果发回原 GPU
4. 加权求和得到最终输出
```

**EP 与 TP 的结合**：
- EP 切分 Expert 维度（不同 Expert 放不同卡）
- TP 切分 Expert 内部（单个 Expert 的权重矩阵切分到多卡）
- 适合超大 Expert（如 DeepSeek-V3 的 256 个 Expert，每个 Expert 本身很大）

**DeepEP**：DeepSeek 开源的高性能 All-to-All 通信库，针对 MoE 场景优化了动态 token 数下的通信效率。
