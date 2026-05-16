---
title: '美团 AI Infra 实习'
date: 2026-04-16 12:09:11
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 训练优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Zero-Bubble 流水线调度的原理？

Zero-Bubble Pipeline Parallelism 旨在消除流水线并行中的 bubble（空闲时间），这是 Pipeline Parallel 效率的核心瓶颈。

**传统 PP 的 Bubble 问题**：

以 1F1B（One Forward One Backward）调度为例：
```
传统 1F1B (P=4 stages, M=8 micro-batches):

Stage 0: F F F F B B B B _ _ _ _ (warmup 阶段有 bubble)
Stage 1: _ F F F F B B B B _ _ _
Stage 2: _ _ F F F F B B B B _ _
Stage 3: _ _ _ F F F F B B B B _
                                    ↑ bubble = (P-1)/M × 时间
```
- Bubble 比例 = (P-1) / M，P=4, M=8 时 bubble = 37.5%
- 要降低 bubble 就要增大 M，但 M 增大意味着更多 micro-batch 的激活值需要同时保存

**Zero-Bubble 的核心洞察——拆分反向传播**：

传统 backward 包含两部分：
- **B**（输入梯度计算）：`dX = dY × W^T`——依赖下游传来的梯度 dY（有流水线依赖）
- **W**（权重梯度计算）：`dW = X^T × dY`——只依赖本层激活 X 和已知的 dY（无流水线依赖！）

```
传统: B 和 W 绑定在一起执行
        ↓
Zero-Bubble: 拆分 B 和 W，将 W 安排到 bubble 时间段

Zero-Bubble 1F1B (P=4):
Stage 0: F F F F B B B B W W W W  ← W 填充原本的 bubble
Stage 1: _ F F F B W B B B W W W
Stage 2: _ _ F F B W B W B B W W
Stage 3: _ _ _ F B W B W B W B W
                    ↑ W 被插入到 bubble 中执行
```

**关键技术细节**：

1. **ILP 求解器调度**：
   - 将调度问题形式化为整数线性规划（ILP）
   - 约束：数据依赖（F 在 B 之前、B 需要下游梯度）、资源约束（同时只能执行一个操作）
   - 目标：最小化 makespan（总完成时间）
   - 求解器输出最优的 F/B/W 调度顺序

2. **内存开销权衡**：
   - 传统 1F1B：每个 stage 最多缓存 P 个 micro-batch 的激活
   - Zero-Bubble：需要延迟 W 的执行，期间激活值不能释放
   - 峰值内存 ≈ 增加 1-2 个 micro-batch 的激活存储
   - 论文提出 ZB-2p（zero-bubble two-pass）进一步控制内存

3. **ZB-H（Handshake Schedule）**：
   - 改进版本：不需要 ILP 求解器，采用启发式调度
   - Stage 之间通过"握手"协议动态决定执行顺序
   - 实现更简单，效果接近最优

**性能对比**：

| 调度方案 | Bubble 比例 | 峰值内存（vs 1F1B） | 实现复杂度 |
|---|---|---|---|
| GPipe | (P-1)/M | 高（存所有激活） | 低 |
| 1F1B | (P-1)/M | 中 | 低 |
| Interleaved 1F1B | (P-1)/(M×V) | 中（V 为虚拟 stage） | 中 |
| Zero-Bubble (ZB-1p) | ~0% | 中（+1-2 micro-batch） | 高（需 ILP） |
| Zero-Bubble (ZB-H) | ~2-5% | 中 | 中 |

**实际收益**：在 P=8、M=24 的配置下，Zero-Bubble 相比 1F1B 可提升训练吞吐 15-30%。

---

### Q: DeepEP 是什么？

DeepEP（DeepSeek Expert Parallelism Library）是 DeepSeek 开源的高性能 MoE All-to-All 通信库，专门解决 MoE 模型中 Expert Parallel 的通信瓶颈。

**背景——MoE 通信挑战**：

MoE 模型每层有一个 router 将 token 分发到不同 expert，Expert Parallel 将 expert 分布在不同 GPU 上：
```
Token Dispatch (All-to-All):
  GPU 0 的 token → 分发到 GPU 0/1/2/3 的 expert
  GPU 1 的 token → 分发到 GPU 0/1/2/3 的 expert
  ...

Expert Compute:
  每个 GPU 处理自己负责的 expert 上的所有 token

Token Combine (All-to-All):
  expert 计算结果 → 回传给原始 GPU
```

传统 NCCL All-to-All 在 MoE 场景下的问题：
- Token 数量动态变化（每个 expert 收到的 token 数不同）
- 通信模式不规则（不是均匀的 All-to-All）
- 延迟敏感（MoE 层间有强依赖）

**DeepEP 核心特性**：

1. **双互联支持**：
   - **NVLink 域内**（intra-node）：SM-mediated 直接内存拷贝，绕过 NCCL 协议栈
   - **RDMA 域间**（inter-node）：基于 InfiniBand RDMA Verbs 实现低延迟跨节点通信

2. **低延迟通信（Latency-optimized mode）**：
   - 针对 Decode 阶段（每步少量 token）优化
   - 使用 RDMA 异步发送，pipeline 化 dispatch/compute/combine
   - 延迟目标：< 100μs 完成一次 expert 通信

3. **高吞吐通信（Throughput-optimized mode）**：
   - 针对 Prefill 阶段（大量 token 并行处理）优化
   - 批量打包 token，最大化带宽利用率
   - 支持 NVLink + RDMA 混合路径选择

4. **动态负载均衡**：
   - 实时统计每个 expert 收到的 token 数
   - 当某个 expert 过载时，可以将部分 token 路由到副本 expert
   - 与 DeepSeek 的 Auxiliary-Loss-Free Load Balancing 配合

5. **计算通信重叠**：
   ```
   Timeline:
   GPU compute: [Expert 0 计算] [Expert 1 计算] [Expert 2 计算]
   Communication:    [Dispatch batch 2] [Dispatch batch 3] [Combine batch 1]
   ```
   - 流水线化：当前 batch 的 expert 计算与下一 batch 的 token dispatch 重叠

**与 NCCL All-to-All 对比**：

| 维度 | NCCL All-to-All | DeepEP |
|---|---|---|
| 通信模式 | 均匀发送/接收 | 支持不均匀（dynamic token routing） |
| 协议栈 | 通用 | MoE 场景定制 |
| 延迟 | ~200-500μs | ~50-100μs（latency mode） |
| 重叠支持 | 需手动实现 | 内置 pipeline |
| 动态负载 | 不感知 | 内置负载均衡 |

**对 DeepSeek-V3 的意义**：DeepSeek-V3 有 256 个 expert，8 个 GPU 上做 EP=8，每次 MoE 层需要两次 All-to-All。DeepEP 使通信开销从训练时间的 30%+ 降低到 10% 以内。

---

### Q: RL（强化学习）异步调度有哪些方案？优缺点？

在 LLM 的 RLHF/GRPO 训练中，强化学习涉及多个模块的协调（Generation、Reward、Training），异步调度是提升训练吞吐的关键。

**LLM-RL 训练的计算流程**：
```
同步方案（简单但低效）:
  Generate → Reward → Train → Generate → Reward → Train ...
  [GPU 空闲]  [GPU 空闲]  [训练]   [GPU 空闲]  [GPU 空闲]  [训练]

异步方案（复杂但高效）:
  Generation: [Gen batch 1] [Gen batch 2] [Gen batch 3] ...
  Reward:         [Reward 1] [Reward 2] [Reward 3] ...
  Training:           [Train 1] [Train 2] [Train 3] ...
```

**主要异步调度方案**：

**1. IMPALA / V-trace 架构**（经典 RL）：
```
多个 Actor（推理 worker）:
  Actor 0: 用当前 policy 生成轨迹 → 存入 replay buffer
  Actor 1: 用当前 policy 生成轨迹 → 存入 replay buffer
  ...

Learner（训练 worker）:
  从 replay buffer 取经验 → V-trace 修正 off-policy 偏差 → 更新 policy → 广播给 Actor
```
- 优点：Actor 和 Learner 完全解耦，吞吐最大化
- 缺点：Actor 使用的 policy 可能已过时（policy lag），需要 V-trace 修正

**2. OpenRLHF 方案**：
```
Ray 集群编排:
  vLLM Generation Server (多 GPU) → 生成 rollout
       ↓ (Ray Object Store)
  Reward Model Server (多 GPU) → 打分
       ↓ (Ray Object Store)
  Training Worker (多 GPU, DeepSpeed) → PPO/GRPO 更新
       ↓ (广播新权重)
  回到 Generation Server
```
- 模块间通过 Ray 的 Object Store 传递数据
- 各模块可以使用不同数量的 GPU
- 支持 Generation 和 Training 的 GPU 复用（时分复用）或独立部署

**3. veRL（Volcano Engine RL）方案**：
```
核心设计——Hybrid Engine:
  同一组 GPU 在不同阶段切换角色:
    推理阶段: GPU 运行 vLLM/SGLang 做高效生成
    训练阶段: GPU 切换为 FSDP/Megatron 做梯度更新

  优势: 无需 GPU 间数据传输（生成的 logprobs 就在本地）
  劣势: 不能真正重叠生成和训练（时分复用）
```

**4. SEED RL 架构**：
- 将推理集中到 Learner 端（Actor 只负责环境交互）
- 减少 policy 传输开销（Actor 不需要完整模型）
- 适合环境计算轻量的场景

**5. AsyncRL / Sample Factory 方案**：
- 单机多 worker，异步环境步进（env stepping）
- 双缓冲（double buffer）：一个 buffer 采集，另一个训练
- 极高的单机吞吐

**优缺点对比**：

| 方案 | 吞吐 | 实现复杂度 | Off-policy 风险 | GPU 利用率 | 适用规模 |
|---|---|---|---|---|---|
| 同步（veRL Hybrid） | 中 | 低 | 无 | 中（时分复用） | 中小规模 |
| 半异步（OpenRLHF） | 高 | 中 | 低（1-2 步 lag） | 高 | 大规模 |
| 全异步（IMPALA 式） | 最高 | 高 | 高（需修正） | 最高 | 超大规模 |

---

### Q: RL 异步调度在算法上需要做哪些改进？

异步调度引入了 **off-policy 问题**：训练使用的经验来自"旧版本"的 policy，而不是当前最新的 policy。这会导致梯度估计有偏，需要算法层面的修正。

**1. 重要性采样修正（Importance Sampling）**：

```
On-policy 梯度:  ∇J = E_{π_θ}[A(s,a) × ∇log π_θ(a|s)]
Off-policy 修正: ∇J = E_{π_old}[ρ × A(s,a) × ∇log π_θ(a|s)]

其中 ρ = π_θ(a|s) / π_old(a|s) 为重要性采样比率
```

- PPO 使用 clip(ρ, 1-ε, 1+ε) 限制修正幅度，防止单个样本权重过大
- V-trace 进一步对 ρ 做截断：`min(ρ, c̄)` 和 `min(ρ, ρ̄)`，c̄/ρ̄ 为截断阈值
- 截断使得 off-policy 时的方差可控，代价是引入少量 bias

**2. Policy Lag 控制**：

| 控制策略 | 方法 | 效果 |
|---|---|---|
| 版本控制 | Actor policy 版本与 Learner 差距超过 K 步则丢弃 | 硬截断，浪费采样 |
| 软约束 | 按版本差距对样本加权（越旧权重越低） | 平滑退化，利用所有样本 |
| 同步点 | 每 N 步强制同步一次 Actor 和 Learner | 限制最大 lag |
| KL 惩罚 | 在 loss 中加入 KL(π_θ \| π_ref) 约束 | 间接限制 policy 偏移 |

**3. 经验优先级（Prioritized Experience Replay）**：
- 按 TD-error 或 advantage 绝对值排序经验
- 高 TD-error 的经验被更频繁采样
- 修正采样偏差：用 importance sampling weight β 修正
- 对 LLM-RL：可按 reward 方差或 advantage 绝对值优先

**4. 梯度累积与聚合策略**：
- **同步聚合**：等所有 worker 的梯度到齐后平均更新（无 staleness，但需等待最慢 worker）
- **异步聚合**：梯度到达即更新（最快，但梯度来自不同版本的 policy）
- **Local SGD**：各 worker 本地更新多步后再同步（折中方案）
- **Staleness-weighted**：根据梯度的"过时程度"加权：`w = 1 / (1 + staleness)`

**5. GRPO 特殊考虑**：
- GRPO 使用组内相对 reward 作为 advantage，不需要 value function
- 异步时同一组的多个 response 应该来自相同版本的 policy
- 如果组内 response 来自不同版本，relative advantage 的比较不公平
- 解决方案：确保一个 group 的所有 response 在同一 policy version 下生成

**6. 实践建议**：
```
保守方案（精度优先）:
  - Policy lag ≤ 1 步
  - PPO clip ε = 0.1-0.2
  - 每次同步后丢弃旧 buffer

激进方案（吞吐优先）:
  - Policy lag ≤ 4-8 步
  - V-trace 修正 + PPO clip
  - 按 staleness 加权
  - 大 replay buffer
```

---

### Q: 手撕：合并 K 个升序链表？

（编程题）
