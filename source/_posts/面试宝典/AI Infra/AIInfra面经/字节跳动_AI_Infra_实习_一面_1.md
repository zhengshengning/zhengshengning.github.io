---
title: '字节跳动 AI Infra 实习 一面 (1)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: MoE模型的优化有哪些经验？RL中MoE优化做过吗？

MoE（Mixture of Experts）模型通过稀疏激活实现"大模型容量、小模型计算量"，但引入了独特的优化挑战：

**核心挑战**：
- All-to-All通信：token需要从源GPU路由到持有目标expert的GPU，再路由回来。
- 负载不均衡：某些expert可能被过多token选中，导致部分GPU过载。
- 显存：虽然每次只激活部分expert，但所有expert权重都需驻留显存。

**优化方向详解**：

**1. Expert并行（EP）**：
```
GPU 0: Expert 0, 1    GPU 1: Expert 2, 3    GPU 2: Expert 4, 5    GPU 3: Expert 6, 7
         ↕ All-to-All通信 (每个MoE层两次: dispatch + combine)
```
- 每GPU持有N/P个expert（N=总expert数, P=GPU数）。
- 需要两次All-to-All：dispatch（token→expert的GPU）和combine（结果→token的GPU）。
- 通信量 = 2 × batch_size × seq_len × hidden_dim × activated_experts/total_experts × sizeof(dtype)。
- 优化：与TP组合（如DeepSeek-V3: TP=8内做EP，跨TP组用DP）。

**2. 负载均衡**：
```python
# Auxiliary Loss (Switch Transformer风格)
# f_i = 该batch中路由到expert_i的token比例
# P_i = 该batch中所有token对expert_i的router概率均值
aux_loss = alpha * N * sum(f_i * P_i)  # 惩罚f和P同时大的expert
```
- alpha通常0.01-0.1。太大影响主任务，太小均衡不够。
- Token丢弃（Capacity Factor）：设expert_capacity = CF × (tokens/num_experts)，超出的token被丢弃或路由到备选expert。CF通常1.0-1.25。

**3. Expert缓存/预取（推理优化）**：
- 对于极大的MoE模型（如DeepSeek-V3 671B），所有expert同时放GPU显存可能不够。
- 策略：只加载top-K被激活的expert到GPU，其他保留在CPU或NVMe。
- 预测性加载：根据router概率预取可能被激活的expert。
- 对于训练，所有expert必须在GPU上（需要梯度）。

**4. 路由优化**：
- Top-K选择：K=1（Switch Transformer）vs K=2（Mixtral/DeepSeek）。K=2精度更好但计算+通信翻倍。
- Shared Expert（DeepSeek-MoE）：保留1-2个永远激活的expert处理通用知识，减少路由冗余。
- Expert-Choice routing：让expert选token（而非token选expert），自动均衡。

**5. RL + MoE**：
- 强化学习中不同状态/任务类型差异大，MoE让不同expert专注不同类型。
- 挑战：RL的不稳定性可能导致router崩溃（所有token路由到1-2个expert）。
- 解决：更强的均衡约束、curriculum learning先稳定router再加RL signal。

### Q: 有没有做过Kernel级别的优化？用CUTE DSL或手写CUDA做Fusion？

**三种Kernel开发方式的定位和选择**：

| 方式 | 开发效率 | 性能上限 | 适用场景 |
|------|---------|---------|---------|
| Triton | 最高（Python） | ~90% cuBLAS | 快速原型、中等复杂度 |
| CUTLASS/CUTE | 中（C++ templates） | ~98% cuBLAS | GEMM变体、极致优化 |
| 手写CUDA | 最低 | 100%（理论最优） | 非标准计算模式 |

**CUTLASS/CUTE DSL详解**：

CUTE（CuTe是CUTLASS 3.x的新抽象层）通过描述数据Layout和计算Pattern来生成kernel：
```cpp
// CUTE核心抽象
auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32{}, // MMA指令
                                 Layout<Shape<_2,_2,_1>>{});    // Warp tiling

// 数据Layout描述
auto smem_layout_A = make_layout(Shape<_128, _32>{},   // Tile shape
                                  Stride<_32, _1>{});   // Row-major

// Copy操作（全局→共享内存）
auto copy_A = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                               Layout<Shape<_32, _4>>{});
```

CUTE的优势：将kernel分解为"数据搬运"和"计算"两个正交维度，通过组合不同的copy atom和mma atom快速生成各种kernel变体。

**Kernel Fusion策略**：

| 融合类型 | 示例 | 实现方式 | 收益 |
|---------|------|---------|------|
| GEMM Epilogue | GEMM+Bias+ReLU | CUTLASS epilogue functor | 几乎零开销 |
| Element-wise chain | Add+Mul+Exp | Triton单kernel | 3-5x加速 |
| Reduce+Element | LayerNorm+Add | 手写CUDA | 2x加速 |
| Attention | QKV+Score+Softmax+O | FlashAttention | 2-4x加速 |

### Q: 做Kernel Fusion时倾向用什么方式？

**选择决策树**：

```
需要GEMM相关fusion?
├── 是 → CUTLASS/CUTE（已有高度优化的GEMM模板）
│         └── Epilogue fusion: 只需定义OutputOp functor
└── 否 → 
    ├── 纯Element-wise链? → Triton（开发最快，性能优秀）
    ├── Reduce + Element-wise? → 手写CUDA（需要精细控制shared memory reduce）
    └── 非标准pattern? → 手写CUDA（最灵活）
```

**Triton示例（Fused SiLU + Element-wise Mul）**：
```python
@triton.jit
def fused_swiglu_kernel(gate_ptr, up_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)
    # SiLU(gate) * up = gate * sigmoid(gate) * up
    result = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, result, mask=mask)
```
开发时间：几分钟。性能：通常达到手写CUDA的85-95%。

**手写CUDA示例（Fused LayerNorm + Residual Add）**：
```cuda
// 需要精细控制：两遍reduce(mean/var) + element-wise normalize + add
// Triton不擅长跨线程reduce的精细控制
__global__ void fused_layernorm_add(float* out, float* input, float* residual,
                                     float* gamma, float* beta, int N) {
    __shared__ float smem[32];  // warp reduce缓冲
    // ... warp shuffle reduce求mean和var
    // ... 归一化 + 残差add在同一kernel中完成
}
```

### Q: 有没有做了Fusion性能反而下降的情况？原因是什么？

**实际案例和原因分析**：

**案例1：寄存器压力导致Occupancy暴跌**：
```
未融合: Kernel A (Occupancy 75%, 32 regs/thread) + Kernel B (Occupancy 75%, 28 regs/thread)
融合后: Fused AB (Occupancy 25%, 96 regs/thread!)
```
- 融合后每线程需要保持A和B的所有中间状态→寄存器用量激增。
- Occupancy从75%降到25%→活跃warp不够隐藏延迟→stall增加。
- **解决**：拆分为两阶段，或在shared memory中暂存中间结果让出寄存器。

**案例2：Shared Memory超限**：
- GEMM tiling需要128KB shared memory + LayerNorm reduce需要32KB。
- 总160KB超过A100单block上限（164KB需特殊配置）。
- 导致每SM只能驻留1个block → Occupancy极低。
- **解决**：减小GEMM tile size，或分两个phase执行。

**案例3：计算密度失衡**：
- 原始：大GEMM(compute-bound, 100us) + 小ReLU(memory-bound, 2us)
- 融合后：大GEMM with epilogue ReLU(100.1us)
- ReLU本身2us的开销可以忽略，但如果融合导致GEMM的tile size被约束（比如output需要ReLU后才能写，限制了流水线深度），可能反而变慢。
- **原则**：memory-bound的小op融合到compute-bound的大op通常无害；反过来可能有问题。

**判断是否应该融合的经验法则**：
1. 融合后每线程寄存器是否超过128（A100）？超过则可能有问题。
2. 融合后shared memory是否超过96KB（通常safe上限）？
3. 两个op的最优block size是否兼容？
4. **始终用benchmark验证**——不要假设融合一定更好。

### Q: Hopper架构的Warp Specialization是什么？底层如何实现？

**核心思想——分工而非全能**：

传统方式（Ampere及之前）：Block内所有Warp既做数据搬运又做计算，轮流执行两种角色。
Warp Specialization（Hopper）：Block内的Warp按角色分组，**持久化专注**做一种工作。

```
传统方式 (所有Warp相同):
  Warp 0-7: [Load] → [sync] → [Compute] → [sync] → [Load] → [sync] → [Compute] → ...
  每次切换角色有sync等待，且计算和加载不能完全overlap

Warp Specialization:
  Producer Warps (0-1): [Load tile0] → [Load tile1] → [Load tile2] → ... (持续搬数据)
  Consumer Warps (2-7): [Compute tile0] → [Compute tile1] → ...         (持续计算)
  通过异步barrier协调，producer完成一块后consumer立即使用
```

**底层硬件支持**：

**1. TMA（Tensor Memory Accelerator）**：
- Hopper新增的专用DMA引擎，独立于SM的计算管线。
- Producer Warp发射TMA指令后可以立即执行其他操作（完全异步）。
- TMA支持多维数据的自动寻址和swizzle，一条指令加载/存储整个2D/3D tile。
- 对比cp.async（Ampere）：cp.async仍占用Load/Store单元，TMA完全不占。

**2. 异步Barrier（mbarrier）**：
```
生产者完成: mbarrier.arrive()  → barrier计数+1
消费者等待: mbarrier.wait()    → 阻塞直到计数达到预期值
```
- 实现零拷贝流水线：Producer填充buffer A → signal → Consumer消费A同时Producer填充B。
- mbarrier是硬件原语，延迟极低（~几个cycle）。

**3. WGMMA（Warp Group Matrix Multiply Accumulate）**：
- Consumer Warp使用WGMMA指令（Warp Group级别的矩阵乘加）。
- WGMMA直接从shared memory读取操作数（无需先load到寄存器），减少寄存器压力。
- 一个Warp Group（4个Warp = 128线程）协作执行大矩阵乘。

**为什么性能更好**：
- Producer Warp持续发射TMA加载，不被计算中断→内存带宽利用率更高。
- Consumer Warp持续计算，不需要等数据加载→计算管线利用率更高。
- 减少了全Block的`__syncthreads()`（只有producer-consumer间的细粒度barrier）。

### Q: 如果去掉Warp Specialization，只保留Tile和Shared Memory优化，性能损失在哪？

**性能损失来源的量化分析**：

| 损失来源 | 机制 | 估计损失 |
|---------|------|---------|
| 计算-加载Overlap不充分 | 同一Warp交替做load和compute，无法完美流水 | 15-25% |
| TMA利用不充分 | 非specialization模式不容易高效使用TMA | 10-15% |
| 更多同步开销 | 全Block syncthreads代替细粒度mbarrier | 5-10% |
| 寄存器效率下降 | 每个Warp需要同时保存load和compute的状态 | 5-10% |

**详细分析**：

**1. Overlap损失**：
```
无Specialization（必须同步切换角色）:
时间: |--load--|--sync--|--compute--|--sync--|--load--|--compute--|
GPU:  |  LDU   | idle   |   FMA    | idle   |  LDU  |   FMA    |
利用率: 各50%

有Specialization（持续流水）:
时间: |--compute tile0--|--compute tile1--|--compute tile2--|
      |--load tile1----|--load tile2----|--load tile3----|
GPU:  |  FMA持续100%  |
      |  TMA持续100%  |
利用率: 接近100%
```

**2. TMA vs cp.async**：
- TMA：1条指令加载整个2D tile（如128×64的FP16 tile = 16KB），由专用硬件执行。
- cp.async：需要多条指令逐行加载，占用Load/Store执行单元。
- TMA的带宽利用率通常高10-20%（更少的指令开销、硬件级优化的内存访问模式）。

**3. 典型性能数据（GEMM, M=N=K=8192, FP16, H100）**：
- CUTLASS with Warp Specialization: ~750 TFLOPS（接近峰值989T的76%）
- CUTLASS without Warp Specialization: ~550 TFLOPS（56%峰值）
- 差距约30%，在大矩阵时更明显（小矩阵瓶颈在launch和occupancy上）。

### Q: 怎么判断一个MoE模型是真的学到了分工，而不是只把Dense模型拆开了？

**验证"有意义的分工"的多维度方法**：

**1. Expert激活模式分析**：
```python
# 统计不同输入类型（语言/领域/任务）的expert激活分布
for category in ["code", "math", "chinese", "english", "science"]:
    expert_counts = count_expert_activations(data[category])
    print(f"{category}: top experts = {expert_counts.argsort()[-3:]}")
    
# 如果真的分工：不同category应该有不同的top experts
# 如果没分工：所有category的top experts基本相同（expert collapse）
```

**2. 消融实验（最直接的证据）**：
```
移除Expert 3 → 数学能力下降20%但语言能力不变 → Expert 3专门处理数学
移除Expert 5 → 代码能力下降25%但其他不变 → Expert 5专门处理代码
移除Expert 0 → 所有能力均匀下降5% → Expert 0是通用expert
```
如果移除任何expert都导致所有能力均匀下降——说明没有分工（只是容量分散）。

**3. Token层面的Expert可解释性**：
- 收集被路由到各expert的token，分析它们的语义特征。
- 真分工的表现：Expert A主要处理数字/运算符token，Expert B主要处理代码关键字等。
- 无分工的表现：各expert处理的token类型分布相似。

**4. Router决策边界分析**：
- 将输入embedding可视化（t-SNE/PCA），着色为被路由到的expert。
- 有意义的分工：embedding空间中形成清晰的expert区域划分。
- 无分工：随机混合，无清晰边界。

**5. 对比Dense基线**：
- 同等激活参数量的Dense模型 vs MoE模型在各任务上的对比。
- 如果MoE只是"拆开"Dense，性能应该≈Dense（甚至因为路由噪声更差）。
- 真的分工使得MoE在总参数量×的条件下超过Dense，说明容量被有效利用。

### Q: RL+MoE中Reward把Routing学坏（所有token集中到少数Expert）怎么处理？

**Expert Collapse/偏好问题的根因**：

RL的reward信号可能无意中强化了"某些expert效果好"→更多token路由过去→该expert更擅长（更多训练机会）→正反馈循环→最终只有1-2个expert被使用（其他退化/死亡）。

**解决方案层次**：

**1. Load Balancing Loss（最标准的方案）**：
```python
# Switch Transformer风格的均衡损失
def load_balance_loss(router_probs, expert_indices, num_experts):
    # f_i: 实际分配给expert_i的token比例
    f = torch.zeros(num_experts)
    for i in range(num_experts):
        f[i] = (expert_indices == i).float().mean()
    
    # P_i: 所有token对expert_i的平均路由概率
    P = router_probs.mean(dim=0)  # [num_experts]
    
    # 均衡损失 (惩罚f和P的相关性)
    return num_experts * (f * P).sum()

total_loss = task_reward_loss + alpha * load_balance_loss  # alpha=0.01
```

**2. Expert容量硬限制**：
```python
capacity = int(capacity_factor * tokens_per_expert)  # 如 1.25 * (total_tokens / num_experts)
# 超出capacity的token:
# - 方案A: 丢弃（不处理，输出为0）
# - 方案B: 路由到第二优先expert
# - 方案C: 路由到shared expert
```

**3. Reward设计中融入均衡性**：
```python
# 多目标reward
reward = task_reward - beta * imbalance_penalty
imbalance_penalty = max_expert_load / avg_expert_load - 1.0  # 0=完美均衡
```
直接将均衡性作为reward的一部分，让RL策略主动学习均衡路由。

**4. Router正则化**：
```python
# 对router输出加entropy正则，鼓励探索（不要太确定地选某个expert）
router_entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1)
loss += -gamma * router_entropy.mean()  # 最大化entropy = 均匀路由
```

**5. 渐进式训练策略**：
- Phase 1：纯SFT训练，让router在监督信号下学到相对均匀的分工。
- Phase 2：加入RL微调，但用较强的均衡约束（大alpha）。
- Phase 3：逐步减小均衡约束，让RL有更多自由度。
- 确保router不会在RL阶段突然崩溃。

**6. Shared Expert保底**：
- 设计1-2个永远激活的shared expert（不参与路由选择）。
- 即使路由expert失衡，shared expert保证基本能力不丢失。
- DeepSeek-MoE的做法：2个shared + 64个routed expert，top-6选择。
