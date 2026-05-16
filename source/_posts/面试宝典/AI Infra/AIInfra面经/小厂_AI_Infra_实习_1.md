---
title: '小厂 AI Infra 实习 (1)'
date: 2026-04-16 12:07:04
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 面经]
---


<!-- more -->

---

### Q: vLLM里Scheduler的调度过程？

**Scheduler的三队列状态机：**

```
                    ┌─────────┐
           新请求 → │ WAITING  │ (等待prefill)
                    └────┬────┘
                         │ 有足够KV blocks
                         ↓
                    ┌─────────┐
                    │ RUNNING  │ (正在生成)
                    └────┬────┘
                    ↙         ↘
         完成(EOS/max)     显存不足(抢占)
                ↓              ↓
           释放资源       ┌──────────┐
           返回结果       │ SWAPPED  │ (KV在CPU)
                         └────┬────┘
                              │ 显存释放后恢复
                              ↓
                         回到RUNNING
```

**每个调度Step的详细流程：**

```python
def schedule(self) -> SchedulerOutputs:
    # Phase 1: 检查running请求的资源需求
    running_scheduled = []
    for seq_group in self.running:
        if seq_group.needs_new_block():  # decode需要新KV block
            if self.block_manager.can_allocate():
                self.block_manager.allocate(seq_group)
                running_scheduled.append(seq_group)
            else:
                # 显存不足! 触发抢占
                self._preempt(seq_group)  # swap或recompute
        else:
            running_scheduled.append(seq_group)
    
    # Phase 2: 尝试从waiting加入新请求(prefill)
    prefill_scheduled = []
    while self.waiting:
        seq_group = self.waiting[0]
        num_blocks_needed = self._get_num_blocks(seq_group)
        if self.block_manager.can_allocate(num_blocks_needed):
            self.waiting.pop(0)
            self.block_manager.allocate(seq_group, num_blocks_needed)
            prefill_scheduled.append(seq_group)
            # 检查是否超过max_num_batched_tokens限制
            if total_tokens > self.max_num_batched_tokens:
                break
        else:
            break  # 资源不足,停止加入新请求
    
    # Phase 3: 尝试恢复swapped请求
    swap_in_scheduled = []
    while self.swapped:
        seq_group = self.swapped[0]
        if self.block_manager.can_swap_in(seq_group):
            self.swapped.pop(0)
            self.block_manager.swap_in(seq_group)
            swap_in_scheduled.append(seq_group)
        else:
            break
    
    return SchedulerOutputs(
        running=running_scheduled,
        prefill=prefill_scheduled,
        swap_in=swap_in_scheduled,
    )
```

**关键调度参数：**

| 参数 | 含义 | 典型值 | 影响 |
|------|------|--------|------|
| max_num_seqs | 最大并发请求数 | 256 | 太大→显存不足触发抢占 |
| max_num_batched_tokens | 每step最大token数 | 4096-8192 | 控制step延迟 |
| preemption_mode | 抢占策略 | "swap"/"recompute" | swap适合长序列 |
| swap_space_bytes | CPU swap空间 | 4-16GB | 决定能swap多少请求 |

---

### Q: vLLM中请求被抢占后续会怎样？

**两种抢占策略的完整流程：**

**Swap策略（适合长序列）：**

```
抢占时:
1. 选择victim请求(通常是最新加入的/最长的)
2. 将其所有KV blocks从GPU显存复制到CPU内存
   GPU Block [7,3,15] → CPU Block [0,1,2] (通过PCIe/NVLink)
3. 释放GPU blocks → 给其他请求使用
4. 请求状态: RUNNING → SWAPPED

恢复时:
1. Scheduler检测到有空闲GPU blocks
2. 将CPU中的KV blocks复制回GPU
   CPU Block [0,1,2] → GPU Block [新分配的物理block]
3. 更新block_table映射
4. 请求状态: SWAPPED → RUNNING
5. 从断点处继续decode(无需重新prefill)

延迟: 取决于KV大小
  1000 tokens × 320KB/token = 320MB
  PCIe Gen4: 320MB / 32GB/s ≈ 10ms swap out + 10ms swap in
```

**Recompute策略（适合短序列）：**

```
抢占时:
1. 直接丢弃victim请求的所有KV blocks
2. 释放GPU blocks
3. 请求状态: RUNNING → WAITING (回到等待队列)

恢复时:
1. 请求重新从waiting进入running
2. 重新执行完整的prefill (重新计算KV-Cache)
3. 从prefill结果继续decode

延迟: 取决于序列长度
  500 tokens的prefill: ~50ms (远小于swap 10GB的KV)
```

**策略选择依据：**

| 场景 | Swap | Recompute | 选择 |
|------|------|-----------|------|
| 短序列(< 256 tokens) | 传输快但overhead比例高 | 重算很快 | Recompute |
| 长序列(> 2048 tokens) | 传输慢但保留了大量计算 | 重算代价高 | Swap |
| CPU内存不足 | 无法swap | 始终可用 | Recompute |
| PCIe带宽拥挤 | swap慢 | 不用PCIe | Recompute |
| 频繁抢占 | swap开销累积 | 重算开销累积 | 根据实际测量选择 |

---

### Q: 投机采样（Speculative Decoding）的流程？vLLM和SGLang中的实现区别？

**投机采样的数学原理：**

```
目标: 保证生成质量等价于target model, 同时加速

核心定理(Rejection Sampling):
  如果 q(x) ≤ p(x), 以概率 q(x)/p(x) 接受token x
  如果 q(x) > p(x), 以概率 p(x)/q(x) 接受, 否则从修正分布采样
  
  其中: p(x) = target model概率, q(x) = draft model概率
  
  结果: 最终生成的token分布 = target model的分布(数学等价!)
```

**完整流程：**

```
Step 1: Draft阶段(小模型快速生成K个token)
  Draft Model (如68M参数):
    x_1 ~ q(·|prefix)
    x_2 ~ q(·|prefix, x_1)
    ...
    x_K ~ q(·|prefix, x_1, ..., x_{K-1})
  保存每步的概率: q(x_1), q(x_2), ..., q(x_K)

Step 2: Verify阶段(大模型一次前向验证K个token)
  Target Model (如70B参数):
    一次前向输入[prefix, x_1, x_2, ..., x_K] → 得到所有位置的p(·)
  
Step 3: 逐位置接受/拒绝
  for i = 1 to K:
    r = random()
    if r < min(1, p(x_i) / q(x_i)):
      接受 x_i, 继续
    else:
      拒绝 x_i
      从修正分布采样替代token: x_i' ~ norm(max(0, p(·) - q(·)))
      丢弃 x_{i+1}...x_K
      break

Step 4: Bonus token
  如果K个全部接受, 再从p(·|..., x_K)采样一个额外token
  
结果: 每次验证平均接受 α·K + 1 个token (α=平均接受率)
  加速比: (α·K + 1) / (1 + draft_time/verify_time)
  典型: α≈0.7, K=5 → 每次验证生成~4.5个token → 加速2-3x
```

**vLLM vs SGLang的实现区别：**

| 维度 | vLLM | SGLang |
|------|------|--------|
| Draft模型类型 | 小模型/ngram/Medusa | 小模型/Eagle/多种 |
| 验证模式 | 线性验证(K个token逐个验证) | 树形验证(多条路径并行验证) |
| KV-Cache管理 | 被拒绝的token需要回退KV | RadixTree中被拒绝分支直接丢弃 |
| 前缀复用 | 标准PagedAttention | RadixAttention自动复用前缀 |
| batch处理 | Draft和Target可分batch | 更紧密集成到serving循环 |
| 调度 | Scheduler统一管理draft/target | draft作为serving的一部分 |

**树形投机解码(SGLang)：**

```
线性(vLLM): draft生成 [A, B, C, D, E] → 验证5个token
  如果C被拒绝 → 只接受A, B → 浪费了D, E的验证计算

树形(SGLang): draft生成树:
       A
      / \
     B   B'
    / \    \
   C   C'   C''
  
  一次验证整棵树(多路径) → 被拒绝路径的兄弟可能被接受
  → 平均接受更多token
  
  KV-Cache: RadixTree天然支持树结构的前缀共享
```

---

### Q: GPTQ量化和SmoothQuant的原理？

**GPTQ——逐层最优权重量化：**

```
核心问题: 量化W为W_q, 最小化输出误差:
  min ||WX - W_q·X||²  (X是校准数据的激活值)

关键insight: 利用Hessian信息(二阶导数)做最优补偿

算法流程:
  对W的每一列(从左到右):
  1. 量化当前列: w_q = quantize(w)
  2. 量化误差: δ = w - w_q
  3. 用Hessian的逆(H^{-1})计算最优补偿:
     未量化的列 += δ · H^{-1}[col, remaining_cols] / H^{-1}[col, col]
  4. 补偿使得后续列补偿前面列的量化误差

数学: min ||WX - W_q·X||² 等价于:
  分组更新(128列一组), 对每组用Cholesky分解H做高效求解
  
结果: INT4/INT3量化精度接近FP16 (PPL增加<0.5)
时间: ~5分钟/层(70B模型总计~4小时)
```

**SmoothQuant——激活感知的平滑量化：**

```
问题: 激活值中有outlier(少数通道值极大)
  激活分布: [-0.1, 0.2, -0.3, 100.0, 0.1, ...]
                              ^^^^^ outlier!
  如果直接INT8量化:
    scale = 100.0/127 ≈ 0.79
    其他值量化后: 0.2/0.79 ≈ 0 (信息完全丢失!)

SmoothQuant的解决方案:
  Y = X · W = (X · diag(s)^{-1}) · (diag(s) · W) = X̃ · W̃
  
  s的选取: s_j = max(|X_j|)^α / max(|W_j|)^(1-α), α∈[0,1]
  
  效果: 
    X̃ = X / s → outlier通道除以大的s → 变平滑, 好量化
    W̃ = W · s → 权重乘以s → 稍微增大, 但权重本身好量化
    
  等价变换: 不改变输出结果, 但让X和W都更容易量化!

优势: 
  - 实现W8A8: 权重和激活都是INT8 → 可用INT8 Tensor Core
  - Prefill阶段有实际加速(compute-bound, INT8算力翻倍)
  - 精度好(outlier问题被"转移"到权重上)
```

**两种方法的对比：**

| 维度 | GPTQ | SmoothQuant |
|------|------|-------------|
| 量化对象 | 仅权重(Weight-only) | 权重+激活(W8A8) |
| 精度 | INT4/INT3 | INT8 |
| 校准需求 | 少量数据(~128样本) | 少量数据(~512样本) |
| 在线量化 | 不需(权重离线量化好) | 激活需在线量化 |
| 加速阶段 | Decode(memory-bound) | Prefill+Decode |
| 加速比 | Decode ~2-4x(带宽减少) | Prefill ~1.5-2x(INT8算力) |
| 数学原理 | Hessian最优补偿 | 等价缩放变换 |

---

### Q: DeepSeek V3中EPLB（Expert-Level Pipeline Load Balancing）推理是什么？

**MoE推理的负载不均问题：**

```
MoE路由: 每个token选top-K experts (如K=2)
实际分布极不均匀:
  Expert 0: 处理 10000 tokens (热门)
  Expert 1: 处理 8000 tokens
  ...
  Expert 63: 处理 50 tokens (冷门)

EP(Expert Parallelism)部署时:
  Device 0: [Expert 0, Expert 1]  → 负载最重, 其他设备等它
  Device 7: [Expert 62, Expert 63] → 很快完成, 空等

All-to-All通信后, 最慢的设备决定整体速度!
```

**EPLB的解决方案：**

```
核心思想: 热门Expert复制到多个设备, 冷门Expert合并

Step 1: 统计负载(离线分析或在线滑动窗口)
  load[i] = 过去N步中expert_i处理的平均token数

Step 2: 负载均衡分配
  目标: 每个设备的总负载尽量相等
  
  方案:
  - Expert 0(负载10000): 复制到Device 0和Device 1
    → 各处理5000 tokens
  - Expert 63(负载50): 与Expert 62合并到同一Device
    → 不浪费一整个设备的资源

  分配前:
    Dev 0: [E0(10000)]         ← 瓶颈!
    Dev 1: [E1(8000)]
    
  分配后(EPLB):
    Dev 0: [E0_copy1(5000), E60(200)]
    Dev 1: [E0_copy2(5000), E1_copy1(4000)]
    Dev 2: [E1_copy2(4000), E62(100), E63(50)]
    ...
    每个设备负载≈相等
```

**EPLB的实现细节：**

| 设计决策 | 方案 | 原因 |
|---------|------|------|
| 复制粒度 | Expert级别 | 比layer级别更细粒度 |
| 路由修改 | Router输出包含replica_id | 告诉All-to-All发给哪个副本 |
| 更新频率 | 每N步(如1000步)重新统计 | 负载分布可能变化 |
| 通信 | All-to-All需要知道每个token发给哪个device | 路由表包含device映射 |
| 权重同步 | 复制的Expert共享权重(只读) | 推理时权重不变 |

**效果：**
- 设备间负载差异从3-5x降到<1.2x
- 整体推理吞吐提升30-50%(消除了等待最慢设备的时间)
- 配合DualPipe进一步隐藏通信延迟

---

### Q: MLA在Prefill和Decode时的计算复杂度区别？MLA矩阵吸收优化？

**MLA(Multi-Head Latent Attention)的设计：**

```
标准MHA: KV-Cache存储完整的K和V
  K, V ∈ R^{seq_len × num_heads × head_dim}
  存储: 2 × seq_len × H × d bytes

MLA: 将KV投影到低维隐空间
  c = W_down · [K;V]  (c ∈ R^{seq_len × d_c}, d_c << H×d)
  KV-Cache只存储c!
  
  存储: seq_len × d_c bytes (减少 2Hd/d_c 倍)
  
  DeepSeek V2: d_c = 512, H×d = 128×128 = 16384
  压缩比: 16384×2 / 512 = 64x !
```

**Prefill时的计算：**

```
Prefill: 需要完整的Q, K, V做Attention
  1. 计算Q: Q = X · W_Q
  2. 从低维c恢复K: K = c · W_up_K  (显式解压)
  3. 从低维c恢复V: V = c · W_up_V  (显式解压)
  4. 标准Attention: O = softmax(QK^T/√d) · V

复杂度: 与标准MHA接近(需要解压K, V)
  但写入KV-Cache时只写c → 写入量小
```

**Decode时的矩阵吸收优化：**

```
Decode时: Q = x_new · W_Q (单个token的Q)
          K_cache = c (低维存储)

直觉方法: 每步decode都解压 K = c · W_up_K → 然后Q·K^T
  每步需要: seq_len × d_c × d 的矩阵乘 (解压开销!)

矩阵吸收: 将解压矩阵"吸收"到Q的投影中
  Q · K^T = Q · (c · W_up_K)^T
           = Q · W_up_K^T · c^T
           = (Q · W_up_K^T) · c^T
           = Q_absorbed · c^T

  令 Q_absorbed = Q · W_up_K^T (在Q侧预计算)
  
  结果: Attention score = Q_absorbed · c^T
  
  好处:
  - 不需要解压K! 直接用低维c做attention
  - Decode时只读取低维c → 带宽需求减少 64x
  - 计算量: seq_len × d_c (vs seq_len × H×d)

类似地, 输出计算:
  O = softmax(score) · V
    = P · c · W_up_V
    = (P · c) · W_up_V  (先算P·c得低维结果,再投影到高维)
```

**吸收后的Decode复杂度对比：**

| 维度 | 标准MHA | MLA(吸收后) | 比值 |
|------|---------|------------|------|
| KV-Cache读取/token | 2×H×d | d_c | 减少64x |
| Attention计算 | seq_len × H × d | seq_len × d_c | 减少32x |
| Q投影(额外) | H×d → H×d | H×d → d_c | 一次性 |
| 总体Decode加速 | 基准 | 显著加速 | Memory-bound→有效 |

---

### Q: DeepSeek V3.2有什么创新点？

**DeepSeek V3的四大核心创新：**

**1. FP8混合精度训练：**

```
传统: BF16/FP32训练 → 高显存、高计算需求
V3创新: 首次在671B MoE模型上成功使用FP8训练

关键技术:
- Tile-wise量化: 将权重/激活切成128×128的tile
  每个tile独立计算scale → 比per-tensor精度好10-100x
- 只在GEMM中使用FP8:
  前向: FP8 × FP8 → FP32累加
  反向: FP8梯度 × FP8激活 → FP32累加
  非GEMM操作(LayerNorm, Softmax)保持BF16
- 延迟缩放(Delayed Scaling):
  scale = FP8_MAX / amax_previous_step  (用上一步的max)
  避免了计算当前step max值的同步开销
  
效果: 节省~40%训练资源, 精度损失<0.1% (vs BF16)
```

**2. 无辅助Loss的MoE负载均衡：**

```
传统方法: 在总loss中加auxiliary loss惩罚负载不均
  L_total = L_main + α × L_balance
  问题: α难调, 过大影响模型质量, 过小不起作用

V3创新: Bias-based routing
  每个expert维护一个bias term:
    routing_score = router(x) + bias[expert_id]
  
  在线更新bias:
    if expert过于繁忙: bias -= Δ (减少被选概率)
    if expert过于空闲: bias += Δ (增加被选概率)
    
  效果: 无需auxiliary loss, 负载自然均衡
        且不干扰主loss的优化方向
```

**3. Multi-Token Prediction (MTP)：**

```
传统: 每个位置只预测下一个token
V3: 每个位置同时预测未来K个token

实现:
  标准输出: h → LM_head → P(x_{t+1})
  MTP输出:  h → MTP_head_1 → P(x_{t+2})
            h → MTP_head_2 → P(x_{t+3})
            ...

训练收益:
  - 提供更强的梯度信号(看得更远)
  - 隐式学习更长距离的依赖

推理收益:
  - MTP heads可作为投机解码的draft source
  - 不需要单独的draft model!
  - 与主模型共享大部分计算
```

**4. DualPipe流水线并行：**

```
传统1F1B Pipeline:
  Stage 0: [F0][F1][F2][F3][B3][B2][B1][B0]
  Stage 1: [  ][F0][F1][F2][F3][B3][B2][B1]
  Bubble:   ^空闲^

DualPipe: 双向流水线 + 计算通信重叠
  正向pipeline: F0→F1→F2→F3
  反向pipeline: 同时 B3←B2←B1←B0
  
  关键: 将每个micro-step分为:
    - 计算部分(不需要通信)
    - 通信部分(All-Reduce/Send-Recv)
  → 一个方向的计算与另一个方向的通信重叠
  
  效果: Bubble ratio < 1F1B, 通信被计算覆盖
```

---

### Q: SGLang中多模态开启TP时，ViT的image embedding在多个进程间怎么高效复用？

**多模态TP部署的挑战：**

```
场景: LLaVA模型 + TP=4
  ViT编码器: 输入image → image_embeddings [num_patches, hidden_dim]
  LLM: 将image_embeddings作为prefix token与text一起处理

问题: 4个TP rank各自是独立进程
  - ViT编码只需做一次(结果对所有rank相同)
  - 但每个rank都需要image_embeddings来做attention
  - 如何避免4次重复编码?
```

**SGLang的高效方案：**

**方案1: 单Rank编码 + Broadcast**

```
Rank 0: ViT_encode(image) → embeddings  // 只Rank 0执行
         ↓ Broadcast
Rank 0,1,2,3: 都获得相同的embeddings
         ↓
各Rank: 正常执行LLM前向(TP切分在head维度)
```

**方案2: 所有Rank独立编码(简单但冗余)**

```
每个Rank独立执行ViT:
  Rank 0: ViT_encode(image) → embeddings
  Rank 1: ViT_encode(image) → embeddings (完全相同!)
  ...
优点: 无通信开销
缺点: 4x冗余计算(ViT通常较小,可接受)
```

**方案3: RadixAttention前缀缓存（SGLang特色）：**

```
SGLang的RadixTree KV-Cache:

首次处理图像:
  image_hash = hash(image_bytes)
  embeddings = ViT_encode(image)
  → 将embeddings对应的KV-Cache存入RadixTree
  → key = image_hash + 位置信息
  
后续请求引用同一图像:
  查找RadixTree → 命中! 直接复用KV-Cache
  不需要重新编码ViT, 也不需要重新计算prefix attention
  
多个请求共享同一张图(如同一system prompt中的图):
  所有请求共享同一个prefix KV-Cache节点
  节省: num_requests × (ViT计算 + prefix attention计算)
```

**TP下各Rank的KV-Cache一致性：**

```
TP切分方式:
  Q: [batch, num_heads, seq_len, head_dim]
     切分num_heads → 每个rank处理 num_heads/TP 个head
  K/V: 同样切分head维度

Image embeddings: 每个rank只需要自己负责的head对应的KV
  → Broadcast时只需传输 head_dim/TP 大小的数据
  或: 每个rank存完整embedding,各自投影为自己负责的KV head

实际SGLang实现:
  1. 图像embedding在所有rank完整存在(不大,如576×4096=2.3MB)
  2. 各rank在自己的forward中将embedding投影为本rank的KV
  3. 投影后的KV自动进入PagedAttention管理
  4. RadixTree中image prefix KV被所有引用该图像的请求共享
```
