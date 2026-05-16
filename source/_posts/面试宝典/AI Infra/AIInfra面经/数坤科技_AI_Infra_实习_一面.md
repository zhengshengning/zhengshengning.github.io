---
title: '数坤科技 AI Infra 实习 一面'
date: 2026-04-16 12:06:32
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 面经]
---


<!-- more -->

---

### Q: Transformer的架构是什么？KV-Cache加速体现在哪里？

**Transformer架构：**

Transformer由Encoder和Decoder两部分组成（原始论文），现代LLM主要使用Decoder-only架构。

```
Encoder Layer (×N):
  Multi-Head Self-Attention (双向) → Add & LayerNorm → FFN → Add & LayerNorm

Decoder Layer (×N):
  Masked Multi-Head Self-Attention (因果) → Add & LayerNorm 
  → Cross-Attention (与Encoder交互) → Add & LayerNorm 
  → FFN → Add & LayerNorm
```

核心组件：
- **Multi-Head Attention**：多组QKV独立计算注意力再拼接
- **FFN（前馈网络）**：通常为两层MLP（先升维4x再降维），现代模型用SwiGLU
- **残差连接**：缓解梯度消失，允许信息直接跳过层传递
- **LayerNorm**：稳定训练（Pre-LN更常用，放在attention/FFN之前）
- **位置编码**：RoPE（旋转位置编码）是主流

**KV-Cache加速原理：**

在自回归推理中，每生成一个新token需要计算它与所有历史token的注意力：

```
不使用KV-Cache：
  Step t: 计算所有t个token的K,V → Attention → 输出第t个token
  Step t+1: 重新计算所有t+1个token的K,V → Attention → 输出第t+1个token
  每步重复计算量：O(t × d)的KV投影

使用KV-Cache：
  Step t: 只计算新token的Q,K,V → 将K,V追加到cache → Q与完整cache做Attention
  每步增量计算量：O(1 × d)的KV投影 + O(t × d)的Attention
```

**加速效果量化：**
- Prefill阶段：所有token并行计算（类似训练），无需cache
- Decode阶段：每步只需计算1个token的QKV投影（而非重算所有历史token的KV）
- 计算量从O(n^2 × d)（重算全部）降为每步O(n × d)（增量attention）
- 代价：需要O(n × d × 2 × num_layers × num_heads)的显存存储KV-Cache

**KV-Cache的显存占用：** 以LLaMA-70B为例，每个token的KV-Cache约1MB（FP16），4K上下文需要4GB/request

### Q: 并行策略有哪几种？

大模型训练中的并行策略用于将单卡无法承载的模型/数据分布到多个设备上：

**1. 数据并行(Data Parallelism, DP)：**
- 每个设备持有完整模型副本，数据均分到各设备
- 前向独立计算，反向梯度通过AllReduce同步（求平均）
- 变体：DDP（PyTorch原生）、ZeRO（分片优化器状态/梯度/参数减少冗余显存）
- 适用：模型能放入单卡，需要加速训练
- 通信量：每步2×模型参数量的数据（一次AllReduce）

**2. 张量并行(Tensor Parallelism, TP)：**
- 将单层的权重矩阵切分到多个设备
- MLP：第一层列切（各设备独立计算）→ 第二层行切（需AllReduce求和）
- Attention：按head切分（各设备处理不同head）→ 输出投影行切+AllReduce
- 每个Transformer层需要2次AllReduce（前向+反向各4次总共4对通信）
- **必须高带宽互联**：NVLink(600GB/s)，PCIe(64GB/s)不够用

**3. 流水线并行(Pipeline Parallelism, PP)：**
- 将模型按层切分到不同设备（如前16层在GPU0，后16层在GPU1）
- 通过micro-batch流水线化减少气泡（bubble）
- 1F1B调度：填充期→稳态期（交替1次前向1次反向）→排空期
- bubble比例：(p-1)/(p-1+m)，p为stage数，m为micro-batch数
- 通信量小（只传激活值），适合跨节点

**4. 序列并行(Sequence Parallelism, SP)：**
- 沿序列维度切分，每个设备处理序列的一部分
- 解决长序列Attention的O(n^2)显存问题
- Ring Attention / DeepSpeed Ulysses等实现

**5. 专家并行(Expert Parallelism, EP)：**
- MoE模型中不同expert分布在不同设备
- 需要All-to-All通信做token dispatch（路由到对应expert）和combine（结果收集回来）

**混合并行策略（3D/4D/5D并行）：**
```
典型配置（如Megatron-LM训练175B模型）：
- 节点内(8卡NVLink): TP=8（需要高带宽）
- 节点间(RDMA): PP=8（通信量较小）
- 跨PP组: DP=16（梯度同步）
```

### Q: 做并行时矩阵切分有什么注意的？数学原理是什么？

**矩阵切分的数学原理：**

对于 Y = X · W（X: [M, K], W: [K, N], Y: [M, N]）：

**列切分(Column Parallel)：W按列切分**
```
W = [W1 | W2]  (W1: [K, N/2], W2: [K, N/2])
Y = X·W = [X·W1 | X·W2]
```
- 每个设备独立计算X·Wi，无需通信
- 结果需要AllGather拼接（或直接传给下一层作为行切输入）
- Megatron MLP第一层（gate_proj/up_proj）使用列切

**行切分(Row Parallel)：W按行切分**
```
W = [W1; W2]  (W1: [K/2, N], W2: [K/2, N])
X = [X1 | X2]  (对应切分输入)
Y = X·W = X1·W1 + X2·W2  (需要AllReduce求和)
```
- 每个设备计算部分积，结果需要AllReduce
- Megatron MLP第二层（down_proj）使用行切

**Megatron-LM的MLP切分方案：**
```
          列切分(无通信)     行切分(AllReduce)
Input X → [Gate_proj列切] → [Down_proj行切] → Output
          各GPU独立计算       各GPU部分积求和
```
- 一对(列切+行切)只需一次AllReduce（前向），巧妙避免中间通信
- 反向传播时同样只需一次AllReduce

**关键注意事项：**
1. **数学等价性**：切分后的分布式计算必须与单设备结果严格一致
2. **维度整除**：切分维度必须被设备数整除（如N=4096按8卡切，每卡512列）
3. **通信类型选择**：列切后需AllGather(拼接), 行切后需AllReduce(求和)
4. **激活值通信**：TP中每层的输入X需要在所有设备上保持一致（forward时需broadcast/allgather）
5. **实际Megatron实现中**：列切输出直接作为行切输入（跳过中间通信），利用SiLU/GeLU的element-wise特性

### Q: 通信时延的影响因素？如何缓解？

**通信时延模型：**
```
通信时间 = 延迟(latency) + 数据量(bytes) / 带宽(bandwidth)
```

**影响因素详解：**

| 因素 | 量化参考 | 影响程度 |
|------|----------|---------|
| 互联带宽 | NVLink:600GB/s, PCIe5:64GB/s, IB:400Gb/s, TCP:25Gb/s | 决定大数据传输速度 |
| 延迟 | NVLink:~1us, RDMA:~2us, TCP:~50us | 决定小消息效率 |
| 数据量 | 梯度/激活值/KV-Cache大小 | 正比影响传输时间 |
| 网络拓扑 | 全连接 vs 环形 vs Fat-tree | 影响集合通信效率 |
| 拥塞 | 多job共享网络 | 抖动和尾延迟 |
| 协议开销 | RDMA(kernel bypass) vs TCP(多次拷贝) | 影响实际利用率 |

**缓解方法：**

**1. 计算通信重叠(Overlap)——最有效：**
```
// 边计算边通信
for layer in layers:
    compute(layer)           # 当前层计算
    async_allreduce(grad[layer-1])  # 上一层梯度已就绪，开始通信
```
- 前向：GEMM分块完成后立即开始该块的通信
- 反向：梯度就绪后立即开始AllReduce

**2. 减少通信量：**
- 梯度压缩/量化（FP32梯度量化为FP16/INT8传输）
- 梯度稀疏化（只传top-k大梯度）
- GQA/MQA减少KV-Cache大小（TP通信减少）

**3. 优化通信效率：**
- Ring AllReduce：将大消息分成chunks流水线传输，带宽利用率(n-1)/n
- 分层通信：节点内NVLink做局部AllReduce，节点间RDMA做全局AllReduce
- 融合小消息：将多个小tensor合并为一个大buffer一次通信

**4. 硬件层面：**
- NVLink/NVSwitch：节点内高带宽（900GB/s on H100）
- InfiniBand NDR/XDR：节点间高带宽低延迟
- SHARP/In-Network Computing：交换机内做AllReduce减少数据传输

### Q: 手撕：在全0方形矩阵中画出一个内切圆（用1填充）？

（编程题）
