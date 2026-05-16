---
title: '阿里巴巴 控股集团 AI Infra 一面'
date: 2026-04-16 12:08:20
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Expert Parallelism中的dispatch和combine是什么？

在MoE的Expert Parallelism（EP）中，不同的专家分布在不同的GPU上。当一个GPU上的token需要被某个远程专家处理时，需要通过**AllToAll通信**进行数据重分布：

**Dispatch（分发）— 前向传播的第一次AllToAll**：

```
Before Dispatch (each GPU has local tokens):
  GPU0: [token_A, token_B, token_C]  (经路由: A→Expert0, B→Expert2, C→Expert1)
  GPU1: [token_D, token_E, token_F]  (经路由: D→Expert1, E→Expert0, F→Expert2)

AllToAll Dispatch:
  GPU0 (holds Expert0): receives [token_A, token_E]   ← 被路由到Expert0的token
  GPU1 (holds Expert1): receives [token_C, token_D]   ← 被路由到Expert1的token
  GPU2 (holds Expert2): receives [token_B, token_F]   ← 被路由到Expert2的token
```

- 每个GPU根据路由结果，将自己的token发送到对应专家所在的GPU
- 通信模式：每个GPU向每个其他GPU发送**不同数量**的token（取决于路由结果）
- 数据量不均匀是性能优化的难点

**Combine（汇聚）— 前向传播的第二次AllToAll**：

```
After Expert Computation:
  GPU0 (Expert0): [Expert0(token_A), Expert0(token_E)]
  GPU1 (Expert1): [Expert1(token_C), Expert1(token_D)]
  GPU2 (Expert2): [Expert2(token_B), Expert2(token_F)]

AllToAll Combine:
  GPU0: receives [Expert0(A), Expert1(C), Expert2(B)]  ← token_A,B,C的专家输出
  GPU1: receives [Expert0(E), Expert1(D), Expert2(F)]  ← token_D,E,F的专家输出

Final (weighted sum by routing probabilities):
  GPU0: output_A = w_A * Expert0(A)
        output_B = w_B * Expert2(B)  (top-k=1 example)
        output_C = w_C * Expert1(C)
```

- 各专家计算完成后，将结果发送**回原始token所在的GPU**
- 按路由权重加权求和（如果top-k>1，需要合并多个专家的输出）

**通信开销分析**：
- 每次AllToAll的数据量 = `num_tokens * hidden_dim * dtype_size`
- 两次AllToAll总通信量 = `2 * batch_size * seq_len * hidden_dim * 2B`（FP16）
- 对于batch_size=1, seq_len=4096, hidden_dim=4096, FP16：每层约64MB × 2 = 128MB通信

**优化方向**：
1. **Overlap计算与通信**：将AllToAll分解为多步P2P通信，与共享专家的计算overlap
2. **减少通信数据量**：对dispatch的activation做FP8量化（DeepSeek-V3：通信量减半）
3. **分层AllToAll**：先节点内NVLink完成本地交换，再跨节点InfiniBand（减少慢链路数据量）
4. **减少活跃专家数**：top-k越小，负载越集中，通信pattern越稀疏
5. **Capacity Factor优化**：限制每个专家接收的最大token数，避免极端不均匀

### Q: vLLM最近有什么好的新特性？

vLLM作为最流行的开源LLM推理引擎，持续演进中的关键特性：

**1. Chunked Prefill（分块预填充）**
- **问题**：长prompt的prefill计算量大（O(N^2)），会长时间独占GPU导致decode请求延迟spike
- **方案**：将长prompt分成多个chunk（如2048 token/chunk），每个调度周期只处理一个chunk，与decode请求交替执行
- **效果**：decode延迟P99降低3-5倍，代价是TTFT略微增加
- **适用**：对decode延迟敏感的在线服务

**2. Prefix Caching（前缀缓存）**
- **原理**：自动检测多个请求之间的公共前缀（如相同的system prompt），复用已计算的KV Cache
- **实现**：对token序列做hash，在KV Cache中维护hash→physical block的映射
- **效果**：如system prompt有2K token，100个请求共享前缀可节省200K token的重复计算
- **SGLang的RadixAttention**：更进一步，用Radix Tree管理所有前缀，支持最长公共前缀匹配

**3. LoRA多适配器动态切换**
- **能力**：同一base模型上加载多个LoRA adapter，按请求路由到不同adapter
- **实现**：LoRA权重存储在CPU/GPU，按需加载到计算路径
- **应用**：多租户场景——同一推理实例服务多个fine-tuned模型
- **性能**：LoRA切换开销很小（只有delta权重参与计算）

**4. Speculative Decoding（投机解码）**
- **原理**：用小模型（draft model）快速生成K个候选token，大模型一次性并行验证
- **效果**：当draft model acceptance rate>70%时，可获得2-3倍加速
- **vLLM支持**：集成draft model推理 + 验证逻辑 + KV Cache回滚机制
- **变体**：Medusa（多头预测）、Eagle（自回归draft）

**5. 多模态支持**
- 支持VLM（Vision-Language Model）推理：LLaVA、Qwen-VL、InternVL等
- 处理：图像预处理 → 视觉编码器 → 视觉token拼接到文本token → LLM推理
- 挑战：不同图片产生不同数量的视觉token，影响调度和batching

**6. FP8量化推理**
- **硬件**：H100/H200原生支持FP8 Tensor Core
- **效果**：相比FP16，吞吐提升~1.5倍，显存减半，质量损失极小
- **实现**：权重+activation都用FP8，KV Cache可选FP8存储
- **需要**：per-tensor或per-channel的scaling factor

**7. Disaggregated Prefill/Decode（PD分离）**
- **动机**：Prefill是compute-bound（大矩阵乘），Decode是memory-bound（小batch GEMV）→二者对硬件需求不同
- **方案**：Prefill和Decode部署在不同GPU/集群上
  - Prefill集群：高计算密度，处理长prompt
  - Decode集群：高内存带宽，处理逐token生成
- **通信**：Prefill计算完的KV Cache通过高速网络传输到Decode集群
- **效果**：Prefill和Decode各自可独立扩缩容，资源利用率更高

### Q: DeepSeek的整体结构是什么？

DeepSeek-V3（671B参数，37B激活）的核心架构创新：

**1. MLA（Multi-head Latent Attention）**

传统Attention的KV Cache = `num_heads * head_dim * seq_len`（如128 * 128 * seq_len = 16K/token）

MLA的设计：
```
Input X → Down_proj → Latent c (低维, 如512维) → Up_proj_K → K
                                                  → Up_proj_V → V
KV Cache只需存储latent c → 512维/token (vs 16K维)，压缩比32x
```

- **关键技巧**：RoPE位置编码与KV压缩解耦
  - 将K分为两部分：content-dependent（从latent解压）+ position-dependent（单独存储rope_dim维的KV）
  - 总KV Cache = `(latent_dim + rope_dim) * seq_len`，仍远小于MHA

**2. DeepSeekMoE（细粒度专家设计）**

与传统MoE（如Mixtral的8个大专家）不同，DeepSeek使用：
- **256个细粒度路由专家**：每个专家的FFN intermediate_dim较小
- **1-2个共享专家**：始终激活，处理通用模式（不参与路由）
- **每token选top-8路由专家** + 共享专家

设计理由：
- 细粒度专家允许更精确的路由（类似"256个小工具"vs"8个大百科"）
- 共享专家确保通用能力不因路由随机性而丢失
- 激活参数量 ≈ 共享专家参数 + 8个路由专家参数 ≈ 37B

**3. Multi-Token Prediction（MTP）**

- 标准LLM只预测下一个token；MTP同时预测未来2-6个token
- 实现：额外的prediction head，每个head预测不同未来位置
- 训练时作为辅助loss（auxiliary objective），提升表示学习质量
- **推理时的bonus**：MTP头可直接复用为Speculative Decoding的draft head
  - 零额外成本获得draft候选token
  - 验证通过率高（因为与主模型同源训练）
  - 实测加速1.8-2.4倍

**4. 其他技术选择**
- **位置编码**：RoPE（与MLA联合设计的解耦版本）
- **归一化**：RMSNorm（Pre-Norm位置）
- **激活函数**：SwiGLU（FFN内部）
- **训练精度**：FP8混合精度训练（前向FP8 + 反向FP8 + 优化器FP32）
- **负载均衡**：Auxiliary-loss-free方法（bias调节而非loss惩罚）
- **训练规模**：14.8T token，2.788M H800 GPU hours

**DeepSeek-V3 vs 其他模型对比**：

| 维度 | DeepSeek-V3 | LLaMA-3 405B | GPT-4 |
|------|-------------|-------------|-------|
| 总参数 | 671B | 405B | ~1.8T(传闻) |
| 激活参数 | 37B | 405B(Dense) | ~220B(传闻MoE) |
| KV Cache效率 | 极高(MLA) | 中(GQA) | 未知 |
| 训练成本 | $5.6M | $30M+(估) | $100M+(估) |

### Q: Tensor Parallelism中AllReduce的次数？

标准Transformer一层中，使用Megatron-LM风格的Tensor Parallelism：

**详细分析**：

**Attention子层**：
```
Input → LayerNorm → [Column Parallel: QKV Projection] → Attention Computation 
→ [Row Parallel: Output Projection] → AllReduce → + Residual → Output
```
- QKV投影列切分：每GPU计算部分head的Q/K/V，**无需通信**（输入broadcast/identity）
- Output投影行切分：每GPU计算部分和，**需要AllReduce**求和
- **Attention子层：1次AllReduce**

**MLP子层**：
```
Input → LayerNorm → [Column Parallel: Up Projection] → GeLU/SwiGLU 
→ [Row Parallel: Down Projection] → AllReduce → + Residual → Output
```
- Up投影列切分：**无需通信**
- Down投影行切分：**需要AllReduce**
- **MLP子层：1次AllReduce**

**每层总计：2次AllReduce（前向）+ 2次AllReduce（反向）= 4次AllReduce**

**实际Megatron的实现细节**：

前向传播中的通信模式（f和g是conjugate操作）：
- `f`操作：前向时identity（不通信），反向时AllReduce
- `g`操作：前向时AllReduce，反向时identity

所以从整体看：
- **前向：2次AllReduce**（Attention后1次 + MLP后1次）
- **反向：2次AllReduce**（对应f操作的反向）

**引入Sequence Parallelism后的变化**：

Sequence Parallelism将LayerNorm和Dropout的计算也并行化（沿sequence维度切分），此时：
- AllReduce被分解为 **Reduce-Scatter + All-Gather**
- Reduce-Scatter在Row Parallel Linear后执行
- All-Gather在Column Parallel Linear前执行
- **总通信量不变**（2M vs 2M），但：
  - 通信可以与LayerNorm/Dropout计算overlap
  - 每次通信的数据量减半（AllReduce的一半）
  - 显存节省：activation不需要每GPU存完整副本

**通信量计算**：
- 每次AllReduce通信量：`2 * hidden_size * seq_len * batch * dtype_size`（Ring AllReduce的2*(N-1)/N≈2倍）
- 对于hidden=8192, seq=4096, batch=1, FP16, TP=8：
  - 每次AllReduce: 2 * 8192 * 4096 * 2 ≈ 128MB
  - 每层4次: 512MB
  - 80层总计: ~40GB通信量/iteration
- NVLink带宽450GB/s (A100) → 通信时间约90ms → 这就是为什么TP只在NVLink内使用

### Q: KV Cache的大小如何计算？

**通用公式**：

```
KV Cache Size = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

各项含义：
- `2`：Key和Value各一份
- `num_layers`：Transformer层数
- `num_kv_heads`：KV head数量（MHA=num_heads, GQA=num_groups, MQA=1）
- `head_dim`：每个head的维度（通常128）
- `seq_len`：序列长度（prompt + generated tokens）
- `batch_size`：并发请求数
- `dtype_bytes`：数据类型字节数（FP16=2, FP8=1, INT8=1）

**主流模型KV Cache计算**：

| 模型 | Layers | KV Heads | Head Dim | 4K seq单条(FP16) |
|------|--------|----------|----------|-----------------|
| LLaMA-7B | 32 | 32(MHA) | 128 | 2×32×32×128×4096×2 = 2.1GB |
| LLaMA-70B | 80 | 8(GQA) | 128 | 2×80×8×128×4096×2 = 1.3GB |
| LLaMA-405B | 126 | 8(GQA) | 128 | 2×126×8×128×4096×2 = 2.1GB |
| DeepSeek-V3 | 61 | MLA(512d) | - | 61×(512+64)×4096×2 = 287MB |

**为什么KV Cache优化至关重要**：

1. **显存占用**：LLaMA-70B在batch=32, seq=4K时，KV Cache = 1.3GB × 32 ≈ 42GB，**超过模型权重本身**（FP16下140GB，TP8后每GPU约17.5GB）
2. **限制并发数**：80GB GPU上，权重占17.5GB，KV Cache每条请求1.3GB→最多只能服务(80-17.5)/1.3≈48条并发请求
3. **带宽瓶颈**：Decode时每步需要读取所有KV Cache（全部seq_len的KV），是decode阶段的主要性能瓶颈

**优化效果对比**：

| 优化方法 | LLaMA-70B单条4K(FP16) | 压缩比 |
|---------|----------------------|--------|
| MHA (基线) | 10.5GB | 1x |
| GQA-8 (默认) | 1.3GB | 8x |
| GQA-8 + INT8量化 | 0.65GB | 16x |
| GQA-8 + INT4量化 | 0.33GB | 32x |
| MLA (DeepSeek) | ~0.3GB | 35x |

这就是为什么GQA/MLA/量化/PagedAttention的组合优化对LLM推理部署如此关键——直接决定了可以服务多少并发用户。

### Q: 手撕：实现MQA和GQA？

（编程题）
