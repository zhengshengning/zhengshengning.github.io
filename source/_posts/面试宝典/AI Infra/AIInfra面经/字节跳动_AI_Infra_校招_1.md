---
title: '字节跳动 AI Infra 校招 (1)'
date: 2026-04-16 12:07:32
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 大模型推理加速方法有哪些？

大模型推理加速是一个系统工程，需要从算法、内存、调度、硬件利用多个维度综合优化。下面按类别详细展开：

**1. KV Cache（缓存历史计算）**：
- 原理：自回归生成中，前面token的Key和Value在后续步骤不变。缓存后每步只需计算新token的KV，避免O(n^2)重复计算。
- 显存占用公式：`2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × sizeof(dtype)`
- 70B模型FP16：每1K tokens约占1.4GB KV Cache。

**2. 量化（减少数据精度）**：

| 方案 | 位宽 | 加速来源 | 精度损失 | 典型用途 |
|------|------|---------|---------|---------|
| W8A8 | 8-bit | 带宽减半+INT8 TC | <1% | 通用部署 |
| W4A16 | 4-bit权重 | 权重带宽↓4x | 1-3% | 单卡大模型 |
| FP8 | E4M3/E5M2 | H100原生支持 | <0.5% | Hopper部署 |
| W4A4 | 4-bit全量化 | 极限压缩 | 3-5% | 边缘端 |

关键洞察：Decode阶段是memory-bound（每步计算量小但需读全部权重），量化直接减少带宽需求从而加速。

**3. FlashAttention（IO感知算法）**：
- 核心思想：将Attention计算分块（tiling），每块在SRAM中完成，避免中间N×N矩阵写回HBM。
- HBM访问从O(N^2)降为O(N^2/SRAM_size)，实际加速2-4x。
- FlashAttention-2进一步优化了并行度（sequence维度并行）和warp间工作分配。
- FlashDecoding：针对Decode阶段（单query多KV）做KV维度并行，利用更多SM。

**4. PagedAttention（显存管理革新）**：
- 传统KV Cache需为每个请求预分配max_seq_len的连续显存→碎片严重、浪费50-70%。
- PagedAttention将KV Cache分为固定大小block（如16 tokens），通过block table维护逻辑→物理映射。
- 效果：显存利用率从30-50%提升到>95%，同样显存可服务2-3x更多并发请求。
- 支持Copy-on-Write：共享前缀的请求复用相同物理block。

**5. 投机解码（Speculative Decoding）**：
```
小模型（7B）快速生成K个候选token（draft）
    ↓
大模型（70B）一次前向验证K个token（并行验证）
    ↓
接受前j个正确token + 重采样第j+1个token
有效输出: j+1个token / 1次大模型前向
```
- 加速比取决于接受率α：加速 ≈ 1/(1-α) × K/(K+1)。通常α=0.7-0.9时加速2-3x。
- 适用：大小模型能力差距适中的场景。差距太大则接受率低。

**6. Continuous Batching（动态批处理）**：
- 传统静态batch：所有请求等最长的完成→短请求被padding浪费。
- Continuous Batching：每个decode step检查完成的请求，移出并填入新请求。
- GPU利用率从30-40%提升到70-90%。吞吐量提升2-3x。
- vLLM/TGI/TensorRT-LLM都实现了此机制。

**7. 算子融合（Kernel Fusion）**：
- 减少kernel launch开销（每次~5us×上百个kernel）。
- 消除中间tensor的HBM写入/读取（最关键收益）。
- 典型融合：QKV投影融合、LayerNorm+残差融合、SiLU×Up投影融合。
- FasterTransformer/TensorRT-LLM中大量应用，单层延迟减少30-50%。

**8. 张量并行（Tensor Parallelism）**：
- 按列切分Q/K/V/Up投影，按行切分Output/Down投影。
- 每层2次AllReduce（前向1次、反向1次→推理只需前向1次）。
- 节点内NVLink（900GB/s）互联时开销小，直接降低单请求延迟。
- TP=8可将70B模型的单步延迟降为TP=1的~1/4（不是1/8，通信有开销）。

**9. PD分离（Prefill-Decode Disaggregation）**：
- Prefill（compute-bound，大batch矩阵乘）和Decode（memory-bound，小batch向量-矩阵乘）的最优配置不同。
- 分离后各自独立优化：Prefill用高算力GPU、Decode用高带宽GPU/多卡并行。
- 消除Prefill挤占Decode的调度干扰，降低P99延迟。
- DistServe/Splitwise等系统已验证30-50%效率提升。

**各优化技术的收益来源总结**：

| 方法 | 针对瓶颈 | 核心收益 |
|------|---------|---------|
| KV Cache | 重复计算 | 消除O(n^2)重计算 |
| 量化 | 内存带宽 | 权重读取量↓2-4x |
| FlashAttention | HBM带宽 | 中间矩阵不落HBM |
| PagedAttention | 显存碎片 | 利用率↑、并发↑ |
| 投机解码 | 自回归串行 | 有效步数↓2-3x |
| Continuous Batching | 调度浪费 | GPU利用率↑ |
| 算子融合 | Kernel开销+中间IO | launch+中间读写消除 |
| 张量并行 | 单卡延迟 | 延迟↓近线性 |
| PD分离 | 混合调度干扰 | 各阶段独立优化 |

---

### Q: 模型压缩量化方法有哪些？量化的优点有哪些？

**量化方法分类**：

**1. 训练后量化（PTQ, Post-Training Quantization）**：
不需要重新训练，直接对已训练好的模型进行量化。

| 方法 | 核心思想 | 实现方式 | 适用场景 |
|------|---------|---------|---------|
| GPTQ | 逐列量化+误差补偿 | OBQ框架，逐列量化权重并将误差分散到未量化列 | W4量化首选 |
| AWQ | 激活感知保护 | 找到对激活影响最大的1%重要权重通道，乘scale保护 | W4质量优先 |
| SmoothQuant | 平滑激活异常值 | 将激活的量化难度（outlier）数学等价转移到权重上 | W8A8部署 |
| GPTQ-R/QuIP | 旋转+向量量化 | 随机旋转消除权重相关性，使量化更均匀 | 极低bit(2-3bit) |
| FP8量化 | 硬件原生格式 | E4M3(前向)/E5M2(反向)，H100/4090原生支持 | Hopper部署 |

**2. 量化感知训练（QAT, Quantization-Aware Training）**：
```python
# 前向：模拟量化（STE直通估计器）
x_quant = fake_quantize(x)  # 量化再反量化，引入量化噪声
y = layer(x_quant)           # 用量化后的值前向
# 反向：梯度直接穿过量化操作（STE假设量化函数导数=1）
loss.backward()              # 梯度不经过量化函数
```
- 精度损失最小（<0.5%），但需要额外训练开销（原始训练的5-20%）。
- 适用于精度要求极高或极低bit（如W2/W3）的场景。

**3. 量化格式详解**：

| 格式 | 数值范围 | 典型用途 | 硬件支持 |
|------|---------|---------|---------|
| INT8 | [-128, 127] | 权重+激活量化 | 所有现代GPU |
| INT4 | [-8, 7] | 仅权重量化 | 需反量化后计算 |
| FP8 E4M3 | [-448, 448] | 前向计算 | H100/4090 |
| FP8 E5M2 | [-57344, 57344] | 梯度/反向 | H100 |
| NF4 | 非均匀4bit | QLoRA基础模型 | 软件反量化 |
| MXFP4 | 微缩放FP4 | 下一代极低bit | Blackwell |

**量化粒度对精度的影响**：
```
Per-tensor:  整个tensor共享1个scale → 精度差，但overhead最小
Per-channel: 每个output channel一个scale → 权重量化的标准选择
Per-group:   每g个元素一个scale(g=128) → INT4必须，平衡精度和开销
Per-token:   每个token一个scale → 激活量化的标准选择（因为不同token的动态范围差异大）
```

**量化的优点详解**：

1. **减少模型体积**：
   - FP16→INT4：模型文件从140GB→35GB（70B模型）。
   - 更小的模型可以在消费级GPU上运行（如4090 24GB跑70B INT4模型）。

2. **降低显存占用**：
   - 同样显存支持更大模型或更多并发。
   - 70B FP16需要140GB（至少2×A100），INT4只需35GB（单卡A100）。

3. **加速推理（根本原因：缓解bandwidth bound）**：
   - Decode阶段每步需读取全部权重（70B = 140GB FP16）。
   - A100 HBM带宽2TB/s → FP16每步至少70ms，INT4每步17.5ms（理论4x加速）。
   - 实际加速2-3x（量化/反量化有开销）。
   - INT8 Tensor Core：直接在低精度下做矩阵乘，计算吞吐也翻倍。

4. **减少内存带宽需求**：
   - 带宽是Decode的核心瓶颈（Arithmetic Intensity < 1 FLOP/Byte）。
   - 量化直接减少每步需读取的字节数，从根本上缓解带宽瓶颈。
   - 在MoE模型中尤其重要：激活的Expert权重需要即时加载。

**量化的常见陷阱**：
- 激活中的outlier：某些channel的值可能是其他的100x，INT8无法表示→需要SmoothQuant/LLM.int8()特殊处理。
- 过度量化（如W2/W3）在小模型上精度崩溃严重，大模型容忍度更高。
- KV Cache量化（INT8/FP8）需注意attention score的数值精度影响。

---

### Q: 权重量化、激活值量化、KV Cache量化分别有什么好处？

三种量化目标针对推理系统的不同瓶颈，各有侧重：

**权重量化（Weight Quantization）**：

| 维度 | 详情 |
|------|------|
| 针对瓶颈 | Decode阶段的权重加载带宽瓶颈 |
| 量化粒度 | Per-channel或Per-group（INT4必须Per-group，g=128） |
| 精度影响 | 较小——权重分布相对稳定，容易量化 |
| 典型方案 | GPTQ W4、AWQ W4、FP8权重 |
| 加速原理 | 减少每步decode需读取的权重字节数 |

数值分析：70B模型Decode一步：
```
FP16: 读取140GB权重 / A100 2TB/s = 70ms
INT4: 读取35GB权重 / 2TB/s = 17.5ms（理论4x加速）
实际: 约2.5-3x（反量化计算+scale读取的overhead）
```

为什么权重容易量化：权重在训练后固定，分布可预先分析和优化（GPTQ通过误差补偿、AWQ通过scale保护，都能将量化误差降到最小）。

**激活值量化（Activation Quantization）**：

| 维度 | 详情 |
|------|------|
| 针对瓶颈 | GEMM计算吞吐（利用INT8/FP8 Tensor Core）和中间激活的显存 |
| 量化粒度 | Per-token（因为不同token的动态范围差异大） |
| 精度影响 | 较大——激活分布动态变化且有outlier |
| 典型方案 | SmoothQuant W8A8、FP8 E4M3 |
| 加速原理 | INT8 GEMM吞吐是FP16的2x + 中间激活显存减半 |

关键挑战——激活outlier问题：
```
正常channel值范围: [-3, 3]
异常channel值范围: [-100, 100]  (存在于~1%的channel中)
INT8量化范围: [-128, 127]

如果用max值确定scale: scale = 100/127
正常值被量化后只有[-3.8, 3.8] → 有效精度仅3-4bit → 精度崩溃！

SmoothQuant解决方案:
将激活的outlier通过数学等价变换"平滑"到权重上:
Y = (X × diag(s)^{-1}) × (diag(s) × W) = X_smooth × W_smooth
X_smooth的range缩小 → 容易量化
W_smooth略微扩大 → 权重本来就容易量化，影响小
```

**KV Cache量化（KV Cache Quantization）**：

| 维度 | 详情 |
|------|------|
| 针对瓶颈 | 长序列/大并发下KV Cache的显存占用 |
| 量化粒度 | Per-token或Per-head |
| 精度影响 | 较小——KV Cache的值分布相对平稳 |
| 典型方案 | INT8 KV、FP8 KV、甚至INT4 KV |
| 加速原理 | 同样显存可服务更多并发或更长序列 |

效果量化：
```
70B模型，2K序列长度，batch=32:
FP16 KV Cache = 2 × 80层 × 8(KV heads) × 128(head_dim) × 2048 × 32 × 2bytes ≈ 42GB
INT8 KV Cache = 21GB（节省21GB，可以多服务约50%请求）
INT4 KV Cache = 10.5GB（节省75%，但需注意attention精度）
```

KV Cache量化的额外收益：减少Attention计算时KV的读取带宽需求（Decode阶段读KV也是瓶颈之一）。

**三者组合策略**：
- 生产环境最常见：W4A16 + FP16 KV（简单有效）。
- 高吞吐追求：W8A8 + INT8 KV（SmoothQuant全链路量化）。
- 极致压缩：W4A8 + INT4 KV（需要精心校准）。

---

### Q: KV Cache原理？PagedAttention为什么效果好？

**KV Cache原理详解**：

自回归生成的核心计算是Attention：
```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

每生成一个新token时：
```
第t步生成:
  Q_t = x_t × W_Q          (只有新token的query, shape [1, d])
  K_t = x_t × W_K          (新token的key)
  V_t = x_t × W_V          (新token的value)
  
  K_cache = concat(K_cache, K_t)   (追加到历史KV)
  V_cache = concat(V_cache, V_t)
  
  Attn_t = softmax(Q_t × K_cache^T / sqrt(d_k)) × V_cache
  // Q_t: [1, d], K_cache: [t, d] → Score: [1, t] → Output: [1, d]
```

无KV Cache时：第t步需要重新计算所有前t个token的KV → 计算量O(t × d^2)
有KV Cache时：第t步只计算新token的KV → 计算量O(d^2)，总节省约t/2倍

**KV Cache的显存挑战**：

```
单请求KV Cache显存 = 2 × L × n_kv × d_h × seq_len × sizeof(dtype)

LLaMA-70B具体数值:
  L=80层, n_kv=8(GQA), d_h=128, seq_len=4096, FP16:
  = 2 × 80 × 8 × 128 × 4096 × 2 bytes = 1.34GB/请求

batch=64时: 1.34GB × 64 = 85.5GB！超过单卡A100的80GB显存
```

这就是为什么KV Cache管理成为推理系统的核心挑战。

**PagedAttention为什么效果好——深入分析**：

**传统KV Cache分配方式的浪费**：
```
传统方案: 为每个请求预分配max_seq_len的连续显存
问题:
  - 请求1实际长度256，预分配4096 → 浪费93.75%
  - 请求2实际长度1024，预分配4096 → 浪费75%
  - 无法利用非连续的空闲碎片
  - 平均显存浪费60-80%（论文实测）
```

**PagedAttention的解决方案**：
```
将KV Cache分为固定大小的block（如每block存16个token的KV）:

Block Table (逻辑→物理映射):
  Request 1: [逻辑block 0→物理block 7] [逻辑block 1→物理block 2] ...
  Request 2: [逻辑block 0→物理block 5] [逻辑block 1→物理block 11] ...

物理block可以分散在显存任意位置（类似OS的虚拟内存页表机制）
```

效果好的具体原因：

| 优势 | 机制 | 量化收益 |
|------|------|---------|
| 消除碎片 | 固定大小block按需分配，不预留 | 显存利用率30%→95% |
| 精确分配 | 生成一个block才分配一个block | 无过度预分配浪费 |
| 共享前缀 | Copy-on-Write机制复用physical block | 多请求共享system prompt节省50%+ |
| 灵活回收 | 请求完成或被抢占时逐block释放 | 碎片化程度从O(请求数)降为0 |
| 高并发 | 同样显存可容纳更多请求 | 吞吐量提升2-4x |

**PagedAttention的kernel实现**：
```python
# 关键: Attention计算需要适配非连续的block布局
# 对每个query token:
for block_idx in block_table[request_id]:
    physical_block = kv_blocks[block_idx]  # 物理地址
    # 计算该block内所有token与query的attention score
    scores = query @ physical_block.keys.T / sqrt(d)
    # 累积softmax分母和加权value
    ...
```

vLLM的实现中，这个Attention kernel经过高度优化（CUDA手写），性能接近连续内存的FlashAttention。

---

### Q: Prefix Cache的作用？

**Prefix Cache的原理和收益详解**：

**问题场景**：实际部署中大量请求共享相同前缀（system prompt / 工具描述 / few-shot示例）：
```
请求1: [system prompt(2000 tokens)] + [用户问题A(100 tokens)]
请求2: [system prompt(2000 tokens)] + [用户问题B(150 tokens)]
请求3: [system prompt(2000 tokens)] + [用户问题C(80 tokens)]

没有Prefix Cache: 每个请求都要重新计算2000 tokens的KV → 3次prefill 2000 tokens
有Prefix Cache:  第一次计算后缓存，后续复用 → 1次prefill + 3次只prefill用户部分
```

**实现机制**：

```
1. Hash匹配: 对prefix内容计算hash → 查找是否有已缓存的KV
   hash(token_ids[0:prefix_len]) → cache_key
   
2. 缓存存储: 将prefix的KV Cache按block存储在GPU显存中
   prefix_kv_blocks = [block_0, block_1, ..., block_n]
   
3. 复用: 新请求到达时，block table直接引用已缓存的block
   new_request.block_table = prefix_blocks + [新分配的block...]
   
4. 淘汰策略: LRU/LFU淘汰低频prefix（显存有限）
```

**收益量化**：

| 指标 | 无Prefix Cache | 有Prefix Cache | 提升 |
|------|---------------|---------------|------|
| TTFT（首token延迟） | 500ms（2K prefix） | 50ms（仅100 token用户输入） | 10x↓ |
| Prefill算力 | 每请求都计算全prefix | 只计算非prefix部分 | 节省90%+（取决于prefix比例） |
| 吞吐量 | 受Prefill算力限制 | Prefill压力大幅减轻 | 2-5x↑ |
| 显存（多请求共享） | N份prefix KV | 1份prefix KV | N倍节省 |

**vLLM中的Prefix Cache实现**：
- Automatic Prefix Caching (APC)：无需用户显式标记prefix。
- 基于token序列内容的hash自动检测共享前缀。
- 与PagedAttention完美配合：共享的prefix blocks通过block table引用，Copy-on-Write保护。
- 支持多级匹配：如果两个请求的前1500 tokens相同但后500不同，也只需重算后500。

**适用场景**：
- ChatBot（固定system prompt占总输入的80%+）。
- RAG应用（共享的retrieval context）。
- 批量评测（相同few-shot prefix + 不同测试问题）。
- 多轮对话（前几轮对话历史作为prefix）。

**与PagedAttention的协同**：Prefix Cache本质是PagedAttention框架下的一个优化——通过block table的物理block共享（reference counting）实现prefix复用，是虚拟内存"共享库映射"思想在推理系统中的应用。

---

### Q: AI Agent的构建包含哪些模块？

AI Agent是基于大语言模型的自主行动系统，其构建涉及多个协同工作的模块。每个模块解决特定问题：

**1. LLM引擎（推理和决策核心）**：
- 作用：理解指令、推理规划、生成文本、调用决策。
- 技术选择：API调用（GPT-4/Claude）或自部署（vLLM/TGI + 开源模型）。
- 关键要求：足够的推理能力、工具调用格式遵循能力、低延迟。
- 优化：针对Agent场景的prompt engineering、function calling微调、长上下文支持。

**2. 工具系统（扩展能力边界）**：
```python
# 工具定义规范（以OpenAI Function Calling格式为例）
tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索互联网获取最新信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询词"},
                "num_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    }
}]
```
- 核心组件：工具注册表、参数验证、执行引擎、结果解析、错误处理。
- 常见工具类型：Web搜索、代码执行（沙箱）、数据库查询、文件操作、API调用。
- 关键挑战：工具描述的清晰度直接影响LLM的调用准确率。

**3. 记忆系统（持久化和检索）**：

| 记忆类型 | 实现方式 | 容量 | 检索方式 | 适用信息 |
|---------|---------|------|---------|---------|
| 短期记忆 | 对话历史(in-context) | 受context window限制 | 无需检索 | 当前对话上下文 |
| 工作记忆 | Scratchpad/State变量 | 小 | 直接引用 | 当前任务中间状态 |
| 长期记忆 | 向量数据库(Embedding) | 无限 | 相似度检索 | 历史经验、知识 |
| 外部记忆 | 文件系统/数据库 | 无限 | 结构化查询 | 结构化数据 |

- 短期记忆管理：Token限制下的滑动窗口/摘要压缩策略。
- 长期记忆实现：将重要信息embed后存入向量数据库（如Chroma/Pinecone），需要时检索top-K。
- 记忆更新：反思机制——任务完成后总结经验存入长期记忆。

**4. 规划模块（任务分解和执行策略）**：

| 规划范式 | 原理 | 优势 | 劣势 |
|---------|------|------|------|
| ReAct | Reasoning+Acting交替 | 可解释、灵活 | 容易偏离、长任务不稳定 |
| CoT | Chain-of-Thought推理 | 简单、推理质量高 | 不适合需要行动的任务 |
| ToT | Tree-of-Thought搜索 | 探索多条路径 | 计算开销大 |
| Plan-then-Execute | 先规划完整计划再执行 | 结构清晰 | 计划难以适应变化 |
| Reflexion | 执行→反思→修正循环 | 自我纠错 | 需要多轮交互 |

**5. RAG检索（知识增强）**：
- 流程：Query → Embedding → 向量检索 → Rerank → 注入Context → LLM生成。
- 关键技术：Chunk策略（大小/重叠）、Embedding模型选择、混合检索（稀疏+稠密）、Reranker质量。
- 与Agent的结合：Agent决定何时需要检索、检索什么、如何整合检索结果。

**6. 编排层（流程控制）**：
- 多轮交互循环：接收任务→规划→执行→观察→判断是否完成→迭代。
- 错误处理：工具调用失败时的重试、回退、替代方案。
- 最大步数限制：防止死循环或token消耗失控。
- 并行执行：识别可并行的子任务同时执行。
- 中间状态管理：长任务的checkpoint和恢复。

**7. 安全护栏（Safety Guardrails）**：
- 输入过滤：检测恶意prompt injection、越权指令。
- 输出过滤：敏感信息检测、毒性过滤、事实性检查。
- 行为边界：限制文件系统访问范围、网络访问白名单、API调用频率限制。
- 人类审批：高风险操作需人工确认（如发送邮件、执行支付）。
- 监控审计：完整的操作日志、异常行为告警。

**模块间的数据流**：
```
用户输入 → 安全护栏(输入过滤) → 记忆系统(加载上下文)
    → 规划模块(分解任务) → LLM引擎(生成action)
    → 工具系统(执行action) → 记忆系统(存储结果)
    → LLM引擎(总结/继续) → 安全护栏(输出过滤) → 用户
```

---

### Q: 手撕：环上n步回到原点的走法数（DP）？

（编程题）
