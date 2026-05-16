---
title: '字节跳动 AI Infra (1)'
date: 2026-04-16 12:17:08
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: KV Cache的原理和优化方法？

**原理——为什么需要KV Cache**：

Transformer自回归生成第t个token时，attention计算需要：
```
Q_t × K_{0:t}^T → Score → Softmax → × V_{0:t} → Output_t
```

没有KV Cache时，每步都要重新计算所有历史token的K和V（因为Q投影依赖当前token，但K/V投影只依赖各自的token embedding——不随新token变化）。

有KV Cache时：
```
Step t:
  1. 计算新token的 K_t = X_t × W_K, V_t = X_t × W_V
  2. 追加到cache: K_cache = [K_0, ..., K_{t-1}, K_t], V同理
  3. 用Q_t与完整的K_cache做attention
```

效果：每步从计算所有token的KV（O(t)次投影）变为只计算1个token的KV（O(1)投影）。总计算量从O(n^3)降为O(n^2)。

**显存占用公式**：
```
单请求KV Cache = 2(K+V) × num_layers × num_kv_heads × head_dim × seq_len × dtype_bytes
```

示例（LLaMA-70B, GQA 8 KV heads, 128 head_dim, 80 layers, FP16, 4K seq）：
```
= 2 × 80 × 8 × 128 × 4096 × 2 bytes = 13.4 GB/请求
```

这就是为什么KV Cache优化如此关键——80GB的A100在加载模型参数（35GB for INT4 70B）后，KV Cache只能支持约3个4K长度的并发请求。

**优化方法详解**：

**1. PagedAttention（内存管理革命）**：
- 灵感：OS虚拟内存分页——逻辑连续不要求物理连续。
- 实现：KV Cache按fixed block（如16 tokens/block）分配，block table做逻辑→物理映射。
- 收益：显存利用率从20-40%→接近100%，同样硬件下并发提升3-5x。
- 额外能力：Copy-on-Write支持beam search/parallel sampling不复制KV。

**2. GQA/MQA（架构级减少KV量）**：
- MHA：Q和KV头数相同（如64Q/64KV）→ KV Cache最大。
- GQA：多个Q头共享一组KV头（如64Q/8KV）→ KV Cache减少8x。
- MQA：所有Q头共享1组KV（如64Q/1KV）→ KV Cache减少64x（但精度损失较大）。
- 现状：LLaMA-3/Qwen2/Mistral均使用GQA，是精度和效率的最佳平衡。

**3. KV Cache量化**：
- KV值的数值范围有限（通常[-3, +3]），INT8量化误差极小。
- Per-head量化：每个attention head独立scale factor。
- 效果：KV显存减半，perplexity增加<0.1。
- 实现：存储时量化（在KV生成后立即量化存入cache），使用时反量化。

**4. Prefix Caching**：
- 场景：1000个请求共享同一个2K token的system prompt。
- 无缓存：每请求都计算2K token的KV → 2000K token计算量。
- 有缓存：只计算一次2K KV，后续999个请求直接引用 → 省99.9%前缀计算。
- 实现：对prefix内容hash → 命中则复用已有physical blocks → miss则正常计算。

**5. 稀疏KV Cache**：
- H2O策略：保留累积注意力权重最高的K个token + 最近W个token的KV。
- 典型配置：K=128, W=256，总KV只保留384个token（vs 原始4K），压缩10x。
- 风险：丢弃的token如果被后续重要query需要，质量下降。

**6. Offloading到CPU**：
- 不活跃请求（用户思考中）的KV Cache卸载到CPU DRAM。
- 重新激活时通过PCIe加载回GPU（Gen4: 32GB/s，4K KV约50ms加载）。
- 适合交互式长session场景。

### Q: PagedAttention的原理？为什么效果好？

**原理——借鉴OS虚拟内存分页**：

传统KV Cache管理的问题：
```
请求A (最终生成100 tokens): 预分配 max_seq_len(2048) tokens的连续显存
请求B (最终生成500 tokens): 预分配 max_seq_len(2048) tokens的连续显存
实际利用率: (100+500) / (2048+2048) = 15%！
```

PagedAttention的解决方案：
```
物理block池: [Block 0] [Block 1] [Block 2] ... [Block N]（每block存16 tokens的KV）

请求A的block table: 逻辑0→物理5, 逻辑1→物理12, ... (按需分配，用多少分多少)
请求B的block table: 逻辑0→物理3, 逻辑1→物理7, ...

Block不需要物理连续！（通过block table做地址转换）
```

**Attention计算时的寻址**：
```cuda
// 传统（连续内存）
float* kv_addr = base_ptr + request_id * max_seq * kv_size;

// PagedAttention（分页）
int block_idx = token_pos / block_size;
int block_offset = token_pos % block_size;
int physical_block = block_table[request_id][block_idx];
float* kv_addr = block_pool + physical_block * block_size * kv_size + block_offset * kv_size;
```

**效果好的深层原因**：

| 优势 | 机制 | 量化收益 |
|------|------|---------|
| 消除碎片 | 按block按需分配，无内部碎片 | 利用率20-40% → 95%+ |
| 消除过度预留 | 不需要预留max_seq空间 | 短请求不浪费空间 |
| 灵活回收 | 请求完成即释放其所有block | 实时释放，无需等batch |
| 共享前缀 | 多请求指向同一物理block（COW） | beam search显存降N倍 |
| 动态增长 | 序列增长时按需追加block | 无需预知最终长度 |

**性能数据**：
- vLLM论文中：同样硬件，PagedAttention使吞吐量相比HuggingFace提升14-24x（主要来自更高并发）。
- 显存利用率对比：传统预分配20-40%，PagedAttention接近100%（仅~4%开销用于block table元数据）。

**Copy-on-Write应用**：
- Beam Search：K个beam共享前缀的block，只有当某beam修改时才复制。
- Parallel Sampling：同一prompt多次采样，prompt的KV block全部共享。

### Q: FlashAttention的原理？

**核心问题——标准Attention的内存瓶颈**：

```
标准实现:
  S = Q × K^T          (N×N矩阵，N=seq_len)
  P = Softmax(S)       (N×N矩阵)
  O = P × V            (最终输出)

问题: S和P都是N×N矩阵，seq_len=4K时单层需32MB(FP16)
      A100 SRAM(shared memory)只有192KB/SM
      → S/P必须存在HBM中，产生O(N²)次HBM读写
```

当N=4K, d=128时：
- 计算量：O(N^2 × d) ≈ 4B FLOPs
- 标准实现HBM I/O：O(N^2) ≈ 32MB × 多次读写
- A100 HBM带宽2TB/s → I/O时间成为瓶颈（比纯计算时间更长）

**FlashAttention的解决方案——Tiling + Online Softmax**：

```
将Q分为Br块, K/V分为Bc块 (每块大小适配SRAM)

for each Q_block (Br tokens):
    初始化: O_block = 0, m = -inf, l = 0  (在线softmax的状态)
    for each K_block, V_block (Bc tokens):
        1. 加载Q_block, K_block到SRAM
        2. 计算局部S = Q_block × K_block^T (在SRAM中，不写HBM!)
        3. 在线更新softmax统计量: 
           m_new = max(m_old, rowmax(S))
           l_new = l_old × exp(m_old - m_new) + rowsum(exp(S - m_new))
        4. 更新O_block: O = O × exp(m_old-m_new)/l_new + exp(S-m_new)/l_new × V_block
    end
    写回O_block到HBM
end
```

**关键创新——Online Softmax**：

标准softmax需要两遍遍历（先求全局max，再算exp/sum）。FlashAttention使用在线算法：
- 逐块计算，每块更新运行中的max和sum统计量。
- 对已计算的部分输出用修正因子rescale。
- 数学上严格等价于标准softmax（证明可见论文Theorem 1）。

**收益分析**：

| 指标 | 标准Attention | FlashAttention |
|------|-------------|----------------|
| HBM I/O | O(N^2) | O(N^2 / M)（M=SRAM大小） |
| 额外显存 | O(N^2)（S和P矩阵） | O(N)（仅统计量m,l） |
| 实际加速 | baseline | 2-4x（A100, seq=2K） |
| 长序列 | OOM(>16K) | 可处理128K+ |

**为什么快（虽然FLOPs不变甚至略多）**：
- FLOPs确实不变（还多了rescale计算），但I/O从O(N^2)降为O(N^2/M)。
- A100的计算-带宽比极高（312T/2T=156），I/O减少后计算管线被充分利用。
- 本质：将Attention从memory-bound变为（接近）compute-bound。

### Q: vLLM的核心技术和推理加速策略？

vLLM是加州大学伯克利分校开发的高性能LLM推理引擎，其核心设计围绕"最大化GPU利用率"：

**技术栈架构**：
```
请求 → Scheduler（调度器）
         ↓
    Continuous Batching（动态组批）
         ↓
    PagedAttention（KV Cache管理）
         ↓
    Model Executor（模型执行）
    ├── Tensor Parallel（多卡并行）
    ├── Quantization（GPTQ/AWQ/FP8）
    └── FlashAttention/FlashInfer（计算）
         ↓
    Token Streaming输出
```

**核心技术详解**：

**1. PagedAttention**：见前述——KV Cache分页管理，利用率接近100%。

**2. Continuous Batching（迭代级调度）**：
```
Step t:   [Req A (decode)] [Req B (decode)] [Req C (prefill)] [空]
Step t+1: [Req A (decode)] [Req B (EOS!)]   [Req C (decode)]  [Req D (新进入!)]
Step t+2: [Req A (decode)] [Req D (prefill)] [Req C (decode)]  [Req E (新进入!)]
```
- 每个decode step后检查完成/新到达的请求。
- 不等待整个batch完成，逐请求进出。
- 效果：vs 静态batch吞吐提升2-3x。

**3. Prefix Caching（APC）**：
- 自动检测请求间的共同前缀（通过token序列hash）。
- 命中时直接引用已有KV Cache block，跳过prefill计算。
- 对大量使用相同system prompt的场景（如企业ChatBot），可减少50-90%的prefill计算。

**4. Chunked Prefill**：
- 长prompt（如32K）的prefill分为多个chunk（如512 tokens/chunk）。
- 每个chunk之间穿插处理decode请求。
- 解决：长prefill不会阻塞所有decode请求，降低P99 TPOT。
- 配置：`--enable-chunked-prefill --max-num-batched-tokens 2048`。

**5. Speculative Decoding集成**：
```python
from vllm import LLM
llm = LLM(model="large-model", speculative_model="small-model", num_speculative_tokens=5)
```
- 内置draft model集成，自动处理验证和回退逻辑。
- 对batch中的每个请求独立做推测-验证。

**6. 多模型格式支持**：
- 原生FP16/BF16、GPTQ(4bit)、AWQ(4bit)、FP8(H100)。
- 加载量化模型零额外步骤（自动识别格式并调用对应kernel）。
- Marlin kernel加速W4A16（比通用kernel快2x）。

**7. 多卡推理策略**：
- Tensor Parallel：`--tensor-parallel-size 4`（单请求低延迟）。
- Pipeline Parallel：`--pipeline-parallel-size 2`（适合跨节点）。
- 二者可组合使用。

**性能对比**（A100-80GB, LLaMA-70B, ShareGPT数据集）：
- vLLM vs HuggingFace: 14-24x吞吐提升。
- vLLM vs TGI: 2-3x吞吐提升。
- 主要来源：PagedAttention的高利用率 + Continuous Batching的高调度效率。

### Q: Prefix Cache的作用？

**场景和动机**：

大量LLM应用中，多个请求共享相同的前缀文本：
- **System Prompt**：所有用户请求前都有相同的系统指令（如"你是一个有帮助的助手..."，可长达数千token）。
- **Few-shot示例**：相同的示例模板+不同的用户查询。
- **文档问答**：同一文档+不同问题。

```
请求1: [System Prompt(2K tokens)] + [用户问题A(100 tokens)]
请求2: [System Prompt(2K tokens)] + [用户问题B(200 tokens)]
请求3: [System Prompt(2K tokens)] + [用户问题C(50 tokens)]
...
1000个请求都有相同的2K token前缀！
```

**无Prefix Cache**：每个请求都要做2K token的prefill计算 → 1000次×2K = 200万token的compute（巨大浪费）。

**有Prefix Cache**：只计算一次2K前缀的KV Cache，后续999个请求直接引用。

**实现机制（vLLM APC）**：

```python
# 1. 对prefix内容做hash
prefix_hash = hash(token_ids[0:prefix_len])

# 2. 查找cache
if prefix_hash in prefix_cache_table:
    # 命中：直接引用已有的KV Cache blocks
    kv_blocks = prefix_cache_table[prefix_hash]
    # 只需要计算用户特定部分的prefill
    compute_prefill(user_specific_tokens)
else:
    # 未命中：完整prefill并缓存
    kv_blocks = compute_full_prefill(all_tokens)
    prefix_cache_table[prefix_hash] = kv_blocks
```

**收益量化**：

| 指标 | 无Prefix Cache | 有Prefix Cache | 提升 |
|------|--------------|----------------|------|
| TTFT(首token延迟) | 200ms(含2K prefill) | 20ms(仅100 token) | 10x |
| Prefill计算量 | 100% | 5%(只算用户部分) | 20x |
| 显存占用 | N × prefix_kv | 1 × prefix_kv | Nx节省 |

**注意事项**：
- 缓存一致性：如果prefix内容变化，hash不同会自动miss。
- 缓存淘汰：LRU策略，不常用的prefix被淘汰释放显存。
- 适用条件：大量请求共享长前缀时收益最大；如果每个请求前缀都不同，则无收益。
- vLLM配置：`--enable-prefix-caching`即可启用。

### Q: AI应用（Agent）的构建大致包含哪些模块？

AI Agent是一个能自主规划、使用工具、持续交互的智能系统，其架构包含以下核心模块：

**整体架构**：
```
┌─────────── User Interface ───────────┐
│  输入处理 → Safety Check → Router    │
└──────────────────┬───────────────────┘
                   ↓
┌─────────── Orchestration Layer ──────┐
│  Planning → Execution → Reflection   │
│  (任务分解)  (执行步骤)  (结果评估)    │
└───┬────────┬────────┬────────┬───────┘
    ↓        ↓        ↓        ↓
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ LLM  │ │Tools │ │Memory│ │ RAG  │
│Engine│ │System│ │System│ │Module│
└──────┘ └──────┘ └──────┘ └──────┘
```

**各模块详解**：

**1. LLM推理引擎（核心大脑）**：
- 负责理解、推理和生成。
- 要求：低延迟（用户等待）、高稳定性（不崩溃）、支持结构化输出（JSON/工具调用格式）。
- 部署：自建(vLLM) 或 API调用(OpenAI/Claude)。
- 关键：prompt工程决定agent行为质量。

**2. 工具系统（Tool Use）**：
- 定义：向LLM描述可用工具的名称、参数、用途。
- 调用：LLM决定用哪个工具+参数→执行→将结果注入上下文。
- 常见工具：代码执行、网络搜索、数据库查询、API调用、文件操作。
- 实现要点：工具描述要清晰（减少调用错误）、超时处理、结果截断（避免context过长）。

**3. 记忆系统（Memory）**：
- **短期记忆**：当前对话上下文（conversation history），受context window限制。
- **长期记忆**：向量数据库存储历史交互/知识，需要时检索注入。
- **工作记忆**：当前任务执行中的中间状态（如多步计划的进度）。
- 挑战：context window有限时如何压缩/选择最相关的历史信息。

**4. 规划模块（Planning）**：
- **ReAct**：Reasoning + Acting交替——先思考再行动，观察结果再思考。
- **Plan-and-Execute**：先生成完整计划，再逐步执行（适合复杂任务）。
- **Tree of Thought**：多路径探索，选最优路径。
- 实现：通过prompt引导LLM输出结构化的计划步骤。

**5. RAG检索增强**：
- 将外部知识库切分为chunk，用embedding model编码存入向量数据库。
- 用户query时检索top-K相关chunk注入prompt。
- 解决LLM知识截止和幻觉问题。
- 关键参数：chunk size(512-1024 token)、overlap(50-100 token)、top-K(3-10)。

**6. 安全护栏（Guardrails）**：
- 输入过滤：检测有害/越权请求（prompt injection防御）。
- 输出过滤：检测并拦截有害/不准确的回复。
- 行为约束：限制工具调用范围（如不允许执行rm -rf）。
- 实现：规则引擎 + 分类模型 + LLM自检。

**7. 编排层（Orchestration）**：
- 控制多轮交互流程（何时调用LLM、何时执行工具、何时返回用户）。
- 错误处理和重试逻辑（工具调用失败时的graceful degradation）。
- 并发控制（多个独立子任务并行执行）。
- 框架：LangChain/LlamaIndex/AutoGen/CrewAI。

**对AI Infra的要求**：
- LLM推理：低延迟（<2s TTFT）、高可用（99.9%）、支持长context。
- 向量检索：ms级延迟、百万级向量库。
- 工具执行：沙箱隔离（代码执行安全）、超时控制。
- 可观测性：trace每步决策和执行结果，便于debug和优化。

### Q: 手撕：环上n步回到原点的走法数？

（编程题）
