---
title: '阿里巴巴 AI Infra 一面 (1)'
date: 2026-04-16 11:44:10
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Agent的架构设计有哪些要素？

一个完整的Agent系统包含以下核心组件和设计要素：

**核心组件**：

**1. 规划（Planning / Reasoning）**
- **任务分解**：将复杂任务拆分为可执行的子任务序列
  - 递归分解：大任务→子任务→子子任务
  - DAG分解：识别可并行的子任务
- **推理范式**：
  - ReAct（Reasoning + Acting）：思考→行动→观察循环
  - Chain-of-Thought：逐步推理后执行
  - Tree-of-Thought：多路径探索后选最优
  - Reflection：执行后自我反思并修正
- **多步决策**：维护目标堆栈，根据中间结果动态调整后续计划

**2. 记忆（Memory）**
- **短期记忆**：当前对话上下文（prompt中的messages），容量受context window限制
- **长期记忆**：
  - 向量数据库存储历史经验和知识（Episodic Memory）
  - 结构化存储用户偏好和实体关系（Semantic Memory）
  - 技能/程序记忆（Procedural Memory）：学到的成功策略
- **记忆管理**：写入（什么信息值得记住）、检索（什么时候调取什么记忆）、遗忘（清理过期/矛盾信息）

**3. 工具（Tools / Actions）**
- API调用：搜索引擎、数据库查询、第三方服务
- 代码执行：Python/Shell代码运行环境
- 文件操作：读写文件、创建/修改文档
- 通信：与其他Agent或人类交互
- **工具描述**：用自然语言描述工具的功能、参数、返回值（function calling schema）
- **工具选择**：Agent根据当前状态和目标选择合适的工具

**4. 观察-行动循环（Observation-Action Loop）**
```
while not task_complete:
  observation = perceive(environment)  # 观察环境/工具返回
  thought = reason(observation, memory, goal)  # 推理
  action = decide(thought, available_tools)  # 决策
  result = execute(action)  # 执行
  memory.update(observation, action, result)  # 更新记忆
  if should_reflect(): adjust_plan()  # 反思调整
```

**架构模式**：

| 模式 | 结构 | 适用场景 | 代表 |
|------|------|---------|------|
| 单Agent（ReAct） | 单一LLM循环推理执行 | 简单任务、工具调用 | ChatGPT Plugins |
| 多Agent协作 | 多个Agent角色分工 | 复杂任务分解 | CrewAI, AutoGen |
| 分层Agent | Manager调度 + Worker执行 | 大规模任务编排 | MetaGPT |
| 辩论式 | 多Agent讨论达成共识 | 需要多角度分析 | ChatDev |

**工程挑战**：
- Token效率：每步推理消耗token→长链推理成本高
- 错误累积：多步执行中早期错误会传播放大
- 幻觉控制：Agent可能"编造"工具参数或虚构执行结果
- 安全边界：限制Agent的行动空间防止有害操作

---

### Q: RAG的检索如何实现？

**RAG（Retrieval-Augmented Generation）完整Pipeline**：

**Stage 1: 索引构建（Offline）**

```
文档集合 → 预处理 → 分块(Chunking) → Embedding → 向量数据库
```

- **预处理**：格式解析（PDF/HTML/Markdown）、清洗（去噪、去格式标记）
- **分块策略**（直接影响检索质量）：
  - 固定大小：512-1024 token/chunk（简单但可能切断语义）
  - 语义分块：按段落/章节/句子边界切分
  - 递归分块：先按大块切，大块太大再递归细分
  - 重叠分块：相邻chunk有20-50%重叠（避免边界信息丢失）
  - 典型chunk size：500-1000 token + 50-100 token overlap
- **Embedding编码**：
  - 模型选择：BGE-large/GTE-large/E5（中文），OpenAI text-embedding-3（英文）
  - 维度：768-1536维
  - 需要注意：query和document可能需要不同的编码策略（如添加instruction prefix）
- **向量数据库**：FAISS（单机高性能）/ Milvus（分布式）/ Pinecone（托管服务）/ ChromaDB（轻量）

**Stage 2: 检索（Online）**

```
用户Query → Query处理 → 向量检索(ANN) → 重排序 → Top-K结果
```

- **ANN（近似最近邻）搜索**：
  - HNSW算法：recall 95%+，延迟<10ms（百万级数据）
  - IVF-PQ：适合十亿级数据，内存开销更小
- **Hybrid Search（混合检索）**：
  - 向量检索（语义相似性）+ BM25（关键词匹配）
  - 互补：向量擅长语义理解（同义词/上下位词），BM25擅长精确匹配（实体名/数字）
  - Reciprocal Rank Fusion（RRF）合并两路结果

**Stage 3: 增强生成（Online）**

```
Retrieved Docs + Original Query → Prompt Construction → LLM → Answer
```

- Prompt模板设计：
  ```
  基于以下参考文档回答用户问题。如果文档中没有相关信息，请说明无法回答。
  
  参考文档：
  {retrieved_chunks}
  
  用户问题：{query}
  ```

**优化技术**：

| 技术 | 作用 | 适用场景 |
|------|------|---------|
| Query Rewriting | 改写用户问题使其更适合检索 | 口语化/模糊查询 |
| HyDE | 先让LLM生成假设性答案，用答案做检索 | 问题和文档表述差异大 |
| Reranker | 用Cross-Encoder对初筛结果精排 | 需要高精度top-K |
| Multi-hop Retrieval | 多轮检索逐步逼近答案 | 复杂推理问题 |
| Contextual Compression | 从长文档中提取与问题相关的片段 | chunk太长 |
| Parent Document Retrieval | 检索小chunk但返回其父文档 | 需要上下文 |

---

### Q: 预训练数据清洗方法有哪些？

预训练数据质量直接决定模型能力。数据清洗是一个多阶段的pipeline：

**1. 去重（Deduplication）— 最重要的步骤之一**

| 方法 | 粒度 | 原理 | 适用规模 |
|------|------|------|---------|
| Exact Dedup | 文档/段落 | Hash完全匹配 | 万亿token |
| MinHash + LSH | 文档 | 多个hash函数估计Jaccard相似度 | 万亿token |
| SimHash | 文档 | 高维特征hash为64bit指纹 | 万亿token |
| Suffix Array | 子串 | 删除高频重复子串（>50字符） | 千亿token |
| Semantic Dedup | 语义 | Embedding相似度去重 | 百亿token |

- 训练数据重复的危害：模型记忆重复内容而非学习泛化→质量下降、隐私泄露风险增加
- 实际比例：Common Crawl去重后通常剩余30-50%的数据

**2. 质量过滤（Quality Filtering）**

**规则基过滤**：
- 长度过滤：删除过短（<50字）或过长的异常文档
- 语言检测：用fasttext语言检测器过滤非目标语言
- 特殊字符比例：删除HTML标签/特殊符号/emoji占比>30%的文档
- 重复行/段落比例：删除内部重复严重的文档（如广告/SEO spam）
- 关键词黑名单：过滤明显的低质内容（导航栏文本、cookie声明等）

**分类器过滤**：
- 训练一个二分类器：正例=高质量数据（Wikipedia/书籍/论文），负例=随机网页
- 用分类器打分，保留score>阈值的文档
- GPT-3使用类似方法，只保留了大约6%的Common Crawl数据

**3. 有害内容过滤**
- 毒性检测模型（Perspective API / 自训练分类器）
- 敏感词/正则匹配过滤明显有害内容
- NSFW分类器过滤成人内容
- 多层过滤：先规则粗筛（快）→再模型精筛（慢但准）

**4. 个人信息移除（PII Removal）**
- 正则匹配：邮箱、电话号码、身份证号、银行卡号、IP地址
- NER模型：识别人名、地址等实体并替换为占位符
- 注意：某些PII格式因地区而异（如中国手机号11位、美国10位）

**5. 领域平衡（Domain Balancing）**
- 不同来源的数据量极不均衡：网页>>书籍>>代码>>论文
- 需要有意上采样高质量稀缺数据（如数学/代码/科学论文）
- 下采样低质量过多的数据（如社交媒体评论）
- 典型配比（参考LLaMA）：网页67% + 代码4.5% + Wikipedia 4.5% + 书籍4.5% + 论文2.5% + 其他

**6. 数据清洗的质量验证**
- 人工抽样检查（随机抽取1000条评估）
- 在小模型上做消融实验验证清洗策略的效果
- 监控模型训练指标（如特定benchmark的score）随数据调整的变化

---

### Q: Group Query Attention的作用？

GQA（Grouped Query Attention）是MHA和MQA之间的折中方案，将原始的多个query head分组，**每组共享一套KV head**。

**核心动机与原理**：

大模型推理的decode阶段是memory-bound的——每生成一个token需要加载所有层的KV Cache，而计算量很小（只做一次向量-矩阵乘）。**减少KV head数量直接减少decode阶段的内存访问量**，从而加速生成。

```
MHA:  Q_heads: [h1,h2,h3,h4,h5,h6,h7,h8]
      K_heads: [k1,k2,k3,k4,k5,k6,k7,k8]  → 8组KV

GQA-4: Q_heads: [h1,h2,h3,h4,h5,h6,h7,h8]
       K_heads: [k1,  k1,  k2,  k2,  k3,  k3,  k4,  k4]  → 4组KV
       (每2个Q head共享1个KV head)

MQA:  Q_heads: [h1,h2,h3,h4,h5,h6,h7,h8]
      K_heads: [k1,k1,k1,k1,k1,k1,k1,k1]  → 1组KV
```

**具体收益**：

**1. KV Cache显存减少**
- GQA-g（g组KV head）：KV Cache = MHA的 `g/num_heads` 倍
- LLaMA-2 70B（32 Q heads, 8 KV heads）：KV Cache是MHA的 8/32 = 25%
- 意味着同等显存可以服务**4倍的并发请求**或**4倍长的序列**

**2. Decode吞吐提升**
- Decode阶段瓶颈是加载KV Cache的带宽
- KV head数减少→加载量减少→单token生成更快
- A100上LLaMA-70B实测：GQA-8比MHA decode快约3.5倍

**3. 模型质量损失极小**
- Google的研究：GQA-8质量只比MHA差0.1-0.3%（困惑度差异）
- MQA（只有1个KV head）质量损失更明显（0.5-1%）
- GQA在多数benchmark上与MHA几乎无差异

**GQA在主流模型中的使用**：

| 模型 | Q heads | KV heads | 分组比例 |
|------|---------|----------|---------|
| LLaMA-2 70B | 64 | 8 | 8:1 |
| LLaMA-3 8B | 32 | 8 | 4:1 |
| LLaMA-3 70B | 64 | 8 | 8:1 |
| Qwen-2.5 72B | 64 | 8 | 8:1 |
| Mistral 7B | 32 | 8 | 4:1 |

**从MHA转换到GQA的方法**：
- 从头训练（最佳质量）
- 从MHA checkpoint继续训练为GQA：将同组内的KV head均值池化后合并，再微调少量steps恢复质量

---

### Q: 手撕：实现LRU Cache？

（编程题）

---

### Q: MoE架构的专家路由是对每个token路由还是对每个序列路由？

**对每个token独立路由**。

**具体过程**：
```python
# 对batch中每个token独立计算路由
# input shape: [batch_size * seq_len, hidden_dim]
router_logits = input @ W_gate  # [batch*seq, num_experts]
router_probs = softmax(router_logits, dim=-1)
topk_values, topk_indices = torch.topk(router_probs, k=top_k, dim=-1)
# 每个token有自己的top-k专家选择，同一序列内不同位置可选不同专家
```

**为什么是token级而非序列级？**

1. **语义粒度**：同一句话中不同token的语义属性差异大
   - "The capital of France is Paris" → "capital"可能路由到地理专家，"France"路由到实体专家，"is"路由到语法专家
2. **容量利用**：token级路由使各专家的负载更均匀分布
3. **模型能力**：细粒度路由让模型学到更精确的专家-语义映射
4. **实验证据**：所有SOTA MoE模型（Switch Transformer、Mixtral、DeepSeek）都使用token级路由

**序列级路由的问题**：
- 一个序列通常包含多个主题/功能→一个专家组合无法覆盖
- 负载均衡更难保证（序列长度不一）
- 浪费专家容量

**但也有变体**：
- **Segment-level routing**：将序列分为固定大小的segment（如128 tokens），每个segment统一路由。是token级和序列级的折中
- **Expert Choice**中每个专家选择token时是跨序列的（但仍是token粒度的选择）

---

### Q: KV Cache的优化方法有哪些？

KV Cache是LLM推理中最大的显存消耗源之一，优化方法涵盖**结构设计、压缩、管理调度**三个层面：

**结构化优化（模型设计时决定）**：

**1. GQA/MQA — 减少KV head数量**
- 效果：KV Cache线性减少（8 KV heads → 1/4显存）
- 限制：需要预训练时确定，无法后期修改
- 推荐：GQA-8（4-8倍压缩，质量损失最小）

**2. MLA — 低秩压缩**
- DeepSeek-V2/V3的方案：将KV投影到低维latent空间
- 压缩比：5-20倍（latent_dim远小于原始KV维度）
- 代价：推理时需要额外的解压计算（但通常很快）

**存储优化（部署时选择）**：

**3. 量化 — 降低每个元素的精度**

| 量化方案 | 压缩比 | 质量影响 | 实现 |
|---------|--------|---------|------|
| FP16→FP8 | 2x | 极小 | H100原生支持 |
| FP16→INT8 | 2x | 小（per-channel） | CUDA kernel |
| FP16→INT4 | 4x | 中等 | 需要group quantization |
| FP16→INT2 | 8x | 较大 | 研究阶段 |

- 关键：KV Cache量化比权重量化更敏感（对attention score直接影响）
- 推荐：INT8 per-channel是目前生产可用的最佳平衡点

**4. 前缀共享（Prefix Caching）**
- 场景：多个请求有相同的system prompt或few-shot examples
- 实现：用Trie树/Hash表索引已计算的前缀KV Cache，新请求直接复用
- 效果：如system prompt占2K token，每个请求节省2K token的KV Cache计算和存储
- vLLM的Automatic Prefix Caching / SGLang的RadixAttention

**5. Offload — 分层存储**
- GPU显存存放活跃请求的KV Cache
- CPU内存存放暂停/等待的请求（如被preempt的请求）
- NVMe SSD存放长期不活跃的历史KV
- 需要时通过PCIe异步预取到GPU
- 延迟：GPU→CPU~10us，CPU→GPU~10us，SSD→GPU~100us

**调度优化（运行时管理）**：

**6. PagedAttention — 消除内存碎片**
- 问题：不同请求的KV Cache长度不同，连续分配造成严重碎片
- 方案：将KV Cache分为固定大小的page（如16 tokens/page），通过page table管理
- 类比：OS的虚拟内存管理（logical address → physical address）
- 效果：显存利用率从20-40%提升到>95%
- 代价：需要修改attention kernel支持非连续KV Cache（scatter/gather）

**7. 驱逐策略 — 有选择地丢弃**
- **H2O（Heavy Hitter Oracle）**：
  - 观察：少量token（~5%）贡献了大部分attention mass（Heavy Hitters）
  - 策略：保留累计attention score最高的token + 最近的token，驱逐其余
  - 效果：保留20%的KV Cache仍能维持>95%的模型质量
- **Scissorhands**：利用attention pattern在不同layer间的一致性
- **StreamingLLM**：保留attention sink（前4个token）+ 滑动窗口
  - 发现：前几个token无论内容如何都积累大量attention（充当"垃圾桶"），删除会导致输出崩溃
  - 适用：超长流式对话场景，固定显存运行无限长度

**8. 投机解码中的KV Cache管理**
- Draft model生成多个候选token的KV Cache
- Target model验证后，被拒绝的token对应的KV Cache需要回滚
- 实现：维护KV Cache的版本/checkpoint，验证失败时快速恢复
