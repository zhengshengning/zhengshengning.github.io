---
title: '阿里巴巴 AI Infra 实习 (2)'
date: 2026-04-16 11:43:54
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 多轮对话超长上下文如何训练？

超长上下文训练的核心挑战是：**显存与计算量随序列长度二次增长**（attention的O(N^2)），以及长序列训练数据的稀缺性。以下是系统化的解决方案：

**1. 长序列并行（Context Parallelism）**

将超长序列切分到多个GPU上分布式计算attention：

- **Ring Attention**：将序列分为chunks分布到多GPU，每个GPU持有完整的Q但只持有部分KV。通过环形通信传递KV blocks——GPU_i计算完与local KV的attention后，接收来自GPU_{i-1}的KV block继续计算，环形传递一圈后得到完整attention结果。通信可与计算完全overlap
- **DeepSpeed Ulysses**：通过AllToAll将attention的head维度而非序列维度分布到多GPU。每个GPU持有所有位置但只有部分head的QKV→本地计算部分head的attention→AllToAll收集结果。适合head数多的场景

两种方案的对比：

| 方案 | 通信原语 | 通信量 | 适用场景 |
|------|---------|--------|---------|
| Ring Attention | P2P (Ring) | O(seq_len * head_dim) per step | 超长序列（>128K） |
| Ulysses | AllToAll | O(seq_len * head_dim * num_heads) | head数多、序列适中 |

**2. 渐进式训练（Progressive Training）**
- 先在短序列（4K-8K）上完成主要预训练
- 然后在中等序列（32K-64K）上继续训练较少steps
- 最后在目标长度（128K-1M）上fine-tune
- 位置编码外推：
  - **RoPE ABF（Adjusted Base Frequency）**：将RoPE的base从10000增大到500000，扩展位置编码的有效范围
  - **YaRN**：对不同频率分量分别做NTK-aware插值和外推
  - **动态NTK**：推理时根据实际序列长度动态调整缩放因子

**3. 显存优化**

| 技术 | 原理 | 显存节省 |
|------|------|---------|
| Gradient Checkpointing | 不存中间激活，反向时重算 | 激活显存降为O(√L) |
| Flash Attention | 分块计算attention，不存N*N矩阵 | attention显存O(N)→O(1) |
| Selective Checkpointing | 只checkpoint计算密集层，保留便宜层的激活 | 折中方案 |

**4. Flash Attention的长序列支持**
- 标准attention需要O(N^2)显存存储attention矩阵
- Flash Attention通过在线softmax（Milakov & Gimelshein 2018）分块计算：
  - 将Q/K/V分成blocks
  - 在SRAM中逐block计算attention并做running softmax
  - 只存储最终输出O和辅助信息（logsumexp），不需要物化完整的N*N矩阵
- 显存复杂度降为O(N)，使得百万token级别的训练成为可能

**5. 数据工程**
- **Packing策略**：将多个短序列拼接到max_seq_len，用attention mask隔离不同序列。避免padding浪费（短序列padding可能浪费50%+计算）
- **长序列数据采集**：长上下文能力需要对应的长训练数据。来源：长文档（论文/书籍/代码仓库）、多轮对话拼接、合成长序列数据
- **课程学习（Curriculum Learning）**：按序列长度从短到长安排训练数据，稳定训练过程
- **数据配比**：长短序列数据混合训练，保证短序列能力不退化

### Q: 上下文记忆策略有哪些？

不同策略在**容量、精度、延迟、成本**之间做不同的trade-off：

**1. 滑动窗口（Sliding Window）**
- 实现：只保留最近N轮对话（或最近K个token）
- 优点：实现最简单、延迟恒定、不需要额外基础设施
- 缺点：丢失早期重要信息（如用户在第一轮声明的偏好）
- 适用：简单对话场景、对历史依赖不强的任务
- 典型窗口：4-8轮或4K-8K token

**2. 摘要压缩（Summarization）**
- 实现：当对话超过阈值时，用模型将早期对话压缩为摘要
- 策略：递归摘要（每满N轮压缩一次）/ 分层摘要（摘要的摘要）
- 优点：保留关键信息的同时控制token数
- 缺点：摘要过程本身有信息损失且产生延迟（需要额外LLM调用）
- 实现细节：摘要prompt需要精心设计，要保留关键实体、数值、决策等

**3. RAG式检索（Retrieval-Augmented Memory）**
- 实现：将历史对话按轮次/主题分块→embedding编码→存入向量数据库（FAISS/Milvus/Pinecone）→新消息到来时检索top-K相关历史
- 优点：理论上无容量限制、按需检索效率高
- 缺点：检索可能遗漏（recall不完美）、增加推理延迟（检索+拼接）、需要维护向量数据库
- 优化：Hybrid检索（向量+关键词BM25）、Reranker精排、Query Expansion
- 适用：知识密集型对话、客服系统

**4. 分层记忆（Hierarchical Memory）**
```
短期记忆：完整的最近3-5轮对话 → 直接放在prompt中
中期记忆：最近20轮的摘要 → 系统消息中
长期记忆：用户画像/偏好/历史决策 → 向量数据库检索补充
```
- 优点：多粒度覆盖，平衡精度和容量
- 缺点：设计复杂、多层之间的信息一致性难保证
- 代表实现：MemGPT的分页虚拟内存设计

**5. KV Cache管理策略**
- **StreamingLLM**：保留attention sink（前几个token的KV）+ 近期token的KV，中间的丢弃
  - 发现前几个token积累了大量attention分数（即使内容不重要），是维持模型稳定输出的"锚点"
  - 窗口外的token完全丢失，不适合需要远距离依赖的任务
- **H2O（Heavy Hitter Oracle）**：保留attention score累计最高的token的KV
  - 根据attention pattern动态选择重要token保留
- **Scissorhands**：基于observation——attention pattern在不同layer间高度一致，可以用浅层attention指导深层的KV驱逐

**选择建议**：

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 简单多轮聊天 | 滑动窗口 | 简单高效 |
| 长期客服对话 | 分层记忆 + RAG | 需要历史和用户画像 |
| 文档问答 | RAG | 知识密集 |
| 流式输出（无限长度） | StreamingLLM | 恒定显存 |
| 资源受限端侧 | 滑动窗口 + 摘要 | 低开销 |

### Q: Skills读取超长SOP的优化有哪些trick？

当SOP（Standard Operating Procedure）文档非常长（数万~数十万token）时，直接塞进context window不现实，需要智能检索和压缩：

**1. 分段加载（Lazy Loading）**
- 将SOP按步骤/章节分段（每段500-2000 token）
- 建立段落索引（标题→段落位置的映射）
- 根据当前对话阶段/用户意图只加载相关段落
- 实现：`current_step`状态追踪 + 条件加载逻辑
- 效果：平均只需加载SOP的10-20%即可完成任务

**2. 摘要索引（Summary-based Index）**
- 为每个SOP段落生成50-100字的摘要
- 运行时先匹配摘要→确定相关段落→再加载详细内容
- 两阶段检索：粗筛（摘要匹配，快速）→精选（详细内容，准确）
- 优点：摘要索引本身很小（可以整体放入prompt），检索精度高于纯向量匹配
- 实现：离线生成摘要并缓存，运行时只做匹配

**3. 结构化存储（Structured Storage）**
- 将SOP解析为结构化数据：
  ```json
  {
    "step_1": {
      "condition": "用户要求退款",
      "action": "确认订单状态...",
      "sub_steps": [...],
      "exception_handling": "如果超过7天..."
    }
  }
  ```
- 查询时只返回匹配条件的分支，避免加载不相关的流程
- 适合逻辑分支多的SOP（如客服话术、审批流程）

**4. Embedding检索（Semantic Retrieval）**
- 将SOP段落编码为向量（embedding模型如BGE/GTE）
- 用当前对话context作为query做ANN检索
- 返回top-K相关段落（K通常3-5）
- 优化：
  - 细粒度分段（按句子or段落）vs 粗粒度（按章节）的权衡
  - 添加metadata过滤（如按SOP版本、适用场景筛选）
  - 多轮对话时用对话摘要而非最后一句做query
- 工具：LangChain + FAISS / LlamaIndex

**5. Prompt压缩（Prompt Compression）**
- 使用LLMLingua/LongLLMLingua等工具对检索到的SOP文本做压缩
- 原理：用小模型（如GPT-2）计算每个token的perplexity，删除低信息量的token
- 可将文本压缩到原来的1/3-1/5且保留关键信息
- 适合检索到的文本仍然太长的情况
- 注意：压缩后的文本可读性下降，需要模型有一定容错能力

**6. 综合方案设计**：
```
用户输入 → 意图识别 → 结构化索引定位章节
                     → Embedding检索补充细节
                     → 合并去重 → Prompt压缩（如仍超限）
                     → 注入prompt
```
