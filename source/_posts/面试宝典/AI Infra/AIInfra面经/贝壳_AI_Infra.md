---
title: '贝壳 AI Infra'
date: 2026-04-16 11:46:14
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 训练优化, 面经]
---


<!-- more -->

---

### Q: Tokenizer怎么优化？

Tokenizer 是 LLM 推理的第一步，虽然计算量小但影响整体效率：

1. **BPE 合并规则优化**：针对领域数据（如代码/医学/法律）重新训练词表，使领域高频词能被单 token 表示。减少平均 token 数 = 减少推理步数（自回归生成的步数直接决定延迟）。

2. **词表大小调整**：增大词表（32K -> 128K）减少序列中的 token 数（信息密度更高），但增加 embedding 参数和 LM head 计算量。需要在 token 数减少（推理步数少）和单步计算增加之间权衡。LLaMA-3 使用 128K 词表。

3. **Fast Tokenizer**：使用 Rust 实现的 tokenizers 库（HuggingFace）替代纯 Python 实现，编码速度提升 10-100x。底层用 Aho-Corasick 自动机或 BPE 缓存加速匹配。

4. **批量并行编码**：使用多线程同时处理多条文本的 tokenize，消除批处理场景下的串行瓶颈。

5. **前缀缓存**：对常用前缀（如 system prompt）缓存 tokenize 结果，避免重复编码。与 Prefix Caching（KV Cache 层面）配合使用。

---

### Q: P-Tuning v2怎么做？和之前版本的区别？

P-Tuning 系列是参数高效微调（PEFT）的重要方法：

**P-Tuning v1**：只在输入 embedding 层添加一组可学习的连续向量（soft prompt），其他层全部冻结。问题：只有输入层有可调参数，对深层表示的控制能力有限，大模型上效果与全参微调差距明显。

**P-Tuning v2（≈ Prefix Tuning）**：在**每一层** Transformer 都插入可学习的 prefix 向量：
- 具体做法：在每层的 Key 和 Value 前拼接 prefix_length 个可学习向量
- Attention 计算变为：Attention(Q, [P_K; K], [P_V; V])
- 等价于每层都有独立的 soft prompt，深层也能被有效调节

**效果对比**：
- v1 在小模型（<1B）上效果尚可，在大模型上远不如全参微调
- v2 在 330M-10B 模型上都能接近全参微调效果（gap<1%），因为深层也有可调参数
- 可调参数量：prefix_length * n_layers * 2 * hidden_size（通常总量为原模型 0.1-1%）

---

### Q: LoRA微调怎么做？

LoRA（Low-Rank Adaptation）在冻结的预训练权重旁添加低秩分解矩阵进行微调：

**核心公式**：W_new = W_frozen + alpha/r * B * A
- W_frozen: 原始权重矩阵 [d_out, d_in]，冻结不更新
- A: [r, d_in] 低秩投影矩阵（随机初始化）
- B: [d_out, r] 低秩投影矩阵（初始化为零）
- r: 秩（rank），通常 8-64，远小于 d

**训练**：只更新 A 和 B（参数量 = r*(d_in + d_out)，远小于 d_in*d_out）。因为 B 初始为零，训练开始时 BA=0 不改变原模型行为。

**推理**：将 BA 合并到 W 中：W_merged = W + alpha/r * BA。无额外推理开销（与原模型计算量完全相同）。

**应用位置**：通常对 attention 的 Q/K/V/O 投影和 FFN 的线性层应用 LoRA。实践中 attention 层的 Q 和 V 投影效果最好。

---

### Q: LoRA两个矩阵怎么初始化？为什么A不能初始化为0？

**初始化策略**：
- **A**：随机初始化（Kaiming uniform 或高斯分布）
- **B**：初始化为全零

**这样设计的原因**：初始时 BA = 0*A = 0，不改变原模型的输出（训练开始时相当于未添加任何适配器），保持预训练模型的初始状态。

**为什么 A 不能为 0**：
从梯度角度分析：
- dL/dB = dL/dY * A^T：如果 A=0，则 B 的梯度为 0，B 永远保持 0
- dL/dA = B^T * dL/dY * X^T：如果 B=0（初始值），A 的梯度也为 0

看似死锁，但实际上训练时 B 的更新发生在 A 非零的前提下：
- 第一步：dL/dA = 0^T * ... = 0（A 不更新），但 dL/dB = dL/dY * A^T ≠ 0（因为 A 随机非零）
- B 更新后变为非零
- 第二步：dL/dA = B^T（非零）* dL/dY * X^T ≠ 0
- A 也开始更新

**关键**：A 的随机初始化打破了对称性，使 B 能收到非零梯度从而开始学习。如果 A=0，整个 LoRA 模块永远保持 BA=0，等于没有添加。

---

### Q: TF-IDF是什么？

TF-IDF（Term Frequency - Inverse Document Frequency）是经典的文本特征权重算法，衡量一个词对文档的重要性：

**TF（词频）**：词在当前文档中出现的频率。公式：TF(t,d) = count(t,d) / |d|。词出现越多说明越相关。

**IDF（逆文档频率）**：log(N / df(t))，N 为总文档数，df(t) 为包含词 t 的文档数。越稀有的词 IDF 越高（"the" IDF 接近 0，"CUDA" IDF 很高）。本质上是惩罚停用词和高频通用词。

**TF-IDF = TF * IDF**：一个词在某文档中频率高（TF 高）且在整个语料中很少出现（IDF 高），才会获得高权重。

**应用**：文本特征提取（作为向量空间模型的维度权重）、关键词抽取（取每篇文档 TF-IDF 最高的 N 个词）、BM25 的基础（搜索引擎中 BM25 是 TF-IDF 的改进版本）。

**局限性**：无法捕获语义相似性（同义词 TF-IDF 为 0）、无法处理词序信息。现代 RAG 系统用 dense embedding 替代，但 BM25（基于 TF-IDF）仍作为混合检索的关键词匹配通道。

---

### Q: GRPO强化学习怎么做？

**GRPO（Group Relative Policy Optimization）** 是 DeepSeek 提出的轻量级 RL 对齐方法：

**核心流程**：
1. 对每个 prompt x 采样 G 个回答（如 G=8）：{y_1, ..., y_G} ~ pi(y|x)
2. 对每个回答计算 reward：r_i = R(x, y_i)
3. 组内归一化计算相对优势：A_i = (r_i - mean(r)) / (std(r) + eps)
4. 策略梯度更新：L = -1/G * sum_i [A_i * sum_t log(pi(y_i_t | x, y_i_{<t})) / |y_i|]
5. 加 KL 惩罚：L += beta * KL(pi || pi_ref)

**关键设计决策**：
- **无需 Critic 网络**：用组内均值作为 baseline（相对排名代替绝对价值估计），大幅简化训练
- **序列级 reward**：适合 outcome-based reward（如数学答案对错、代码测试通过）
- **长度归一化**：除以 |y_i| 防止长回答获得不公平的优势

---

### Q: 奖励函数怎么设计？

奖励函数的设计是 RL 对齐成功的关键，常见策略：

1. **Rule-based Reward**（确定性规则）：格式正确性检查、代码编译/执行通过、数学答案精确匹配。优点是信号清晰无噪声，缺点是只能评估可验证的客观标准。

2. **Model-based Reward**（Reward Model）：用人类偏好数据训练的 RM 对回答打分。Bradley-Terry 模型：P(y_w > y_l) = sigma(r(y_w) - r(y_l))。优点是能评估开放式质量，缺点是 RM 本身有偏差和上限。

3. **Outcome Reward**（结果导向）：最终结果的正确性。如数学答案对错(+1/-1)、代码测试通过率。与 GRPO 配合最自然。

4. **Process Reward Model（PRM）**：对推理过程的每一步打分，不仅看最终结果。能引导模型学习正确的推理路径。训练成本高（需要步骤级标注）。

5. **组合奖励**：多种信号加权组合。如 r = 0.5*correctness + 0.3*format + 0.2*conciseness。权重需要仔细调优。

---

### Q: 如何实现LLM as a Judge？

用强模型（如 GPT-4/Claude）自动评估弱模型输出质量：

**关键设计**：
1. **明确评分标准**：定义具体维度（准确性/流畅性/相关性/安全性），每个维度给出 1-5 分的详细描述
2. **Few-shot 示例**：在 prompt 中提供 2-3 个评分示例（包含不同分数的案例），校准评分尺度
3. **处理位置偏差**：LLM 倾向于偏好第一个（或最后一个）候选。解决：随机交换两个候选的展示顺序，取两次评分的平均
4. **处理长度偏差**：LLM 倾向于偏好更长的回答。解决：在评分指令中明确"简洁且完整的回答应该获得高分"
5. **处理自我偏好**：模型倾向于偏好自己生成的风格。解决：使用与被评模型不同的 Judge 模型

**输出格式**：要求 Judge 先给出逐维度分析，再给出总分和理由（CoT 提升评分一致性）。

---

### Q: RAG怎么做？

RAG（Retrieval-Augmented Generation）的完整流程：

**离线阶段**（知识库构建）：
1. 文档预处理：格式转换 -> 清洗去噪 -> 切分 chunk（256-512 token，overlap 50-100）
2. 向量化：用 embedding 模型（如 bge-large）将 chunk 转为向量
3. 建立索引：存入向量数据库（Milvus/FAISS），建立 HNSW/IVF 等 ANN 索引

**在线阶段**（检索增强生成）：
1. Query 处理：原始 query 可做改写/扩展（HyDE、多查询生成）
2. 检索：query embedding 与文档 embedding 相似度匹配，召回 top-k（混合检索：向量 + BM25）
3. 重排序：用 Cross-Encoder（如 bge-reranker）对 top-k 精排，取 top-3~5
4. 增强：将检索文档拼接到 prompt 中（注意放置位置避免 Lost in the Middle）
5. 生成：LLM 基于增强上下文生成答案

**优化方向**：多路召回、查询改写、迭代检索、知识图谱融合、长上下文模型。

---

### Q: 向量检索怎么做？

高效向量检索（ANN - Approximate Nearest Neighbor）是 RAG 的核心技术：

**索引算法**：
- **HNSW（Hierarchical Navigable Small World）**：多层图结构，查询从顶层快速定位到底层邻域。准确率高（recall >95%），构建慢但查询快。最常用。
- **IVF（Inverted File）**：先 K-Means 聚类划分 Voronoi cells，查询时只搜索最近的 nprobe 个 cells。适合大规模数据（10亿+）。
- **Flat（暴力搜索）**：精确但慢，适合 <100K 文档的小规模场景。

**相似度度量**：余弦相似度（归一化后等价内积）、L2 距离、内积（IP）。选择取决于 embedding 模型的训练方式。

**压缩加速**：PQ（Product Quantization，将向量分段量化）减少内存占用和计算量；SQ（Scalar Quantization，标量量化为 INT8）简单高效。

**混合检索**：dense vector（语义匹配）+ sparse BM25（关键词匹配）融合。RRF（Reciprocal Rank Fusion）或学习的融合权重合并两路结果。互补性强，通常 +5-10% recall。

---

### Q: LLM预训练和后训练分别做什么？

**预训练（Pre-training）**：
- 目标：在海量无标注文本上学习语言知识和世界知识
- 任务：CLM（下一个 token 预测），自监督学习
- 数据：数万亿 token（Common Crawl/书籍/代码/Wiki 等），数据质量和配比是关键
- 规模：数千 GPU 训练数周到数月
- 产出：具有广泛知识但不会遵循指令的"基座模型"

**后训练（Post-training）**：
- **SFT（Supervised Fine-Tuning）**：用高质量指令-回答对教模型遵循指令。数据量 10K-1M，训练 1-5 epoch。
- **RLHF/DPO 对齐**：用人类偏好数据训练模型区分好坏回答，学习有帮助且安全。PPO/DPO/GRPO 等算法。
- **能力增强**：特定领域数据 continue pretrain（如代码/数学/医学），或长上下文适应训练。

**后训练的核心目标**：不是注入新知识（预训练已完成），而是"教会模型如何使用已有知识"——对齐人类意图。

---

### Q: 量化具体做什么？

量化是将高精度数值映射为低精度数值的过程：

**核心步骤**：
1. **确定量化范围（Calibration）**：用校准数据（100-1000条）跑一遍推理，统计权重/激活的数值范围（min/max 或 percentile）
2. **计算量化参数**：scale = range / (2^bits - 1)，zero_point = round(-min/scale)
3. **量化（Quantize）**：q = clamp(round(x / scale + zp), 0, 2^bits-1)
4. **推理时**：用量化后的整数做计算（INT8 GEMM），或实时反量化为 FP16 计算（W4A16 方案）
5. **反量化（Dequantize）**：x_approx = (q - zp) * scale

**目标**：减少模型大小（W4 比 FP16 小 4x）、加速推理（INT8 Tensor Core 吞吐是 FP16 的 2x）、降低显存占用（KV Cache 也可量化）。

**精度保证**：好的量化方法（GPTQ/AWQ）在 4-bit 量化下任务指标下降 <1%。

---

### Q: 手撕：字符串转整数（LeetCode 8）？

（编程题）
