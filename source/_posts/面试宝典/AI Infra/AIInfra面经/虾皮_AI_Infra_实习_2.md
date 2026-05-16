---
title: '虾皮 AI Infra 实习 (2)'
date: 2026-04-16 11:45:46
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 面经]
---


<!-- more -->

---

### Q: RAG应用中如何做效果评估？具体用了哪些指标？

**RAG评估的三个维度：**

```
┌──────────────────────────────────────────────────────┐
│                    RAG Pipeline                        │
│  Query → [Retriever] → Documents → [Generator] → Answer │
│           ↑ 检索质量评估              ↑ 生成质量评估       │
│                    ← 端到端评估 →                        │
└──────────────────────────────────────────────────────┘
```

**1. 检索质量指标：**

| 指标 | 公式/定义 | 含义 | 目标 |
|------|----------|------|------|
| Recall@K | 相关文档被检索到的比例 | top-K中包含了多少相关文档 | >0.9 |
| MRR | 1/第一个相关文档的排名 | 相关文档排多高 | >0.7 |
| Hit Rate | 至少命中1个相关文档的查询比例 | 检索的可靠性 | >0.95 |
| NDCG@K | 归一化折损累积增益 | 相关文档排序质量 | >0.8 |
| Context Precision | 相关chunk占检索chunk的比例 | 检索的精确度 | >0.7 |

**2. 生成质量指标：**

| 指标 | 评估内容 | 评估方法 | 典型问题 |
|------|---------|---------|---------|
| Faithfulness | 答案是否基于检索文档 | LLM-as-Judge判断是否有文档支撑 | 幻觉检测 |
| Answer Relevance | 答案与问题相关性 | 用答案反推问题,比较与原问题的相似度 | 答非所问 |
| Correctness | 答案是否正确 | 与gold answer比较(如果有) | 事实错误 |
| Hallucination Rate | 生成了文档中没有的信息 | 逐句检查是否能追溯到源文档 | 编造内容 |

**3. 端到端评估框架(RAGAS)：**

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 忠实度: 答案是否有文档支撑
    answer_relevancy,    # 答案相关性
    context_precision,   # 检索精确度
    context_recall,      # 检索召回
)

results = evaluate(
    dataset=eval_dataset,  # {question, answer, contexts, ground_truth}
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
# 输出每个指标的分数和整体RAGAS score
```

**4. LLM-as-Judge评估（无Ground Truth时）：**

```
给评估LLM:
  "请评估以下回答的质量:
   问题: {question}
   检索文档: {contexts}
   模型回答: {answer}
   
   请从以下维度打分(1-5):
   1. 准确性: 答案是否基于文档内容
   2. 完整性: 是否回答了问题的所有方面
   3. 简洁性: 是否有冗余信息"
```

---

### Q: 如何训练用于RAG场景的生成模型？

**RAG生成模型训练的完整策略：**

**1. 训练数据构造：**

```python
# 训练样本格式:
{
    "instruction": "基于以下文档回答问题",
    "input": f"[文档1]: {doc1}\n[文档2]: {doc2}\n\n问题: {question}",
    "output": f"根据文档[1]，{answer}。参考来源: [文档1第3段]"
}

# 数据构造策略:
# 1. 正常样本: 检索到相关文档 → 正确引用回答
# 2. 干扰样本: 混入不相关文档 → 模型学会忽略噪声
# 3. 无证据样本: 文档不包含答案 → 模型学会说"文档中没有相关信息"
# 4. 多文档综合: 答案需要综合多个文档 → 学会跨文档推理
```

**2. 训练策略对比：**

| 策略 | 方法 | 目标 | 适用场景 |
|------|------|------|---------|
| SFT基础 | 在(query+docs, answer)对上微调 | 学会利用上下文 | 所有RAG场景 |
| 引用训练 | 要求输出标注引用来源 | 可追溯性 | 需要可解释性 |
| DPO对齐 | 有引用的回答 > 无引用的回答 | 偏好引用证据 | 减少幻觉 |
| Rejection训练 | 检索失败时输出"不知道" | 减少幻觉 | 高可靠性场景 |
| RLHF | 基于事实性的奖励模型 | 综合质量 | 最终优化 |

**3. 关键训练技巧：**

```python
# 技巧1: Document Dropout — 训练时随机丢弃部分文档
# 让模型不依赖特定数量的文档,增强鲁棒性
if random.random() < 0.2:
    contexts = contexts[:1]  # 20%时间只给1个文档

# 技巧2: Negative Document Injection
# 30%的时间插入不相关文档,训练模型甄别能力
if random.random() < 0.3:
    contexts.append(get_random_irrelevant_doc())

# 技巧3: Loss Masking
# 只在answer部分计算loss,不在文档/指令部分计算
# 避免模型"记住"文档内容(应该在推理时读取)

# 技巧4: 长上下文适配
# RAG输入可能很长(多个文档), 需要:
# - 使用RoPE外推(NTK-aware scaling)
# - 在长序列上继续预训练
# - Flash Attention支持长序列
```

---

### Q: vLLM的核心原理是什么？

**vLLM的完整技术栈：**

```
┌─── vLLM Server ────────────────────────────┐
│ HTTP/gRPC API                               │
│    ↓                                        │
│ AsyncLLMEngine (异步请求管理)                 │
│    ↓                                        │
│ Scheduler:                                  │
│  ├── 三队列: waiting → running → swapped    │
│  ├── Block Manager: 分配/回收物理block       │
│  └── 策略: FCFS, 抢占(swap/recompute)       │
│    ↓                                        │
│ Model Runner:                               │
│  ├── 准备输入tensor (padding/concatenation) │
│  ├── 执行前向计算 (PyTorch + custom kernels)│
│  └── PagedAttention Kernel (非连续KV读取)   │
│    ↓                                        │
│ Sampler: 采样下一个token                     │
└─────────────────────────────────────────────┘
```

**PagedAttention的关键机制：**

| 特性 | 原理 | 显存收益 |
|------|------|---------|
| 非连续Block存储 | 物理block散落在显存任意位置 | 消除外部碎片 |
| 按需分配 | 每16 tokens分配1个新block | 消除内部碎片(vs预分配max) |
| Block Table间接索引 | 类似页表的逻辑→物理映射 | O(1)管理开销 |
| Copy-on-Write | beam search共享block,写时复制 | 节省(beam-1)/beam |
| Prefix Sharing | 相同前缀引用同一block(引用计数) | 减少重复prefill |

**Continuous Batching的实现：**

```python
# 每个iteration:
while True:
    # 1. 调度: 决定哪些请求参与本step
    scheduled = scheduler.schedule()  # 考虑显存预算
    
    # 2. 执行: 混合prefill和decode
    # prefill请求: 处理全部/一个chunk的input tokens
    # decode请求: 生成1个新token
    outputs = model_runner.execute(scheduled)
    
    # 3. 更新: 检查完成/加入/抢占
    for req in scheduled:
        if req.is_finished():   # EOS或max_len
            scheduler.free(req)  # 释放KV blocks
            yield req.output     # 返回结果
    
    # 4. 新请求随时加入(异步)
    new_reqs = check_new_requests()
    scheduler.add_to_waiting(new_reqs)
```

**vLLM vs SGLang性能对比(典型场景)：**

| 场景 | vLLM | SGLang | 原因 |
|------|------|--------|------|
| 简单单轮对话 | 基准 | +10-20% | SGLang调度优化 |
| 长共享前缀 | +20% | +50-80% | RadixAttention自动复用 |
| 结构化输出 | 基准 | +30-50% | SGLang原生支持 |
| Beam Search | +CoW优化 | +CoW+树验证 | SGLang树形结构更优 |

---

### Q: 如何增强模型的多轮对话能力？

**多轮对话的核心挑战：**

| 挑战 | 表现 | 原因 |
|------|------|------|
| 上下文遗忘 | 忽略早期对话内容 | 长序列中的信息衰减(Lost in the Middle) |
| 指代消解 | "它"、"那个"理解错误 | 缺乏指代关系的训练 |
| 一致性 | 前后矛盾 | 训练时各轮独立优化 |
| 话题切换 | 混淆不同话题的上下文 | 缺乏话题边界感知 |
| 累积错误 | 基于错误理解继续对话 | 模型不会主动澄清 |

**解决方案：**

**1. 训练数据设计：**

```python
# 多轮对话训练样本:
[
    {"role": "user", "content": "帮我推荐一款手机"},
    {"role": "assistant", "content": "您的预算和需求是什么？"},
    {"role": "user", "content": "3000左右，拍照好的"},  # 追问
    {"role": "assistant", "content": "推荐小米14，3999起..."},
    {"role": "user", "content": "有没有更便宜的？"},  # 指代(它=手机)
    {"role": "assistant", "content": "同价位还有..."},
    {"role": "user", "content": "算了，换个话题"},  # 话题切换
]
# Loss只在assistant部分计算
# 模型需要学会: 追问、指代消解、话题切换
```

**2. 上下文管理策略：**

| 策略 | 方法 | 适用场景 |
|------|------|---------|
| 滑动窗口 | 保留最近N轮 | 短对话 |
| 摘要压缩 | LLM总结早期对话为摘要 | 超长对话(>50轮) |
| 关键信息提取 | 提取实体/偏好/约束 | 任务型对话 |
| KV-Cache管理 | Prefix Caching共享对话前缀 | 系统效率 |
| 分段attention | 不同轮次不同attention权重 | 研究方向 |

**3. 对齐训练：**

```
DPO数据构造:
  chosen: 与前文一致的回答
  rejected: 与前文矛盾的回答
  
  例:
  用户: "我不吃辣" → 后续推荐辣菜 = rejected
                   → 推荐清淡菜 = chosen

RLHF奖励:
  一致性奖励: 回答与历史对话无矛盾 +1
  引用奖励: 正确引用前文信息 +0.5
  澄清奖励: 不确定时主动确认 +0.3
```

---

### Q: CoT（Chain-of-Thought）训练数据如何构造？

**CoT数据构造的五种主流方法：**

**1. 强模型蒸馏（最常用）：**

```python
# 用GPT-4/Claude生成推理链:
prompt = f"""
请一步一步思考并解答以下问题:
{question}

要求:
1. 分步骤清晰展示推理过程
2. 每步说明为什么这样推理
3. 最后给出答案
"""
response = gpt4(prompt)  # 获得高质量推理链

# 筛选: 只保留最终答案正确的推理链
if verify_answer(response, gold_answer):
    training_data.append({
        "input": question,
        "output": response  # 包含推理过程+答案
    })
```

**2. 自举(Self-consistency + Filter)：**

```python
# 让模型自身生成多条推理路径:
responses = [model.generate(question, temperature=0.7) for _ in range(32)]

# 提取答案并投票:
answers = [extract_answer(r) for r in responses]
majority_answer = Counter(answers).most_common(1)[0][0]

# 保留得到多数答案的推理链:
filtered = [r for r, a in zip(responses, answers) if a == majority_answer]
# 选择最短/最清晰的作为训练数据
```

**3. 程序验证（数学/代码题）：**

```python
# 对于有确定答案的题目:
# 生成推理链 → 执行代码验证 → 只保留验证通过的

cot_response = model.generate(math_problem)
# 提取最终数学表达式
expr = extract_expression(cot_response)
# 程序验证
if eval(expr) == gold_answer:
    # 推理链正确, 纳入训练集
    data.append(cot_response)
```

**4. 逆向构造(Answer → Steps)：**

```
给定: 问题 + 正确答案
任务: 生成合理的推理步骤连接问题和答案

Prompt: "以下问题的答案是{answer}，请反推出完整的解题步骤"
验证: 让另一个模型只看步骤(不看答案)能否得出相同答案
```

**5. 人工标注（最高质量但成本高）：**

| 环节 | 做法 |
|------|------|
| 标注员 | 领域专家(数学博士/程序员) |
| 标注格式 | 每步写"因为...所以..."的推理 |
| 质量控制 | 交叉验证 + 逻辑检查 |
| 成本 | ~$50-100/条(复杂推理) |

**CoT训练的关键注意事项：**
- 推理步骤要**自然**(像人的思维过程)，不能跳跃
- 中间步骤要**可验证**(每步可以检查对错)
- 错误推理也有价值(可以做DPO中的rejected)
- 推理链不宜过长(>1000 token时模型容易迷失)

---

### Q: 介绍vLLM、量化、KV-Cache优化技巧？

**三大优化技术的协同关系：**

```
完整优化栈:
┌──── 模型层 ────┐
│ 权重量化(INT4)  │ → 模型体积减少4x，decode带宽减少4x
│ GQA/MQA结构    │ → KV head数减少，KV-Cache天然更小
└────────────────┘
┌──── KV层 ──────┐
│ PagedAttention  │ → 显存利用率96%+
│ KV-Cache量化   │ → 每token KV减少2-4x
│ Prefix Caching │ → 避免重复prefill
└────────────────┘
┌──── 系统层 ────┐
│ Continuous Batch│ → GPU利用率80%+
│ CUDA Graph     │ → decode step延迟减少30%
│ PD分离         │ → P99延迟改善3-5x
└────────────────┘
```

**量化技术选择指南：**

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 70B模型单卡部署 | INT4权重(AWQ/GPTQ) | 140GB→35GB，可上单80GB卡 |
| Decode加速 | INT4权重(Weight-only) | Memory-bound阶段带宽减4x |
| Prefill加速 | W8A8(SmoothQuant) | Compute-bound阶段INT8 TC加速 |
| H100推理 | FP8(Transformer Engine) | 硬件原生支持，精度最好 |
| 长上下文 | KV-Cache INT8/FP8 | 显存是主要瓶颈 |

**KV-Cache优化技术全景：**

| 技术 | 原理 | 效果 | 适用条件 |
|------|------|------|---------|
| GQA(训练时) | 减少KV head到1-8个 | KV减少4-32x | 需要重新训练 |
| PagedAttention | 分页管理消除碎片 | 利用率提升到96% | 通用 |
| KV-Cache量化 | INT8/FP8存储KV | 每token减少2x | 精度损失可控 |
| Prefix Caching | 共享前缀复用 | 减少重复计算 | 共同前缀场景 |
| Token Eviction | 驱逐不重要的历史token | 限制KV大小 | 超长序列 |
| Sliding Window | 只保留最近W个token | KV上限=W | 训练时设计 |
| KV Offloading | 长期KV移到CPU/NVMe | 支持更长序列 | 延迟容忍 |
| MLA(低秩KV) | 投影到低维存储 | KV减少8-16x | DeepSeek架构 |

---

### Q: "packing"形式和"多轮对话"形式训练有何区别？

**Packing训练（效率优化）：**

```
传统Padding方式(浪费GPU):
  样本A(100 tokens): [A A A ... A PAD PAD PAD ... PAD]  ← max_len=2048
  样本B(50 tokens):  [B B B ... B PAD PAD PAD ... PAD]  ← 大量padding浪费
  GPU利用率: (100+50) / (2048*2) = 3.7% ← 极低!

Packing方式(高效):
  [A A A...A <sep> B B B...B <sep> C C C...C <sep> D D D...]  = 2048 tokens
  多个短样本拼成一个固定长度序列
  GPU利用率: 接近100%
  
关键: Attention Mask防止跨样本注意力!
  样本A的token只能attend到样本A的token
  样本B的token只能attend到样本B的token
  → Block Diagonal Attention Mask
```

**多轮对话训练（语义需求）：**

```
格式:
  [System] You are a helpful assistant.
  [User] 什么是GPU？
  [Assistant] GPU是图形处理单元...   ← 计算loss
  [User] 它和CPU有什么区别？          ← 不计算loss (mask掉)  
  [Assistant] 主要区别是...           ← 计算loss
  
特点:
  - 各轮之间有因果attention联系(后面轮可以看到前面轮)
  - Loss只在assistant回复token上计算
  - 模型需要理解上下文依赖(指代、追问)
```

**核心区别总结：**

| 维度 | Packing | 多轮对话 |
|------|---------|---------|
| 目的 | 训练效率(减少padding) | 学习对话能力 |
| 样本间关系 | 独立(不互相影响) | 依赖(后轮依赖前轮) |
| Attention Mask | Block Diagonal(互相阻断) | Causal(可看到前文) |
| Loss计算 | 所有token(或各样本的target) | 只有assistant token |
| Position ID | 每个样本独立从0开始 | 整个对话连续递增 |
| 典型场景 | 预训练/SFT短样本 | 对话SFT/RLHF |

---

### Q: 手撕：LeetCode 72 编辑距离？

（编程题）
