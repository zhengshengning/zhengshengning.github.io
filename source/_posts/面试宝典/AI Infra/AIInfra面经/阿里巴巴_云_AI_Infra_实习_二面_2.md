---
title: '阿里巴巴 云 AI Infra 实习 二面 (2)'
date: 2026-04-16 11:44:06
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: vLLM有哪些优化技术？

vLLM是当前最流行的开源LLM推理引擎，其核心优化技术涵盖内存管理、调度策略、计算加速三个层面：

**1. PagedAttention（核心创新）**
- **问题**：KV Cache长度随生成动态增长，传统连续分配导致严重内存碎片（浪费60-80%显存）
- **方案**：借鉴OS虚拟内存，将KV Cache分为固定大小的page（通常16 tokens/page）
  - 逻辑连续但物理不连续
  - 通过page table做地址翻译
  - 只在需要时分配新page
- **效果**：显存利用率从~20%提升到>95%，同等硬件可服务2-4倍的并发请求
- **代价**：Attention kernel需要支持非连续内存访问（gather/scatter）

**2. Continuous Batching（连续批处理）**
- **传统Static Batching**：一个batch内所有请求必须等最慢的完成→GPU空闲等待
- **Continuous Batching**：每个调度步（iteration）都可以动态加入新请求或移除已完成请求
- **效果**：GPU利用率大幅提升（从30-50%→70-90%），吞吐量提高2-3倍
- **实现**：scheduler每步检查完成/新到的请求，动态调整当前batch组成

**3. Prefix Caching（前缀缓存）**
- **场景**：多个请求共享相同前缀（system prompt / few-shot examples）
- **实现**：对token序列的prefix做hash，在全局KV Cache pool中查找已有计算结果
- **效果**：如2K token的system prompt被100个请求共享→节省200K token的重复Prefill计算
- **SGLang进一步优化**：RadixAttention用Radix Tree支持最长公共前缀匹配（而非精确前缀匹配）

**4. Chunked Prefill（分块预填充）**
- **问题**：长prompt（如32K token）的prefill需要数秒，期间decode请求被阻塞
- **方案**：将长prompt分成固定大小chunk（如2048 token），每个调度周期处理一个chunk + decode请求混合执行
- **效果**：decode延迟P99从数秒降到<100ms
- **trade-off**：TTFT（首token延迟）略微增加（需要多个周期完成prefill）

**5. Tensor Parallelism（多卡推理）**
- 大模型（>13B）无法单卡推理→通过TP将模型切分到多GPU
- vLLM集成Megatron-style TP：AllReduce通过NCCL实现
- 支持TP = 2/4/8，通常在NVLink连接的GPU间使用
- 近期也支持Pipeline Parallelism（PP）和EP用于更大模型

**6. 量化支持**
- **GPTQ**：训练后权重量化（INT4/INT8），使用校准数据集
- **AWQ**：Activation-aware Weight Quantization，保护重要权重通道
- **FP8**：H100原生支持，质量接近FP16但吞吐提升1.5倍
- **INT8 KV Cache量化**：减少KV Cache显存开销
- **SqueezeLLM/AQLM**：更激进的压缩方案

**7. CUDA Graph**
- **问题**：Decode阶段每步计算量很小但kernel launch开销固定（~10us/kernel）→launch占总时间30%+
- **方案**：将一次完整的decode step的所有kernel录制为CUDA Graph，后续一次launch执行整个graph
- **效果**：减少kernel launch开销~10倍（~10us → ~1us），对小batch/短模型改善显著
- **限制**：要求每步的kernel序列固定（不支持动态shape）→通过padding到固定size解决

**8. Speculative Decoding（投机解码）**
- **原理**：用小快模型（draft）一次生成K个候选token → 大模型一次并行验证所有K个token
- **收益**：如果K个token中前M个被接受，则一次forward生成了M+1个token（vs 普通的1个）
- **条件**：draft model的acceptance rate需>60%才有加速效果
- **vLLM支持**：Eagle（自回归draft）、Medusa（多头并行预测）、Lookahead Decoding等

**技术栈总览**：
```
请求到达 → Scheduler (Continuous Batching + 优先级调度)
         → Prefix Match (Prefix Caching查找)
         → KV Cache Alloc (PagedAttention分页分配)
         → Model Forward (TP + CUDA Graph + FP8/量化)
         → Sampling → 返回token
         → KV Cache Free (完成的请求释放page)
```

### Q: 如何利用大模型蒸馏一个3B的小模型？

**知识蒸馏（Knowledge Distillation）的完整Pipeline**：

**Step 1: 选择教师模型**
- 选择高性能大模型作为教师（如LLaMA-70B、GPT-4）
- 关键：教师模型的能力上界决定学生模型的能力上界
- 考虑教师模型是否开源（需要logits则必须是开源/自有模型）

**Step 2: 蒸馏策略选择**

| 策略 | 说明 | 数据需求 | 效果 |
|------|------|---------|------|
| Logit蒸馏 | 学生学习教师的输出概率分布 | 需要教师logits | 最好 |
| Black-box蒸馏 | 只用教师的输出文本训练学生 | 只需要输出 | 次好 |
| 特征蒸馏 | 匹配教师的中间层表示 | 需要中间层激活 | 适合特定场景 |

**Step 3: Logit蒸馏（最常用、效果最好）**

```python
# 蒸馏Loss = α * 硬标签Loss + (1-α) * 软标签Loss
hard_loss = CrossEntropy(student_logits, ground_truth_labels)
soft_loss = KL_Divergence(
    softmax(student_logits / T),   # T = temperature (通常2-10)
    softmax(teacher_logits / T)
)
total_loss = alpha * hard_loss + (1-alpha) * T^2 * soft_loss
```

- **Temperature T的作用**：T>1使softmax分布更平滑，暴露教师的"暗知识"
  - T=1：等同于普通training
  - T=4-8：常用范围，让学生学到教师对错误答案的细粒度排序
  - T越大→分布越均匀→学到更多相对关系
- **α的选择**：通常0.3-0.5（给软标签更大权重）

**Step 4: 数据策略**

**高质量合成数据生成（Black-box蒸馏核心）**：
- 用教师模型在种子任务上生成高质量response
- 筛选：只保留质量好的输出（自评分/人工过滤/reward model过滤）
- 多样性：使用不同temperature采样 + prompt变体 + domain覆盖
- 规模：通常需要100K-1M高质量instruction-response对
- 经典案例：Alpaca（GPT-4数据蒸馏到LLaMA-7B）

**数据配方**：
- 通用能力数据：50%（对话、问答、创作）
- 推理能力数据：20%（数学、代码、逻辑）
- 知识数据：20%（百科、专业领域）
- 安全对齐数据：10%（拒绝、安全回复）

**Step 5: 渐进蒸馏（Progressive Distillation）**
- 直接从70B→3B跨度太大，能力损失严重
- 渐进方案：70B → 13B → 7B → 3B，每步蒸馏保留更多能力
- 或多教师集成：用多个中间大小的模型ensemble作为教师

**Step 6: 训练配置**
- 学习率：比预训练低（通常1e-5 ~ 5e-5）
- Batch size：较大batch（稳定蒸馏信号）
- Epochs：2-5个epoch（避免过拟合教师分布）
- Warm-up：较长的warm-up期（10%）

**质量评估**：
- 蒸馏后3B模型通常能达到同规模预训练模型的质量
- 好的蒸馏3B模型可能接近6-7B预训练模型的效果
- 在特定垂直领域蒸馏效果更好（领域收窄后知识更容易压缩）

### Q: Logistic回归的模型原理和Loss？

**模型原理**：

Logistic回归是广义线性模型（GLM），通过Sigmoid函数将线性输出映射为概率：

```
z = w^T x + b                     # 线性组合
P(y=1|x) = σ(z) = 1/(1+exp(-z))  # Sigmoid映射到[0,1]
P(y=0|x) = 1 - σ(z)
```

**几何解释**：w^Tx + b = 0 定义了一个超平面（决策边界），z的符号决定分类，|z|表示"确信度"

**为什么用Sigmoid而非直接阈值？**
- Sigmoid是连续可微的→可以用梯度下降优化
- 输出是概率→有良好的概率解释性
- 是指数族分布的自然参数化→有统计学最优性

**Loss函数——二元交叉熵（Binary Cross-Entropy）**：

```
L = -[y * log(p) + (1-y) * log(1-p)]

展开：
- 当y=1时：L = -log(p)      → p越接近1，loss越小
- 当y=0时：L = -log(1-p)    → p越接近0，loss越小
```

**与最大似然估计的等价性**：
- 假设y服从伯努利分布：P(y|x) = p^y * (1-p)^(1-y)
- 对数似然：log P(y|x) = y*log(p) + (1-y)*log(1-p)
- 最大化对数似然 = 最小化交叉熵（前面加负号）

**梯度（简洁优美）**：
```
∂L/∂w = (σ(w^Tx+b) - y) * x = (p - y) * x
∂L/∂b = p - y
```
- 梯度 = (预测值 - 真实值) * 输入
- 当预测完全正确时（p=y），梯度为0
- 梯度大小正比于预测误差→自然的自适应学习率

**为什么不用MSE作为Loss？**
- MSE + Sigmoid组合的loss landscape非凸（有大片平坦区域），梯度消失严重
- Cross-entropy + Sigmoid是凸函数（全局最优可保证）
- Cross-entropy的梯度更"尖锐"，收敛更快

**与深度学习的联系**：
- 多类扩展：Softmax回归（Cross-entropy loss在K类上的自然推广）
- LLM的最后一层本质上就是一个超大规模的Softmax回归（vocab_size维）
- 每个token位置的loss = Cross-entropy(predicted_logits, true_token)

### Q: 给定时间序列如何进行特征筛选和基于规则的建模？

**特征工程——从原始时间序列提取有意义的特征**：

**1. 时域特征（Temporal Features）**
```python
# 滑动窗口统计
mean_7d = series.rolling(7).mean()      # 7天均值
std_7d = series.rolling(7).std()        # 7天波动率
skew_7d = series.rolling(7).skew()      # 偏度
max_min_7d = rolling_max - rolling_min  # 极差

# 差分特征
diff_1 = series.diff(1)   # 一阶差分（速度）
diff_2 = series.diff(2)   # 二阶差分（加速度）

# 滞后特征
lag_1 = series.shift(1)   # 1步滞后
lag_7 = series.shift(7)   # 7步滞后（周期性）
```

**2. 频域特征（Frequency Features）**
- FFT主频率和能量分布
- 小波变换的多尺度特征
- 自相关系数（ACF）/ 偏自相关系数（PACF）

**3. 统计检验特征**
- ADF检验的统计量（判断平稳性）
- 变点检测（CUSUM / Bayesian Online Change Point Detection）
- 异常度分数（Z-score / Isolation Forest score）

**特征筛选方法**：

| 方法 | 类型 | 优点 | 适用 |
|------|------|------|------|
| XGBoost Feature Importance | 模型驱动 | 考虑特征交互 | 通用 |
| L1正则化 (Lasso) | 嵌入式 | 自动置零不重要特征 | 线性关系强 |
| 互信息 (Mutual Information) | 过滤式 | 捕捉非线性关系 | 复杂依赖 |
| SHAP values | 模型解释 | 可解释性强 | 需要模型可信度高 |
| Pearson/Spearman相关系数 | 过滤式 | 简单快速 | 线性/单调关系 |
| Recursive Feature Elimination | 包装式 | 综合考虑特征组合 | 特征数不太多 |

**实操流程**：
1. 生成大量候选特征（50-200个）
2. 去除高相关性重复特征（correlation > 0.95 的保留一个）
3. XGBoost/LightGBM拟合 → feature importance排序
4. 前向/后向选择确定最终特征子集（通常10-30个）
5. 验证选定特征在时间外样本上的稳定性

**基于规则的建模**：

当模型可解释性比精度更重要时（如金融风控、运维告警），使用规则建模：

**方法1：决策树可视化规则**
```python
# 训练浅决策树（depth=3-5），提取规则
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_selected_features, y)
# 每条从根到叶的路径就是一条规则
# 如：IF mean_7d > 100 AND diff_1 > 10 AND std_7d > 5 THEN alarm=True
```

**方法2：业务领域知识构建规则**
```python
# 结合特征分析和领域知识
rules = {
    "急剧下降": lambda x: x['diff_1'] < -threshold_1 and x['magnitude'] > base,
    "持续异常": lambda x: all(x['zscore_rolling'].iloc[-5:] > 2.5),
    "周期性突变": lambda x: abs(x['value'] - x['lag_7']) > seasonal_threshold,
}
```

**方法3：规则优化与验证**
- 在训练集上确定阈值→验证集上调优→测试集上评估
- 用Precision-Recall曲线确定最优阈值
- 确保规则可解释且可维护（每条规则需有业务含义）
- 建立规则白名单/黑名单机制处理例外情况

### Q: DeepSeek-MTP是用在训练阶段还是推理阶段？具体过程？

**训练和推理都用，但目的不同**：

**训练阶段——作为辅助训练目标（Auxiliary Training Objective）**：

```
标准目标：Next-Token Prediction (NTP)
  Input:  [t1, t2, ..., t_n]
  Target: [t2, t3, ..., t_{n+1}]   (预测下一个token)

MTP目标：同时预测未来K个token
  Input:  [t1, t2, ..., t_n]
  Target_1: [t2, t3, ..., t_{n+1}]   (下一个, 主头)
  Target_2: [t3, t4, ..., t_{n+2}]   (下两个, 辅助头)
  Target_3: [t4, t5, ..., t_{n+3}]   (下三个, 辅助头)
  ...
```

**训练实现细节**：
- 主Transformer backbone共享，额外添加K-1个轻量级prediction head（通常是一层Transformer block + LM head）
- 每个辅助头接收backbone的输出和前一个头的预测embedding，预测对应位置
- 损失函数：`L_total = L_NTP + Σ_{k=2}^{K} α_k * L_MTP_k`（α通常递减，如0.3, 0.2, 0.1）

**训练收益**：
1. **更好的表示学习**：迫使模型在当前位置就"预见"更远的未来→学到更深层的语义结构
2. **隐式规划能力**：模型需要在内部表示中编码multi-step信息
3. **训练效率**：每个token提供更多监督信号（一个token产生K个loss），等效于更大的训练数据量

**推理阶段——作为Speculative Decoding的Draft Head**：

传统Speculative Decoding需要一个额外的小模型（draft model），而MTP头可以**零成本复用**：

```
Standard Speculative Decoding:
  Draft Model (外部小模型) → 生成K个候选token → 大模型一次验证

MTP-based Speculative Decoding:
  MTP Heads (内置) → 生成K个候选token → 主模型头验证
  
具体过程：
  Step 1: 主模型forward → 得到hidden states + 主头预测token_1
  Step 2: MTP head_2 利用hidden states → 预测token_2 (draft)
  Step 3: MTP head_3 → 预测token_3 (draft)
  ...
  Step K: 将[token_1, ..., token_K]一次性送入主模型验证
  Step K+1: 验证通过的token直接accept，第一个被拒绝的位置重新采样
```

**推理加速效果**：
- 接受率（Acceptance Rate）：因为MTP头与主模型同源训练，接受率通常>70%（远高于独立draft model的50-60%）
- 实际加速：1.8-2.4倍（取决于任务和K的设定）
- **零额外成本**：不需要加载额外的draft model，不额外占用显存

**与传统Speculative Decoding的对比**：

| 维度 | 外部Draft Model | MTP Head |
|------|----------------|----------|
| 额外显存 | 需要加载小模型(~1-3B) | 极小（几层MLP） |
| 接受率 | 50-60% | 70-80% |
| 加速比 | 1.5-2.0x | 1.8-2.4x |
| 部署复杂度 | 需要管理两个模型 | 集成在一个模型中 |
| Draft质量 | 独立训练，可能与大模型分布不一致 | 与主模型联合训练，分布一致性好 |

### Q: 如何通过Agent方法训练金融领域Coder模型？

**核心思路**：利用Agent的代码执行能力构建自动化的"数据生成→训练→验证"闭环（Data Flywheel）。

**完整Pipeline设计**：

**Phase 1: 数据收集（Agent作为数据收集器）**
```
Agent → 爬取金融代码仓库（量化策略、风控模型、数据分析脚本）
     → 收集金融API文档（交易所API、数据服务API）
     → 收集金融领域Q&A（StackOverflow金融tag、量化论坛）
     → 清洗+去重+质量过滤
```

**Phase 2: 任务生成（Agent作为任务设计者）**
- Agent分析收集的代码/文档，自动生成编程任务：
  - "实现一个计算夏普比率的函数"
  - "使用pandas重采样日线数据为周线OHLC"
  - "实现一个事件驱动的回测引擎"
- 从简单到复杂的课程设计（课程学习）
- 生成对应的参考答案和测试用例

**Phase 3: SFT微调（初始能力建立）**
- 用Agent生成的高质量instruction-code对做SFT
- 基座模型：CodeLLaMA-3B 或 DeepSeek-Coder-3B
- 数据量：~50K-200K金融编程instruction对

**Phase 4: RL优化（Agent提供奖励信号）— 核心创新**

```python
# Agent作为自动化验证器
def agent_reward(prompt, generated_code):
    # 1. 语法检查
    if not syntax_valid(generated_code):
        return -1.0
    
    # 2. 执行测试
    test_result = agent.execute_code(generated_code, test_cases)
    if not test_result.passed:
        return -0.5
    
    # 3. 功能验证（Agent理解prompt要求，验证输出正确性）
    output = agent.run_code(generated_code, sample_data)
    correctness = agent.verify(prompt, output)  # Agent判断输出是否符合预期
    
    # 4. 代码质量评估
    quality = agent.assess(generated_code, criteria=["efficiency", "readability", "best_practices"])
    
    return 0.3 * correctness + 0.3 * test_result.score + 0.2 * quality + 0.2 * safety_score
```

**关键优势**：金融代码有明确的正确性标准（运行结果可验证）→Agent的代码执行能力提供了高质量、低成本的奖励信号

**Phase 5: 迭代发现与补强（Agent作为评估者）**
```
Agent评估模型 → 发现薄弱领域（如衍生品定价代码正确率低）
             → 针对性补充该领域数据
             → 生成更多衍生品相关编程任务
             → 重新SFT + RL
             → 再评估...（循环迭代）
```

**数据飞轮的闭环**：
```
Agent收集/生成数据 → SFT训练 → Agent执行验证提供奖励 → RL优化 
     ↑                                                    |
     └──── Agent评估发现弱点 ← 模型能力提升 ←─────────────┘
```

**技术挑战**：
- **安全执行环境**：金融代码可能涉及真实API调用→沙箱隔离
- **奖励信号质量**：Agent自身的判断可能不完美→需要人工抽检calibrate
- **数据多样性**：避免Agent生成的数据过于同质化→引入随机性和外部数据
- **领域知识注入**：确保Agent理解金融特有概念（如T+1结算、涨跌停限制）
