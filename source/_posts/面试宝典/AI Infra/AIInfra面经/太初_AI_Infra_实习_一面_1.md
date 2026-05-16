---
title: '太初 AI Infra 实习 一面 (1)'
date: 2026-04-16 12:16:43
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 训练优化, 面经]
---


<!-- more -->

---

### Q: 数据并行的原理？

数据并行（Data Parallelism, DP）是最基础的分布式训练并行策略，核心思想是**数据分片、模型复制、梯度同步**。

**工作原理：**

```
         GPU 0          GPU 1          GPU 2          GPU 3
模型:     完整模型M       完整模型M       完整模型M       完整模型M
数据:     batch_0        batch_1        batch_2        batch_3
前向:     loss_0         loss_1         loss_2         loss_3
反向:     grad_0         grad_1         grad_2         grad_3
                    ↓ AllReduce (平均梯度) ↓
同步:     avg_grad       avg_grad       avg_grad       avg_grad
更新:     M' = M - lr*avg_grad  (所有设备执行相同更新，保持一致)
```

**主要变体对比：**

| 变体 | 显存占用 | 通信模式 | 特点 |
|------|---------|---------|------|
| DP (DataParallel) | 完整模型+优化器 | 参数Server模式 | PyTorch旧API，单进程多线程，GIL瓶颈 |
| DDP (DistributedDataParallel) | 完整模型+优化器 | Ring AllReduce | 多进程，通信与计算重叠，主流 |
| ZeRO Stage 1 | 优化器状态1/N | AllGather+ReduceScatter | 减少优化器显存(如Adam的m,v) |
| ZeRO Stage 2 | +梯度1/N | AllGather+ReduceScatter | 进一步减少梯度显存 |
| ZeRO Stage 3 (FSDP) | +参数1/N | AllGather(前向)+ReduceScatter(反向) | 等效于模型并行的显存效率 |

**DDP的关键优化——计算通信重叠：**
```
# 梯度AllReduce与反向传播重叠
反向传播第L层 → 第L层梯度就绪 → 立即启动第L层的AllReduce
反向传播第L-1层 → ...（与第L层通信并行）
```
通过Bucket机制将小梯度合并为大buffer一次通信，减少通信次数。

**通信量分析：**
- Ring AllReduce：每个设备发送和接收 2×(N-1)/N × model_size ≈ 2×model_size的数据
- 8卡DDP训练7B模型(FP16)：每步通信约2×7B×2bytes = 28GB数据

### Q: FasterTransformer(FT)框架的特点？

FasterTransformer是NVIDIA开源的Transformer推理加速库（现已被**TensorRT-LLM**取代），代表了LLM推理优化的核心技术集合。

**核心技术栈：**

**1. 高度优化的CUDA Kernel：**
- 融合QKV线性投影为一个GEMM（减少3个kernel为1个）
- 融合 LayerNorm + Residual + Bias Add（减少多次HBM读写）
- 针对不同head_dim和seq_len的attention kernel特化版本
- FP16/INT8/FP8混合精度kernel

**2. 多级并行支持：**
- 张量并行(TP)：MLP列切+行切，Attention按head切分
- 流水线并行(PP)：按层切分到不同GPU
- 支持NVLink/NCCL高效通信

**3. KV-Cache管理：**
- 预分配连续KV-Cache buffer
- 支持beam search场景下的KV-Cache共享和复制
- 内存池管理避免频繁分配释放

**4. 量化推理：**
- SmoothQuant INT8量化（W8A8）
- Weight-Only INT8/INT4量化
- FP8推理（H100）

**FT vs TensorRT-LLM（继任者）：**

| 维度 | FasterTransformer | TensorRT-LLM |
|------|-------------------|--------------|
| API | C++ API为主 | Python API + C++后端 |
| 模型支持 | 需要手动适配 | 自动模型转换 |
| 动态batch | 有限支持 | 原生Continuous Batching |
| PagedAttention | 不支持 | 支持 |
| 量化 | INT8/FP16 | INT4/INT8/FP8全系列 |
| 维护状态 | 已停止维护 | NVIDIA主推 |

### Q: 随机森林的原理？

随机森林是基于**Bagging**思想的集成学习方法，通过构建多棵随机化的决策树并聚合预测结果。

**训练过程：**

```
原始训练集D (N个样本, F个特征)
    ↓ 重复T次（构建T棵树）
    ├── Bootstrap采样: 有放回抽取N个样本(约63.2%被选中, 36.8%为OOB)
    ├── 特征随机选择: 每次分裂从F个特征中随机选取m个(分类m=√F, 回归m=F/3)
    ├── 构建决策树: 在m个特征中选最优分裂(信息增益/Gini/MSE)
    └── 不剪枝: 让每棵树充分生长
    
预测：
    分类 → 多数投票
    回归 → 取均值
```

**为什么有效？——偏差-方差分析：**
- 单棵决策树：低偏差、高方差（过拟合）
- 随机森林：通过平均多棵低相关性的树，**方差降低而偏差不增**
- 关键：树之间的相关性越低，集成效果越好（Bootstrap+特征随机化降低相关性）

**核心优势：**
- 不易过拟合（可以加任意多棵树而不会过拟合）
- 可以评估特征重要性（通过置换特征后OOB误差的变化）
- 对缺失值和异常值鲁棒
- 支持并行训练（各树独立）
- 自带交叉验证（OOB数据可估计泛化误差）

### Q: GBDT的原理？

GBDT（Gradient Boosting Decision Tree）是基于**Boosting**思想的集成方法，核心是**每棵新树拟合前面所有树的残差（负梯度方向）**。

**算法流程：**

```
初始化: F_0(x) = argmin_c Σ L(y_i, c)  （如均值）

For m = 1 to M:
    1. 计算负梯度(伪残差): r_im = -∂L(y_i, F_{m-1}(x_i))/∂F_{m-1}(x_i)
    2. 用决策树拟合伪残差: h_m(x) ← fit(X, r)
    3. 学习率缩放后更新: F_m(x) = F_{m-1}(x) + η · h_m(x)

最终模型: F_M(x) = F_0(x) + η·Σ h_m(x)
```

**与随机森林的本质区别：**

| 维度 | 随机森林(Bagging) | GBDT(Boosting) |
|------|-----------------|----------------|
| 树的构建 | 并行独立构建 | 串行，每棵依赖前一棵 |
| 每棵树的目标 | 拟合原始标签 | 拟合残差/负梯度 |
| 每棵树的深度 | 深（不剪枝） | 浅（通常4-8层） |
| 偏差vs方差 | 降低方差 | 降低偏差 |
| 过拟合风险 | 低 | 高（需要early stopping/正则化） |
| 并行性 | 天然并行 | 难以并行 |

**XGBoost/LightGBM的改进：**
- XGBoost：加入正则化项（叶节点数+叶值L2），二阶泰勒展开更精确
- LightGBM：GOSS(梯度采样)+EFB(互斥特征捆绑)加速训练，Histogram-based分裂

### Q: 优化器了解哪些？

**主流优化器对比：**

| 优化器 | 更新规则核心 | 显存(每参数) | 适用场景 |
|--------|------------|------------|---------|
| SGD+Momentum | v=βv+g; w-=lr·v | +1份(v) | CV任务、需要好泛化性 |
| Adam | m,v指数移动平均; w-=lr·m/(√v+ε) | +2份(m,v) | 通用，NLP/生成模型 |
| AdamW | Adam + 正确的weight decay | +2份 | 大模型训练标配 |
| LAMB/LARS | 层级自适应lr(按层norm缩放) | +2份 | 超大batch训练(batch>8K) |
| Adafactor | 分解二阶矩为行/列向量 | +约0.5份 | 大模型省显存 |
| Lion | sign(β1·m+(1-β1)·g) | +1份(m) | 更少显存，Google推荐 |
| 8-bit Adam | 量化m/v为INT8存储 | +0.5份 | 微调省显存 |

**Adam为什么是默认选择？**
- 一阶矩m（动量）：平滑梯度方向，减少震荡
- 二阶矩v（自适应学习率）：对不同参数自适应调整步长（梯度大的参数步长小）
- 偏差校正：训练初期m/v还不准，除以(1-β^t)修正

**AdamW vs Adam的关键区别：**
```python
# Adam (错误的weight decay实现)
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g²
w = w - lr * m/(√v+ε) - lr*λ*w  # weight decay与自适应lr耦合

# AdamW (解耦weight decay)
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g²
w = w - lr * m/(√v+ε) - λ*w     # weight decay独立于adam更新
```
AdamW中weight decay不经过Adam的自适应缩放，正则化效果更稳定。

### Q: BERT和GPT的区别？

**架构与训练范式对比：**

| 维度 | BERT (Encoder-only) | GPT (Decoder-only) |
|------|---------------------|-------------------|
| 注意力方向 | 双向（看到所有token） | 因果（只看左侧token） |
| 预训练任务 | MLM(随机mask15%预测) + NSP | 自回归(预测下一个token) |
| 推理方式 | 整句一次编码 | 逐token生成 |
| 擅长任务 | 理解任务(分类/NER/QA抽取) | 生成任务(对话/代码/创作) |
| 代表模型 | BERT-base(110M), RoBERTa | GPT-3(175B), GPT-4, LLaMA |
| Scaling趋势 | 参数量较小(百M级) | 越大越强(几B~万亿) |

**为什么现代大模型选择GPT路线？**
1. **统一范式**：自回归可以统一理解和生成（生成包含理解）
2. **Scaling Law更好**：Decoder-only的loss随模型/数据增大持续下降
3. **Zero-shot/Few-shot能力**：预训练自然获得in-context learning能力
4. **训练效率**：每个token都贡献loss（BERT只有15%的mask token贡献loss）

**Encoder-only仍有优势的场景：**
- 分类/检索任务（双向编码信息更丰富）
- 推理延迟敏感（一次前向vs逐token生成）
- Embedding生成（句子向量表示）

### Q: Transformer的结构？

**完整Transformer架构：**

```
输入 → Token Embedding + Positional Encoding
  ↓
Encoder (×N layers):
  ├── Multi-Head Self-Attention
  │     Q = X·W_Q, K = X·W_K, V = X·W_V
  │     Attention = softmax(QK^T/√d_k)·V
  │     MultiHead = Concat(head_1,...,head_h)·W_O
  ├── Add & LayerNorm (残差连接)
  ├── Feed-Forward Network
  │     FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2  (或SwiGLU)
  └── Add & LayerNorm
  ↓
Decoder (×N layers):
  ├── Masked Multi-Head Self-Attention (因果mask)
  ├── Add & LayerNorm
  ├── Cross-Attention (Q来自decoder, K/V来自encoder输出)
  ├── Add & LayerNorm
  ├── Feed-Forward Network
  └── Add & LayerNorm
  ↓
Linear → Softmax → 输出概率
```

**关键设计决策在现代模型中的演进：**

| 组件 | 原始Transformer | 现代LLM(如LLaMA) |
|------|----------------|-----------------|
| LayerNorm位置 | Post-LN | Pre-LN(训练更稳定) |
| 位置编码 | Sinusoidal/Learned | RoPE(相对位置,支持外推) |
| FFN结构 | ReLU两层MLP | SwiGLU(3层,效果更好) |
| 注意力 | MHA | GQA(减少KV-Cache) |
| Norm类型 | LayerNorm | RMSNorm(更快,无bias) |

### Q: 线上服务推理如何提高吞吐量？

**系统级优化——从请求到响应的完整优化链：**

**1. 调度与Batching优化（最大化GPU利用率）：**

| 技术 | 原理 | 效果 |
|------|------|------|
| Continuous Batching | 每个decode step级别动态增删请求 | 吞吐提升2-5x |
| Chunked Prefill | 长prefill分块与decode穿插 | P99延迟降低 |
| PD分离 | Prefill/Decode分到不同GPU | 资源匹配更优 |
| 优先级调度 | decode优先保证生成不中断 | 延迟更平稳 |

**2. 显存与KV-Cache优化：**
- PagedAttention：分页管理KV-Cache，消除碎片，利用率接近100%
- Prefix Caching：共享前缀复用KV-Cache（如system prompt）
- KV-Cache量化(INT8/FP8)：减少显存占用和读取带宽
- Token eviction：驱逐不重要的历史token KV

**3. 计算优化：**
- 量化(INT8/FP8/INT4)：降低计算量和带宽需求
- FlashAttention：减少HBM IO，加速Attention计算
- CUDA Graph：消除kernel launch开销（decode阶段固定pattern）
- 算子融合：减少中间tensor的HBM读写

**4. 生成加速：**
- 投机解码(Speculative Decoding)：小模型draft+大模型verify，加速1.5-3x
- Medusa/EAGLE：多头并行预测多个token
- Lookahead Decoding：利用Jacobi迭代并行解码

**5. 分布式与扩展：**
- 张量并行(TP)：单次推理延迟降低
- Disaggregated Serving：prefill和decode节点池化，弹性伸缩
- 负载均衡：请求路由到负载最低的实例

### Q: 手撕：链表加法？

（编程题）
