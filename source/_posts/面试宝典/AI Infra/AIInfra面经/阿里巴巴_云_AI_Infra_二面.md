---
title: '阿里巴巴 云 AI Infra 二面'
date: 2026-04-16 11:43:50
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 什么是交叉熵？

交叉熵`H(p,q) = -Σ p(x)log(q(x))`衡量真实分布p与预测分布q的差异。在分类中p为one-hot标签，退化为`-log(q_y)`。值越小表示预测越准确。信息论意义：用分布q编码分布p中采样的数据所需的平均比特数。

### Q: vLLM有哪些优化技术？

- **PagedAttention**：KV Cache分页管理，消除内存碎片。
- **Continuous Batching**：请求完成立即退出，新请求随时加入。
- **Prefix Caching**：相同前缀的请求共享KV Cache。
- **Chunked Prefill**：长prompt分块处理避免阻塞decode。
- **Speculative Decoding**：投机解码加速生成。
- **量化支持**：GPTQ/AWQ/FP8等多种量化格式。
- **Tensor Parallelism**：多卡推理支持。
- **CUDA Graph**：减少kernel launch开销。

### Q: Logistic回归的模型原理和Loss？

**原理**：线性模型的输出经sigmoid函数映射到(0,1)概率区间：`P(y=1|x) = σ(w^Tx + b) = 1/(1+exp(-(w^Tx+b)))`。

**Loss**：二元交叉熵（Binary Cross Entropy）：`L = -[y*log(p) + (1-y)*log(1-p)]`。等价于最大似然估计。梯度形式简洁：`∂L/∂w = (p-y)*x`。

### Q: 给定时间序列，如何通过机器学习筛选重要特征并进行建模？

**特征工程**：提取统计特征（均值/方差/趋势/周期性）、滑动窗口特征、频域特征（FFT）、差分特征。

**特征选择**：
- 相关性分析（Pearson/Spearman）。
- 基于树模型的特征重要性（XGBoost/LightGBM feature importance）。
- L1正则化（Lasso）自动稀疏选择。
- 递归特征消除（RFE）。

**基于规则建模**：筛选出重要特征后，设定阈值/组合条件构建规则引擎，结合领域知识定义告警/决策逻辑。

### Q: DeepSeek-MTP是用在训练阶段还是推理阶段？具体过程？

MTP（Multi-Token Prediction）**训练时使用，推理时可选择性使用**。

**训练过程**：在标准next-token prediction基础上，增加额外的预测头（共享backbone特征），同时预测未来多个token（如2-4个）。多个头的loss加权求和作为训练目标。

**推理中的应用**：MTP训练的额外预测头可作为Speculative Decoding的draft head，以近零额外成本生成草稿token，主模型验证后加速生成。

### Q: 如何通过Agent方法训练一个金融领域的Coder模型？

1. **数据收集**：用Agent从金融开源项目、量化策略库、API文档中收集代码和文档。
2. **数据生成**：Agent自动生成金融编程任务（如"写一个动量策略"）及参考答案。
3. **SFT**：用收集和生成的金融代码数据微调base模型。
4. **RLHF/RLAIF**：Agent执行生成的代码验证正确性，以执行结果作为奖励信号进行RL训练。
5. **迭代优化**：Agent评估模型输出，发现薄弱领域，补充训练数据，循环改进。
