---
title: '字节跳动 抖音 AI Infra'
date: 2026-04-16 11:43:07
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 过拟合怎么解决？L1和L2正则的区别？L2的导数是什么？

**过拟合的本质**：模型在训练集上学到了噪声和细节而非真正的规律，导致泛化能力差。根本原因是模型复杂度相对于数据量过高。

**系统化的解决方案（从数据、模型、训练三个维度）**：

| 维度 | 方法 | 原理 | 适用场景 |
|------|------|------|---------|
| 数据 | 增加数据量 | 更多数据覆盖真实分布 | 数据获取成本低时 |
| 数据 | 数据增强 | 人工扩充数据多样性 | CV/NLP通用 |
| 模型 | 减小模型 | 降低模型capacity | 模型明显过大时 |
| 模型 | Dropout | 随机屏蔽神经元 | 深度网络 |
| 训练 | L1/L2正则 | 约束权重大小 | 所有模型 |
| 训练 | Early Stopping | 验证集loss上升时停止 | 通用 |
| 训练 | Batch Normalization | 减少内部协变量偏移+轻微正则效果 | 深度网络 |
| 训练 | Label Smoothing | 软化标签防止过度自信 | 分类任务 |
| 训练 | Weight Decay | 每步衰减权重 | Adam系优化器 |
| 训练 | Mixup/CutMix | 混合样本和标签 | CV任务 |

**L1 vs L2正则——深入对比**：

总loss = 原始loss + λ × 正则项

| 维度 | L1 (Lasso) | L2 (Ridge) |
|------|-----------|-----------|
| 正则项 | λ × Σ\|w_i\| | λ × Σw_i^2 |
| 导数/梯度 | λ × sign(w_i) | λ × 2w_i |
| 权重衰减行为 | 每步减少固定常数 | 每步减少比例值 |
| 结果分布 | 稀疏（很多权重精确为0） | 均匀缩小（接近0但不为0） |
| 几何解释 | L1球（菱形）角点在坐标轴上 | L2球（圆形）无角点 |
| 特征选择 | 自动选择重要特征 | 保留所有特征但权重缩小 |
| 适用场景 | 高维稀疏特征 | 特征都有用、防止权重过大 |

**为什么L1产生稀疏解（几何直觉）**：
```
优化问题: min Loss(w) s.t. ||w||_1 ≤ C (等价于L1正则)

等高线(Loss的) 与 L1球(菱形) 的切点:
  菱形的角在坐标轴上 → 切点大概率在角上 → 某些w_i=0
  
L2对比: 圆形无角点 → 切点在任意位置 → w_i接近0但不为0
```

**L2的导数推导**：
```
正则项: R = λ × w^2
dR/dw = λ × 2w

在梯度下降中:
w_new = w - lr × (dLoss/dw + 2λw)
      = w(1 - 2λ×lr) - lr × dLoss/dw
      ↑ 这就是"weight decay"——每步将w乘以(1-2λ×lr)<1

当使用Adam时，L2正则和Weight Decay不等价！
- L2: 梯度2λw参与Adam的自适应学习率计算
- Weight Decay (AdamW): 直接在更新后 w = w - η×2λ×w，不经过自适应
- AdamW的理论更正确（Loshchilov & Hutter, 2019）
```

**常见误区**：
- "Weight Decay = L2正则"：只在SGD中等价，在Adam中不等价。
- "L1正则让权重为0"：严格来说只在全局最优解处为0，SGD实践中可能在0附近震荡。

### Q: Dropout训练和测试的区别？

**Dropout的完整机制**：

**训练时（Inverted Dropout，现代标准实现）**：
```python
# PyTorch实现等价代码
def dropout_train(x, p=0.5):
    # 1. 生成mask: 每个元素以概率p被置0
    mask = (torch.rand_like(x) > p).float()
    # 2. 应用mask
    x_dropped = x * mask
    # 3. 缩放: 除以(1-p)保持期望不变
    x_scaled = x_dropped / (1 - p)
    return x_scaled

# 为什么要除以(1-p)?
# 训练时期望: E[x_scaled] = E[x * mask / (1-p)] = x * (1-p) / (1-p) = x
# 这样测试时不需要做任何修改！
```

**测试时**：
```python
def dropout_test(x, p=0.5):
    return x  # 不做任何操作！（因为训练时已经用inverted scaling补偿了）
```

**历史版本（原始Dropout，已不常用）**：
```
训练: x_out = x * mask（不缩放）
测试: x_out = x * (1-p)（整体缩放以匹配训练时的期望）
问题: 测试时需要修改前向传播 → 不方便、容易忘记 → 被inverted版本取代
```

**Dropout的多重理论解释**：

| 解释角度 | 内容 |
|---------|------|
| 集成学习 | 每次前向是不同子网络，等价于指数级子网络的近似集成 |
| 噪声注入 | 对隐藏层添加乘性噪声(Bernoulli)，相当于数据增强 |
| 共适应打破 | 防止神经元过度依赖特定其他神经元，迫使学习更鲁棒的特征 |
| 贝叶斯近似 | 可以证明Dropout近似贝叶斯推断(MC Dropout) |
| 正则化 | 等价于自适应的L2正则（与输入相关的正则强度） |

**Dropout位置和超参数选择**：
```
典型配置:
- 全连接层后: p=0.5（最常见）
- CNN中: Spatial Dropout, p=0.1-0.3
- RNN中: 只在时间步间drop，不在时间步内drop
- Transformer中:
  - Attention权重后: p=0.1
  - FFN中间层后: p=0.1
  - 残差连接前: p=0.1
  
大模型(>1B参数)通常不用Dropout: 数据量足够大时正则效果有限，反而降低训练效率
```

**Dropout的变体**：
- **DropConnect**：随机置零权重而非激活值。
- **Spatial Dropout**：整个channel一起drop（CNN中更合理）。
- **DropBlock**：随机drop连续区域而非单个元素。
- **MC Dropout**：测试时也开启Dropout，多次前向取均值→不确定性估计。
- **Stochastic Depth**：随机跳过整个层（ResNet中使用）。

### Q: 常见优化器有哪些？详细讲讲。

优化器是深度学习训练的核心组件，其选择直接影响收敛速度、最终性能和稳定性：

**1. SGD（Stochastic Gradient Descent）**：
```python
# 最朴素的更新
w = w - lr * gradient

# 特点:
# - 每步只用mini-batch估计梯度 → 噪声大但有正则效果
# - 收敛慢、容易卡在鞍点
# - 但最终解的泛化性通常最好（sharp vs flat minima理论）
# - 大模型预训练中仍有使用（配合lr warmup + cosine decay）
```

**2. SGD + Momentum**：
```python
# 引入动量（指数移动平均）
v = beta * v + gradient           # beta通常0.9
w = w - lr * v

# 物理直觉: 小球在loss surface上滚动，有惯性
# 优势:
#   - 加速收敛: 一致方向的梯度被累积放大
#   - 减少振荡: 垂直方向的梯度正负抵消
#   - 帮助逃离小的局部最优和鞍点

# Nesterov Momentum (略优):
v = beta * v + gradient_at(w - beta * v)  # 先"看一步"再算梯度
```

**3. AdaGrad（Adaptive Gradient）**：
```python
# 对每个参数自适应学习率
G += gradient^2                   # 累积历史梯度平方和
w = w - lr / (sqrt(G) + eps) * gradient

# 频繁更新的参数: G大 → 有效lr小（已经学得差不多了）
# 稀疏特征: G小 → 有效lr大（难得出现，要多学）
# 致命缺陷: G单调递增 → 后期lr趋近0 → 训练停止
```

**4. RMSProp（Root Mean Square Propagation）**：
```python
# 用指数移动平均代替累积（解决AdaGrad的lr单调衰减问题）
v = beta * v + (1-beta) * gradient^2    # beta通常0.99
w = w - lr / (sqrt(v) + eps) * gradient

# v是梯度平方的移动平均 → 不会无限累积
# 效果: 自适应学习率但不会过度衰减
# 广泛用于RNN训练
```

**5. Adam（Adaptive Moment Estimation）**：
```python
# 结合Momentum(一阶矩) + RMSProp(二阶矩) + Bias Correction
m = beta1 * m + (1-beta1) * gradient     # 一阶矩（动量）, beta1=0.9
v = beta2 * v + (1-beta2) * gradient^2   # 二阶矩（自适应lr）, beta2=0.999

# Bias correction（消除初始化偏差）
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)

w = w - lr * m_hat / (sqrt(v_hat) + eps)  # eps=1e-8

# 为什么需要bias correction?
# 初始m=0, 第1步: m = 0.1*g (远小于真实均值g)
# 校正: m/(1-0.9^1) = 0.1*g/0.1 = g ✓

# Adam是最常用的默认选择——收敛快、对超参数鲁棒
```

**6. AdamW（Adam with Decoupled Weight Decay）**：
```python
# Adam + L2正则时的问题:
# 标准Adam中 gradient += 2*lambda*w → 这个"正则梯度"也被自适应缩放了！
# 导致大梯度的参数正则效果反而弱（不正确）

# AdamW: 将weight decay从梯度计算中分离
m, v = adam_update(gradient)  # 只用任务梯度更新m,v
w = w - lr * m_hat / (sqrt(v_hat) + eps)  # Adam步
w = w - lr * lambda * w                    # Weight decay步（独立！）

# 好处: weight decay强度不受自适应学习率影响，每个参数的正则力度一致
# 现在大模型训练的标准选择: AdamW + cosine lr schedule + warmup
```

**7. LAMB/LARS（大Batch训练专用）**：
```python
# 问题: batch size从256增到32K时，训练不收敛
# 原因: 不同层的梯度范数差异大，统一lr导致部分层更新过大

# LARS (Layer-wise Adaptive Rate Scaling):
for each layer l:
    local_lr = trust_ratio * ||w_l|| / ||grad_l||  # 按层缩放
    w_l = w_l - lr * local_lr * grad_l

# LAMB: LARS + Adam的结合
# 让batch size可以扩展到32K-64K而不损失收敛性
# BERT预训练从3天缩短到76分钟（batch 64K + LAMB）
```

**优化器选择决策**：

| 场景 | 推荐优化器 | 原因 |
|------|-----------|------|
| 大模型预训练 | AdamW | 收敛快、稳定 |
| 大batch预训练 | LAMB/LARS | 支持超大batch |
| 微调(SFT/LoRA) | AdamW | 标准选择 |
| CNN图像分类 | SGD+Momentum | 泛化最好 |
| 显存受限 | Adafactor | 分解二阶矩省内存 |
| 小数据微调 | SGD+小lr | 避免过拟合 |

### Q: 怎么筛选特征？

特征选择旨在从原始特征集中选出最有价值的子集，减少维度、降低过拟合风险、加速训练并提高模型可解释性：

**1. 过滤法（Filter Methods）——独立于模型**：

| 方法 | 原理 | 适用数据类型 | 复杂度 |
|------|------|------------|--------|
| 方差选择 | 移除低方差特征(近常数) | 数值型 | O(n) |
| 相关系数 | Pearson/Spearman衡量与目标的线性/单调关系 | 数值型 | O(n×d) |
| 卡方检验 | 检验特征与标签的独立性 | 分类变量 | O(n×d) |
| 互信息(MI) | 非线性依赖关系(信息论) | 通用 | O(n×d) |
| F-test/ANOVA | 组间方差/组内方差 | 数值特征+分类标签 | O(n×d) |

```python
# 实践中的Filter方法
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# 选择互信息最高的K个特征
selector = SelectKBest(mutual_info_classif, k=50)
X_selected = selector.fit_transform(X, y)

# 移除高度相关的冗余特征(特征间)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
```

优点：速度快、与模型无关。缺点：忽略特征组合效应（两个单独无用的特征组合可能很有用）。

**2. 包装法（Wrapper Methods）——基于模型表现**：

```python
# 递归特征消除(RFE): 反复训练模型，每次移除最不重要的特征
from sklearn.feature_selection import RFECV

rfe = RFECV(estimator=RandomForestClassifier(), step=1, cv=5, scoring='accuracy')
rfe.fit(X, y)
selected = X.columns[rfe.support_]

# 前向选择: 从空集开始，每次加入提升最大的特征
# 后向消除: 从全集开始，每次移除损失最小的特征
```

优点：考虑了特征组合和模型特性。缺点：计算开销大（需要多次训练模型），O(d^2 × 训练成本)。

**3. 嵌入法（Embedded Methods）——模型内置**：

```python
# L1正则（自动特征选择）
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5).fit(X, y)
selected = X.columns[lasso.coef_ != 0]  # 系数不为0的特征被选中

# 树模型特征重要性
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier().fit(X, y)
importances = model.feature_importances_  # 基于信息增益/GINI减少
# 选择importance > threshold的特征

# XGBoost/LightGBM的多种重要性指标:
# - weight: 特征被用于分裂的次数
# - gain: 特征带来的平均信息增益
# - cover: 特征覆盖的样本数
# gain通常最有参考价值
```

优点：计算效率高（训练一次即可）、考虑特征交互。

**4. 深度学习中的特征选择**：

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| Attention权重 | 高注意力=高重要性 | NLP/序列数据 |
| Embedding层 | 自动学习稀疏特征的表示 | 推荐系统 |
| Feature Gate | 可学习的特征门控 | 多模态融合 |
| AutoML/NAS | 自动搜索最优特征+结构 | 有计算预算时 |
| SHAP值 | 基于博弈论的特征贡献 | 模型解释 |

**特征选择的实践建议**：
1. 先做Filter快速去掉明显无用的特征（方差=0、缺失>90%）。
2. 用树模型的feature_importance做初步排序。
3. 用RFECV或Permutation Importance精细筛选。
4. 注意特征泄漏（data leakage）——不要用未来信息做特征。
5. 验证：特征子集在测试集上的表现才是最终标准。

### Q: 树模型和线性模型的区别？

**全面对比——从假设、能力、工程到应用场景**：

**基本原理对比**：
```
线性模型: y = w^T × x + b
  决策边界是超平面(直线/平面)
  假设输出是特征的线性组合

树模型: if x1 > 3.5 and x2 < 7.0 then class=A
  决策边界是轴对齐的阶梯函数
  通过递归分裂学习规则
```

**多维度详细对比**：

| 维度 | 线性模型 | 树模型(GBDT/XGBoost/RF) |
|------|---------|----------------------|
| 假设空间 | 线性关系 | 任意非线性(充分深) |
| 决策边界 | 超平面(光滑) | 轴对齐阶梯(非光滑) |
| 特征交互 | 需手动构造交叉特征 | 天然学习特征交互 |
| 特征缩放 | 必须(归一化/标准化) | 不需要(只看排序) |
| 缺失值处理 | 需要填充 | 天然处理(学习分裂方向) |
| 类别特征 | 需要独热编码 | 可直接处理(LightGBM) |
| 可解释性 | 权重直接解释(每个特征贡献) | 特征重要性+SHAP |
| 训练速度 | 极快(解析解或几步收敛) | 中等(迭代建树) |
| 预测速度 | 极快(矩阵乘) | 快(树遍历) |
| 高维稀疏 | 非常适合(如文本TF-IDF) | 较差(分裂效率低) |
| 数据量小 | 不容易过拟合 | 容易过拟合 |
| 数据量大 | 拟合能力不足 | 表现优秀 |
| 正则化 | L1/L2 | 树深度/叶子数/采样率 |
| 外推能力 | 有(线性延伸) | 无(只能预测训练范围内的值) |

**为什么推荐系统偏爱线性模型+树模型组合**：
```
实际系统架构:
  1. 线性模型(LR): 处理超高维稀疏特征(用户ID×物品ID的交叉, 10亿维)
     - 训练快、支持在线学习、容易分布式
  2. GBDT: 处理中维稠密特征(统计特征、embedding)
     - 自动学习非线性和特征交互
  3. 深度模型(DNN): 学习高阶特征交互
     - Wide&Deep = LR + DNN
     - DeepFM = FM(交叉) + DNN(高阶)
```

**选择决策指南**：

| 场景特征 | 推荐模型 | 原因 |
|---------|---------|------|
| 高维稀疏(>100K维, 如NLP) | 线性模型(LR/SVM) | 树模型在稀疏高维上效率差 |
| 中维稠密(10-1000维) | 树模型(XGBoost/LightGBM) | 自动处理非线性交互 |
| 需要外推(预测训练范围外) | 线性模型 | 树模型无外推能力 |
| 小数据(<1000样本) | 线性模型+正则 | 避免过拟合 |
| 大数据+非线性 | 集成树模型 | 拟合能力强+不过拟合 |
| 实时在线学习 | 线性模型(FTRL) | 增量更新简单 |
| 表格数据竞赛 | XGBoost/LightGBM | 几乎总是最优 |
| 图像/文本/语音 | 深度模型(CNN/Transformer) | 需要学习表示 |

**常见误区**：
- "深度学习总是比树模型好"：在表格数据上，GBDT仍然在多数benchmark上优于或持平DNN（2022年多篇论文验证）。
- "线性模型太简单"：加入特征交叉和多项式特征后，线性模型也能拟合复杂关系。
- "树模型不需要特征工程"：虽然不需要缩放，但好的特征仍然显著提升效果。

### Q: 手撕：非递归实现二叉树中序遍历？

（编程题）
