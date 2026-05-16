---
title: '理想汽车 AI Infra 校招 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: PyTorch DDP（DistributedDataParallel）的原理？

DDP 是 PyTorch 的数据并行方案，核心思路是**模型复制 + 梯度同步**：

**架构**：
- 每张卡持有完整模型副本，处理不同 mini-batch 数据
- 前向：各卡独立计算，无通信
- 反向：梯度计算完成后通过 AllReduce 同步，确保各卡参数更新一致

**关键优化——Gradient Bucketing（梯度分桶 + 通信重叠）**：
- 将模型参数按反向计算顺序分成多个 bucket（默认 25MB/bucket）
- 当一个 bucket 内所有参数的梯度都已计算完成，立即发起该 bucket 的 AllReduce
- 不等全部梯度算完再通信——**反向计算与通信重叠**
- 效果：通信时间几乎完全被计算掩盖（前提：计算时间 > 通信时间）

**通信后端**：
- NCCL：GPU 间通信（NVLink 内/IB 跨节点），默认首选
- Gloo：CPU 或备选后端
- 通信算法：Ring AllReduce（大消息带宽最优）或 Tree AllReduce（小消息延迟最低）

**通信量**：每次 AllReduce 传输 2 × model_size × (P-1)/P ≈ 2×model_size（与 GPU 数量 P 无关！Ring 算法的优势）

**局限**：模型必须放进单卡显存（包括参数+梯度+优化器状态+激活）。8B 模型 Adam 训练需要约 100+GB/卡——需要 ZeRO 配合。

---

### Q: CV 的发展路径？

**深度学习时代 CV 的演进主线**：

```
手工特征 → CNN 爆发 → 更深网络 → 高效架构 → Vision Transformer → 大规模预训练
```

| 年份 | 里程碑 | 核心贡献 |
|---|---|---|
| 2012 | AlexNet | GPU + ReLU + Dropout，深度学习在 CV 爆发 |
| 2014 | VGGNet | 证明深度（3×3 堆叠）比宽度更重要 |
| 2014 | GoogLeNet/Inception | 多尺度并行分支，减少参数 |
| 2015 | ResNet | 残差连接解决退化问题，可训练 100+层 |
| 2017 | DenseNet | 密集连接，特征复用最大化 |
| 2019 | EfficientNet | NAS 搜索最优的宽度/深度/分辨率配比 |
| 2020 | ViT | 纯 Transformer 做图像分类，大数据下超越 CNN |
| 2021 | Swin Transformer | 窗口注意力+层次化，兼顾效率和全局建模 |
| 2022 | MAE/BEiT | 自监督预训练（mask image modeling） |
| 2023 | SAM/DINO v2 | 大规模基础视觉模型，zero-shot 泛化 |

**核心演进方向**：网络深度 → 连接方式创新 → 注意力机制引入 → 自监督大规模预训练 → 多模态统一。

---

### Q: NLP 的发展路径？RNN 和 Transformer 的优缺点？

**NLP 发展路径**：
```
RNN → LSTM/GRU → Seq2Seq+Attention → Transformer → BERT/GPT → GPT-3/LLM → ChatGPT/指令微调 → Reasoning (o1/R1)
```

**RNN vs Transformer 详细对比**：

| 维度 | RNN/LSTM | Transformer |
|---|---|---|
| 序列处理 | 串行逐步 | 全并行 |
| 长距离依赖 | O(n) 路径，梯度衰减 | O(1) 路径（self-attention） |
| 训练并行度 | 不可并行（时间步依赖） | 完全并行（所有 token 同时计算） |
| 计算复杂度 | O(n×d²) per step | O(n²×d) per layer |
| 内存 | O(1) 状态（固定大小 hidden state）| O(n²) attention 矩阵 |
| 参数效率 | 参数量小 | 需要更多参数和数据 |
| 归纳偏置 | 天然的序列局部性 | 无（需要位置编码、更多数据） |
| Scaling | Scaling law 不明确 | Scaling law 明确（越大越好） |
| 推理（生成） | O(1) per step（只用 hidden state）| O(n) per step（需读 KV Cache） |

**为什么 Transformer 胜出**：
- 训练效率：GPU 天然适合并行矩阵运算
- Scaling Law：Transformer + 更多数据 + 更大模型 → 持续涨点
- FlashAttention 解决了 O(n²) 内存瓶颈
- KV Cache 解决了推理效率

---

### Q: 模型训练的并行方式有哪些？

| 并行方式 | 切分维度 | 通信操作 | 适用场景 |
|---|---|---|---|
| DP (Data Parallel) | 数据 | AllReduce 梯度 | 模型放得进单卡 |
| TP (Tensor Parallel) | 层内权重 | AllReduce/ReduceScatter | 单层太大放不进单卡 |
| PP (Pipeline Parallel) | 层间 | P2P Send/Recv | 模型层数多 |
| SP (Sequence Parallel) | 序列维度 | AllGather/ReduceScatter | 减少 activation 显存 |
| EP (Expert Parallel) | MoE Expert | All-to-All | MoE 模型 |
| ZeRO | 优化器/梯度/参数 | AllGather/ReduceScatter | 突破单卡显存限制 |

**典型组合（3D 并行 + ZeRO）**：
- 训练 70B 模型，64 张 H100：DP=8 × TP=4 × PP=2
- 训练 405B 模型：DP=8 × TP=8 × PP=16 + ZeRO-1

**选择原则**：
- TP 度 ≤ 单机 GPU 数（NVLink 内通信）
- PP 跨机（通信量小，只传激活/梯度）
- DP 度 = 总 GPU 数 / (TP × PP)

---

### Q: Transformer 与 CV 的结合（ViT）？

**ViT（Vision Transformer）核心设计**：

```
输入图像 [3, 224, 224]
    ↓ 切 patch（16×16）
196 个 patch → Linear Projection → 196 个 token embedding [196, 768]
    ↓ 加入 CLS token + Position Embedding
[197, 768] → Standard Transformer Encoder (12层) → CLS token 输出 → 分类头
```

**关键设计决策**：
- Patch Size：16×16（论文默认）。越小 token 越多，精度高但计算量 O(n²) 增长
- Position Embedding：可学习的 1D positional embedding（197个位置）
- CLS Token：全局聚合的特征，用于最终分类

**ViT 的优缺点**：
- **优点**：全局建模能力强（每个 patch 直接关注所有其他 patch）；大数据下超越 CNN；架构简洁统一
- **缺点**：缺乏 CNN 的局部归纳偏置（locality + translation equivariance）；小数据集上效果不如 CNN；计算量随分辨率平方增长

**后续改进**：
- **DeiT**：无需大数据，通过知识蒸馏从 CNN 教 ViT
- **Swin Transformer**：窗口注意力（局部性）+ Shifted Window（跨窗口信息交互）+ 层次化结构（类似 FPN）

---

### Q: 模型轻量化部署方式有哪些？

**七种主流轻量化方法及其权衡**：

| 方法 | 压缩比 | 精度损失 | 硬件友好度 | 开发成本 |
|---|---|---|---|---|
| INT8 量化 | 2x | 微小 | 高（Tensor Core） | 低 |
| INT4 量化 | 4x | 小 | 中（需 dequant） | 低 |
| 结构化剪枝 | 1.5-3x | 中 | 高（形状规则） | 中 |
| 知识蒸馏 | 自定义 | 取决于小模型 | 高 | 高（需训练） |
| 低秩分解 | 1.5-2x | 中 | 高 | 低 |
| 算子融合 | 1.5-3x加速 | 无 | 高 | 中 |
| 编译优化(TRT) | 2-5x加速 | 微小 | NVIDIA only | 低 |

**实际部署建议**：
- **追求极致速度**：INT4 量化 + TensorRT/vLLM + 算子融合
- **精度敏感**：INT8 (SmoothQuant W8A8) + 编译优化
- **端侧部署**：知识蒸馏小模型 + INT4 + llama.cpp
- **灵活性优先**：ONNX Runtime + 动态量化

---

### Q: TensorRT 的优缺点？

**优点（为什么性能最强）**：
- **图优化**：算子融合（Conv+BN+ReLU 合一）、常量折叠、死代码消除
- **Auto-Tuning**：对每个子图穷举所有可能的 kernel 实现，实际 benchmark 选最快的。这是 TRT 比手写 kernel 还快的原因——它的搜索空间包含 NVIDIA 内部未公开的高效 kernel
- **量化支持**：INT8/FP8 量化 + 自动校准
- **动态 shape**：通过 Optimization Profile 适配不同输入
- **极致推理性能**：CV 模型推理通常比 PyTorch 快 3-10x

**缺点（工程痛点）**：
- **仅支持 NVIDIA GPU**：完全不跨平台
- **算子覆盖不全**：新算子/自定义算子需要写 **Plugin**（C++ 实现 + 注册 + 序列化），开发成本高
- **模型转换复杂**：ONNX→TRT 转换中不支持的 op 需要逐个处理
- **Build Engine 耗时长**：Auto-Tuning 需要实际运行 benchmark（大模型可能 30 分钟+）
- **版本兼容性差**：不同 TRT 版本的 engine 不兼容，CUDA 版本也有耦合
- **调试困难**：engine 是黑盒，精度问题定位困难（需要逐层对比）

---

### Q: BatchNorm 的作用？

**BatchNorm 的计算**：
```
对 mini-batch B 的每个 feature channel:
  μ = mean(B)           # batch 均值
  σ² = var(B)           # batch 方差
  x̂ = (x - μ) / √(σ² + ε)  # 标准化
  y = γ × x̂ + β         # 可学习的缩放和偏移
```

**四大作用及原理**：

1. **加速收敛**：
   - 减少 Internal Covariate Shift（每层输入分布随训练变化）
   - 归一化后允许使用更大学习率（lr 可提高 10x），因为梯度不会因输入 scale 变化而失控

2. **正则化效果**：
   - Batch 内统计（μ、σ）引入了随机噪声（不同 batch 的统计量略有不同）
   - 类似 Dropout 的效果，轻微防止过拟合
   - 但 batch_size 越大效果越弱

3. **缓解梯度消失/爆炸**：
   - 保持每层输出在合理范围（零均值、单位方差）
   - 防止因层数加深导致的数值偏移累积

4. **降低对初始化的敏感性**：
   - 即使初始权重偏大/偏小，BN 也能将中间特征归一化
   - 使训练对初始化方案更鲁棒

**注意**：LLM 中不用 BatchNorm 而用 LayerNorm/RMSNorm，因为：
- 序列长度不固定，batch 维度语义不一致
- 推理时 batch_size=1，BN 的 running stats 不可靠
- LayerNorm 对每个 token 独立归一化，更适合生成任务

---

### Q: 手撕：到右下角的最短路径（简单 DP）？

（编程题）
