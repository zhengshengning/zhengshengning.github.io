---
title: '小米 AI Infra 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 手撕：最长回文子串？

（编程题）

---

### Q: YOLO模型如何进行剪枝？

YOLO剪枝的目标是在保持检测精度的前提下减小模型体积和加速推理，核心是**结构化剪枝**（移除整个channel/filter，而非单个权重）。

**主流方法：基于BN层gamma的结构化剪枝**

原理：BatchNorm层的可学习缩放因子gamma反映了对应channel的重要性——gamma越小说明该channel对输出的贡献越小，可以安全移除。

**完整流程**：

1. **正常训练**：先训练一个baseline模型达到目标精度。
2. **稀疏训练**：对BN层的gamma参数施加L1正则（`loss += lambda × sum(|gamma|)`），迫使不重要channel的gamma趋近于0。lambda通常取1e-4~1e-3。
3. **确定剪枝阈值**：统计所有BN层gamma的分布，选择合适的阈值（如全局排序取后30-70%）。
4. **执行剪枝**：移除gamma小于阈值的channel——包括对应的卷积核、BN参数、以及相关层的输入通道。
5. **微调恢复**：在剪枝后的模型上fine-tune 10-30个epoch恢复精度。

**关键注意事项**：
- **跳跃连接约束**：YOLO中有残差连接/concat操作的位置，相连层的channel数必须一致或可整除，剪枝时需保持对齐。
- **检测头适配**：剪枝后backbone的输出channel变化，检测头的输入channel需对应调整。
- **分层剪枝比例**：不同层的敏感度不同，前几层和检测头附近层通常更敏感，应少剪或不剪。
- **渐进式剪枝**：多轮"剪枝-微调"比一次性大幅剪枝效果更好。

**典型效果**：YOLOv5s剪枝50% channel后，FLOPs减少约60%，mAP@0.5下降1-2%，推理速度提升40-60%。

---

### Q: 模型部署加速有哪些技术？

模型部署加速是一个多层次优化的系统工程，涵盖模型压缩、图优化、硬件利用三个维度：

**1. 量化（最直接有效）**：
- FP32→FP16：2倍压缩，Tensor Core加速，精度几乎无损。
- FP16→INT8：再2倍压缩，利用INT8 Tensor Core，需校准。
- INT4/NF4：4倍压缩，仅量化权重（W4A16），大模型推理主流。
- 实际加速比：A100上FP16→INT8加速约1.8-2.2倍（GEMM），Decode阶段加速更明显（带宽受限）。

**2. 算子融合**：
- 典型模式：Conv+BN+ReLU融合（BN参数合并到Conv中，消除独立BN kernel）。
- LLM中：QKV投影融合（3个GEMM→1个大GEMM）、LayerNorm+残差add融合。
- 效果：减少kernel launch开销（~5us/次）、消除中间tensor的HBM读写。

**3. 图优化（编译器级别）**：
- 常量折叠：预计算静态子图。
- 死代码消除：移除无用节点。
- Layout优化：选择硬件最优数据格式（如NHWC for Tensor Core）。
- 形状推导和静态化：尽量消除动态shape的运行时判断。

**4. 推理引擎**：
- TensorRT：NVIDIA GPU专用，自动量化+融合+代码生成，通常比PyTorch快3-5倍。
- ONNX Runtime：跨平台，支持多种EP（CUDA/TensorRT/OpenVINO）。
- TVM/Apache TVM：编译器方法，自动搜索最优schedule。
- vLLM/TensorRT-LLM：LLM专用推理引擎。

**5. 知识蒸馏**：用大模型（Teacher）指导小模型（Student）训练，获得接近Teacher的精度但Student的速度。

**6. 剪枝**：结构化剪枝移除冗余channel/head，非结构化剪枝（稀疏化）需硬件支持（如A100的结构化稀疏2:4）。

**7. 硬件适配**：Tensor Core利用（要求维度对齐到8/16）、内存对齐（128字节）、多Stream并行。

---

### Q: 知识蒸馏的原理？

知识蒸馏（Knowledge Distillation）的核心思想是让小模型（Student）学习大模型（Teacher）的**软标签分布**（soft labels/dark knowledge），而非仅学习硬标签（one-hot labels）。

**为什么软标签更有价值**：
- 硬标签只告诉模型"正确答案是猫"。
- 软标签告诉模型"这是猫（概率0.7），但它也有点像老虎（0.2）和豹子（0.1）"——这种类间相似度信息是额外的知识。
- Teacher的logits中编码了数据分布的结构信息，Student通过模仿这种分布学到更好的泛化能力。

**训练Loss公式**：
```
Loss = alpha × KD_Loss + (1 - alpha) × CE_Loss

其中:
- KD_Loss = KL_divergence(Student_soft || Teacher_soft)
- CE_Loss = CrossEntropy(Student_hard, ground_truth)
- Student_soft = softmax(student_logits / T)
- Teacher_soft = softmax(teacher_logits / T)
```

**温度T的作用**：
- T=1：标准softmax，Teacher的概率分布尖锐（高置信度类dominate）。
- T>1：分布变平滑，低概率类的相对关系被放大，暗知识更明显。
- T越大：暗知识信号越强，但过大会导致分布过于均匀失去区分度。
- 典型值T=4-20，T=1退化为普通标签学习。

**蒸馏变体**：
- **Feature Distillation**：Student模仿Teacher中间层特征（如FitNets）。
- **Self-Distillation**：模型自身作为Teacher蒸馏（如DeiT中BERT作Teacher）。
- **Online Distillation**：Teacher和Student同时训练，互相学习。
- **LLM蒸馏**：用GPT-4生成高质量response作为训练数据（实际是一种数据蒸馏）。

**实际效果**：典型场景下Student可达到Teacher 95-99%的精度，但参数量仅为1/4-1/10。

---

### Q: 模型量化的方法和原理？

量化将浮点参数/激活映射为低位整数表示，是模型部署加速中**投入产出比最高**的技术。

**量化原理（线性映射）**：
```
对称量化: q = round(clamp(x / scale, -Qmax, Qmax))
         scale = max(|x|) / Qmax

非对称量化: q = round(clamp(x / scale + zero_point, 0, Qmax))
           scale = (max(x) - min(x)) / (Qmax - Qmin)
```

**PTQ（训练后量化）—— 快速但精度可能受损**：

| 方法 | 原理 | 适用 |
|------|------|------|
| MinMax | 直接用min/max确定range | 简单场景 |
| Percentile | 截断outlier（如99.99分位） | 有离群值时 |
| KL散度 | 搜索使分布差异最小的阈值 | TensorRT默认 |
| GPTQ | 逐列量化+Hessian指导误差补偿 | LLM 4bit量化 |
| AWQ | 保护对激活影响大的salient通道 | LLM 4bit量化 |
| SmoothQuant | 将量化难度从激活迁移到权重 | W8A8推理 |

**QAT（量化感知训练）—— 精度最优但需重训**：
- 训练时在前向中插入伪量化节点（fake quantize），模拟量化误差。
- 反向传播时用STE（Straight-Through Estimator）让梯度穿过不可微的round操作。
- 模型在训练中学会适应量化噪声，最终精度接近浮点模型。

**常见量化方案及效果**：

| 方案 | 权重精度 | 激活精度 | 模型压缩比 | 精度损失 |
|------|---------|---------|-----------|---------|
| W16A16 | FP16 | FP16 | 2x | 几乎无 |
| W8A8 | INT8 | INT8 | 4x | <1% |
| W4A16 | INT4 | FP16 | 8x | 1-3% |
| W4A8 | INT4 | INT8 | 8x | 2-5% |

**量化粒度对比**：per-tensor（最粗，开销最小，精度最差）< per-channel（权重标准做法）< per-group（G=128，大模型4bit标配，精度最好但每组需额外存scale）。
