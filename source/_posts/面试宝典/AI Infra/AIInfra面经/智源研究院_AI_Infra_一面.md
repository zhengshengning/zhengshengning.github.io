---
title: '智源研究院 AI Infra 一面'
date: 2026-04-16 12:20:57
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 卷积算子如何高效实现？

卷积是CNN的核心算子，不同实现方式在不同场景下各有优势：

| 实现方式 | 原理 | 适用场景 | 优缺点 |
|---------|------|---------|--------|
| Im2col + GEMM | 将卷积展开为矩阵乘法 | 通用（cuBLAS加速） | 额外内存开销，但利用高度优化的GEMM |
| Direct Conv | 直接滑窗实现 | 小kernel、特殊shape | 灵活但难以优化到极致 |
| Winograd | 减少乘法次数 | 3×3 kernel | 快2-4倍，但数值精度稍差 |
| FFT | 频域卷积 | 大kernel（>7×7） | 大kernel快，小kernel开销大 |
| 隐式GEMM | 不显式im2col | 通用（cuDNN默认） | 省内存+高效 |

**Im2col + GEMM（最经典方法）**：

将输入按卷积窗口展开为2D矩阵，卷积变为矩阵乘法：
- 展开后输入矩阵：[output_h×output_w, C_in×kH×kW]
- 权重矩阵：[C_out, C_in×kH×kW]
- 输出 = 权重 × 展开输入^T
- 利用cuBLAS/Tensor Core的高度优化GEMM实现加速。
- 代价：展开后矩阵比原始输入大 kernel_h×kernel_w 倍（3×3 conv展开9倍）。

**Winograd（3×3卷积的王者）**：

原理：通过数学变换将3×3卷积的乘法次数从9减少到4（F(2,3)）或2.25（F(4,3)）次/输出元素。
- F(m,r)：在m×m输出tile上，用(m+r-1)×(m+r-1)的输入tile计算，减少乘法。
- 代价：需要额外的变换矩阵乘法和加法；r越大数值稳定性越差。
- 效果：3×3 conv配合Winograd在cuDNN中通常是最快的（比直接GEMM快30-50%）。
- 限制：不适合stride>1、dilation>1的情况。

**隐式GEMM（cuDNN的Implicit GEMM Precomputed）**：

不做显式的im2col（避免额外内存分配），而是在GEMM kernel内部通过索引计算隐式地从原始输入中读取数据，等效于im2col+GEMM但省去中间buffer。是cuDNN的默认高效实现。

**选择建议**：
- 3×3 conv + stride=1：Winograd（cuDNN自动选择）。
- 1×1 conv：退化为GEMM，直接用cuBLAS。
- 大kernel/任意参数：隐式GEMM。
- 最佳实践：让cuDNN auto-tune选择（`cudnnFindConvolutionForwardAlgorithm`）。

---

### Q: 端侧大模型部署的关键技术？

端侧（手机/车载/IoT）部署大模型面临严格的**内存、功耗、延迟**三重约束，需要综合多种技术：

**1. 量化（最核心的技术）**：
- INT4/INT8量化将模型体积压缩4-8倍：7B模型从14GB(FP16)→3.5GB(INT4)，可放入手机RAM。
- 方法：GPTQ/AWQ离线量化权重为4bit，推理时反量化后与FP16激活计算。
- 精度影响：4bit量化通常perplexity增加0.5-2%，对大多数应用可接受。

**2. 模型裁剪/压缩**：
- 层剪枝：去掉不重要的Transformer层（如LLM-Pruner）。
- 注意力头剪枝：减少attention头数（如从32头减到16头）。
- 宽度缩减：减小hidden_size和intermediate_size。
- 结构化蒸馏：用大模型蒸馏到小模型。

**3. KV Cache优化（端侧内存限制的关键）**：
- GQA/MQA：减少KV头数，如Llama-3用GQA（8个KV头 vs 32个Q头），KV Cache减少4倍。
- KV量化：INT8存储KV Cache。
- 滑动窗口注意力：只保留最近N个token的KV（如Mistral的4096窗口）。
- Token剪枝：丢弃注意力权重低的历史token的KV。

**4. 轻量模型选择**：
- 1-3B参数的小模型：Qwen-1.5B、Phi-3-mini(3.8B)、Gemma-2B。
- 这些模型专为端侧设计，在有限参数量下最大化能力。

**5. 推理引擎优化**：
- **llama.cpp**：C/C++实现，支持GGUF格式、2-8bit量化、ARM NEON/Apple Metal加速。iPhone上跑7B模型约5-10 tokens/s。
- **MLC-LLM**：Apache TVM编译优化，支持多端（Android/iOS/Web）。
- **MediaPipe LLM**：Google的端侧推理方案。

**6. 投机解码（端侧适配版）**：
- 用参数极少的draft model（如100M参数）配合主模型，加速1.5-2倍。
- 或使用MTP头（主模型自带的额外预测头）。

**7. 内存管理**：
- mmap按需加载：不一次性将所有权重加载到内存，使用mmap让OS按需page in。减少峰值内存占用。
- Layer-wise加载：每次只加载当前计算需要的层的权重。适合RAM极有限的设备。

**端侧部署典型数据**：
- 7B模型INT4量化：模型约3.5GB，推理需4-6GB RAM，iPhone 15 Pro约8-12 tokens/s。
- 1.5B模型INT4：模型约1GB，推理需2GB RAM，中端手机约20-30 tokens/s。

---

### Q: 手撕：树形DP？

（编程题）
