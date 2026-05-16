---
title: '百度 AI Infra 校招 二面 (1)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: RT-DETR相比DETR做了哪些优化？

DETR（DEtection TRansformer）虽然实现了端到端检测（无需 NMS），但速度慢、收敛慢。RT-DETR（百度提出）通过以下优化实现实时检测：

1. **高效 Encoder**：用多尺度特征融合（类似 FPN + AIFI 跨尺度注意力）替代 DETR 的 6 层 Transformer Encoder 全注意力。将 Encoder 计算量降低数倍的同时保持多尺度特征提取能力。

2. **IoU-aware Query Selection**：不再随机初始化 decoder query，而是从 encoder 输出中根据 IoU 质量评分选择 top-K 特征作为 query。这大幅加速了收敛（从 500 epoch 降到 ~72 epoch）。

3. **去除 NMS 后处理**：继承 DETR 的端到端检测优势，推理 pipeline 更简洁，在 GPU 上避免了 NMS 的串行瓶颈。

4. **灵活的速度-精度 tradeoff**：通过调整 Decoder 层数（如 3 层 vs 6 层）实现不同速度档位，适配不同部署需求。

**效果**：RT-DETR-L 在 COCO 上达到 53.0% AP，T4 GPU 上 114 FPS，首次实现了 DETR 类模型的实时推理。

---

### Q: 大模型推理加速的方法有哪些？

大模型推理加速是一个系统性工程，从模型层面到系统层面都有优化空间：

**模型/算法层面**：
1. **量化**：W8A8（2x 加速）、W4A16（4x 模型压缩，Tensor Core INT4 加速）。GPTQ/AWQ/SmoothQuant 等方法在精度损失 <1% 的前提下显著提速。
2. **投机解码（Speculative Decoding）**：用小模型（如 68M 参数）快速生成 K 个草稿 token，大模型一次性验证。验证通过率高时可获得 2-3x 加速，且输出分布与大模型完全一致。
3. **稀疏化**：结构化剪枝（去除冗余 head/FFN 神经元）、MoE（只激活部分专家）。

**系统/工程层面**：
4. **KV Cache**：缓存历史 Key/Value 避免重复计算，是自回归推理的基础优化。
5. **FlashAttention/Flash Decoding**：IO 感知的 attention 实现，内存效率和速度双提升。
6. **连续批处理（Continuous Batching）**：不同请求动态组 batch（vLLM/TGI），避免短请求等待长请求。
7. **PagedAttention**：分页管理 KV Cache，消除内存碎片，提升服务吞吐。
8. **模型并行**：TP（减少单卡显存压力）+ PP（多卡流水线）。
9. **算子融合**：将多个小 kernel 合并为一个，减少 launch 开销和中间内存读写。
10. **CUDA Graph**：预录制 kernel 执行图，消除反复 launch 的 CPU 开销。

---

### Q: Attention算子加速方法？

Attention 是 Transformer 的核心瓶颈，占推理延迟的 30-60%。加速方法：

1. **FlashAttention**：分块计算 + online softmax，避免 N^2 attention 矩阵写 HBM。Prefill 阶段标准选择。
2. **Flash Decoding**：沿 KV 序列维度并行切分，解决 Decode 阶段 Q 只有 1 行时并行度不足的问题。长序列 decode 提速 3-8x。
3. **MQA/GQA**：减少 KV head 数量，直接降低 KV Cache 读取的带宽需求和显存占用。GQA 在质量和效率间取得良好平衡。
4. **稀疏 Attention**：只计算部分位置的注意力（如 Longformer 的局部 + 全局 attention、BigBird、StreamingLLM 的 sink attention）。适合超长序列但可能损失质量。
5. **线性 Attention**：用核函数近似 softmax（如 elu+1），将复杂度从 O(N^2) 降到 O(N)。但精度通常不如标准 attention（目前非主流）。
6. **KV Cache 压缩**：量化 KV Cache（如 KV8）、Head-wise 压缩、Token 淘汰（H2O）等降低 Cache 带宽需求。

---

### Q: 手撕：字符串相乘（LeetCode 43）？

（编程题）

---

### Q: 手撕：验证栈序列（LeetCode 946）？

（编程题）
