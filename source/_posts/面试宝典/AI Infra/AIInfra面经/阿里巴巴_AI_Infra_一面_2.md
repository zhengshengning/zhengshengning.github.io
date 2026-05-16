---
title: '阿里巴巴 AI Infra 一面 (2)'
date: 2026-04-16 11:44:14
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Qwen-VL的训练流程是怎样的？

Qwen-VL训练分为三个阶段：
1. **视觉编码器预训练**：ViT在大规模图文对数据上预训练（如CLIP）。
2. **跨模态对齐**：冻结LLM，训练视觉编码器和投影层（将视觉特征映射到LLM的embedding空间）。
3. **联合微调**：解冻全部参数，在指令数据上进行多模态SFT。

---

### Q: 多模态大模型的视觉特征传递给LLM有哪些方法？

- **线性投影**：直接用线性层映射视觉特征到LLM维度（LLaVA）。
- **Q-Former/Perceiver**：用可学习query通过交叉注意力从视觉特征中提取固定数量token（BLIP-2）。
- **MLP Adapter**：多层MLP做映射（LLaVA-1.5）。
- **Cross Attention**：在LLM层间插入交叉注意力层处理视觉特征（Flamingo）。
- **直接拼接**：将视觉token与文本token拼接作为LLM输入。

---

### Q: ViT一般怎么预训练？

- **对比学习**：CLIP方式，图文对的contrastive loss。
- **MAE（Masked Autoencoder）**：随机mask 75%的patch，重建被mask的像素。
- **DINO/DINOv2**：自蒸馏，student和teacher网络的特征一致性学习。
- **监督预训练**：在ImageNet等分类数据上做有监督训练。

---

### Q: 多模态RAG介绍？

多模态RAG将检索扩展到图文混合场景：
- **索引**：对文档中的文本和图片分别编码（或联合编码），存入向量库。
- **检索**：支持以文本/图片为query检索相关的文本/图片（跨模态检索）。
- **生成**：将检索到的多模态内容作为上下文输入多模态LLM生成答案。

挑战：跨模态对齐质量、图片内容理解精度、多模态文档的分块策略。

---

### Q: 对于不同形状的图片或视频，位置编码怎么设计？

- **2D RoPE**：对图片的行列位置分别做旋转位置编码，支持任意分辨率。
- **动态分辨率**：将图片切分为固定大小patch（如14x14像素），patch数量随分辨率变化，位置编码按实际patch位置分配。
- **插值**：将预训练的固定位置编码做双线性插值到目标分辨率。
- **视频**：在2D空间位置编码基础上增加时间维度编码（3D位置）。
- **NTK-Aware RoPE**：通过调整RoPE的base频率支持更长序列。

---

### Q: 残差连接（Residual Connection）的作用？

- **缓解梯度消失**：为梯度提供"短路"通道，梯度可直接回传到浅层。
- **恒等映射**：网络只需学习残差F(x)=H(x)-x，学习难度降低。
- **信息保留**：原始信息不经变换直接传递到深层，避免信息退化。
- **训练稳定性**：使得训练极深网络（如100+层ResNet、GPT）成为可能。

在Transformer中，每个attention和FFN子层后都有残差连接：`output = LayerNorm(x + Sublayer(x))`。

---

### Q: 大模型训练和推理时显存不够，有哪些优化方法？

**训练**：
- 混合精度（FP16/BF16）减少一半显存。
- ZeRO Stage 1/2/3切分优化器/梯度/参数。
- Gradient Checkpointing以时间换空间。
- Offload到CPU/NVMe。
- 梯度累积减少活跃batch显存。

**推理**：
- 模型量化（INT8/INT4）。
- PagedAttention管理KV Cache。
- KV Cache量化/压缩。
- 模型并行（TP）分散到多卡。
- Flash Attention减少中间显存。
- Token pruning/early exit减少计算。

---

### Q: 手撕：数组中的第K个最大元素？

（编程题）
