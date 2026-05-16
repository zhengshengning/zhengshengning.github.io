---
title: 'Teleai AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: vLLM和SGLang有什么区别？SGLang相对于vLLM有什么优势？

**vLLM**：以PagedAttention为核心，虚拟内存管理KV Cache减少碎片，Continuous Batching动态调度。

**SGLang**：在vLLM基础上的优化：
- **RadixAttention**：前缀共享的KV Cache管理，相同前缀的请求自动复用KV Cache。
- **更高效的调度**：基于Radix Tree的前缀感知调度策略。
- **编程接口**：提供结构化生成的高效原语（fork/join）。
- **推理模型更适配**：对需要长上下文和多轮推理的场景（如DeepSeek-R1）更优，因前缀复用收益大。

---

### Q: 大模型量化算法有哪些？有没有实际部署经验？

**主要算法**：
- **GPTQ**：逐层权重量化，用Hessian矩阵信息补偿量化误差。
- **AWQ**：激活感知权重量化，保护对输出影响大的权重通道。
- **SmoothQuant**：将激活的量化难度迁移到权重上（乘以平滑因子）。
- **AutoAWQ**：AWQ的自动化实现，支持4-bit量化打包。

部署时关注：量化前后精度对比（perplexity/任务指标）、推理速度提升比例、显存节省量。

---

### Q: CLIP的原理和推理流程？

**原理**：对比学习框架，同时训练图像编码器（ViT/ResNet）和文本编码器（Transformer），使匹配的图文对在嵌入空间中距离最近（InfoNCE loss）。

**推理流程**：输入图片通过图像编码器得到图像embedding → 输入文本通过文本编码器得到文本embedding → 计算cosine similarity → 用于零样本分类/检索。

---

### Q: 常见大模型算子优化思路有哪些？

**访存优化**：
- 合并访存（coalesced access）、向量化加载（float4）。
- 使用共享内存减少全局内存访问次数。
- 算子融合减少中间结果写回全局内存。

**并行优化**：
- 提高occupancy（调整block size、减少寄存器）。
- 利用Tensor Core加速矩阵运算。
- Warp-level原语（shuffle）减少共享内存依赖。

---

### Q: Ascend CANN MindIE框架有哪些组成？

CANN（Compute Architecture for Neural Networks）是华为昇腾AI处理器的计算架构：
- **AscendCL**：底层算子开发接口（类似CUDA Runtime API）。
- **CANN算子库**：预实现的高性能算子。
- **图编译器**：将计算图优化编译为昇腾硬件指令。
- **MindIE**：推理引擎，包含模型转换、量化、调度等功能。支持大模型高效推理（类似TensorRT-LLM for Ascend）。

---

### Q: V100显存多少？DeepSeek/Qwen 32B INT8量化能部署吗？怎么部署？

V100有16GB和32GB两个版本。32B模型INT8量化后约32GB（32B * 1字节），单张V100(32GB)勉强放下参数但无KV Cache空间。

**部署方案**：使用2-4张V100(32GB)做张量并行；或使用A100/H100等更大显存卡。部署工具：vLLM + AutoAWQ/GPTQ量化，设置`--tensor-parallel-size`配置多卡。

---

### Q: 并发场景下怎么测试最大并发数？需要关注哪些指标？

**测试方法**：逐步增加并发请求数，直到系统指标达到瓶颈。

**关注指标**：
- **TTFT（Time To First Token）**：首token延迟，反映用户等待。
- **TPOT（Time Per Output Token）**：每token生成延迟。
- **Throughput（tokens/s）**：系统总吞吐。
- **P50/P90/P99延迟**：关注长尾。
- **显存使用率**：是否OOM。
- **请求排队长度**：调度器压力。

---

### Q: vLLM怎么去支持自研模型？

1. 在`vllm/model_executor/models/`下新增模型文件，实现模型前向逻辑。
2. 继承vLLM的模型基类，实现`forward()`和权重加载方法。
3. 在模型注册表中注册新模型。
4. 确保attention层使用vLLM的PagedAttention实现。
5. 处理好权重名称映射（HuggingFace权重到vLLM格式）。
6. 如有自定义算子需以插件形式集成。
