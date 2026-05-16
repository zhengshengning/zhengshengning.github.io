---
title: '格灵深瞳 AI Infra 一面'
date: 2026-04-16 12:16:03
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 推理优化, 训练优化, 面经]
---


<!-- more -->

---

### Q: 给一个新模型，要求在特定硬件上做推理优化，应该怎么下手？

系统化推理优化方法论（从全局到局部）：

**Phase 1: Profiling 与瓶颈分析**
1. 先跑 baseline benchmark：测量端到端延迟、各 kernel 耗时分布
2. 用硬件 profiler（Nsight Compute/Systems）分析热点 kernel
3. 确定瓶颈类型：compute-bound（TFLOPS 接近峰值）/ memory-bound（带宽接近峰值）/ launch-bound（kernel 数过多）

**Phase 2: 模型级分析**
4. 统计各层 FLOPs、参数量、访存量（arithmetic intensity）
5. 确定热点算子（通常 GEMM/Attention 占 80%+ 时间）
6. 分析模型结构特点：是否可量化？是否有可稀疏化的层？

**Phase 3: 图级优化**
7. 算子融合：将 element-wise 链（如 LayerNorm + Linear）融合为单 kernel
8. 常量折叠：编译期计算不变量
9. 布局转换：选择适配硬件的内存布局（如 NHWC vs NCHW）

**Phase 4: 量化评估**
10. 尝试 INT8/FP8 量化，评估精度影响
11. 逐层敏感度分析，敏感层保持高精度
12. 选择合适的量化方案（PTQ 快速上线 / QAT 精度最优）

**Phase 5: 算子级优化**
13. 对热点算子做定制化实现：tiling 适配 cache/shared memory
14. 向量化加载、利用硬件特殊单元（Tensor Core/矩阵引擎）
15. Auto-tuning 搜索最优参数配置

**Phase 6: 系统级优化**
16. Batch 调度策略、流水线并行、异步计算与通信重叠

---

### Q: 推理优化领域有哪些业界还没人做的方向？

（开放性问题，展示技术视野和思考深度）

**前沿方向分析**：

1. **动态稀疏 Attention**：根据输入内容自适应决定关注哪些位置，而非固定的稀疏模式（如 Longformer 的固定窗口）。挑战在于动态稀疏模式下的 kernel 实现效率（不规则访存）。

2. **异构协同推理**：CPU+GPU+NPU 联合调度，不同算子分发到最适合的硬件执行。当前各硬件各自为政，缺乏统一的动态调度框架和高效的跨设备数据传输。

3. **端侧大模型的自适应精度**：根据输入复杂度动态调整推理精度——简单问题用 W2/W4 快速响应，复杂问题用 W8/FP16 保证质量。类似 early exit 的思路但在精度维度。

4. **长上下文 KV Cache 压缩与检索**：100K+ 上下文中大部分 KV 对当前 token 无关，如何实现"注意力检索"——只加载相关的 KV 子集？StreamingLLM/H2O 是初步探索。

5. **编译器自动发现结构化稀疏性**：让编译器自动分析模型权重/激活的稀疏模式（如 N:M 稀疏、block 稀疏），生成对应的高效 kernel，无需人工指定。
