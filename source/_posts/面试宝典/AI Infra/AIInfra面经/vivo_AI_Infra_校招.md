---
title: 'vivo AI Infra 校招'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: TVM中图优化和算子优化分别做什么？设备如何提供支持？

**图优化（Relay级 / Graph-level）**：

操作对象是**计算图中的算子节点和边**，关注算子间的关系：

| 优化类型 | 说明 | 效果 |
|---------|------|------|
| 算子融合（Fusion） | 将多个小算子合并为一个kernel（如Conv+BN+ReLU） | 消除中间tensor的HBM读写，减少kernel launch开销 |
| 常量折叠（Constant Folding） | 编译期计算常量表达式 | 减少运行时计算 |
| 死代码消除 | 移除不影响输出的计算 | 减少无效计算 |
| 布局转换（Layout Transform） | NCHW→NHWC或其他设备友好的数据布局 | 匹配硬件最优访存模式 |
| 形状推断与静态化 | 将动态shape尽可能静态化 | 使后续优化可应用 |

**设备对图优化的支持**：通过`target`描述提供硬件信息（如"cuda -arch=sm_80"），告知编译器可用的融合规则（如Tensor Core要求的shape约束）、最优数据布局等。

**算子优化（TIR级 / Operator-level）**：

操作对象是**单个算子内部的循环结构和内存访问**：

| 调度原语 | 说明 | 目标 |
|---------|------|------|
| split/tile | 循环分块 | 适配缓存层次（寄存器→SMEM→L2→HBM） |
| reorder | 改变循环嵌套顺序 | 提高数据局部性 |
| vectorize | 向量化内层循环 | 利用SIMD/向量加载指令 |
| unroll | 循环展开 | 减少循环开销，暴露ILP |
| parallel | 标记可并行维度 | 映射到多核/多线程 |
| bind | 将循环绑定到GPU线程层次 | blockIdx/threadIdx映射 |
| cache_read/write | 数据缓存到更快层次 | 利用共享内存/寄存器 |
| compute_at | 控制计算嵌入位置 | 减少中间buffer大小 |

**设备对算子优化的支持**：通过硬件参数指导调度决策——SM数量（决定grid大小）、每SM共享内存大小（48KB-164KB，决定tile大小）、寄存器数量（64K/SM，决定每线程的数据重用量）、向量宽度（float4=16B）、Warp大小（32）等。

**两级优化的协作关系**：
```
模型 → [Relay图优化: 融合+布局] → 优化后的子图 → [TIR算子优化: tiling+调度] → 高性能kernel → 目标代码
```
图优化减少算子数量和中间数据，算子优化使每个算子内部执行效率最大化。

### Q: TVM的Auto-Tuning方案设计是什么？与手动优化的区别和优劣？

**Auto-Tuning的完整流程**：

```
1. 定义搜索空间（Schedule Template）
   - tile sizes: [1,2,4,8,16,32,64,128]的组合
   - unroll factors: [1,2,4,8,16]
   - 并行策略: [parallel, vectorize, bind_thread]
   - 循环顺序的所有排列
   → 搜索空间通常有10⁶~10⁹个配置

2. 搜索策略
   - AutoTVM: XGBoost代价模型预测性能 + 模拟退火/GA探索
   - Ansor (Auto-Scheduler): 自动生成搜索空间 + 进化搜索 + 基于sketch的程序采样
   - Meta-Schedule (TVM Unity): 可组合的调度规则 + 代价模型

3. 硬件实测
   - 将候选配置编译为可执行kernel
   - 在目标硬件上实际运行测量性能（避免代价模型误差）
   - 测量结果反馈给代价模型更新

4. 最优配置选择
   - 经过数百~数千次试验后选择最优配置
   - 保存调优日志（tuning log），后续可复用
```

**与手动优化的对比**：

| 维度 | Auto-Tuning | 手动优化（CUDA专家） |
|------|-------------|-------------------|
| 适配性 | 自动适配不同shape/硬件 | 特定shape/硬件极致优化 |
| 开发效率 | 定义搜索空间后自动搜索 | 需要深厚的硬件知识 |
| 性能上限 | 受限于搜索空间定义 | 可使用任意优化技巧 |
| 搜索时间 | 数小时~数天 | 人工数天~数周 |
| 可迁移性 | 换硬件只需重新tuning | 需要重新适配 |
| 可维护性 | 代码自动生成，修改困难 | 代码可读可维护 |
| 复杂优化 | Warp Specialization等难以表达 | 无限制 |

**AutoTVM vs Ansor vs 手动优化的性能对比（以GEMM为例）**：
- cuBLAS (NVIDIA手动优化): 100% 性能基线
- Ansor: 通常达到cuBLAS的80-95%
- AutoTVM (需要手写template): 达到70-90%
- 未优化的naive实现: 5-15%

**适用场景建议**：
- **Auto-Tuning最适合**：大量不同shape的算子、非NVIDIA硬件（无高质量库）、快速原型验证
- **手动优化最适合**：核心热点算子（如FlashAttention）、需要极致性能、复杂的优化模式（如persistent kernel、warp specialization）
- **最佳实践**：对标准算子用Auto-Tuning快速达到80%性能，对关键算子手动优化追求最后的20%

### Q: 其他编译框架有哪些？对比TVM有何不同？

**主流编译框架全景对比**：

**XLA（Accelerated Linear Algebra, Google）**：
- **定位**：面向TPU/GPU的图编译器，与TensorFlow/JAX深度绑定
- **核心特点**：
  - 自动算子融合能力强（HLO IR上的fusion pass）
  - 对TPU优化极致（毕竟同源于Google）
  - 支持自动分布式并行（GSPMD）
- **vs TVM**：XLA是封闭生态（TF/JAX only），TVM支持任意前端；XLA对NVIDIA GPU的优化不如TVM/Triton灵活
- **适用场景**：Google Cloud TPU上的训练/推理

**Triton（OpenAI）**：
- **定位**：GPU kernel编程DSL，介于CUDA和Python之间的抽象层
- **核心特点**：
  - 用Python-like语法写kernel，自动处理tiling、内存管理、并行映射
  - 程序员只需指定tile大小和内存访问模式，编译器自动生成高效CUDA代码
  - 编译到LLVM IR再到PTX
- **vs TVM**：Triton是kernel-level DSL（手写单个kernel），TVM是端到端编译器（从模型到部署）。Triton给程序员更多控制权但需要理解GPU编程概念
- **性能**：手写Triton kernel可达cuBLAS 90-100%性能，开发效率是CUDA的5-10倍
- **适用场景**：快速开发自定义kernel（如FlashAttention最初用Triton实现原型）

**MLIR（Multi-Level Intermediate Representation, Google）**：
- **定位**：通用的编译器基础设施框架，提供可扩展的方言（Dialect）机制
- **核心特点**：
  - 多层IR支持渐进式lowering：High-level Dialect → Mid-level → Low-level → LLVM
  - 可定义领域特定的Dialect（如Linalg for 线性代数、GPU for GPU操作）
  - 强大的pattern rewrite和pass infrastructure
- **vs TVM**：MLIR是构建编译器的**框架的框架**，TVM是完整的端到端DL编译器。用MLIR可以从零构建类似TVM的编译器，但需要大量开发工作
- **适用场景**：构建新的领域编译器、硬件厂商的编译栈

**TensorRT（NVIDIA）**：
- **定位**：闭源推理优化器，对NVIDIA GPU极致优化
- **核心特点**：
  - 自动选择最优kernel实现（从预编译的kernel库中选择）
  - 支持INT8量化（需要校准）、FP16、结构化稀疏
  - Layer fusion、kernel auto-tuning
  - 支持动态shape（dynamic shape profile）
- **vs TVM**：TensorRT对NVIDIA硬件优化通常优于TVM（闭源手调kernel），但完全不跨平台。TVM开源可扩展但NVIDIA上性能稍逊
- **典型加速**：对推理场景，TensorRT通常比PyTorch eager快3-10倍
- **适用场景**：NVIDIA GPU上的生产推理部署

**Halide**：
- **定位**：计算与调度分离的先驱（2012年MIT）
- **核心贡献**：提出 Algorithm（计算逻辑）和 Schedule（执行策略）分离的编程模型——TVM的Schedule设计直接受其启发
- **vs TVM**：Halide主要面向图像处理pipeline（2D stencil），TVM扩展到通用张量计算和深度学习
- **适用场景**：图像处理、计算摄影

**框架选择决策树**：
```
需要部署推理？
├── NVIDIA GPU only → TensorRT（最优性能）
├── 多平台/边缘设备 → TVM / Apache TVM Unity
└── 自定义kernel需求
    ├── 快速开发 → Triton
    └── 极致性能 → CUDA + cutlass

需要训练编译？
├── JAX/TPU生态 → XLA
├── PyTorch生态 → torch.compile (Inductor/Triton backend)
└── 自研框架 → MLIR自建

需要新硬件支持？
└── MLIR构建定制编译栈
```
