---
title: '燧原科技 AI Infra 社招 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: MLIR的表示和优化是怎么做的？

MLIR（Multi-Level Intermediate Representation）是LLVM子项目，一个**可扩展的多层编译器基础设施**。核心创新是通过Dialect机制支持在任意抽象层级定义和优化IR。

**核心概念：**

**1. Dialect（方言）——可扩展的IR层级：**
```
每个Dialect定义一组Operation（算子）、Type（类型）、Attribute（属性）
```
常用Dialect层级（从高到低）：

| Dialect | 抽象级别 | 包含的操作 | 优化重点 |
|---------|---------|-----------|---------|
| linalg | 高层线性代数 | matmul, conv, generic | 分块、融合策略 |
| tensor/memref | 张量/内存引用 | 张量操作、内存分配 | 内存规划 |
| affine | 仿射循环 | affine.for, affine.load | 循环变换(tile/fuse/unroll) |
| scf | 结构化控制流 | scf.for, scf.if | 标准循环优化 |
| gpu | GPU映射 | gpu.launch, gpu.thread_id | 线程映射 |
| llvm | LLVM IR | LLVM指令集 | 标准LLVM优化 |

**2. 渐进式Lowering（Progressive Lowering）：**
```
linalg.matmul → (tiling) → affine.for loops → (mapping) → gpu.launch + scf.for
    → (lowering) → llvm dialect → LLVM IR → PTX/SASS
```
每一步lowering只做有限的变换，保留当前层有用的信息。相比传统编译器一次性从高层降到低层，MLIR可以在每层做更精确的优化。

**3. Pass机制：**
```mlir
// Pass pipeline示例
func.func @matmul(%A: tensor<256x512xf32>, %B: tensor<512x256xf32>)
                                            → tensor<256x256xf32> {
  // linalg层面：决定分块策略
  // → tile(64, 64, 32)
  // affine层面：决定循环顺序和并行
  // → interchange(0, 1, 2)
  // gpu层面：映射到CUDA线程
  // → map(blockIdx.x, blockIdx.y, threadIdx.x)
}
```

**MLIR相比传统编译器的优势：**
- **信息保留**：高层语义（如"这是一个矩阵乘"）一直保留到需要具体实现的层级
- **统一基础设施**：不同框架(TF/PyTorch/TVM)的IR可以统一到MLIR生态
- **可组合优化**：不同Dialect的Pass可以自由组合，形成定制化的编译流水线
- **硬件适配**：为不同硬件定义专用Dialect（如NVGPU dialect for Tensor Core）

### Q: XLA动态性不好如何解决？

XLA（Accelerated Linear Algebra）是Google的ML编译器，基于**静态shape**编译优化。核心问题：遇到不同shape的输入需要重新编译（compilation cache miss），导致严重的性能开销。

**动态shape带来的问题：**
- 不同序列长度 → 不同shape → 触发重编译（首次可能需要数十秒）
- 动态batch size → 每种batch size需要独立编译
- 控制流依赖数据 → `while_loop`的trip count不确定

**解决方案对比：**

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| Padding固定shape | 所有输入pad到同一大小 | 只需编译一次 | 浪费计算，短序列效率低 |
| Bucketing(分桶) | 将shape聚类为几个桶(如128/256/512/1024) | 编译次数有限 | 仍有padding浪费，桶内最长序列决定计算量 |
| Bounded Dynamic Shape | 指定shape的上界，编译时使用上界shape | 支持动态但有上界 | 仍可能浪费 |
| Dynamic shapes (StableHLO) | IR支持符号shape维度 | 真正的动态 | 编译器优化受限 |
| torch.compile dynamic=True | PyTorch编译器的动态shape模式 | 灵活 | 图断裂多，部分优化无法做 |

**实际工程中的最佳实践：**
- 训练：Padding + Bucketing组合（训练时shape变化不大）
- 推理Prefill：Bucketing（预编译几种常见长度）
- 推理Decode：shape固定（batch×1），非常适合XLA/CUDA Graph
- 对于需要真正动态性的部分：用Triton/手写CUDA实现动态shape kernel，绕过XLA

**JAX/XLA中的缓解技巧：**
```python
# jax.jit中指定static_argnums避免不必要的重编译
@jax.jit(static_argnums=(2,))  # 第3个参数作为静态常量
def forward(params, x, config):
    ...

# 或使用donate_argnums优化内存
```

### Q: TVM的原理和现状？NPU手写算子 vs 自动生成？

**TVM核心原理——计算与调度分离 + 自动搜索：**

```python
# 计算描述（不变的数学定义）
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))

# 调度空间（可搜索的实现策略）
# - tile大小: {8, 16, 32, 64, 128}
# - 循环顺序: {ijk, ikj, jik, ...}
# - 向量化维度: {inner_i, inner_j}
# - 并行维度: {outer_i, outer_j}
# - 展开因子: {1, 2, 4, 8}
```

AutoTVM/Ansor通过**cost model（XGBoost）+ 实际硬件测量**在这个巨大的搜索空间中找到最优配置。

**现状与趋势：**

TVM在NPU上的实际使用逐渐减少，原因：

| 因素 | TVM自动生成 | NPU手写算子 |
|------|------------|------------|
| 性能 | 理论峰值的60-80% | 理论峰值的85-95% |
| 硬件特性利用 | 通用搜索难覆盖特殊指令 | 精确利用DMA、CUBE等 |
| 开发效率 | 高（自动搜索） | 低（需深入理解ISA） |
| 调优时间 | 数小时搜索 | 数天~数周开发 |
| 适用算子范围 | 通用 | 核心算子 |

**当前主流方案——混合策略：**
```
核心算子(GEMM, Conv, Attention等): 手写（占计算量80%+，必须极致优化）
长尾算子(数百种element-wise等): TVM/编译器自动生成（数量多但单个占比小）
```

**NPU手写算子的挑战：**
- NPU架构差异大（华为昇腾的CUBE/Vector、寒武纪的MLU Core、天数智芯的GPGPU-like）
- 缺乏统一的编程模型和调试工具
- ISA文档不如NVIDIA完善
- 生态工具链（profiler、debugger）不成熟
