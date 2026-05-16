---
title: '商汤 AI Infra 面试'
date: 2026-04-16 12:19:48
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: GEMM优化怎么做？为什么一个线程要计算C矩阵8x8个元素？寄存器会溢出吗？

**GEMM优化核心思路：**

GEMM（C = A * B）优化的本质是通过**多级分块（Tiling）**最大化数据复用，减少对HBM的访问次数。

```
全局内存 → Block Tile (加载到共享内存, ~128x128)
              → Warp Tile (分配给warp, ~64x64) 
                 → Thread Tile (每个线程计算, ~8x8)
```

**为什么一个线程计算8x8个元素？——提高计算访存比（Arithmetic Intensity）：**

以外积(Outer Product)方式计算为例：
- 每个线程从共享内存读取A的一列片段(8个float)和B的一行片段(8个float) → 共读取16个float
- 用这16个float计算8×8=64次FMA → 计算访存比 = 64/16 = 4:1
- 如果每个线程只算1×1：读取2个float做1次FMA → 计算访存比 = 1:2
- **8x8的计算访存比是1x1的16倍**，大幅减少共享内存带宽压力

**寄存器使用分析——会溢出吗？**

| 用途 | 寄存器数量 |
|------|-----------|
| C矩阵结果 (8x8 float) | 64个 |
| A片段缓存 (8 float) | 8个 |
| B片段缓存 (8 float) | 8个 |
| 循环变量、指针、临时值 | ~10-20个 |
| **合计** | **~90-100个** |

A100每个线程最多可用255个寄存器（每SM共65536个），100个寄存器远未溢出。但需注意：
- 寄存器使用量直接影响**occupancy**：用100个寄存器时，每SM最多驻留65536/100≈655个线程（约20个warp），occupancy约30%
- 这是性能权衡：较低occupancy但高计算效率（data reuse）通常比高occupancy但低效率更优
- Cutlass/cuBLAS中的高性能kernel通常在occupancy 25-50%时达到峰值性能

### Q: #pragma unroll的作用？

`#pragma unroll`指示编译器将循环体展开为顺序执行的代码，消除循环控制开销。

**展开前后对比：**
```cuda
// 展开前
for (int i = 0; i < 4; i++)
    c[i] = a[i] + b[i];

// 展开后（编译器生成的等价代码）
c[0] = a[0] + b[0];
c[1] = a[1] + b[1];
c[2] = a[2] + b[2];
c[3] = a[3] + b[3];
```

**为什么能提升性能？**

1. **消除循环控制开销**：减少了比较、分支跳转、计数器递增指令
2. **提高ILP（指令级并行）**：展开后的独立指令可以被warp调度器同时发射到不同功能单元
3. **使编译器更好优化**：展开后编译器能看到更多数据依赖关系，做更好的寄存器分配和指令重排
4. **减少分支预测失败**：GPU上分支比CPU更昂贵（可能导致warp divergence）

**使用方式：**
```cuda
#pragma unroll        // 完全展开（编译器自动判断循环次数）
#pragma unroll 4      // 部分展开（展开因子为4）
#pragma unroll 1      // 禁止展开
```

**注意事项：**
- 循环次数必须是编译时常量才能完全展开
- 过度展开会增加寄存器压力和指令cache miss
- CUDA中内层循环（如K维度的累加循环）展开效果最佳

### Q: 与Cutlass/cuBLAS相比，手写GEMM kernel性能如何？

**性能层次对比：**

| 实现 | 典型性能 | 特点 |
|------|---------|------|
| cuBLAS | 硬件峰值90-95% | NVIDIA黑盒库，针对各GPU架构和shape高度调优 |
| Cutlass | 硬件峰值85-93% | 开源C++模板库，提供定制灵活性 |
| 手写优化Kernel | 硬件峰值75-90% | 需要深入理解硬件，开发周期长 |
| 朴素实现 | 硬件峰值<10% | 对照基准 |

**cuBLAS的优势：**
- 针对每个GPU架构（sm_80/sm_89/sm_90）有独立优化的kernel库
- 内置heuristic自动选择最优kernel（根据M/N/K shape和数据类型）
- 利用未公开的硬件特性和指令
- 包含thousands of pre-tuned kernel变体

**Cutlass的优势：**
- 模板化设计，支持自定义epilogue（如GEMM+Bias+ReLU融合）
- 支持自定义数据布局和Tensor Core MMA指令
- 提供不同粒度的组件（从GEMM到warp-level MMA到thread-level copy）
- 性能接近cuBLAS，但完全可控

**手写kernel有优势的场景：**
- 非标准shape（如极端的M=1或K很小的Skinny GEMM）
- 自定义融合（GEMM + 复杂后处理逻辑）
- 非标准数据类型或布局
- 与框架特定的内存管理集成（如PagedAttention中的非连续KV读取）

### Q: 了解Tensor Core吗？

Tensor Core是NVIDIA GPU上的专用**矩阵乘加(MMA)**硬件单元，通过在单个时钟周期内完成小矩阵乘法来大幅提升算力。

**工作原理：**
```
D = A * B + C
其中A/B/C/D是小矩阵（如16x16x16的FP16矩阵乘得到FP16/FP32结果）
```

**各代GPU的Tensor Core对比：**

| 架构 | GPU | 支持精度 | 峰值算力(FP16) | MMA指令形状 |
|------|-----|----------|---------------|------------|
| Volta (sm_70) | V100 | FP16 | 125 TFLOPS | 8x8x4 |
| Ampere (sm_80) | A100 | FP16/BF16/TF32/INT8/FP64 | 312 TFLOPS | 16x8x16 |
| Hopper (sm_90) | H100 | +FP8(E4M3/E5M2) | 989 TFLOPS(FP16) | 16x8x16/64x... |
| Blackwell (sm_100) | B200 | +FP4 | 2.5 PFLOPS(FP4) | 更大 |

**使用方式（从高到低）：**
1. **cuBLAS/cuDNN**：自动使用Tensor Core（最简单）
2. **Cutlass**：模板级别配置MMA指令
3. **WMMA API (C++)**：`nvcuda::wmma::mma_sync()`，warp级别操作
4. **PTX/SASS MMA指令**：最细粒度，如`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`

**使用约束：**
- 矩阵维度必须是特定值的倍数（如16的倍数）
- 数据需要满足对齐要求（通常128-bit对齐）
- 只支持特定数据类型组合（如FP16输入+FP32累加）
- 需要整个warp协同执行MMA操作

### Q: 做过低比特位的GEMM吗？

低比特GEMM（INT8/INT4/FP8）是模型推理加速的关键技术，核心是用低精度计算换取更高吞吐和更低带宽需求。

**为什么低比特GEMM更快？**
1. **带宽节省**：INT8数据量是FP16的一半，memory-bound场景下直接快2倍
2. **计算吞吐翻倍**：A100 INT8 Tensor Core算力624 TOPS vs FP16 312 TFLOPS
3. **显存节省**：模型权重减小一半（INT8）或四分之一（INT4）

**量化GEMM的计算流程：**
```
Y_fp32 = dequant(X_int8) * dequant(W_int8)
       = (X_int8 * scale_x) * (W_int8 * scale_w)
       = X_int8 * W_int8 * (scale_x * scale_w)  // INT8矩阵乘 + scale后处理
```

**关键技术挑战：**

| 挑战 | 解决方案 |
|------|----------|
| 量化精度损失 | Per-channel/per-group scale；校准数据集选择 |
| 激活值outlier | SmoothQuant转移难度到权重；per-token动态量化 |
| 混合精度累加 | INT8乘+INT32/FP32累加（Tensor Core原生支持） |
| 数据布局对齐 | Interleaved layout/CUTLASS最优布局 |
| 不同量化粒度 | Per-tensor(简单) vs per-channel(精度好) vs per-group(折中) |

**FP8 vs INT8对比（H100）：**

| 特性 | INT8 | FP8 (E4M3) |
|------|------|------------|
| 动态范围 | [-128, 127] | [-448, 448] |
| 精度 | 均匀分布 | 近似对数分布（小值精度高） |
| 需要zero-point | 常需要 | 不需要 |
| H100峰值 | 1979 TOPS | 3958 TOPS |
| 量化误差 | 对outlier敏感 | 更好处理不同scale |
