---
title: CUDA WGMMA指令详解
date: 2026-06-01 14:00:00
mathjax: true
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程高阶]
tags: [CUDA, WGMMA, Hopper, Tensor Core, 矩阵乘法]
---

Hopper 架构（SM90）引入了 WGMMA（Warpgroup Matrix Multiply-Accumulate）指令，这是 NVIDIA Tensor Core 编程模型的一次重大升级——从单 Warp 操作跃升为 Warpgroup（4 个 Warp、128 线程）级别的协作矩阵运算。本文从硬件演进动机出发，系统介绍 WGMMA 的工作原理、指令格式、编程方式与性能特征。

<!-- more -->

## 📑 目录

- [1. 从 WMMA 到 WGMMA：Tensor Core 指令演进](#1-从-wmma-到-wgmma-tensor-core-指令演进)
- [2. Warpgroup 的概念](#2-warpgroup-的概念)
- [3. WGMMA 指令详解](#3-wgmma-指令详解)
- [4. WGMMA 的异步执行模型](#4-wgmma-的异步执行模型)
- [5. 使用 CuTe 编程 WGMMA](#5-使用-cute-编程-wgmma)
- [6. 与前代指令的对比](#6-与前代指令的对比)
- [7. 性能优化要点](#7-性能优化要点)
- [总结](#-总结)

---

## 1. 从 WMMA 到 WGMMA：Tensor Core 指令演进

### 1.1 Tensor Core 指令的三代演进

NVIDIA 的矩阵运算指令经历了三代发展：

| 架构 | 指令 | 操作粒度 | 典型矩阵形状 |
|------|------|---------|-------------|
| Volta/Turing (SM70/75) | `wmma` | 单 Warp (32 线程) | 16×16×16 |
| Ampere (SM80) | `mma` (PTX) | 单 Warp (32 线程) | 16×8×16 |
| Hopper (SM90) | `wgmma` | Warpgroup (128 线程) | 64×N×16 |

> **白话理解**：如果把矩阵乘法比作搬砖砌墙，`wmma` 是一个 32 人小队搬一小块砖；`mma` 是同样的小队换了更灵活的工具；而 `wgmma` 则是把 4 个小队组成 128 人大队，一次搬一整面墙——协调成本均摊到更多计算量上，整体效率大幅提升。

### 1.2 为什么要升级到 Warpgroup 粒度

单 Warp 粒度的 `mma` 指令存在两个核心瓶颈：

1. **指令发射开销大**：要完成一个较大的矩阵分块（如 64×64），需要发射大量 `mma` 指令，每条指令都有调度开销
2. **寄存器压力高**：操作数 A、B 都必须在寄存器中，大矩阵分块需要极大的寄存器占用

WGMMA 的设计哲学是：**将更多工作打包进单条指令，同时允许操作数直接来自 Shared Memory，减少寄存器占用**。

## 2. Warpgroup 的概念

### 2.1 定义与组成

**Warpgroup** 是 Hopper 架构引入的线程组织层级，由 4 个连续的 Warp 组成，共 128 个线程：

```text
Warpgroup = Warp0 + Warp1 + Warp2 + Warp3 = 4 × 32 = 128 线程
```

在一个 Thread Block 中，Warpgroup 按 warp index 自然划分：

- Warpgroup 0：Warp 0~3（线程 0~127）
- Warpgroup 1：Warp 4~7（线程 128~255）
- 以此类推

### 2.2 Warpgroup 与 SM 的关系

Hopper 架构的每个 SM 包含 4 个处理块（Processing Block），每个处理块有独立的 Warp Scheduler 和 Tensor Core。一个 Warpgroup 的 4 个 Warp 分布在 4 个处理块中，使得 WGMMA 指令可以同时利用所有 4 个 Tensor Core 单元协同执行。

> **白话理解**：SM 就像一个有 4 条生产线的车间，以前每条线各干各的（单 Warp MMA）；Warpgroup 相当于把 4 条线联动起来，接同一张大订单——4 个 Tensor Core 同时为同一条 WGMMA 指令干活。

## 3. WGMMA 指令详解

### 3.1 基本运算语义

WGMMA 执行的核心运算为：

$$
D = A \times B + C
$$

其中：
- $A$ 的形状为 $M \times K$，$M$ 固定为 64
- $B$ 的形状为 $K \times N$，$N$ 可选 8、16、24、...、256（8 的倍数）
- $C/D$ 的形状为 $M \times N = 64 \times N$

### 3.2 操作数来源

WGMMA 指令最关键的设计特性之一是操作数的来源灵活性：

| 操作数 | 来源 | 说明 |
|--------|------|------|
| A | 寄存器 **或** Shared Memory | 双模式可选 |
| B | **仅** Shared Memory | 必须在 SMEM 中 |
| C/D | 寄存器 | 累加器始终在寄存器 |

📌 **关键点**：B 操作数强制来自 Shared Memory 是 WGMMA 与前代 `mma` 指令的根本区别。这意味着 B 矩阵不需要先加载到寄存器再参与计算，显著降低了寄存器压力。

### 3.3 支持的数据类型与形状

WGMMA 支持多种数据类型组合，$K$ 维度大小随数据类型变化：

| A 类型 | B 类型 | D 类型 | K 维度 | 算力（每 WGMMA） |
|--------|--------|--------|--------|------------------|
| FP16 | FP16 | FP16/FP32 | 16 | 64×N×16 |
| BF16 | BF16 | FP16/FP32 | 16 | 64×N×16 |
| TF32 | TF32 | FP32 | 8 | 64×N×8 |
| FP8 (E4M3) | FP8 (E4M3/E5M2) | FP16/FP32 | 32 | 64×N×32 |
| INT8 | INT8 | INT32 | 32 | 64×N×32 |

💡 **提示**：FP8 的 $K=32$ 意味着单条 WGMMA 指令处理的计算量是 FP16 的两倍，这也是 Hopper 在 FP8 训练/推理场景下性能飙升的硬件基础。

### 3.4 PTX 指令格式

WGMMA 在 PTX 层面的指令格式为：

```text
wgmma.mma_async.sync.aligned.shape.dtype_d.dtype_a.dtype_b  d, a_desc, b_desc, scale_d, imm_scale_a, imm_scale_b, imm_trans_a, imm_trans_b;
```

各字段含义：

- `shape`：矩阵形状，如 `m64n128k16`
- `dtype_d/a/b`：数据类型
- `d`：累加器寄存器组
- `a_desc/b_desc`：矩阵描述符（当操作数来自 Shared Memory 时使用 64-bit 描述符）
- `scale_d`：是否将 D 初始化为零（`scale_d = 0` 表示 $D = A \times B$，忽略 C）
- `imm_scale_a/b`：正/负号（1 或 -1）
- `imm_trans_a/b`：是否转置

一个具体的 PTX 示例：

```text
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {d0, d1, ..., d63},       // D: 64 个 FP32 寄存器
    a_desc,                    // A 矩阵描述符（SMEM）
    b_desc,                    // B 矩阵描述符（SMEM）
    1,                         // scale_d = 1，执行 D = A*B + C
    1, 1,                      // scale_a = +1, scale_b = +1
    0, 0;                      // trans_a = 0, trans_b = 0
```

### 3.5 矩阵描述符（Matrix Descriptor）

当 A 或 B 来自 Shared Memory 时，需要通过 **64-bit 矩阵描述符**来描述其在 Shared Memory 中的布局：

```text
描述符格式（64 bits）:
[63:62] - Swizzle 模式 (00=none, 01=32B, 10=64B, 11=128B)
[61:49] - 保留
[48:32] - Leading dimension stride (字节)
[31:16] - Stride dimension offset
[15:4]  - Base address（Shared Memory 偏移，16B 对齐）
[3:0]   - 保留
```

Swizzle 模式用于避免 Shared Memory 的 Bank Conflict，是 WGMMA 高效访存的关键。

## 4. WGMMA 的异步执行模型

### 4.1 异步执行的含义

WGMMA 是一条**异步指令**——发射后立即返回，计算在后台由 Tensor Core 执行。这意味着：

1. 发射 WGMMA 后，线程可以继续执行其他工作（如数据搬运）
2. 在读取结果之前，必须显式等待 WGMMA 完成

> **白话理解**：WGMMA 就像在餐厅点了一道菜——你下单（发射指令）后不用站在厨房门口干等，可以先去倒水、聊天（做其他事情），等菜好了（计算完成）再过来取。

### 4.2 异步屏障与 Commit/Wait

WGMMA 使用专用的 fence 和 commit/wait 机制管理异步执行：

```cuda
// 1. Fence：声明后续 WGMMA 的操作数已就绪
wgmma_fence();

// 2. 发射多条 WGMMA 指令
wgmma_mma_async(...);  // 第一条
wgmma_mma_async(...);  // 第二条
// ... 可以连续发射多条

// 3. Commit：将当前所有待执行的 WGMMA 提交为一个组
wgmma_commit_group();

// 4. Wait：等待指定数量的 group 完成
wgmma_wait_group<0>();  // 等待所有 group 完成
```

关键 API 说明：

- **`wgmma_fence()`**：内存屏障，确保 WGMMA 读取的 Shared Memory 数据已写入完毕
- **`wgmma_commit_group()`**：将从上次 commit 到现在发射的所有 WGMMA 打包成一个"组"
- **`wgmma_wait_group<N>()`**：等待直到最多还有 N 个组未完成（N=0 表示全部完成）

### 4.3 与 TMA 的协作流水线

WGMMA 设计为与 TMA（Tensor Memory Accelerator）紧密配合。一个典型的 GEMM 主循环流水线如下：

{% mermaid graph LR %}
    A["TMA 加载<br/>Stage K+2"] --> B["WGMMA 计算<br/>Stage K"]
    B --> C["TMA 加载<br/>Stage K+3"]
    C --> D["WGMMA 计算<br/>Stage K+1"]
{% endmermaid %}

```cuda
// Hopper GEMM 主循环伪代码（多阶段流水线）
for (int k = 0; k < num_k_tiles; k++) {
    // 等待 TMA 加载完成（Stage k 的数据就绪）
    barrier_wait(k % NUM_STAGES);

    // 发射 WGMMA
    wgmma_fence();
    for (int n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        wgmma_mma_async(accum[n_tile], smem_A[k % NUM_STAGES],
                        smem_B[k % NUM_STAGES][n_tile]);
    }
    wgmma_commit_group();

    // 异步发起下一阶段的 TMA 加载
    if (k + NUM_STAGES < num_k_tiles) {
        tma_load_async(smem_A[(k + NUM_STAGES) % NUM_STAGES],
                       gmem_A, k + NUM_STAGES);
        tma_load_async(smem_B[(k + NUM_STAGES) % NUM_STAGES],
                       gmem_B, k + NUM_STAGES);
    }

    // 等待当前 WGMMA 组完成（为下一次迭代的 fence 做准备）
    wgmma_wait_group<0>();
}
```

## 5. 使用 CuTe 编程 WGMMA

### 5.1 为什么用 CuTe

直接使用 PTX 内联汇编编写 WGMMA 极为复杂（需要手动管理描述符、Swizzle、寄存器映射）。NVIDIA 的 **CuTe**（CUDA Templates）库提供了高层抽象，是实践中使用 WGMMA 的主要方式。

### 5.2 CuTe 中的 WGMMA 接口

```cpp
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>

using namespace cute;

// 定义 WGMMA 的 TiledMMA
// M=64, N=128, K=16, FP16 输入, FP32 累加
using MMA = decltype(make_tiled_mma(
    SM90_64x128x16_F32F16F16_SS{}  // SS = 两个操作数都来自 Shared Memory
));

// 或者 A 来自寄存器、B 来自 Shared Memory
using MMA_RS = decltype(make_tiled_mma(
    SM90_64x128x16_F32F16F16_RS{}  // RS = A from Register, B from Shared
));
```

命名约定解释：`SM90_64x128x16_F32F16F16_SS`

- `SM90`：Hopper 架构
- `64x128x16`：M=64, N=128, K=16
- `F32F16F16`：D=FP32, A=FP16, B=FP16
- `SS`：Source A 和 Source B 都来自 Shared Memory（`RS` 表示 A 来自 Register）

### 5.3 完整 GEMM Kernel 骨架

```cpp
template <class ProblemShape, class TiledMMA,
          class SmemLayoutA, class SmemLayoutB>
__global__ void wgmma_gemm_kernel(
    ProblemShape shape_MNK,
    const half_t* gmem_A, const half_t* gmem_B, float* gmem_D)
{
    using namespace cute;

    // 1. 创建 TMA 描述符和 Shared Memory
    extern __shared__ char smem[];
    auto smem_A = make_tensor(make_smem_ptr<half_t>(smem), SmemLayoutA{});
    auto smem_B = make_tensor(make_smem_ptr<half_t>(smem + size(SmemLayoutA{}) * sizeof(half_t)),
                              SmemLayoutB{});

    // 2. 初始化 TiledMMA 和累加器
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto accum = partition_fragment_C(tiled_mma, select<0,1>(shape_MNK));
    clear(accum);

    // 3. 主循环：TMA 加载 + WGMMA 计算
    auto K = get<2>(shape_MNK);
    for (int k = 0; k < K; k += Int<16>{}) {
        // TMA 搬运数据到 Shared Memory（省略细节）
        // ...

        // 同步等待数据就绪
        __syncthreads();

        // 创建 SMEM tensor 的 MMA 视图
        auto tCsA = thr_mma.partition_A(smem_A);
        auto tCsB = thr_mma.partition_B(smem_B);

        // 执行 WGMMA
        cute::gemm(tiled_mma, tCsA, tCsB, accum);
    }

    // 4. 写回结果
    // ...
}
```

### 5.4 Swizzle 布局

WGMMA 要求 Shared Memory 中的数据使用特定的 Swizzle 模式以避免 Bank Conflict。CuTe 提供了标准的 Swizzle 布局：

```cpp
// 128-byte Swizzle 布局（适用于 FP16，K=16 时每行 32 字节）
using SmemLayoutAtom = decltype(
    composition(Swizzle<3, 3, 3>{},
                Layout<Shape<_8, _32>, Stride<_32, _1>>{}));

using SmemLayout = decltype(
    tile_to_shape(SmemLayoutAtom{},
                  Shape<_64, _16>{}));  // 64 行 × K=16 列
```

💡 **提示**：Swizzle 的三个参数 `<B, M, S>` 分别表示 Base bits、Mask bits 和 Shift bits。`Swizzle<3,3,3>` 是 128B 粒度的 Swizzle，恰好匹配 Shared Memory 32 个 Bank 的全宽度，可完全消除 Bank Conflict。

## 6. 与前代指令的对比

| 特性 | WMMA (SM70+) | MMA PTX (SM80+) | WGMMA (SM90+) |
|------|-------------|-----------------|---------------|
| 操作粒度 | 1 Warp | 1 Warp | 1 Warpgroup (4 Warps) |
| 每条指令计算量 | 16×16×16 | 16×8×16 | 64×N×K (N 最大 256) |
| B 操作数来源 | 寄存器 | 寄存器 | Shared Memory |
| 执行模型 | 同步 | 同步 | **异步** |
| 寄存器压力 | 高 | 高 | 低（B 不占寄存器） |
| 编程接口 | CUDA C++ API | PTX 内联汇编 | CuTe / PTX |
| 与 TMA 配合 | 不支持 | 不支持 | 原生设计 |

⚠️ **注意**：虽然 WGMMA 在 Hopper 上提供了最高性能，但它**只能在 SM90 及以上架构运行**。如果需要兼容 Ampere 等旧架构，仍需使用 `mma` PTX 指令。

## 7. 性能优化要点

### 7.1 充分重叠计算与访存

WGMMA 异步执行的核心价值在于让 Tensor Core 的计算与 TMA 的数据搬运并行进行。实践中应确保：

- 使用**多阶段流水线**（通常 3~4 个 Stage），让 TMA 加载数据的延迟被计算完全隐藏
- Commit Group 的粒度要适当——太细（每条 WGMMA 一个 Group）增加管理开销，太粗（所有 WGMMA 一个 Group）减少重叠机会

### 7.2 选择合适的操作数模式

| 模式 | 场景 | 优势 |
|------|------|------|
| SS（A、B 都来自 SMEM） | 标准 GEMM | 寄存器压力最低 |
| RS（A 来自寄存器，B 来自 SMEM） | A 矩阵复用率高的场景 | A 常驻寄存器避免重复加载 |

💡 **提示**：当 N 维度远大于 M 维度时（如 Attention 中 Q×K^T 的计算），使用 RS 模式让 Q 矩阵常驻寄存器、K 矩阵从 Shared Memory 流式读入，可以获得更好的性能。

### 7.3 注意 Warpgroup 对齐

- Thread Block 的线程数必须是 128 的倍数（至少一个完整 Warpgroup）
- 每个 Warpgroup 独立执行自己的 WGMMA，Block 内多个 Warpgroup 可以并行处理不同的输出 tile

### 7.4 Swizzle 模式与 Bank Conflict

WGMMA 的 B 矩阵直接从 Shared Memory 读取，如果布局不当会产生严重的 Bank Conflict。必须使用匹配的 Swizzle 模式。

Swizzle 模式的选择取决于 tile 在 Shared Memory 中的 **Leading Dimension 步长**（连续两行之间的字节距离），而非单纯看 K 维度行宽：

- Leading Dimension ≥ 128B：使用 128B Swizzle（最常用，如 CuTe 的 `Swizzle<3,3,3>`）
- Leading Dimension ≥ 64B 且 < 128B：使用 64B Swizzle
- Leading Dimension ≥ 32B 且 < 64B：使用 32B Swizzle

💡 **提示**：实践中 CUTLASS 的 SM90 GEMM Kernel 普遍使用 128B Swizzle，因为矩阵 tile 的行步长通常 ≥ 128B（例如 FP16 下 64 列 × 2B = 128B）。只有极窄 tile 才需要降级到更小的 Swizzle 粒度。

## 📝 总结

WGMMA 是 Hopper 架构 Tensor Core 编程的核心指令，其设计理念可以归纳为三点：

1. **更大粒度**：Warpgroup（128 线程）协同执行，单条指令处理 64×N×K 的矩阵块，摊薄指令发射开销
2. **更低寄存器压力**：B 操作数直接从 Shared Memory 读取，释放宝贵的寄存器资源用于扩大 tile 或提高 Occupancy
3. **异步执行**：与 TMA 配合构建流水线，实现计算与访存的完全重叠

在实践中，建议通过 CuTe 库或 CUTLASS 3.x 框架使用 WGMMA，避免直接编写 PTX。理解 WGMMA 的工作原理有助于设计更高效的 Kernel 流水线结构和选择合适的 tile size。

## 🎯 自我检验清单

- 能解释 WGMMA 与 WMMA/MMA 的核心区别（粒度、操作数来源、执行模型）
- 能说明 Warpgroup 的组成（4 Warps、128 线程）及其与 SM 处理块的对应关系
- 能描述 WGMMA 异步执行模型中 fence/commit/wait 各自的作用
- 能区分 SS 模式和 RS 模式的适用场景
- 能解释为什么 B 操作数来自 Shared Memory 可以降低寄存器压力
- 能说明 WGMMA 与 TMA 配合的多阶段流水线工作流程
- 能理解 Swizzle 模式对 WGMMA 性能的影响
- 能使用 CuTe 的命名约定读懂 `SM90_64x128x16_F32F16F16_SS` 的含义

## 📚 参考资料

- [NVIDIA PTX ISA - Warpgroup Level Matrix Multiply-Accumulate](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
- [NVIDIA CUTLASS 3.x - CuTe Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md)
- [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [CUDA C++ Programming Guide - Asynchronous Data Movement](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Lei Mao's Blog - NVIDIA CUDA WGMMA](https://leimao.github.io/blog/NVIDIA-CUDA-WGMMA/)
