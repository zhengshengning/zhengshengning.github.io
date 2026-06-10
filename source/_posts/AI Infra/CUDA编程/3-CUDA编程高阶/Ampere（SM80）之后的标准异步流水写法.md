---
title: Ampere（SM80）之后的标准异步流水写法
date: 2026-05-26 15:30:00
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程高阶]
tags: [CUDA, 异步拷贝, Pipeline, Ampere, 性能优化]
---

Ampere 架构引入了硬件级异步拷贝指令 `cp.async`，让数据从 Global Memory 到 Shared Memory 的搬运可以完全绕过寄存器，与计算真正并行。本文介绍 SM80 之后标准的多阶段异步流水线（Multi-Stage Async Pipeline）编程模式，从原理到可运行的代码模板，帮你掌握现代高性能 Kernel 的数据搬运范式。

<!-- more -->

## 📑 目录

- [1. 为什么需要异步流水线](#1-为什么需要异步流水线)
- [2. Ampere 的硬件基础：cp.async](#2-ampere-的硬件基础cpasync)
- [3. 多阶段流水线模型](#3-多阶段流水线模型)
- [4. 标准写法：完整代码模板](#4-标准写法完整代码模板)
- [5. cuda::pipeline API 详解](#5-cudapipeline-api-详解)
- [6. Hopper 架构的进一步演进：TMA](#6-hopper-架构的进一步演进tma)
- [7. 常见陷阱与调试技巧](#7-常见陷阱与调试技巧)
- [总结](#-总结)

---

## 1. 为什么需要异步流水线

### 1.1 传统双缓冲的局限

在 Ampere 之前，实现计算与访存重叠的标准做法是**手动双缓冲**（Double Buffering）：

```cuda
// Pre-Ampere 双缓冲伪代码
__shared__ float smem[2][TILE_SIZE];
int buf = 0;

// 预加载第一块
load_tile(smem[buf], global_ptr, 0);
__syncthreads();

for (int k = 0; k < num_tiles; k++) {
    // 加载下一块到另一个 buffer
    load_tile(smem[1 - buf], global_ptr, k + 1);
    // 计算当前 buffer
    compute(smem[buf]);
    __syncthreads();
    buf = 1 - buf;
}
```

⚠️ **注意**：上面的 `load_tile` 和 `compute` 在代码结构上是**顺序执行**的。虽然编译器可以将 `LDG`（非阻塞指令）重排到 `compute` 之前发射，让单个 warp 内实现部分指令级重叠，但这要求 LDG 的目标寄存器在整个 compute 期间被占用——本质上是用寄存器换重叠。更主要的重叠来自 **warp 间调度**：当 warp A 在等待内存返回时，调度器切换到 warp B 去执行计算。这意味着双缓冲的效果**强依赖于高 Occupancy**（足够多的活跃 warp 来填满内存延迟），而寄存器压力恰恰限制了 Occupancy——形成矛盾。

> **白话理解**：双缓冲就像餐厅的传菜窗口有两个托盘——厨师往一个托盘放菜的同时，服务员从另一个托盘端菜上桌。两边互不等待，效率翻倍。

下面是一个完整的真实案例——用双缓冲实现 SAXPY（$y = \alpha x + y$）elementwise 操作：

```cuda
// Pre-Ampere 双缓冲 SAXPY Kernel
constexpr int TILE = 256;

__global__ void saxpy_double_buffer(float alpha, const float* x,
                                    float* y, int N) {
    __shared__ float sx[2][TILE];
    __shared__ float sy[2][TILE];

    int block_start = blockIdx.x * (TILE * ((N / TILE + gridDim.x - 1) / gridDim.x));
    int num_tiles = (N - block_start + TILE - 1) / TILE;
    int buf = 0;

    // 预加载第一个 tile 到 buffer 0
    int idx = block_start + threadIdx.x;
    if (idx < N) {
        sx[0][threadIdx.x] = x[idx];
        sy[0][threadIdx.x] = y[idx];
    }
    __syncthreads();

    for (int t = 0; t < num_tiles; t++) {
        // 加载下一个 tile 到另一个 buffer（与计算重叠）
        int next_idx = block_start + (t + 1) * TILE + threadIdx.x;
        if (t + 1 < num_tiles && next_idx < N) {
            sx[1 - buf][threadIdx.x] = x[next_idx];
            sy[1 - buf][threadIdx.x] = y[next_idx];
        }

        // 计算当前 buffer
        float result = alpha * sx[buf][threadIdx.x] + sy[buf][threadIdx.x];

        // 写回
        int out_idx = block_start + t * TILE + threadIdx.x;
        if (out_idx < N) {
            y[out_idx] = result;
        }

        __syncthreads();
        buf = 1 - buf;
    }
}
```

💡 **提示**：对于 SAXPY 这类计算强度极低的 elementwise 操作（计算/访存比接近 1:1），双缓冲的收益有限，因为瓶颈在带宽而非延迟隐藏。但当 elementwise 操作变复杂（如多个输入 + 激活函数 + 归一化），双缓冲的价值就体现出来了。

这种写法有两个根本问题：

**问题 1：数据搬运必须经过寄存器**

上面代码中 `sx[0][threadIdx.x] = x[idx]` 看起来是一条赋值语句，但编译到 SASS 层面实际上是两条指令串行执行：

```text
LDG.E R4, [R2]        // Global Memory → Register（耗时 ~200-800 cycles）
STS [R6], R4           // Register → Shared Memory
```

寄存器 `R4` 在这里充当了"中转站"。这意味着：
- 每搬运一个 `float`，就要占用一个寄存器来暂存数据，搬运 `float4` 则占用 4 个寄存器
- 当 tile 较大时，大量寄存器被"搬运中"的数据占据，挤压了计算可用的寄存器数量
- `LDG` 完成前线程会 stall（或切换到其他 warp），`STS` 必须等 `LDG` 返回后才能执行——两条指令之间存在硬依赖

**问题 2：同步粒度过粗**

`__syncthreads()` 是一个 Block 级别的全量屏障——它等待 Block 内**所有线程的所有操作**完成，而不仅仅是数据搬运。这带来两个后果：

- **无法区分"搬运完成"和"计算完成"**：即使只想确认"下一个 tile 的数据已经到达 Shared Memory"，也必须等待所有线程的所有工作（包括计算、写回）全部结束
- **流水线被强制序列化**：理想情况下，我们希望"tile N 的计算"和"tile N+1 的搬运"完全重叠；但 `__syncthreads()` 在两者之间插入了一道硬墙，强制先完成一个再开始另一个

⚠️ **注意**：在上面的 SAXPY 双缓冲示例中，虽然加载和计算在代码顺序上是交错的，但由于 `LDG` → `STS` 的寄存器依赖和 `__syncthreads()` 的全量同步，实际的计算/访存重叠程度远不如看起来那么理想。GPU 的 warp 调度器可以通过切换 warp 来部分隐藏延迟，但这依赖于有足够多的活跃 warp（即高 Occupancy），而寄存器压力恰恰限制了 Occupancy。

### 1.2 异步拷贝的核心优势

Ampere 的 `cp.async` 指令实现了 **Global Memory → Shared Memory 的直接搬运**，不经过寄存器文件：

📥 Global Memory → ⚙️ `cp.async` → 📤 Shared Memory（绕过寄存器）

这带来三个好处：

- **释放寄存器**：不再需要中转寄存器，降低寄存器压力，提高 Occupancy
- **减少指令数**：一条 `cp.async` 替代 `LDG` + `STS` 两条指令
- **细粒度同步**：通过 `cp.async.commit_group` 和 `cp.async.wait_group` 实现分组等待，不必全量同步

## 2. Ampere 的硬件基础：cp.async

### 2.1 PTX 指令

`cp.async` 的 PTX 形式：

```text
cp.async.ca.shared.global [dst_smem], [src_gmem], cp_size;
cp.async.commit_group;
cp.async.wait_group N;
```

- `cp.async.ca.shared.global`：从 Global Memory 异步拷贝到 Shared Memory，`ca` 表示 cache-all（经过 L1 cache）
- `cp.async.cg.shared.global`：`cg` 表示 cache-global（绕过 L1，仅走 L2）
- `cp_size`：每次拷贝的字节数，支持 4、8、16 字节
- `commit_group`：将之前所有未提交的 `cp.async` 打包为一个 group
- `wait_group N`：等待直到最多还有 N 个 group 未完成

💡 **提示**：`cp_size = 16`（即 128-bit）效率最高，对应一次 `float4` 或 `int4` 的搬运宽度，与 Global Memory 的事务粒度对齐。

### 2.2 CUDA C++ 封装

NVIDIA 提供了 C++ 层面的封装，无需直接写 PTX：

```cuda
#include <cuda/pipeline>
#include <cooperative_groups.h>

// 单次异步拷贝
cuda::memcpy_async(dst_shared_ptr, src_global_ptr, sizeof(float4), pipeline);

// 或使用更底层的 __pipeline 内建函数
__pipeline_memcpy_async(dst_shared_ptr, src_global_ptr, 16); // 16 bytes
__pipeline_commit();
__pipeline_wait_prior(N);
```

### 2.3 与同步拷贝的对比

| 📊 维度 | 同步拷贝（Pre-Ampere） | 异步拷贝（Ampere+） |
|---------|----------------------|-------------------|
| 数据路径 | Global → Register → Shared | Global → Shared（直接） |
| 寄存器开销 | 需要中转寄存器 | 无 |
| 指令数 | 2 条（LDG + STS） | 1 条（cp.async） |
| 同步方式 | `__syncthreads()` | `wait_group` 细粒度等待 |
| 最小粒度 | 1 字节（LDG.U8） | 4 / 8 / 16 字节 |
| 计算重叠 | 需要手动交错指令 | 硬件自动异步执行 |

## 3. 多阶段流水线模型

### 3.1 从双缓冲到多阶段

双缓冲只有 2 个阶段（stage），在某些场景下不足以完全隐藏访存延迟。Ampere 的异步机制天然支持 **N 阶段流水线**（N-Stage Pipeline），通过增加 stage 数量来更充分地重叠计算与访存。

> **白话理解**：双缓冲是两个托盘轮换，多阶段流水线是传送带——上面同时有 3、4 个托盘在不同位置流转，厨师不停地放菜，服务员不停地取菜，中间永远有菜在路上。

### 3.2 流水线时序图

以 3 阶段（`NUM_STAGES = 3`）为例，时间轴上的行为：

{% mermaid sequenceDiagram %}
    participant G as Global Memory
    participant S as Shared Memory
    participant C as Compute Unit
    Note over G,C: Stage 0,1,2 预填充（Prologue）
    G->>S: async copy tile 0
    G->>S: async copy tile 1
    G->>S: async copy tile 2
    Note over G,C: 主循环（Main Loop）
    S->>C: compute tile 0
    G->>S: async copy tile 3
    S->>C: compute tile 1
    G->>S: async copy tile 4
    Note over G,C: 尾部排空（Epilogue）
    S->>C: compute tile N-2
    S->>C: compute tile N-1
{% endmermaid %}

### 3.3 关键参数选择

`NUM_STAGES` 的选择需要权衡：

| ✅ 增加 stages 的好处 | ❌ 增加 stages 的代价 |
|---------------------|---------------------|
| 更好地隐藏访存延迟 | 占用更多 Shared Memory |
| 流水线更满，利用率更高 | 可能降低 Occupancy |
| 对不规则访存模式更鲁棒 | 代码复杂度略增 |

📌 **关键点**：实践中 `NUM_STAGES = 3` 或 `4` 是最常见的选择。对于 A100（SM80），Shared Memory 最大 164 KB，通常 3-4 个 stage 就能充分隐藏 Global Memory 的 \~400 cycle 延迟。

## 4. 标准写法：完整代码模板

### 4.1 基本结构

一个标准的多阶段异步流水线 Kernel 由三部分组成：

1. **Prologue（序幕）**：预填充前 `NUM_STAGES` 个 tile
2. **Main Loop（主循环）**：每次迭代等待最早的 stage 完成 → 计算 → 发起新的异步拷贝
3. **Epilogue（尾声）**：处理流水线中剩余的 tile

### 4.2 完整代码示例

以一维向量处理为例，展示标准的 3-stage 异步流水线模板：

```cuda
#include <cuda_pipeline.h>

constexpr int TILE_SIZE = 256;   // 每个 tile 的 float 数量
constexpr int NUM_STAGES = 3;
constexpr int COPY_SIZE = 16;    // 每次拷贝 16 字节 = 1 个 float4

__global__ void async_pipeline_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int N) {
    __shared__ float smem[NUM_STAGES][TILE_SIZE];

    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * TILE_SIZE;
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // 每个线程负责拷贝 4 个 float（float4 = 16 bytes）
    const int copy_idx = tid * 4;

    // ========== Prologue: 预填充 NUM_STAGES 个 tile ==========
    for (int stage = 0; stage < NUM_STAGES && stage < num_tiles; stage++) {
        int global_base = (block_start + stage * gridDim.x * TILE_SIZE) + copy_idx;
        if (global_base + 3 < N) {
            __pipeline_memcpy_async(&smem[stage][copy_idx],
                                    &input[global_base],
                                    COPY_SIZE);
        }
        __pipeline_commit();
    }

    // ========== Main Loop ==========
    for (int tile = 0; tile < num_tiles; tile++) {
        int stage = tile % NUM_STAGES;

        // 等待当前 stage 的数据就绪
        __pipeline_wait_prior(NUM_STAGES - 1);
        __syncthreads();

        // 计算（每个线程处理多个元素）
        for (int i = tid; i < TILE_SIZE; i += blockDim.x) {
            float val = smem[stage][i];
            int out_idx = block_start + tile * gridDim.x * TILE_SIZE + i;
            if (out_idx < N) {
                output[out_idx] = val * val + val;
            }
        }

        // 发起下一轮异步拷贝（填充刚用完的 stage）
        int next_tile = tile + NUM_STAGES;
        if (next_tile < num_tiles) {
            int next_base = (block_start + next_tile * gridDim.x * TILE_SIZE) + copy_idx;
            if (next_base + 3 < N) {
                __pipeline_memcpy_async(&smem[stage][copy_idx],
                                        &input[next_base],
                                        COPY_SIZE);
            }
        }
        __pipeline_commit();
        __syncthreads();
    }
}
```

### 4.3 GEMM 场景的流水线模板

矩阵乘法是异步流水线最典型的应用场景。以下是简化的 GEMM 流水线骨架：

```cuda
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int NUM_STAGES = 3;

__global__ void gemm_async_pipeline(const float* A, const float* B,
                                    float* C, int M, int N, int K) {
    __shared__ float smem_A[NUM_STAGES][BM][BK];
    __shared__ float smem_B[NUM_STAGES][BK][BN];

    float acc[BM / blockDim.y][BN / blockDim.x] = {0.0f};
    int num_k_tiles = K / BK;

    // Prologue: 预加载前 NUM_STAGES 个 K-tile
    for (int s = 0; s < NUM_STAGES && s < num_k_tiles; s++) {
        async_load_tile_A(smem_A[s], A, s);
        async_load_tile_B(smem_B[s], B, s);
        __pipeline_commit();
    }

    // Main Loop
    for (int k = 0; k < num_k_tiles; k++) {
        int stage = k % NUM_STAGES;

        __pipeline_wait_prior(NUM_STAGES - 1);
        __syncthreads();

        // 计算当前 tile 的矩阵乘法
        compute_tile(acc, smem_A[stage], smem_B[stage]);

        // 预取下一个 tile
        int prefetch_k = k + NUM_STAGES;
        if (prefetch_k < num_k_tiles) {
            async_load_tile_A(smem_A[stage], A, prefetch_k);
            async_load_tile_B(smem_B[stage], B, prefetch_k);
        }
        __pipeline_commit();
        __syncthreads();
    }

    // 写回结果
    store_result(C, acc);
}
```

⚠️ **注意**：实际 GEMM Kernel 还需要处理向量化加载（`float4`）、Warp 级别的 tile 划分、寄存器 tiling 等细节。上面的代码展示的是流水线骨架，不是生产级实现。

## 5. cuda::pipeline API 详解

### 5.1 API 层次

CUDA 提供了两层 API 来使用异步流水线：

| 📊 层次 | API | 适用场景 |
|---------|-----|---------|
| 底层内建函数 | `__pipeline_memcpy_async` / `__pipeline_commit` / `__pipeline_wait_prior` | 简单场景，直接控制 |
| C++ Pipeline 类 | `cuda::pipeline<cuda::thread_scope_block>` | 复杂场景，类型安全，支持 cooperative groups |

### 5.2 cuda::pipeline 用法

```cuda
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel_with_pipeline(const float4* src, float4* dst, int N) {
    __shared__ float4 smem[NUM_STAGES][TILE_SIZE / 4];

    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES> pipe_state;
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    for (int stage = 0; stage < NUM_STAGES; stage++) {
        pipe.producer_acquire();
        cuda::memcpy_async(block, smem[stage], src + stage * TILE_SIZE / 4,
                           sizeof(float4) * TILE_SIZE / 4, pipe);
        pipe.producer_commit();
    }

    for (int tile = 0; tile < num_tiles; tile++) {
        int stage = tile % NUM_STAGES;

        pipe.consumer_wait();
        // ... 计算 ...
        pipe.consumer_release();

        // 发起下一轮拷贝
        pipe.producer_acquire();
        // ... async copy ...
        pipe.producer_commit();
    }
}
```

### 5.3 关键语义

| 📊 操作 | 含义 |
|---------|------|
| `producer_acquire()` | 获取一个空闲 stage 的写权限 |
| `producer_commit()` | 提交当前 stage，标记为"数据在路上" |
| `consumer_wait()` | 等待最早提交的 stage 数据就绪 |
| `consumer_release()` | 释放当前 stage，标记为"可重用" |

💡 **提示**：`cuda::pipeline` 的 producer/consumer 语义更清晰地表达了流水线的意图，编译器也能据此做更好的优化。在复杂 Kernel 中推荐使用这套 API。

## 6. Hopper 架构的进一步演进：TMA

### 6.1 从 cp.async 到 TMA

Hopper（SM90）在 Ampere 的异步拷贝基础上引入了 **TMA（Tensor Memory Accelerator）**，进一步将数据搬运卸载到专用硬件单元：

| 📊 特性 | cp.async（Ampere） | TMA（Hopper） |
|---------|-------------------|--------------|
| 地址计算 | 每个线程独立计算地址 | TMA 硬件自动计算多维地址 |
| 发起方式 | 每个线程发起自己的拷贝 | 单个线程发起整个 tile 的拷贝 |
| 同步机制 | `wait_group` | `arrive/wait` on barrier |
| 数据布局 | 线性 | 支持多维 tensor 描述符 |
| Swizzle | 需手动处理 | 硬件自动 swizzle 避免 bank 冲突 |

### 6.2 Hopper 的 Barrier-based Pipeline

Hopper 使用 `cuda::barrier` 替代 `commit/wait_group` 来同步异步操作：

```cuda
#include <cuda/barrier>

__global__ void hopper_pipeline_kernel(...) {
    __shared__ float smem[NUM_STAGES][TILE_SIZE];
    __shared__ cuda::barrier bar[NUM_STAGES];

    if (threadIdx.x == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            init(&bar[s], blockDim.x);
        }
    }
    __syncthreads();

    for (int tile = 0; tile < num_tiles; tile++) {
        int stage = tile % NUM_STAGES;

        // 等待当前 stage 可用
        bar[stage].arrive_and_wait();

        // TMA 异步加载（仅线程 0 发起）
        if (threadIdx.x == 0) {
            cuda::memcpy_async(smem[stage], global_ptr + tile * TILE_SIZE,
                               sizeof(float) * TILE_SIZE, bar[stage]);
        }

        // 计算前一个 stage
        // ...
    }
}
```

📌 **关键点**：Hopper 的 TMA + Barrier 模式是 Ampere cp.async 模式的自然演进。理解 Ampere 的多阶段流水线是学习 Hopper 编程的基础。

## 7. 常见陷阱与调试技巧

### 7.1 典型错误

| ❌ 错误 | 📝 原因 | ✅ 修复 |
|---------|---------|---------|
| 数据读到旧值 | `wait_prior` 的参数 N 设置错误 | N 应为 `NUM_STAGES - 1`，表示最多允许 N 个 group 未完成 |
| Shared Memory 被覆盖 | 计算未完成就复用了同一 stage | 确保 `__syncthreads()` 在 commit 之前 |
| 性能无提升 | 拷贝粒度太小（如单个 float） | 使用 `float4`（16 字节）对齐拷贝 |
| 编译错误 | 未指定 SM80+ 架构 | 添加 `-arch=sm_80` 编译选项 |

### 7.2 wait_prior 参数详解

`__pipeline_wait_prior(N)` 的语义是：**等待直到未完成的 group 数量 ≤ N**。

```text
假设 NUM_STAGES = 3，当前已 commit 了 group 0, 1, 2, 3, 4

__pipeline_wait_prior(2):
  等待直到未完成 group ≤ 2
  → group 0, 1, 2 必须完成，group 3, 4 可以还在路上
  → 此时可以安全读取 group 0, 1, 2 对应的 smem 数据
```

⚠️ **注意**：`__pipeline_wait_prior(0)` 等价于等待所有已提交的 group 完成，是最保守的同步方式，但会破坏流水线效果。正确的值通常是 `NUM_STAGES - 1`。

### 7.3 对齐要求

`cp.async` 对源地址和目标地址有对齐要求：

- 拷贝 4 字节：源和目标 4 字节对齐
- 拷贝 8 字节：源和目标 8 字节对齐
- 拷贝 16 字节：源和目标 16 字节对齐

不满足对齐要求时，硬件会将请求拆分为多个较小粒度的事务来完成，功能上仍然正确，但性能会显著下降。始终确保地址对齐是最佳实践。

### 7.4 Nsight 性能验证

使用 Nsight Compute 验证流水线是否生效：

```bash
ncu --set full -o pipeline_profile ./my_kernel
```

关注以下指标：
- **L1/TEX Hit Rate**：`cp.async.cg` 应绕过 L1，`cp.async.ca` 应利用 L1
- **Shared Memory Throughput**：应接近理论峰值
- **Warp Stall - Wait**：如果 wait 占比过高，说明 stage 数不够或计算量不足以隐藏延迟

## 📝 总结

Ampere 之后的标准异步流水线写法是现代高性能 CUDA Kernel 的基础范式：

1. `cp.async` 实现 Global → Shared 的直接搬运，绕过寄存器，减少指令数和寄存器压力
2. 多阶段流水线（通常 3-4 个 stage）通过 `commit_group` / `wait_group` 实现细粒度同步
3. 标准结构为 Prologue（预填充）→ Main Loop（计算 + 预取交替）→ Epilogue（排空）
4. `cuda::pipeline` C++ API 提供了更清晰的 producer/consumer 语义
5. Hopper 的 TMA + Barrier 是这一模式的自然演进，但核心思想不变
6. 实践中注意 `wait_prior` 参数设置、地址对齐、以及拷贝粒度（优先 16 字节）

## 🎯 自我检验清单

- 能解释 `cp.async` 相比传统 `LDG + STS` 的三个核心优势
- 能画出 3-stage 异步流水线的时序图，标注 Prologue / Main Loop / Epilogue
- 能独立编写一个使用 `__pipeline_memcpy_async` 的多阶段流水线 Kernel
- 能正确设置 `__pipeline_wait_prior(N)` 的参数 N，并解释其语义
- 能说明 `NUM_STAGES` 选择的权衡（延迟隐藏 vs Shared Memory 占用）
- 能区分 `cp.async.ca` 和 `cp.async.cg` 的缓存行为差异
- 能使用 `cuda::pipeline` API 重写底层内建函数版本的流水线
- 能解释 Hopper TMA 相对于 Ampere cp.async 的改进点

## 📚 参考资料

- [NVIDIA CUDA C++ Programming Guide - Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-cuda-pipeline)
- [NVIDIA CUTLASS - Efficient GEMM Pipelining](https://github.com/NVIDIA/cutlass)
- [CUDA Samples - cp.async Pipeline](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/globalToShmemAsyncCopy)
- [GTC 2020 - Developing CUDA Kernels to Push Tensor Cores to the Absolute Limit on NVIDIA A100](https://developer.nvidia.com/gtc/2020/video/s21745)
- [Lei Mao's Blog - CUDA Shared Memory Async Copy](https://leimao.github.io/blog/CUDA-Shared-Memory-Async-Copy/)
