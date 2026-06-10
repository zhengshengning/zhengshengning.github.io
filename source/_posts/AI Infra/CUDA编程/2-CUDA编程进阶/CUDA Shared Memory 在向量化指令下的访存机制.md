---
title: CUDA Shared Memory 在向量化指令下的访存机制
date: 2026-05-28 15:00:00
mathjax: true
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程进阶]
tags: [CUDA, Shared Memory, Bank Conflict, 向量化访存, GPU架构]
---

本文深入探讨当使用向量化访存指令（LDS.64 / LDS.128）访问 Shared Memory 时，Memory Transaction 的拆分规则、广播机制的触发条件以及 Bank Conflict 的计算方式——这些内容在 NVIDIA 官方文档中几乎没有覆盖，却是高性能 Kernel 开发的关键。

<!-- more -->

## 📑 目录

- [1. 背景与动机](#1-背景与动机)
- [2. Shared Memory 基础模型回顾](#2-shared-memory-基础模型回顾)
- [3. 向量化指令下的访存机制](#3-向量化指令下的访存机制)
- [4. 64 位宽访存指令（LDS.64）](#4-64-位宽访存指令lds64)
- [5. 128 位宽访存指令（LDS.128）](#5-128-位宽访存指令lds128)
- [6. 实践验证方法](#6-实践验证方法)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)
- [参考资料](#-参考资料)

---

## 1. 背景与动机

NVIDIA GPU 的存储层次从慢到快依次为：Global Memory → L2 Cache → L1TEX Cache / Shared Memory → Register File。自 Volta/Turing 架构起，L1TEX Cache 与 Shared Memory 共享同一块片上 SRAM，二者拥有相近的访问延迟和带宽，这使得 Shared Memory 的高效利用成为 Kernel 性能优化的核心环节。

然而，NVIDIA 官方 CUDA 编程手册中关于 Shared Memory 的讲解，主要围绕**每个线程访问单个 4 字节（32 位）元素**的场景展开——在此场景下讨论 Bank Conflict 和广播（Broadcast）机制。但在实际高性能 Kernel 中，我们经常使用向量化访存指令：

- **LDS.64**：单线程一次读取 8 字节（64 位，相当于 2 个 Bank 宽度的元素）
- **LDS.128**：单线程一次读取 16 字节（128 位，相当于 4 个 Bank 宽度的元素）

对于这类向量化访存场景下 Shared Memory 的行为，官方文档几乎没有给出说明。本文结合社区讨论和 Microbenchmark 实验，试图还原向量化指令下 Shared Memory 的访存机制细节。

⚠️ **注意**：本文结论仅在 **Turing 架构** GPU 上验证，其他架构（Ampere、Hopper 等）的行为可能存在差异。

## 2. Shared Memory 基础模型回顾

在深入向量化访存之前，先回顾标准的 Shared Memory 访存模型。

### 2.1 存储结构

可以将 Shared Memory 抽象为一个长度为 $N$ 的数组，每个元素宽 4 字节（一个 `int` 或 `float`）。这块存储对同一 Thread Block 内的所有线程可见，不同 Block 间互不可见。

Shared Memory 的关键硬件特征：被划分为 **32 个 Bank**，数组中第 $i$ 个元素映射到第 $i \mod 32$ 个 Bank。

### 2.2 两条核心规则

Shared Memory 的访存行为归结为两条规则：

| 场景 | 行为 | 对性能的影响 |
|------|------|------------|
| 同一 Warp 中多个线程访问**同一元素** | 触发**广播机制**，合并为一次访问 | ✅ 无损耗 |
| 同一 Warp 中多个线程访问**同一 Bank 的不同元素** | 产生 **Bank Conflict**，访问被串行化 | ❌ 降低吞吐 |

### 2.3 Wavefront 与性能度量

在 Nsight Compute 中，Shared Memory 的访存性能通过 **Wavefront** 数目来衡量——Wavefront 越多，表示该访存指令需要越多的串行步骤来完成，性能越差。理想情况下一次 Shared Memory 访存只产生 1 个 Wavefront。

举几个 32 位访存的经典场景帮助回顾：

- **场景 A**：Warp 中多个线程读同一个地址 → 广播机制触发，无 Bank Conflict → 1 Wavefront
- **场景 B**：32 个线程各访问不同 Bank → 无冲突 → 1 Wavefront
- **场景 C**：多个线程访问同一 Bank 中的不同元素（如 Bank 18 被 4 个线程同时访问不同行） → 4-way Bank Conflict → 4 Wavefront

## 3. 向量化指令下的访存机制

### 3.1 为什么标准模型不再适用

标准 Shared Memory 模型以"每个线程访问一个元素"为前提进行分析。当使用 LDS.64 或 LDS.128 时，单个线程一次就读取了 2 个或 4 个连续的 4 字节元素，标准模型无法直接描述这种情况。

### 3.2 正确的分析粒度：Memory Transaction

分析向量化访存时，应以每个 **Memory Transaction** 为单位进行 Bank Conflict 判定，而非以 Warp 或指令为单位。

核心原则：

📌 **关键点**：一个 Warp 中 32 个线程同时执行一条 Shared Memory 访存指令时，会产生 **1 个或多个** Memory Transaction。单个 Memory Transaction 的最大数据量为 **128 字节**。当总访存需求超出 128 字节时，会被拆分为多个 Transaction 串行执行。

由于同一 Warp 内所有线程在同一时刻执行的访存指令位宽相同（不会出现线程 0 执行 LDS.32 而线程 1 执行 LDS.128 的情况），我们只需分别讨论 64 位和 128 位两种情况。

## 4. 64 位宽访存指令（LDS.64）

### 4.1 Transaction 拆分规则

对于 LDS.64 指令，Transaction 的拆分以 **Half-Warp**（16 个线程）为基本单位：

$$
\text{Transaction 数} = \text{活跃 Half-Warp 数} - \text{广播合并数}
$$

具体规则：

- 一个 Warp 被分为 2 个 Half-Warp（线程 0\~15 和线程 16\~31）
- 每个 Half-Warp 内只要有至少 1 个活跃线程，就产生 1 个 Transaction
- 若满足广播条件，两个 Half-Warp 的 Transaction 可合并为 1 个

### 4.2 广播机制的触发条件

在 64 位访存场景下，广播合并需要满足以下两个条件中的**至少一个**（且必须是所有活跃线程**全局**满足同一条件）：

- **条件 1**：对于 Warp 内所有活跃的第 $i$ 号线程，第 $i \oplus 1$ 号线程要么不活跃，要么与其访问相同地址
- **条件 2**：对于 Warp 内所有活跃的第 $i$ 号线程，第 $i \oplus 2$ 号线程要么不活跃，要么与其访问相同地址

其中 $\oplus$ 表示按位异或（XOR）运算。

💡 **提示**：这里的"全局满足同一条件"是关键约束——如果前半 Warp 满足条件 1 但后半 Warp 满足条件 2（而非条件 1），则广播合并**不会**触发。

### 4.3 实例分析

**Case 1：所有活跃线程在同一 Half-Warp 内**

假设只有线程 0\~15 中的部分线程活跃（线程 16\~31 全部不活跃），则只有 1 个 Half-Warp 活跃：

- Transaction 数 = 1
- Bank Conflict 在该 Transaction 内独立计算
- Wavefront = 1（无 Bank Conflict 时）

**Case 2：活跃线程跨越两个 Half-Warp，无广播**

假设线程 15 和线程 16 分别活跃（分属两个 Half-Warp），且访问不同地址：

- 两个 Half-Warp 各有活跃线程 → 产生 2 个 Transaction
- 无 Bank Conflict 时 Wavefront = 2

**Case 3：跨 Half-Warp 但触发广播条件 1**

假设活跃线程分散在两个 Half-Warp 中，但对所有活跃线程 $i$，线程 $i \oplus 1$ 要么不活跃，要么地址相同：

- 满足广播条件 1 → 两个 Transaction 合并为 1
- Wavefront = 1

**Case 4：看似满足广播条件但实际不满足**

前半 Warp 中的线程满足条件 1（XOR 1 邻居同地址），后半 Warp 中的线程满足条件 2（XOR 2 邻居同地址），但**没有一个条件被全局满足**：

- 不触发广播合并
- Transaction 数 = 2，Wavefront = 2

**Case 5：跨 Half-Warp，不满足任何广播条件**

活跃线程分散在两个 Half-Warp，地址各不相同且 XOR 邻居关系不满足广播条件：

- Transaction 数 = 2，Wavefront = 2

### 4.4 Bank Conflict 的计算

在 64 位访存下，Bank Conflict 的计算方式是：

1. 先确定 Transaction 的数量和划分
2. **在每个 Transaction 内部**独立计算 Bank Conflict
3. 将所有 Transaction 的 Wavefront 累加（因为 Transaction 间串行执行）

### 4.5 验证代码

以下代码可配合 Nsight Compute 验证上述 5 个 Case：

```cuda
#include <cstdint>

__global__ void smem_64_case1(uint32_t *a) {
    __shared__ uint32_t smem[64];
    uint32_t tid = threadIdx.x;
    smem[tid] = tid;
    smem[tid + 32] = tid + 32;
    __syncthreads();
    // 只有线程 0~15 活跃，都在第 1 个 Half-Warp
    if (tid < 16) {
        reinterpret_cast<uint2 *>(a)[tid] =
            reinterpret_cast<const uint2 *>(smem)[tid];
    }
}

__global__ void smem_64_case2(uint32_t *a) {
    __shared__ uint32_t smem[64];
    uint32_t tid = threadIdx.x;
    smem[tid] = tid;
    smem[tid + 32] = tid + 32;
    __syncthreads();
    // 线程 15 和 16 活跃，跨越两个 Half-Warp
    if (tid == 15 || tid == 16) {
        reinterpret_cast<uint2 *>(a)[tid] =
            reinterpret_cast<const uint2 *>(smem)[tid];
    }
}

__global__ void smem_64_case3(uint32_t *a) {
    __shared__ uint32_t smem[64];
    uint32_t tid = threadIdx.x;
    smem[tid] = tid;
    smem[tid + 32] = tid + 32;
    __syncthreads();
    // 满足广播条件 1：XOR 1 邻居同地址
    if (tid == 0 || tid == 1 || tid == 16 || tid == 17) {
        reinterpret_cast<uint2 *>(a)[tid] =
            reinterpret_cast<const uint2 *>(smem)[tid & ~1u];
    }
}

__global__ void smem_64_case4(uint32_t *a) {
    __shared__ uint32_t smem[64];
    uint32_t tid = threadIdx.x;
    smem[tid] = tid;
    smem[tid + 32] = tid + 32;
    __syncthreads();
    // 前半满足条件 1(XOR 1 邻居同地址)但不满足条件 2
    // 后半满足条件 2(XOR 2 邻居同地址)但不满足条件 1
    // 全局：条件 1 和条件 2 都不满足 → 不触发广播
    if (tid < 4 || (tid >= 16 && tid < 20)) {
        uint32_t addr;
        if (tid < 16) {
            addr = tid / 2;      // 0,0,1,1 → XOR 1 同地址, XOR 2 不同
        } else {
            addr = 8 + tid % 2;  // 8,9,8,9 → XOR 2 同地址, XOR 1 不同
        }
        reinterpret_cast<uint2 *>(a)[tid] =
            reinterpret_cast<const uint2 *>(smem)[addr];
    }
}

__global__ void smem_64_case5(uint32_t *a) {
    __shared__ uint32_t smem[64];
    uint32_t tid = threadIdx.x;
    smem[tid] = tid;
    smem[tid + 32] = tid + 32;
    __syncthreads();
    // 跨 Half-Warp，无广播
    if (tid == 0 || tid == 16) {
        reinterpret_cast<uint2 *>(a)[tid] =
            reinterpret_cast<const uint2 *>(smem)[tid];
    }
}

int main() {
    uint32_t *d_a;
    cudaMalloc(&d_a, sizeof(uint32_t) * 64);
    smem_64_case1<<<1, 32>>>(d_a);
    smem_64_case2<<<1, 32>>>(d_a);
    smem_64_case3<<<1, 32>>>(d_a);
    smem_64_case4<<<1, 32>>>(d_a);
    smem_64_case5<<<1, 32>>>(d_a);
    cudaFree(d_a);
    cudaDeviceSynchronize();
    return 0;
}
```

使用 Nsight Compute 的 profiling 命令：

```bash
ncu --set full -k smem_64 ./your_binary
```

观察 `l1tex__data_pipe_lsu_wavefronts_mem_shared` 指标即可验证。

## 5. 128 位宽访存指令（LDS.128）

### 5.1 Transaction 拆分规则

LDS.128 的分析层次比 LDS.64 多一级，采用**两级划分**：

| 层级 | 单位 | 线程范围 | 作用 |
|------|------|---------|------|
| 第一级 | Half-Warp | 线程 0\~15 / 16\~31 | 独立计算各自的 Transaction 数 |
| 第二级 | Quarter-Warp | 每 8 个线程一组 | 确定每个 Half-Warp 内的 Transaction 数 |

规则为：对于每个 Half-Warp，其产生的 Transaction 数等于该 Half-Warp 内**活跃 Quarter-Warp 的数目**（减去广播合并数）。整个 Warp 的 Transaction 总数是两个 Half-Warp 的 Transaction 数之和。

### 5.2 广播机制

128 位访存的广播条件与 64 位相同（条件 1：XOR 1 或条件 2：XOR 2），且以**整个 Warp** 为单位判定。如果广播触发，每个 Half-Warp 内的两个 Quarter-Warp 的 Transaction 可合并为一个。

📌 **关键点**：即使 32 个线程全部活跃且访问完全相同的地址，由于广播合并只能将 Quarter-Warp 对合并（不能跨 Half-Warp 合并），最终仍然需要至少 **2 个 Transaction**（每个 Half-Warp 各 1 个）。

### 5.3 实例分析

**Case 1：2 个 Half-Warp 各 1 个 Quarter-Warp 活跃**

线程 15（Half-Warp 0 的 Quarter-Warp 1）和线程 16（Half-Warp 1 的 Quarter-Warp 0）活跃：

- Half-Warp 0：1 个 Quarter-Warp 活跃 → 1 Transaction
- Half-Warp 1：1 个 Quarter-Warp 活跃 → 1 Transaction
- 总计 2 Transaction，Wavefront = 2

**Case 2：1 个 Half-Warp 的 2 个 Quarter-Warp 都活跃，触发广播**

线程 0 和线程 15 活跃（同属 Half-Warp 0，分属 Quarter-Warp 0 和 1），满足广播条件：

- 2 个 Quarter-Warp 合并为 1 个 Transaction
- 仅 Half-Warp 0 活跃，总计 1 Transaction
- Wavefront = 1

**Case 3：所有线程活跃，满足广播条件**

32 个线程全部活跃，XOR 1 邻居访问相同地址：

- Half-Warp 0：2 个 Quarter-Warp 活跃，广播合并 → 1 Transaction
- Half-Warp 1：2 个 Quarter-Warp 活跃，广播合并 → 1 Transaction
- 总计 2 Transaction，Wavefront = 2

**Case 4：所有线程活跃，不满足广播条件**

32 个线程全部活跃，地址各不相同且不满足广播条件：

- Half-Warp 0：2 个 Quarter-Warp → 2 Transaction
- Half-Warp 1：2 个 Quarter-Warp → 2 Transaction
- 总计 4 Transaction，Wavefront = 4

**Case 5：广播触发但存在 Bank Conflict**

32 个线程活跃，同时满足条件 1 和条件 2 触发广播，但合并后的 Transaction 内部产生 Bank Conflict：

- 原本每个 Half-Warp 合并为 1 Transaction（共 2 个）
- 但每个 Transaction 内有 1 个 Bank Conflict → 实际拆分为 2 Wavefront
- 总计 4 Wavefront

**Case 6：所有线程活跃，无广播，无 Bank Conflict**

4 个 Quarter-Warp 各独立产生 1 个 Transaction：

- 总计 4 Transaction，Wavefront = 4

### 5.4 验证代码

```cuda
#include <cstdint>

__global__ void smem_128_case1(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    if (tid == 15 || tid == 16) {
        reinterpret_cast<uint4 *>(a)[tid] =
            reinterpret_cast<const uint4 *>(smem)[4];
    }
}

__global__ void smem_128_case2(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    if (tid == 0 || tid == 15) {
        reinterpret_cast<uint4 *>(a)[tid] =
            reinterpret_cast<const uint4 *>(smem)[4];
    }
}

__global__ void smem_128_case3(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
        smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}

__global__ void smem_128_case4(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    uint32_t addr;
    if (tid < 16) {
        addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
    } else {
        addr = (tid / 8) * 2 + ((tid % 8) % 2);
    }
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[addr];
}

__global__ void smem_128_case5(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[(tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8];
}

__global__ void smem_128_case6(uint32_t *a) {
    __shared__ uint32_t smem[128];
    uint32_t tid = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        smem[i * 32 + tid] = tid;
    }
    __syncthreads();
    uint32_t addr = (tid / 16) * 4 + (tid % 16 / 8) * 8;
    if (tid < 16) {
        addr += (tid % 4 / 2) * 2;
    } else {
        addr += (tid % 4 % 2) * 2;
    }
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[addr];
}

int main() {
    uint32_t *d_a;
    cudaMalloc(&d_a, sizeof(uint32_t) * 128);
    smem_128_case1<<<1, 32>>>(d_a);
    smem_128_case2<<<1, 32>>>(d_a);
    smem_128_case3<<<1, 32>>>(d_a);
    smem_128_case4<<<1, 32>>>(d_a);
    smem_128_case5<<<1, 32>>>(d_a);
    smem_128_case6<<<1, 32>>>(d_a);
    cudaFree(d_a);
    cudaDeviceSynchronize();
    return 0;
}
```

## 6. 实践验证方法

### 6.1 使用 Nsight Compute 验证

通过以下步骤可验证上述所有 Case：

```bash
# 编译（以 Turing 架构为例）
nvcc -arch=sm_75 -o smem_test smem_test.cu

# 使用 Nsight Compute 分析
ncu --set full --kernel-name-base function ./smem_test
```

重点关注指标：

| 指标名 | 含义 |
|--------|------|
| `l1tex__data_pipe_lsu_wavefronts_mem_shared` | Shared Memory 访存的 Wavefront 总数 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | Bank Conflict 次数 |

### 6.2 规则速查表

| 指令位宽 | 拆分单位 | Transaction 计算方式 | 广播作用范围 |
|---------|---------|---------------------|------------|
| LDS.32 | Warp（32 线程） | 1 Transaction / Warp | 同地址元素合并 |
| LDS.64 | Half-Warp（16 线程） | 活跃 Half-Warp 数 | 两个 Half-Warp 合并 |
| LDS.128 | Quarter-Warp（8 线程） | 每个 Half-Warp 内活跃 Quarter-Warp 数之和 | 同 Half-Warp 内 Quarter-Warp 合并 |

## 📝 总结

向量化指令下 Shared Memory 的访存机制可以用三句话概括：

1. **拆分粒度随位宽升级**：LDS.64 以 Half-Warp 为单位产生 Transaction，LDS.128 以 Quarter-Warp 为单位（在每个 Half-Warp 内独立计算）
2. **广播条件统一但判定全局**：无论位宽如何，广播触发条件都是 XOR 1 或 XOR 2 邻居地址相同（或不活跃），且必须整个 Warp 全局满足同一条件
3. **Bank Conflict 按 Transaction 独立计算后累加**：先确定 Transaction 划分，再在每个 Transaction 内部套用标准 Bank Conflict 规则

掌握这些规则后，在编写使用 `uint2`（64 位）或 `uint4`/`float4`（128 位）向量类型的 Kernel 时，就能准确预判 Shared Memory 的访存效率，从而做出更好的数据布局和访问模式设计决策。

## 🎯 自我检验清单

- 能说出 Shared Memory 的 32 Bank 映射规则和 Bank Conflict 的产生条件
- 能解释 Wavefront 指标的含义及其与性能的关系
- 能计算 LDS.64 指令下一个 Warp 产生的 Memory Transaction 数
- 能判断 LDS.64 场景下广播条件是否满足（XOR 1 / XOR 2 规则）
- 能计算 LDS.128 指令下每个 Half-Warp 产生的 Transaction 数
- 能区分"前半 Warp 满足条件 1 + 后半满足条件 2"与"全局满足条件 1"的不同结果
- 能使用 Nsight Compute 的 `wavefronts_mem_shared` 指标验证分析结论
- 能根据 Transaction 拆分规则独立计算带 Bank Conflict 的 Wavefront 总数
- 能在实际 Kernel 中选择合适的向量化宽度以最小化 Wavefront 数

## 📚 参考资料

- [NVIDIA CUDA C++ Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [CUDA Shared Memory Bank Conflicts - Stack Overflow Discussion](https://stackoverflow.com/questions/3841877/what-is-a-bank-conflict-doing-cuda-opencl-programming)
- [Understanding Shared Memory Bank Conflicts in Vectorized Access - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/shared-memory-bank-conflict-for-64-bit-and-128-bit-access/283718)
