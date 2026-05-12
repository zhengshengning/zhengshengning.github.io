---
title: CUDA 13.1新特性：Tile编程模型
date: 2026-05-11 15:00:00
mathjax: true
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程高阶]
tags: [CUDA, cuTile, Tensor Core, GPU编程, 性能优化]
---

NVIDIA 在 CUDA 13.1 中推出了 **cuTile**——一种全新的 Tile 编程模型，被称为自 CUDA 诞生以来最重大的编程范式革新。它让开发者用"数据块"而非"单个线程"来思考 GPU 编程，自动利用 Tensor Core 和 TMA 等硬件加速单元，大幅降低高性能 Kernel 的编写门槛。本文从概念到实战，全面拆解这一新编程模型。

<!-- more -->

## 📑 目录

- [1. 为什么需要 Tile 编程模型](#1-为什么需要-tile-编程模型)
- [2. cuTile 核心概念](#2-cutile-核心概念)
- [3. 环境搭建与安装](#3-环境搭建与安装)
- [4. 第一个 cuTile Kernel：向量加法](#4-第一个-cutile-kernel向量加法)
- [5. 核心 API 详解](#5-核心-api-详解)
- [6. 实战：矩阵乘法 Kernel](#6-实战矩阵乘法-kernel)
- [7. 进阶优化技巧](#7-进阶优化技巧)
- [8. cuTile 与传统 CUDA 编程对比](#8-cutile-与传统-cuda-编程对比)
- [9. 架构支持与生态](#9-架构支持与生态)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)
- [参考资料](#-参考资料)

---

## 1. 为什么需要 Tile 编程模型

### 1.1 传统 SIMT 模型的痛点

传统 CUDA 编程采用 **SIMT**（Single Instruction, Multiple Threads）模型，开发者需要从单个线程的视角出发编写代码。写一个矩阵乘法 Kernel，你至少需要处理以下工作：

- 手动计算 `threadIdx.x + blockIdx.x * blockDim.x` 来确定每个线程的全局索引
- 手动选择 Block 大小并进行越界检查
- 手动管理 Shared Memory 的分配、加载、同步
- 手动调用 `wmma` 或 `mma.sync` PTX 指令来使用 Tensor Core
- 手动使用 TMA（Tensor Memory Accelerator）进行异步数据搬运

这就好比你要指挥一千个工人搬砖，传统 SIMT 要求你给每个工人写一份独立的指令——"你是第 347 号，你负责从第 347 个位置取砖，放到第 347 个位置"。工人越多，指令越复杂。

### 1.2 Tile 编程模型的思路转变

cuTile 换了一种思路：**你只需要说"把这一块砖墙搬过去"**，至于怎么分配工人、每个人搬哪块砖，由编译器自动决定。

具体来说，cuTile 将编程抽象从**线程级别**提升到了**数据块（Tile）级别**：

| 🔑 维度 | SIMT 模型 | Tile 模型 |
|---------|-----------|-----------|
| 编程视角 | 单个线程 | 数据块（Tile） |
| 索引计算 | 手动 `threadIdx + blockIdx * blockDim` | 自动，按 Tile 坐标寻址 |
| Block 大小 | 开发者指定 | 编译器自动选择 |
| Tensor Core | 手动调用 wmma/mma 指令 | `ct.mma()` 自动映射 |
| 内存管理 | 手动分配 Shared Memory | 运行时自动管理 |
| 越界处理 | 手动 `if (idx < N)` | 自动 padding |

---

## 2. cuTile 核心概念

cuTile 的编程模型围绕五个核心概念构建：

### 2.1 Array（数组）

Array 是存储在 GPU 设备内存（HBM）中的多维张量，是 Kernel 操作的数据源。cuTile 兼容 **CuPy** 和 **PyTorch** 的张量对象，不需要自定义数据类型。

### 2.2 Tile（数据块）

Tile 是 cuTile 的灵魂概念——**一个固定形状的数据切片**，是 Kernel 中一切计算和数据搬运的基本单元。

> 把 Array 想象成一整面墙，Tile 就是你从墙上切下来的一块固定大小的瓷砖。Kernel 的工作就是：取一块瓷砖、加工、贴回去。

Tile 的形状在编译时确定，例如一个 `(128, 64)` 的 2D Tile 表示 128 行 64 列的数据块。

### 2.3 Kernel（核函数）

用 `@ct.kernel` 装饰器标记的 Python 函数，定义了每个 Block 对 Tile 执行的操作逻辑。

```python
@ct.kernel
def my_kernel(input_tensor, output_tensor):
    # 每个Block执行这段逻辑，操作的是Tile而非单个元素
    tile = ct.load(input_tensor, index=(ct.bid(0),), shape=(TILE_SIZE,))
    result = tile * 2.0
    ct.store(output_tensor, index=(ct.bid(0),), tile=result)
```

### 2.4 Block（执行单元）

与传统 CUDA 的 Thread Block 对应，但在 cuTile 中你不需要关心 Block 内部的线程组织。编译器根据 Tile 形状和硬件资源自动决定最优的 Block 大小和线程分配方案。

### 2.5 Compile-time Constant（编译期常量）

用 `ct.Constant[int]` 或 `ct.Constant[bool]` 声明的参数，在编译时确定值。编译器利用这些常量进行循环展开、选择最优的 Tensor Core 指令等优化。

```python
@ct.kernel
def gemm_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    # tm, tn, tk 在编译时已知，编译器据此生成特化的高效代码
    ...
```

---

## 3. 环境搭建与安装

### 3.1 硬件与驱动要求

- **GPU**：Blackwell 架构（B200、RTX 5080/5090）— CUDA 13.1+ 支持
- **驱动**：NVIDIA Driver r580 或更高版本
- **Python**：3.10+

💡 **提示**：CUDA 13.2 进一步将支持扩展到了 Ampere（A100）和 Ada Lovelace（RTX 4090）架构。Hopper（H100）的支持将在后续版本中加入。

### 3.2 安装方式

**方式一：pip 安装（推荐）**

```bash
# 安装 cuTile 及 tileiras 编译器
pip install cuda-tile[tileiras]

# 安装 CuPy 用于设备内存管理
pip install cupy-cuda13x
```

**方式二：搭配系统 CUDA Toolkit**

如果系统已安装 CUDA Toolkit 13.1+，可以单独安装 Python 包：

```bash
pip install cuda-tile
```

**方式三：Debian 系统包**

```bash
apt-get install cuda-tileiras-13.2 cuda-compiler-13.2
```

### 3.3 验证安装

```python
import cuda.tile as ct
print(ct.__version__)  # 应输出 1.x.x
```

---

## 4. 第一个 cuTile Kernel：向量加法

用一个最简单的向量加法来感受 cuTile 的编程方式。

### 4.1 cuTile 实现

```python
import cuda.tile as ct
import cupy
import numpy as np

TILE_SIZE = 16

@ct.kernel
def vector_add(a, b, result):
    # 1. 获取当前Block的ID
    block_id = ct.bid(0)

    # 2. 按Tile粒度加载数据
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))

    # 3. Tile级别的加法运算
    result_tile = a_tile + b_tile

    # 4. 将结果Tile写回
    ct.store(result, index=(block_id,), tile=result_tile)

# 准备数据
N = 128
rng = cupy.random.default_rng(42)
a = rng.random(N, dtype='float32')
b = rng.random(N, dtype='float32')
result = cupy.zeros(N, dtype='float32')

# 计算Grid大小并启动Kernel
grid = (ct.cdiv(N, TILE_SIZE), 1, 1)  # cdiv = 向上取整除法
ct.launch(cupy.cuda.get_current_stream(), grid, vector_add, (a, b, result))

# 验证结果
expected = cupy.asnumpy(a) + cupy.asnumpy(b)
np.testing.assert_array_almost_equal(cupy.asnumpy(result), expected)
print("验证通过！")
```

### 4.2 与传统 CUDA 对比

来看同一个功能的传统 CUDA C++ 实现：

```cpp
// 传统CUDA：需要手动处理线程索引和越界检查
__global__ void vector_add(float* a, float* b, float* result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {  // 必须手动越界检查
        result[idx] = a[idx] + b[idx];
    }
}

// 启动时必须手动指定Block大小
vector_add<<<(N + 255) / 256, 256>>>(a, b, result, N);
```

对比要点：

- ✅ cuTile 无需计算线程索引，直接用 `ct.bid(0)` 获取 Block ID
- ✅ cuTile 无需手动越界检查，`ct.load` 支持自动 padding
- ✅ cuTile 无需指定 Block 大小，编译器自动选择
- ❌ 传统 CUDA 需要手动 `threadIdx.x + blockIdx.x * blockDim.x`
- ❌ 传统 CUDA 需要 `if (idx < N)` 防止越界

---

## 5. 核心 API 详解

### 5.1 Kernel 定义与启动

```python
# 定义Kernel
@ct.kernel
def my_kernel(tensor_a, tensor_b, scalar_param: ct.Constant[int]):
    ...

# 启动Kernel
ct.launch(stream, grid, my_kernel, (arg1, arg2, const_val))
```

| 📊 参数 | 📝 说明 |
|---------|---------|
| `stream` | CUDA Stream 对象（CuPy 或 PyTorch 的 stream） |
| `grid` | 三元组 `(grid_x, grid_y, grid_z)`，定义 Block 网格 |
| `my_kernel` | 被 `@ct.kernel` 装饰的函数 |
| `args` | Kernel 参数元组 |

### 5.2 数据搬运

**加载 Tile**：

```python
tile = ct.load(tensor, index=(bx, by), shape=(M, N))
```

- `tensor`：源张量（CuPy/PyTorch 设备张量）
- `index`：Tile 空间坐标，而非元素坐标。`(bx, by)` 表示第 `bx` 行、第 `by` 列的 Tile
- `shape`：Tile 的形状，如 `(128, 64)` 表示加载 128×64 的数据块

`ct.load` 的高级选项：

```python
# 转置加载：交换最后两个维度
tile_t = ct.load(B, index=(k, bidy), shape=(tk, tn), order=(0, 1, 3, 2))

# 延迟提示：用于软件流水线预取
tile = ct.load(A, index=(bidx, k), shape=(tm, tk), latency=4)

# 自动越界填零
tile = ct.load(tensor, index=(bid,), shape=(TILE,), padding_mode="zero_pad")
```

**存储 Tile**：

```python
ct.store(tensor, index=(bx, by), tile=result_tile)
```

💡 **提示**：`ct.load` / `ct.store` 在硬件支持时自动使用 **TMA**（Tensor Memory Accelerator）进行异步批量数据搬运，无需手动编排。

### 5.3 计算操作

**矩阵乘加（MMA）**：

```python
# 自动映射到Tensor Core
accumulator = ct.mma(tile_a, tile_b, accumulator)
```

这是 cuTile 最核心的优势之一——调用 `ct.mma()` 时，编译器根据操作数的形状和数据类型，自动选择最优的 Tensor Core 指令（如 Blackwell 的 FP16 MMA 或 FP8 MMA），开发者无需了解底层 PTX 指令。

**逐元素运算**：

```python
c = a + b       # 加法
c = a * b       # 逐元素乘
c = a - b       # 减法
c = ct.truediv(a, b)  # 除法（支持fast-math标志）
```

**归约操作**：

```python
row_max = ct.max(tile, axis=1)    # 按行取最大值
col_sum = ct.sum(tile, axis=0)    # 按列求和
```

**特殊数学函数**：

```python
result = ct.exp2(tile)                # 2^x 快速指数
result = ct.exp2(tile, flush_to_zero=True)  # 非规格化数置零，避免慢速微码
```

### 5.4 Tile 创建与类型转换

```python
# 创建常量Tile
zeros = ct.full((128, 64), 0, dtype=ct.float32)

# 生成索引范围
indices = ct.arange(0, 128)

# 条件选择
result = ct.where(mask_tile, tile_a, tile_b)

# 类型转换
fp16_tile = ct.astype(fp32_tile, ct.float16)
```

### 5.5 网格信息

```python
bx = ct.bid(0)  # X维度的Block ID
by = ct.bid(1)  # Y维度的Block ID
```

### 5.6 辅助工具

```python
# 向上取整除法：cdiv(7, 4) = 2
num_blocks = ct.cdiv(N, TILE_SIZE)

# 计算某个轴上的Tile数量
num_k_tiles = ct.num_tiles(A, axis=1, shape=(tm, tk))
```

---

## 6. 实战：矩阵乘法 Kernel

矩阵乘法是 GPU 编程的"Hello World Plus"，也是展示 cuTile 威力的最佳场景。

### 6.1 基础版本

```python
import cuda.tile as ct

@ct.kernel
def matmul_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    """
    A: (M, K), B: (K, N), C: (M, N)
    tm, tn: 输出Tile的行列大小
    tk: K维度的分块大小
    """
    # 每个Block负责C矩阵中一个 (tm, tn) 大小的Tile
    # 使用swizzle将1D Block ID映射为2D坐标（见7.1节）
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    # K维度需要遍历多少个Tile
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # 初始化累加器（float32保证数值精度）
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

    # 沿K维度迭代
    for k in range(num_tiles_k):
        # 加载A和B的Tile
        a_tile = ct.load(A, index=(bidx, k), shape=(tm, tk))
        b_tile = ct.load(B, index=(k, bidy), shape=(tk, tn))

        # 矩阵乘加 → 自动使用Tensor Core
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    # 将结果写回C
    ct.store(C, index=(bidx, bidy), tile=accumulator)
```

### 6.2 启动 Kernel

```python
import torch

M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)
C = torch.zeros(M, N, device='cuda', dtype=torch.float32)

# Tile尺寸
tm, tn, tk = 128, 128, 32

# Grid大小 = 输出矩阵被划分为多少个Tile
grid_x = ct.cdiv(M, tm)
grid_y = ct.cdiv(N, tn)
grid = (grid_x * grid_y, 1, 1)

ct.launch(torch.cuda.current_stream(), grid, matmul_kernel, (A, B, C, tm, tn, tk))
```

### 6.3 计算流程图解

{% mermaid graph TD %}
    A["Global Memory: A (M×K)"] --> L1["ct.load → A Tile (tm×tk)"]
    B["Global Memory: B (K×N)"] --> L2["ct.load → B Tile (tk×tn)"]
    L1 --> MMA["ct.mma(a, b, acc) → Tensor Core"]
    L2 --> MMA
    MMA --> ACC["Accumulator (tm×tn, FP32)"]
    ACC -->|"K轮迭代"| MMA
    ACC -->|"迭代完成"| ST["ct.store → C Tile (tm×tn)"]
    ST --> C["Global Memory: C (M×N)"]
{% endmermaid %}

每个 Block 沿 K 维度循环加载 A 和 B 的 Tile，通过 `ct.mma` 在 Tensor Core 上执行矩阵乘加，最终将累加结果写回全局内存中对应位置。

---

## 7. 进阶优化技巧

### 7.1 2D Block Swizzle

当 Grid 很大时，相邻 Block 访问的数据可能分布在 HBM 的不同位置，L2 Cache 命中率低。**Swizzle** 重新映射 Block ID 到 2D 坐标，让相邻 Block 访问相邻数据。

```python
def swizzle_2d(M, N, tm, tn, group_size):
    """将1D的Block ID重映射为2D坐标，提升L2缓存命中率"""
    grid_m = ct.cdiv(M, tm)
    grid_n = ct.cdiv(N, tn)

    linear_id = ct.bid(0)

    # 按group_size对行进行分组
    group_id = linear_id // (group_size * grid_n)
    first_row_in_group = group_id * group_size
    group_size_actual = min(grid_m - first_row_in_group, group_size)

    within_group = linear_id % (group_size_actual * grid_n)
    bidx = first_row_in_group + (within_group % group_size_actual)
    bidy = within_group // group_size_actual

    return bidx, bidy
```

> Swizzle 的效果类似于把一维流水线重排成二维网格——原来 Block 0、1、2、3 沿一行排开，现在变成 2×2 的方阵，相邻 Block 在空间上也相邻，更容易复用 L2 Cache 中已经加载的数据。

📌 **关键点**：在 RTX 5080 上的 FP16 GEMM 测试中，2D Swizzle 可将内存访问总量降低约 **20\%**，直接提升 Kernel 吞吐。

### 7.2 Fast-Math 标志

cuTile 支持在运算中开启快速数学模式，以微小的精度代价换取显著的性能提升：

```python
# flush_to_zero: 将非规格化浮点数（denormals）置为零
# 避免硬件进入慢速微码路径处理极小值
result = ct.exp2(tile, flush_to_zero=True)

# 近似计算模式：跳过硬件近似后的迭代精化步骤
result = ct.truediv(a, b, rounding_mode=ct.RMd.APPROX)
```

| ⚙️ 标志 | 📝 作用 | 适用场景 |
|---------|---------|---------|
| `flush_to_zero=True` | 非规格化数置零 | 几乎所有深度学习场景 |
| `rounding_mode=APPROX` | 跳过迭代精化 | 不需要严格精度的中间计算 |

### 7.3 软件流水线预取

通过 `latency` 参数提示编译器提前预取下一轮迭代的数据，隐藏内存延迟：

```python
for k in range(num_tiles_k):
    # latency=4 提示编译器提前4轮迭代预取数据
    a_tile = ct.load(A, index=(bidx, k), shape=(tm, tk), latency=4)
    b_tile = ct.load(B, index=(k, bidy), shape=(tk, tn), latency=4)
    accumulator = ct.mma(a_tile, b_tile, accumulator)
```

### 7.4 自动调优（Autotuning）

不同输入规模下的最优 Tile 尺寸不同。cuTile 提供实验性的自动调优功能：

```python
from cuda.tile_experimental import autotune_launch, clear_autotune_cache

# autotune_launch 会遍历多组Tile参数，选择最快的配置
# 结果按张量形状缓存，后续调用零开销
autotune_launch(
    stream, grid, matmul_kernel,
    (A, B, C),
    tune_params={
        'tm': [64, 128, 256],
        'tn': [64, 128, 256],
        'tk': [16, 32, 64],
    }
)
```

⚠️ **注意**：`autotune_launch` 目前处于实验阶段（`cuda.tile_experimental`），API 可能在后续版本中变动。

### 7.5 K 循环拆分

对于长序列的 Flash Attention 等场景，可以将 K 维度的循环拆分为多个阶段，让编译器更好地调度计算和内存操作的重叠：

```python
@ct.kernel
def flash_attn_kernel(Q, K, V, O, tm: ct.Constant[int], tn: ct.Constant[int],
                      split_k: ct.Constant[bool]):
    if split_k:
        # 前半段和后半段分别处理，减少寄存器压力
        ...
    else:
        # 常规路径
        ...
```

---

## 8. cuTile 与传统 CUDA 编程对比

### 8.1 编程复杂度对比

以矩阵乘法为例：

| 📊 方面 | 传统 CUDA C++ | cuTile Python |
|---------|--------------|---------------|
| 索引计算 | 手动计算行列偏移、Tile 内坐标 | `ct.bid()` + `ct.load()` 自动寻址 |
| Shared Memory | `__shared__` 声明 + 手动加载 + `__syncthreads()` | 编译器自动管理 |
| Tensor Core | 调用 `wmma` API 或内联 PTX | `ct.mma()` 一行搞定 |
| TMA 搬运 | 手动配置 TMA 描述符 + 异步拷贝 | `ct.load()` 自动触发 |
| 越界处理 | `if (row < M && col < N)` | `padding_mode` 自动处理 |
| Block 大小 | 手动调参 128/256/512 | 编译器自动选择 |
| 代码行数 | 100-200 行 | 20-40 行 |

### 8.2 性能表现

在 RTX 5080 上，FP16 矩阵乘法的性能测试结果：

- cuTile 的矩阵乘法 Kernel 可达 **PyTorch/cuBLAS 性能的 90\% 以上**（矩阵大小 1024 到 16384）
- 在 B200 上进行 Flash Attention（FP16，Causal），优化后的 cuTile Kernel 达到 **918 TFLOPS**（seqlen=16384）
- 经过 Swizzle + Fast-Math + K 循环拆分 + Autotuning 的完整优化栈，相比朴素 cuTile 实现有 **1.6 倍加速**

### 8.3 适用场景分析

✅ **cuTile 适合的场景**：

- 快速原型开发和算法验证
- 深度学习算子开发（GEMM、Attention、Softmax、RMSNorm 等）
- 需要快速利用 Tensor Core 而不想深入 PTX 的场景
- 跨架构可移植的 Kernel（同一代码跑 Blackwell 和 Ampere）

❌ **仍需传统 CUDA 的场景**：

- 极致性能调优（需要控制到 warp 级别的细节）
- 非规则计算模式（稀疏矩阵、图计算等不适合 Tile 抽象的场景）
- 需要与现有 C++ CUDA 代码库深度集成

---

## 9. 架构支持与生态

### 9.1 硬件支持矩阵

| 🖥️ GPU 架构 | 代表型号 | Compute Capability | 最低 CUDA 版本 |
|------------|---------|-------------------|---------------|
| Blackwell（数据中心） | B200 | 10.0 | CUDA 13.1 |
| Blackwell（消费级） | RTX 5080/5090 | 12.0 | CUDA 13.1 |
| Ampere | A100, A10 | 8.x | CUDA 13.2 |
| Ada Lovelace | RTX 4090, L40 | 8.x | CUDA 13.2 |
| Hopper | H100, H200 | 9.x | 后续版本支持 |

💡 **提示**：cuTile 的一大优势是**跨架构可移植**——同一份 Kernel 代码无需修改即可在所有支持的 GPU 上运行，编译器针对不同硬件生成最优指令。

### 9.2 编译流程

cuTile 的代码不是直接编译为 PTX，而是经过一条专用的编译管线：

{% mermaid graph LR %}
    A["@ct.kernel Python 代码"] --> B["cuTile 前端"]
    B --> C["Tile IR（中间表示）"]
    C --> D["tileiras 编译器"]
    D --> E["GPU 机器码"]
{% endmermaid %}

**Tile IR** 是 cuTile 引入的新型中间表示，类似于 SIMT 模型中 PTX 的角色，但运作在 Tile 抽象层面。它是一套虚拟 ISA，面向 DSL 和编译器开发者，可作为其他语言后端的编译目标。

### 9.3 语言生态

| 🌐 语言 | 项目 | 📝 说明 |
|---------|------|---------|
| Python | `cuda-tile`（官方） | 主要开发接口 |
| Julia | `cuTile.jl`（JuliaGPU） | Julia 社区绑定 |
| Triton | Tile IR Backend | 将 Triton 程序编译到 Tile IR |

### 9.4 TileGym：Kernel 库与教程

NVIDIA 同步开源了 **TileGym** 项目，提供大量基于 cuTile 的 Kernel 示例和教程：

- **深度学习算子**：GEMM、Flash Attention、Softmax、RMSNorm、RoPE 等
- **端到端 LLM 推理**：Llama 3.1-8B、DeepSeek V2 的集成示例
- **性能基准**：与 cuBLAS、Triton 的对比测试

安装与使用：

```bash
pip install tilegym[tileiras]

# 运行性能基准测试
cd tests/benchmark
bash run_all.sh
```

---

## 📝 总结

cuTile 是 CUDA 编程范式的一次重大跃迁。它的核心价值在于：

1. **抽象提升**：从"管理线程"到"操作数据块"，大幅降低 GPU 编程心智负担
2. **硬件自动映射**：`ct.mma()` 自动使用 Tensor Core，`ct.load()` 自动使用 TMA，开发者无需了解硬件指令细节
3. **跨架构可移植**：同一份代码运行在 Blackwell、Ampere、Ada 等多代 GPU 上
4. **Python 原生**：直接在 Python 生态中编写高性能 Kernel，与 PyTorch/CuPy 无缝集成
5. **接近极致性能**：优化后的 cuTile Kernel 可达到手写 CUDA 和 cuBLAS 90\% 以上的性能

对于 AI Infra 工程师来说，cuTile 让"不精通 CUDA 底层也能写出高效 Kernel"成为可能，而精通底层的工程师则可以用更少的代码量完成同样的优化工作。这不是取代传统 CUDA 编程，而是在大多数深度学习场景中提供了一条更高效的路径。

## 🎯 自我检验清单

- 能解释 cuTile 的 Tile 编程模型与传统 SIMT 模型的核心区别
- 能说出 cuTile 的五个核心概念：Array、Tile、Kernel、Block、Compile-time Constant
- 能独立使用 `ct.load()` / `ct.store()` 和算术运算编写一个完整的向量加法 Kernel
- 能使用 cuTile 编写矩阵乘法 Kernel，并理解 K 维度迭代累加的工作流程
- 能解释 `ct.mma()` 如何自动映射到 Tensor Core 而无需手动编写 wmma 指令
- 能描述 cuTile 的编译管线：Python → Tile IR → tileiras → GPU 机器码
- 能使用 `ct.Constant[int]` 声明编译期常量，并解释其对性能优化的意义
- 能说出至少三种 cuTile 的进阶优化技巧（Swizzle、Fast-Math、软件流水线、Autotuning）
- 能判断一个计算任务是否适合用 cuTile 实现，还是需要回退到传统 CUDA 编程

## 📚 参考资料

- [NVIDIA cuTile Python - GitHub](https://github.com/NVIDIA/cutile-python)
- [TileGym: CUDA Tile Kernel Library - GitHub](https://github.com/NVIDIA/TileGym)
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Technical Blog - CUDA Tag](https://developer.nvidia.com/blog/tag/cuda/)
- [CUDA C++ Programming Guide - Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
