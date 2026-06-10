---
title: Hopper核心新特性
date: 2026-06-03 14:00:00
mathjax: true
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程高阶]
tags: [CUDA, Hopper, H100, TMA, SM90, GPU架构]
---

NVIDIA Hopper 架构（SM90，代表产品 H100）是面向大规模 AI 训练和推理场景设计的一代 GPU 架构，引入了多项突破性硬件特性。本文系统梳理 Hopper 的核心创新——从 TMA 异步数据搬运引擎、第四代 Tensor Core 与 WGMMA 指令、Thread Block Cluster 与分布式共享内存，到异步事务屏障和新的内存层次——帮助你建立对这代架构的完整认知。

<!-- more -->

## 📑 目录

- [1. Hopper 架构总览](#1-hopper-架构总览)
- [2. Tensor Memory Accelerator（TMA）](#2-tensor-memory-acceleratortma)
- [3. 第四代 Tensor Core 与 WGMMA 指令](#3-第四代-tensor-core-与-wgmma-指令)
- [4. Thread Block Cluster 与分布式共享内存](#4-thread-block-cluster-与分布式共享内存)
- [5. 异步事务屏障（Async Transaction Barrier）](#5-异步事务屏障async-transaction-barrier)
- [6. FP8 数据类型支持](#6-fp8-数据类型支持)
- [7. 其他重要特性](#7-其他重要特性)
- [8. Hopper 编程范式：特性协同](#8-hopper-编程范式特性协同)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)

---

## 1. Hopper 架构总览

### 1.1 架构定位

Hopper 架构（SM90/SM90a）是 NVIDIA 在 2022 年发布的数据中心 GPU 架构，主要面向大模型训练与推理。它在前代 Ampere（SM80）基础上做了大幅度的架构革新，核心目标是**降低数据搬运开销，提升计算单元利用率**。

> **白话理解**：如果说 Ampere 是把"计算引擎"升级了一档（引入异步拷贝 `cp.async`），那 Hopper 是把整条"物流系统"重新设计了——给 GPU 配了专职搬运工（TMA）、打通了仓库隔断（分布式共享内存）、升级了流水线调度协议（异步事务屏障），让计算单元几乎不需要自己操心数据在哪、怎么搬。

### 1.2 关键参数对比

| 📊 指标 | Ampere A100（SM80） | Hopper H100（SM90） |
|---------|-------------------|-------------------|
| SM 数量 | 108 | 132 |
| FP16 Tensor TFLOPS | 312 | 989 |
| FP8 Tensor TFLOPS | - | 1979 |
| HBM 带宽 | 2.0 TB/s | 3.35 TB/s |
| L2 缓存 | 40 MB | 50 MB |
| Shared Memory / SM | 最大 164 KB | 最大 228 KB |
| NVLink 带宽 | 600 GB/s | 900 GB/s |
| 新增硬件单元 | - | TMA、DPX |

### 1.3 核心创新一览

Hopper 的创新不是某个单点突破，而是一组**协同设计**的硬件特性：

1. **TMA（Tensor Memory Accelerator）**：硬件异步数据搬运引擎，解放线程的访存职责
2. **第四代 Tensor Core + WGMMA**：支持异步矩阵计算，吞吐翻倍
3. **Thread Block Cluster**：新增线程层级，硬件支持跨 Block 协作
4. **Distributed Shared Memory**：Cluster 内跨 SM 共享内存直接互访
5. **异步事务屏障（Async Transaction Barrier）**：精细化的异步操作同步机制
6. **FP8 原生支持**：E4M3/E5M2 两种格式，推理吞吐翻倍

---

## 2. Tensor Memory Accelerator（TMA）

### 2.1 为什么需要 TMA

在 Ampere 架构中，`cp.async` 指令已经实现了从 Global Memory 到 Shared Memory 的异步拷贝，绕过了寄存器中转。但它仍然有明显的局限：

- **地址计算由线程负责**：每个线程需要自己计算源地址和目标地址
- **线程数量受限于数据量**：搬运 N 个元素需要 N 个线程参与
- **多维数据需要手动处理**：矩阵的 Tile 切分、跨步访问需要复杂的索引逻辑

> **白话理解**：`cp.async` 就像让工人（线程）自己搬砖——虽然不用中途放下再捡起（绕过寄存器），但每个人还是得亲自跑腿、自己记住往哪搬。TMA 则是给工地配了一台自动传送带——你只需要告诉它"把第 3 排第 5 列那一块搬到仓库 B 的第 2 号位置"，它自己就搞定了，工人可以安心去做计算。

### 2.2 TMA 的核心能力

TMA 是一个独立于 SM 计算管线的**硬件数据搬运引擎**，它提供以下能力：

**单线程发起批量搬运**：只需一个线程发出 TMA 指令，硬件自动完成整个 Tile 的搬运，其余线程可以同时做有用计算。

**原生多维寻址**：TMA 天然理解 1D-5D 张量布局，通过 Tensor Map（描述符）记录张量的 shape、stride、数据类型等元信息，搬运时直接按坐标索引。

**支持双向传输**：
- `cp.async.bulk.tensor`：Global → Shared（load）
- `cp.async.bulk.tensor` store 变体：Shared → Global（store）

**支持多播（Multicast）**：一次 TMA load 可以将数据同时分发到 Cluster 内多个 Block 的 Shared Memory，避免重复搬运。

### 2.3 Tensor Map 描述符

TMA 的核心抽象是 **Tensor Map**，它是一个预先创建的描述符对象，记录了张量的完整布局信息：

```cpp
#include <cuda.h>

// 在 Host 端创建 Tensor Map 描述符
CUtensorMap tensor_map;
CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;

// 定义张量形状和访问参数
uint64_t global_dim[2] = {M, K};      // 全局张量维度
uint64_t global_stride[1] = {K * 2};  // 字节跨步（仅需 rank-1 个）
uint32_t box_dim[2] = {TILE_M, TILE_K}; // 每次搬运的 Tile 尺寸
uint32_t elem_stride[2] = {1, 1};     // 元素步长

// 创建 2D TMA 描述符
cuTensorMapEncodeTiled(
    &tensor_map,
    dtype,
    2,                        // 维度数
    (void*)d_tensor,          // 全局内存基址
    global_dim,
    global_stride,
    box_dim,
    elem_stride,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B,        // Swizzle 模式
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
```

📌 **关键点**：Tensor Map 在 Host 端创建后传入 Kernel，由硬件使用。它将张量的"逻辑布局"和"物理地址计算"封装在一起，Kernel 代码只需指定坐标，无需手动计算偏移。

### 2.4 TMA 基本用法

```cpp
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;

__global__ void tma_load_kernel(const __grid_constant__ CUtensorMap tensor_map) {
    __shared__ alignas(128) half smem_buf[TILE_M][TILE_K];
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        // 初始化 barrier：期望 blockDim.x 个线程 arrive
        init(&bar, blockDim.x);
    }
    __syncthreads();

    barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // thread 0: arrive 并设置期望的事务字节数，同时发起 TMA
        token = cuda::barrier_arrive_tx(bar, 1, TILE_M * TILE_K * sizeof(half));

        uint32_t smem_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(smem_buf));

        int tile_row = blockIdx.x * TILE_M;
        int tile_col = 0;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :
            : "r"(smem_addr), "l"(&tensor_map),
              "r"(tile_col), "r"(tile_row), "r"(smem_addr_of_bar)
            : "memory"
        );
    } else {
        // 其余线程仅 arrive，不参与搬运
        token = cuda::barrier_arrive(bar, 1);
    }

    // 所有线程使用各自的 token 等待 TMA 搬运完成
    bar.wait(std::move(token));

    // 现在可以安全使用 smem_buf 中的数据
}
```

### 2.5 TMA Multicast

TMA 的多播功能是 Cluster 场景下的杀手级特性——一次 Global Memory 读取，同时写入 Cluster 内多个 Block 的 Shared Memory：

```text
                   ┌─── Block 0 Shared Memory
                   │
Global Memory ─── TMA Multicast ──┼─── Block 1 Shared Memory
                   │
                   └─── Block 2 Shared Memory
```

💡 **提示**：在 GEMM 中，矩阵 A 的同一个 Tile 往往被同一行的多个 Block 共用。使用 TMA Multicast 可以将这一次读取同时分发给所有需要它的 Block，HBM 读取量直接除以 Cluster 中共享该数据的 Block 数。

### 2.6 TMA vs cp.async 对比

| ✅ TMA 优势 | ❌ cp.async 局限 |
|------------|----------------|
| 单线程发起整个 Tile 搬运 | 每个线程搬运自己负责的元素 |
| 硬件自动计算多维地址 | 线程手动计算地址偏移 |
| 原生支持 Swizzle | 需要手动实现 Swizzle |
| 支持 Multicast 到多个 Block | 只能搬运到本 Block |
| 解放线程做计算 | 每个线程需参与发射搬运指令 |

---

## 3. 第四代 Tensor Core 与 WGMMA 指令

### 3.1 Tensor Core 演进

| 📊 代际 | 架构 | 关键提升 |
|--------|------|---------|
| 第一代 | Volta（SM70） | 引入 Tensor Core，FP16 HMMA |
| 第二代 | Turing（SM75） | 增加 INT8/INT4/Binary |
| 第三代 | Ampere（SM80） | TF32、BF16、Double Tensor |
| 第四代 | Hopper（SM90） | FP8、异步执行、WGMMA |

Hopper 的第四代 Tensor Core 不仅增加了 FP8 数据类型支持使得计算吞吐翻倍，更重要的架构变化是引入了 **WGMMA（Warp Group Matrix Multiply-Accumulate）** 指令，使矩阵运算从"同步阻塞"演进为"异步流水"。

### 3.2 从 HMMA 到 WGMMA

**Ampere 的 HMMA（`mma.sync`）**：
- 以单个 Warp（32 线程）为执行单位
- 操作数必须在寄存器中就绪
- 同步执行：发射后 Warp 阻塞直到结果写回寄存器

**Hopper 的 WGMMA（`wgmma.mma_async`）**：
- 以 **Warp Group**（128 线程 = 4 个 Warp）为执行单位
- 操作数 A 可以直接来自 Shared Memory（不需要先加载到寄存器）
- 异步执行：发射后线程可以继续执行其他指令

> **白话理解**：HMMA 就像一个人做菜——得把所有食材（操作数）都摆到案板（寄存器）上才能开始切。WGMMA 更像一个自动炒菜机——你把食材放冰箱（Shared Memory）里告诉它位置就行，它自己去拿、自己炒，你可以去准备下一道菜的食材。

### 3.3 WGMMA 操作数来源

WGMMA 最大的创新是操作数来源的灵活性：

```text
操作数 A 来源：
  ├── 寄存器（传统方式）
  └── Shared Memory（新增，直接从 SMEM 取数）★

操作数 B 来源：
  ├── 寄存器
  └── Shared Memory

累加器 D：
  └── 寄存器（始终在寄存器中累加）
```

当操作数 A 直接来自 Shared Memory 时，省去了 "SMEM → 寄存器 → Tensor Core" 这一步的寄存器占用和搬运延迟，同时释放了宝贵的寄存器资源给累加器使用。

### 3.4 WGMMA 指令形态

```text
wgmma.mma_async.sync.aligned.shape.dtype_d.dtype_a.dtype_b  d, a_desc, b_desc, ...
```

- **shape**：支持的矩阵规模，如 `m64n256k16`、`m64n128k16` 等
- **dtype**：支持 FP16、BF16、TF32、FP8（E4M3/E5M2）、INT8
- **描述符（desc）**：当操作数来自 Shared Memory 时，使用 64 位矩阵描述符指定地址和布局

典型 WGMMA 使用模式：

```cpp
// 伪代码示意 WGMMA 流水线
__global__ void wgmma_gemm_kernel() {
    // ... TMA 加载 A、B 到 Shared Memory ...

    // 创建 WGMMA 描述符（指向 Shared Memory 中的矩阵块）
    uint64_t desc_a = make_smem_desc(smem_A_ptr);
    uint64_t desc_b = make_smem_desc(smem_B_ptr);

    // 发射异步矩阵计算
    wgmma_fence();                    // 表明即将使用新的 SMEM 数据
    wgmma_async(desc_a, desc_b, acc); // 异步执行，不阻塞
    wgmma_commit();                   // 提交到异步组
    wgmma_wait<0>();                  // 等待完成（可以延迟等待）
}
```

### 3.5 WGMMA 的异步流水线优势

WGMMA 的异步特性使得数据搬运和计算可以形成真正的**双重流水线**：

```text
时间 →
────────────────────────────────────────────
TMA:    [Load Tile 0] [Load Tile 1] [Load Tile 2] ...
WGMMA:        [Compute Tile 0] [Compute Tile 1] ...
────────────────────────────────────────────
```

在 Ampere 架构上，即使 `cp.async` 实现了异步加载，计算阶段的 `mma.sync` 仍然需要先将数据从 SMEM 搬到寄存器再执行——这一步是同步的。Hopper 上 WGMMA 直接消费 SMEM 中的数据，与 TMA 形成端到端的异步流水。

### 3.6 Warp Specialization

WGMMA 的异步特性天然支持 **Warp Specialization** 编程模式——将 Block 中的 Warp 分为"生产者"和"消费者"角色：

- **Producer Warp（1 个 Warp）**：专门负责发射 TMA 指令，将数据搬入 Shared Memory
- **Consumer Warp Group（4 个 Warp）**：专门负责发射 WGMMA 指令，执行矩阵计算

```text
Producer Warp:     TMA[0] → TMA[1] → TMA[2] → TMA[3] → ...
                       ↓         ↓         ↓         ↓
Consumer WarpGroup:  WGMMA[0] → WGMMA[1] → WGMMA[2] → WGMMA[3] → ...
```

💡 **提示**：Warp Specialization 不是简单的软件设计模式，而是与 Hopper 硬件协同设计的——TMA 由专用硬件执行不消耗计算资源，WGMMA 由 Tensor Core 异步执行不阻塞 CUDA Core。两条硬件流水线通过异步屏障协调，实现真正的计算/访存重叠。

---

## 4. Thread Block Cluster 与分布式共享内存

### 4.1 新增的线程层级

Hopper 在传统的线程层次中新增了 **Thread Block Cluster** 层级：

```text
Thread → Warp → Warp Group → Thread Block → Cluster → Grid
                   (新增)                      (新增)
```

一个 Cluster 由 1-16 个 Thread Block 组成（实际推荐 2-8 个），这些 Block 被硬件保证**调度到物理相邻的 SM 上**。

```cpp
// 声明 Cluster 维度
// 方法一：Kernel 属性
__global__ void __cluster_dims__(4, 1, 1) my_kernel() { ... }

// 方法二：启动时动态指定
cudaLaunchConfig_t config;
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {4, 1, 1};
config.attrs = attrs;
config.numAttrs = 1;
cudaLaunchKernelEx(&config, my_kernel, ...);
```

### 4.2 Distributed Shared Memory（DSMEM）

Cluster 的核心价值是 **Distributed Shared Memory**——Cluster 内的任何 Block 可以直接访问其他 Block 的 Shared Memory，延迟接近本地 SMEM 访问。

> **白话理解**：传统架构中每个 SM 的 Shared Memory 就像私人保险柜，只有本 SM 上运行的线程能打开。DSMEM 则是把同一个 Cluster 中所有 SM 的保险柜用内部通道连通了——你可以直接从隔壁 SM 的保险柜取东西，不需要先存到银行总部（Global Memory）再转。

DSMEM 的硬件实现基于 SM 之间的**片上互连网络（On-chip Crossbar）**，提供比 L2 缓存更低的延迟和更高的带宽：

| 📊 访问路径 | 典型延迟 | 带宽 |
|------------|---------|------|
| 本地 Shared Memory | ~20-30 cycles | ~128 B/cycle/SM |
| DSMEM（Cluster 内跨 SM） | ~30-50 cycles | ~64 B/cycle/SM |
| L2 缓存 | ~200+ cycles | 受限 |
| HBM（Global Memory） | ~400-800 cycles | 3.35 TB/s 共享 |

### 4.3 DSMEM 的编程模型

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(4, 1, 1) dsmem_example() {
    extern __shared__ int smem[];
    auto cluster = cg::this_cluster();
    int block_rank = cluster.block_rank();

    // 每个 Block 写入自己的 Shared Memory
    smem[threadIdx.x] = block_rank * 1000 + threadIdx.x;
    cluster.sync();

    // 读取相邻 Block 的 Shared Memory（通过 DSMEM）
    int neighbor = (block_rank + 1) % cluster.num_blocks();
    int* remote_smem = cluster.map_shared_rank(smem, neighbor);
    int remote_val = remote_smem[threadIdx.x]; // 直接读取，硬件路由
}
```

### 4.4 Cluster 的实际应用场景

**GEMM 中的数据复用**：矩阵乘法 $C = A \times B$ 中，同一行的多个 Block 共享矩阵 A 的同一行 Tile。利用 Cluster + TMA Multicast，一次 HBM 读取同时送达多个 Block。

**Reduction 操作**：Cluster 内的多个 Block 可以通过 DSMEM 直接汇总局部结果，无需通过 Global Memory 做跨 Block 归约。

**Attention 计算**：FlashAttention 的 online softmax 需要跨 Tile 共享 max 值和 sum 值，Cluster 内 Block 可以通过 DSMEM 低延迟交换这些标量。

---

## 5. 异步事务屏障（Async Transaction Barrier）

### 5.1 传统同步的问题

Ampere 架构中 `cp.async` 配合 `cp.async.wait_group<N>` 实现异步等待，但这种机制粒度较粗——只能按"组"等待，不能精确追踪"某个特定的数据块是否到达"。

Hopper 引入了 **mbarrier（异步事务屏障）**，它是一种硬件支持的同步原语，核心创新在于：可以追踪**字节级别**的异步事务完成状态。

> **白话理解**：传统 `__syncthreads()` 就像老师点名——确认所有人都到齐了才上课。`mbarrier` 更像快递柜——你期望收到 3 个包裹共 15 公斤，快递柜会自动累计已收到的重量，当重量达标时通知你"全到了"。不需要知道是哪个快递员送的、什么时候送的，只关心最终结果。

### 5.2 mbarrier 的工作原理

mbarrier 维护两个计数：

1. **到达计数（Arrive Count）**：跟踪多少个线程/操作已经"签到"
2. **事务计数（Transaction Count）**：跟踪预期接收的字节数，当异步操作（如 TMA）完成数据写入时，硬件自动递减

```text
mbarrier 状态机：
┌─────────────────┐
│ Phase 0 (等待)   │ ←─ init(expected_arrive_count)
│ arrive_count = N │     set_transaction_bytes(expected_bytes)
│ tx_count = B     │
└────────┬────────┘
         │ 所有线程 arrive + 所有字节到达
         ▼
┌─────────────────┐
│ Phase 1 (就绪)   │ ←─ 自动翻转 phase
│ 可以安全消费数据  │
└─────────────────┘
```

### 5.3 mbarrier 与 TMA 的配合

这是 Hopper 上最典型的使用模式——TMA 搬运完成后自动通知 mbarrier：

```cpp
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__global__ void tma_with_mbarrier(const __grid_constant__ CUtensorMap tma_desc) {
    __shared__ alignas(128) half tile[TILE_M * TILE_K];

    // 多阶段流水线 barrier 数组
    __shared__ barrier bars[NUM_STAGES];

    int stage = 0;
    if (threadIdx.x == 0) {
        // 初始化 barrier：期望 1 个线程 arrive + TMA 写入的字节数
        for (int s = 0; s < NUM_STAGES; s++) {
            init(&bars[s], 1);
        }

        // 设置事务字节数：TMA 完成后硬件自动减少这个计数
        cde::barrier_arrive_tx(bars[stage], 1, TILE_M * TILE_K * sizeof(half));

        // 发起 TMA（完成后硬件自动通知 barrier）
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            tile, &tma_desc, coord_x, coord_y, bars[stage]);
    }

    // 所有线程等待数据就绪
    bars[stage].wait(bars[stage].arrive());

    // 安全使用 tile 数据
}
```

### 5.4 多阶段流水线示例

实际的高性能 Kernel 会使用多个 mbarrier 组成流水线：

```cpp
constexpr int STAGES = 4;  // 4 级流水

__shared__ half smem_A[STAGES][TILE_M * TILE_K];
__shared__ half smem_B[STAGES][TILE_N * TILE_K];
__shared__ barrier load_bars[STAGES];  // 加载完成信号
__shared__ barrier comp_bars[STAGES];  // 计算完成信号（生产者可复用 buffer）

// Producer: 预填充流水线
for (int s = 0; s < STAGES; s++) {
    issue_tma_load(smem_A[s], smem_B[s], load_bars[s], k_iter + s);
}

// Consumer: 主循环
for (int k = 0; k < num_k_tiles; k++) {
    int s = k % STAGES;

    // 等待当前阶段的数据加载完毕
    load_bars[s].wait(...);

    // 执行 WGMMA 计算
    wgmma_async(smem_A[s], smem_B[s], accum);
    wgmma_wait<0>();

    // 通知 Producer 可以复用当前 buffer
    comp_bars[s].arrive();

    // Producer 同时预取下一批数据（在后续阶段）
}
```

### 5.5 与 Ampere 同步机制的对比

| ✅ Hopper mbarrier | ❌ Ampere cp.async + wait\_group |
|-------------------|-------------------------------|
| 追踪具体字节完成量 | 只能按"组"粗粒度等待 |
| 与 TMA 硬件集成 | 与 cp.async 配合 |
| 支持跨 Block 同步（Cluster） | 仅限 Block 内 |
| Phase-flip 机制天然支持多阶段 | 需手动管理 fence |
| 可精确区分 Producer/Consumer | 全员统一等待 |

---

## 6. FP8 数据类型支持

### 6.1 FP8 的两种格式

Hopper 原生支持两种 8 位浮点格式：

| 📊 格式 | 指数位 | 尾数位 | 动态范围 | 精度 | 典型用途 |
|--------|-------|-------|---------|------|---------|
| E4M3 | 4 | 3 | ±448 | 较高 | 前向推理、权重存储 |
| E5M2 | 5 | 2 | ±57344 | 较低 | 反向传播梯度 |

> **白话理解**：E4M3 像精确但量程有限的温度计（适合数值分布集中的权重），E5M2 像量程大但刻度粗的温度计（适合偶尔出现极端值的梯度）。

### 6.2 性能提升

FP8 相比 FP16 的理论吞吐提升为 **2 倍**：

$$
\text{H100 FP8 TFLOPS} = 1979 \quad vs \quad \text{H100 FP16 TFLOPS} = 989
$$

这不仅仅是因为数据量减半带来的带宽节省，更重要的是 Tensor Core 在相同周期内可以处理两倍数量的 FP8 元素。

### 6.3 FP8 GEMM 编程

```cpp
// FP8 WGMMA 示例（概念代码）
// 操作数 A：E4M3 格式
// 操作数 B：E4M3 格式
// 累加器 D：FP32 格式（保持精度）
wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 d, a_desc, b_desc, ...;
```

📌 **关键点**：FP8 计算使用 FP32 累加器，避免低精度累加导致的数值发散。典型的训练流程是：权重和激活以 FP8 存储和计算，梯度更新时转回 FP16/FP32。

### 6.4 FP8 训练策略

FP8 训练并非简单地将所有数据截断为 8 位，需要配合**动态缩放（Dynamic Scaling）**：

1. 前向传播：权重和激活使用 E4M3，输出用 FP16/FP32
2. 反向传播：梯度使用 E5M2（更大动态范围容纳梯度尖峰）
3. 每层维护一个 Scale Factor，将数值映射到 FP8 可表示的范围
4. Scale Factor 按统计量动态调整（delayed scaling 或 just-in-time scaling）

⚠️ **注意**：并非所有层都适合 FP8。Attention 的 Softmax 输出、LayerNorm 等对精度敏感的计算通常保持 FP16 或更高精度。

---

## 7. 其他重要特性

### 7.1 DPX 指令

Hopper 新增 **DPX（Dynamic Programming X）** 加速指令，为动态规划算法提供硬件加速：

- `__vimin3_relu`：三路比较取最小值 + ReLU
- `__vimax3`：三路比较取最大值
- `__vibfind`：位搜索

典型应用：Smith-Waterman 序列比对算法（生物信息学）、最短路径算法等。DPX 指令将这些操作从多条指令压缩为单条硬件指令，性能提升可达 7 倍。

### 7.2 扩大的 Shared Memory

Hopper 将每个 SM 的 Shared Memory 容量提升至最大 **228 KB**（Ampere 为 164 KB）。更大的 SMEM 直接影响：

- GEMM 的 Tile 尺寸可以更大，减少主循环迭代次数
- 多阶段流水线可以使用更多 Stage（如 4-8 阶段），更好地隐藏延迟
- Attention 算子可以容纳更大的 QKV 块

### 7.3 新的 L2 缓存控制

Hopper 增强了 L2 缓存的软件控制能力：

- **Residency Control**：可以"钉住"特定数据在 L2 中不被驱逐
- **访问优先级**：为不同数据流设置缓存优先级
- **TMA 集成**：TMA 搬运时可以指定 L2 驻留策略

### 7.4 NVLink 4.0 与 NVSwitch

虽然不属于 SM 架构层面，但 Hopper 配套的通信升级对多卡场景意义重大：

- NVLink 4.0：每 GPU 900 GB/s 双向带宽（相比 A100 的 600 GB/s）
- 第三代 NVSwitch：支持 8 GPU 全互连
- 支持 **NVLink SHARP**：网络内归约，减少多卡通信开销

---

## 8. Hopper 编程范式：特性协同

### 8.1 一个完整的 Hopper GEMM 流水线

Hopper 的各项特性不是孤立使用的，它们协同构成了全新的编程范式。以 GEMM 为例：

{% mermaid graph LR %}
    A["Host: 创建 TMA 描述符"] --> B["Producer Warp: 发射 TMA"]
    B --> C["TMA Engine: 异步搬运 + Multicast"]
    C --> D["mbarrier: 通知数据就绪"]
    D --> E["Consumer WarpGroup: 发射 WGMMA"]
    E --> F["Tensor Core: 异步计算"]
    F --> G["累加器: FP32 结果"]
{% endmermaid %}

### 8.2 Hopper 高性能 Kernel 的典型结构

```text
┌──────────────────────────────────────────────────┐
│ Kernel 启动：Cluster (4 Blocks × 128 threads)     │
├──────────────────────────────────────────────────┤
│ Block 内角色划分：                                  │
│   Producer Warp (Warp 0)：TMA 搬运               │
│   Consumer WarpGroup (Warp 1-4)：WGMMA 计算      │
├──────────────────────────────────────────────────┤
│ 流水线阶段（4-Stage Pipeline）：                   │
│                                                  │
│   Stage 0: [TMA Load] ──→ [mbarrier] ──→ [WGMMA]│
│   Stage 1: [TMA Load] ──→ [mbarrier] ──→ [WGMMA]│
│   Stage 2: [TMA Load] ──→ [mbarrier] ──→ [WGMMA]│
│   Stage 3: [TMA Load] ──→ [mbarrier] ──→ [WGMMA]│
├──────────────────────────────────────────────────┤
│ Cluster 协作：                                    │
│   TMA Multicast：A 矩阵一次加载，多 Block 共用     │
│   DSMEM：跨 Block 交换部分结果                     │
└──────────────────────────────────────────────────┘
```

### 8.3 与 Ampere 编程范式的对比

| 📊 维度 | Ampere 范式 | Hopper 范式 |
|--------|------------|------------|
| 数据搬运 | 所有线程参与 cp.async | 1 个 Warp 发 TMA，其余专注计算 |
| 矩阵计算 | mma.sync（同步，Warp 级） | wgmma.mma\_async（异步，WarpGroup 级） |
| 操作数来源 | 必须先 SMEM → Register | 可直接从 SMEM 消费 |
| 跨 Block 协作 | 通过 Global Memory | DSMEM 直接通信 |
| 同步机制 | __syncthreads + fence | mbarrier（事务感知） |
| 编程模式 | 同质线程（所有线程做同样的事） | Warp Specialization（角色分工） |
| 理论计算上限 | 312 TFLOPS (FP16) | 989 TFLOPS (FP16) / 1979 (FP8) |

### 8.4 CUTLASS 3.x 中的 Hopper 支持

NVIDIA 的 CUTLASS 库从 3.0 版本开始全面支持 Hopper 特性，使用 **CuTe**（CUDA Templates）抽象层封装底层细节：

```cpp
// CUTLASS 3.x 中使用 Hopper 特性的 GEMM 示例（简化）
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<
        STAGES,                            // Pipeline stages
        ClusterShape,                      // Thread Block Cluster 形状
        KernelSchedule                     // Warp Specialization 调度
    >,
    TileShape,                             // CTA Tile 维度
    ElementA, LayoutA,                     // A 矩阵类型和布局
    ElementB, LayoutB                      // B 矩阵类型和布局
>;
```

💡 **提示**：对于大多数应用场景，直接使用 CUTLASS 3.x 或 cuBLAS（其底层也基于 Hopper 特性优化）是最佳选择。手写 Hopper PTX 主要适用于需要极致定制的场景（如自定义融合 Kernel）。

---

## 📝 总结

Hopper 架构的核心创新可以用一句话概括：**让线程从数据搬运中解放出来，专注于计算**。

实现这一目标的手段是一组协同设计的硬件特性：

1. **TMA** 将数据搬运从线程职责变为硬件自动完成
2. **WGMMA** 让矩阵计算异步化，操作数可直接来自 SMEM
3. **Thread Block Cluster + DSMEM** 打破了 Block 间的内存隔离
4. **mbarrier** 以字节精度追踪异步事务完成状态
5. **FP8** 在保持训练精度的前提下将吞吐翻倍
6. **Warp Specialization** 利用角色分工最大化硬件利用率

这些特性共同将 GPU 编程从"线程级并行计算"推向了"异步流水线编排"的新范式。对于 AI Infra 工程师而言，理解 Hopper 架构是掌握现代高性能 Kernel 开发（如 FlashAttention-3、FP8 GEMM）的基础。

## 🎯 自我检验清单

- 能画出 Hopper 的线程层次结构（Thread → Warp → WarpGroup → Block → Cluster → Grid）并解释每一层的硬件意义
- 能解释 TMA 相比 cp.async 的三个核心优势，并说明为什么 TMA 只需一个线程发起
- 能描述 Tensor Map 描述符的作用，以及为什么它需要在 Host 端创建
- 能区分 HMMA（mma.sync）和 WGMMA（wgmma.mma\_async）在执行单元、操作数来源、同步语义上的区别
- 能解释 WGMMA 操作数 A 直接来自 SMEM 带来的寄存器压力减轻效果
- 能说明 Distributed Shared Memory 的访问延迟大致在什么量级，以及它与 Global Memory 通信的性能差异
- 能描述 mbarrier 的"事务计数"机制如何与 TMA 配合实现精确的异步同步
- 能区分 FP8 E4M3 和 E5M2 的适用场景（前向 vs 反向）
- 能画出 Hopper GEMM 的 Warp Specialization 结构：Producer Warp + Consumer WarpGroup
- 能用 TMA + mbarrier + WGMMA 三者协同的视角，描述一个 4-Stage 流水线 GEMM 的执行时序

## 📚 参考资料

- [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [CUDA C++ Programming Guide - Hopper Features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)
- [NVIDIA PTX ISA - wgmma Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
- [CUTLASS 3.x Documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [GTC 2022: NVIDIA Hopper Architecture In-Depth](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/)
- [GTC 2023: How CUDA Programming Works on Hopper GPUs](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51398/)
- [FP8 Formats for Deep Learning (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433)
