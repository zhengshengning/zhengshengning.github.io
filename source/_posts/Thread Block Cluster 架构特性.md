---
title: Thread Block Cluster 架构特性
date: 2026-03-25 16:00:00
categories:
  - [AI Infra, CUDA编程]
tags: [CUDA, Hopper, Thread Block Cluster, GPU, 并行计算]
---

NVIDIA Hopper架构（从H100 GPU开始）引入了 **Thread Block Cluster** 这一重要特性，在传统的线程层次结构中新增了一层，为大规模Block间协作提供了硬件原生支持。本文介绍其核心机制、编程接口和典型应用场景。

<!-- more -->

## 1. 线程层次结构升级

传统GPU的并行层次为：

```text
Thread → Warp → Thread Block → Grid
```

Hopper架构新增了 Thread Block Cluster 层级：

```text
Thread → Warp → Thread Block → Thread Block Cluster → Grid
```

一个Cluster由多个Thread Block组成（最多8个），这些Block被调度到物理上相邻的SM上，可以通过硬件互连直接通信，无需经过Global Memory。

## 2. Distributed Shared Memory (DSM)

Cluster最核心的能力是 **Distributed Shared Memory**，允许一个Block直接访问同Cluster内其他Block的Shared Memory。

```cpp
// 传统 Shared Memory（仅限Block内）
__shared__ float smem[256];

// Hopper Distributed Shared Memory（Cluster内跨Block访问）
__shared__ __cluster__ float cluster_smem[256];

// 访问其他Block的共享内存
// cluster_rank: 当前block在cluster中的ID
// target_rank: 目标block在cluster中的ID
float* remote_smem = cluster_smem + cluster.map_shared_rank(target_rank);
```

## 3. 核心编程接口

### 3.1 跨Block数据共享

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(2, 2, 1) kernel() {
    __shared__ __cluster__ int data[128];

    // 获取cluster对象
    cg::cluster_group cluster = cg::this_cluster();

    // Cluster信息
    unsigned int block_rank = cluster.block_rank();
    unsigned int num_blocks = cluster.num_blocks();

    // 当前Block写入数据
    data[threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x;
    cluster.sync(); // Cluster级别同步

    // 访问相邻Block的共享内存
    int neighbor_rank = (block_rank + 1) % num_blocks;
    float* neighbor_data = cluster.map_shared_rank(data, neighbor_rank);
    int remote_value = neighbor_data[threadIdx.x];
}
```

### 3.2 硬件加速的Cluster同步

```cpp
// Cluster级别的barrier同步
cooperative_groups::this_cluster().sync();

// 比通过global memory协调快得多
```

## 4. 典型应用场景

### 4.1 矩阵乘法优化

多个Block协作计算更大的Tile，通过DSM共享数据，减少Global Memory访问。

```cpp
__global__ void __cluster_dims__(2, 2, 1)
matmul_cluster(float* A, float* B, float* C) {
    __shared__ __cluster__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ __cluster__ float tileB[TILE_SIZE][TILE_SIZE];

    auto cluster = cooperative_groups::this_cluster();

    // Cluster内的Blocks协作加载更大的数据块
    // 通过DSM共享，减少global memory访问

    // Block 0,1加载A的不同部分
    // Block 2,3加载B的不同部分
    // 所有Blocks通过DSM访问完整数据
}
```

### 4.2 卷积操作优化

相邻Block的Halo区域通过DSM直接访问，避免重复从Global Memory加载。

```cpp
__global__ void __cluster_dims__(2, 2, 1)
conv2d_cluster() {
    __shared__ __cluster__ float input_tile[TILE_H][TILE_W];

    // 相邻Blocks的边界数据通过DSM直接访问
    // 无需重复从global memory加载halo区域
}
```

## 5. 性能对比

| 特性 | 传统方式 | Thread Block Cluster |
|------|---------|---------------------|
| Block间通信 | Global Memory | Distributed Shared Memory |
| 延迟 | 高（数百cycles） | 低（类似L1缓存） |
| 带宽 | 受限于HBM | 芯片内互连带宽 |
| 同步开销 | 需要多次kernel启动 | 硬件原生支持 |

## 6. 硬件支持

- **H100**：首次引入，最多8个Blocks per Cluster
- **专用互连网络**：连接Cluster内的SM
- **硬件加速同步原语**：原生支持Cluster级别barrier
- **低延迟跨SM访问**：DSM访问延迟接近本地Shared Memory

## 总结

Thread Block Cluster是Hopper架构的重要创新，为大规模协作并行计算提供了更高效的硬件支持。其核心价值在于通过Distributed Shared Memory实现低延迟的跨Block数据共享，特别适合需要大量Block间数据交换的算法（如大型矩阵运算、3D卷积等）。

## 参考资料

- [CUDA C Programming Guide - Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters)
- [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core)
