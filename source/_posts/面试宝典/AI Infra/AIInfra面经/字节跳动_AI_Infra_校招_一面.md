---
title: '字节跳动 AI Infra 校招 一面'
date: 2026-04-16 12:10:14
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 手撕：两个有序数组找中位数（LeetCode 4）？

（编程题）

---

### Q: 手撕：CUDA实现GEMM，讲优化方法和评估指标？

（编程题）

GEMM（General Matrix Multiply）是AI推理/训练中最核心的算子，优化从naive到极致有多个层次：

**优化层次**：

**Level 0: Naive（每线程计算C的一个元素）**：
```cuda
// 每线程独立从全局内存读A的一行和B的一列
C[row][col] = 0;
for (int k = 0; k < K; k++)
    C[row][col] += A[row][k] * B[k][col];
// 问题: 每个乘加需2次全局内存读取(~400 cycles)，计算:访存 = 1:2
```
性能：通常只达峰值的1-5%。

**Level 1: Tiling + Shared Memory**：
```cuda
// Block协作将A和B的tile加载到shared memory
__shared__ float As[TILE][TILE], Bs[TILE][TILE];
for (int t = 0; t < K/TILE; t++) {
    As[ty][tx] = A[row][t*TILE+tx];  // 协作加载
    Bs[ty][tx] = B[t*TILE+ty][col];
    __syncthreads();
    for (int k = 0; k < TILE; k++)
        sum += As[ty][k] * Bs[k][tx];  // 从shared memory读(~20 cycles)
    __syncthreads();
}
```
核心思想：数据从HBM加载到shared memory后，被TILE×TILE个线程共享复用。数据复用率 = TILE，HBM访问减少TILE倍。

**Level 2: 寄存器级Tiling（每线程计算小块）**：
```cuda
// 每线程负责TM×TN个输出元素（如4×4）
float c[TM][TN] = {0};  // 在寄存器中累加
for (int k = 0; k < TILE; k++) {
    float a[TM], b[TN];
    // 从shared memory加载到寄存器
    for (int i = 0; i < TM; i++) a[i] = As[ty*TM+i][k];
    for (int j = 0; j < TN; j++) b[j] = Bs[k][tx*TN+j];
    // 寄存器间运算（零延迟）
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            c[i][j] += a[i] * b[j];
}
```
每线程多元素（TM×TN）→ 增加每线程的计算量 → 提高Compute/Memory比 → 减少总线程数和shared memory压力。

**Level 3: 向量化访存（float4/LDS.128）**：
```cuda
// 128位单次传输（4个float或8个half）
float4 a4 = *reinterpret_cast<float4*>(&A[row][col]);
// 减少4倍的load指令数和内存事务
```

**Level 4: Double Buffering（隐藏延迟）**：
```cuda
// 计算当前tile时，异步加载下一个tile
cp_async(&smem_next[...], &global[...]);  // DMA不占计算管线
compute_tile(smem_current);               // 同时计算
cp_async_wait();                          // 等待加载完成
swap(smem_current, smem_next);            // 切换buffer
```

**Level 5: Tensor Core（WMMA/WGMMA）**：
```cuda
// 使用wmma API（Volta+）
wmma::fragment<wmma::matrix_a, 16, 16, 16, half> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::load_matrix_sync(a_frag, smem_a, 16);
wmma::load_matrix_sync(b_frag, smem_b, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // 单条指令完成16×16×16矩阵乘
```
Tensor Core单条指令完成数百次FMA，吞吐比CUDA Core高16x。

**Level 6: Warp级协作（Warp Tile）**：
- 32个线程协作处理一个较大的输出tile（如64×64）。
- 通过warp内的寄存器共享（`__shfl_sync`）减少shared memory读取。
- CUTLASS/CUTE DSL的核心抽象。

**评估指标**：

| 指标 | 计算公式 | 优秀标准(A100 FP16) |
|------|---------|-------------------|
| TFLOPS | 2×M×N×K / time(s) / 1e12 | >250 TFLOPS |
| 效率% | actual TFLOPS / 312 × 100% | >80% |
| HBM带宽利用 | bytes_transferred / time / peak_BW | Memory-bound shape时>80% |
| 达峰比 | actual / cuBLAS性能 | >90% |

不同矩阵规模的性能特征：
- 大矩阵（M,N,K>4096）：compute-bound，追求高TFLOPS。
- 小矩阵（M或N<256）：可能memory-bound，关注带宽利用。
- Batch小GEMM：可能latency-bound（kernel launch开销占比高）。

---

### Q: 手撕：实现MLA（Multi-head Latent Attention）？

（编程题）
