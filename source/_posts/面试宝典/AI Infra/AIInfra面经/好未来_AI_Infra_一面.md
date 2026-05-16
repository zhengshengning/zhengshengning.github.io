---
title: '好未来 AI Infra 一面'
date: 2026-04-16 12:16:11
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 手撕：两个字符串的编辑距离/转换？

（编程题）

### Q: 手撕：二叉树根节点到叶节点组成数字的和？

（编程题）

### Q: CUDA算子优化的通用方法？

CUDA 算子优化遵循"分析瓶颈 -> 针对性优化"的原则，以下是系统化的优化方法论：

**Step 0: 确定瓶颈类型**：
- 使用 Nsight Compute 的 Roofline Model 分析 kernel 是 compute-bound 还是 memory-bound
- **Compute-bound**（算术强度高，如 GEMM）：优化方向是提高计算效率（Tensor Core、ILP、减少冗余计算）
- **Memory-bound**（算术强度低，如 elementwise/reduction）：优化方向是减少访存量、提高带宽利用率

**1. 合并全局内存访问（Coalesced Access）**：
- 同一 warp 的 32 线程访问连续 128 字节对齐的地址 -> 合并为一次 128B 事务
- 非合并访问（stride、random）可能导致实际带宽利用率降至 10-20%
- 修复方法：调整数据布局使线程 ID 对应连续地址（如 AoS -> SoA 转换）

**2. 使用 Shared Memory 缓存复用数据**：
- 对 tile 内多次使用的数据，先从 Global Memory 搬到 Shared Memory（一次全局读取），然后在 Shared Memory 中多次读取
- 典型场景：GEMM 的 K 维度循环中，A/B tile 被复用 N_tile/M_tile 次
- 注意避免 bank conflict（padding 或 swizzle）

**3. 避免 Bank Conflict 和 Warp Divergence**：
- Bank Conflict：shared memory 32 bank 冲突导致串行化。Padding +1 解决
- Warp Divergence：if/else 分支在 warp 内走不同路径，串行执行两个分支。优化：让连续 threadIdx 走相同分支路径

**4. 增加 ILP（Instruction Level Parallelism）**：
- 每个线程内多个独立操作可以流水线执行
- 方法：循环展开（`#pragma unroll`）、每线程处理多个元素
- 效果：隐藏指令延迟（尤其是内存访问延迟），提升每线程的有效吞吐

**5. 向量化加载（Vectorized Load/Store）**：
- 使用 `float4`/`int4` 等 128-bit 宽加载指令
- 每条加载指令搬运 4 个 float（16 字节），减少指令数和地址计算开销
- 效果：对 memory-bound kernel 通常提升 20-40% 性能
- 要求：访问地址必须对齐到向量宽度

**6. Kernel Fusion 减少读写和 launch 开销**：
- 将多个连续的 elementwise/reduction kernel 合并为一个
- 中间结果保留在寄存器/shared memory 中，避免写出再读入 Global Memory
- 减少 kernel launch 数（每次 launch ~5us 开销）
- 典型融合：LayerNorm = reduce(mean) + reduce(var) + normalize + scale，可融合为单 kernel

**优化效果参考**：一个 naive GEMM 可能只有峰值性能的 1-5%，经过 tiling + shared memory + 向量化 + double buffer 可达 50-70%，再加 Tensor Core 可达 80-90%（接近 cuBLAS 水平）。

### Q: CUDA Reduce优化过程？

Reduce（归约）是最基础的并行算子，其优化过程是理解 GPU 并行编程的经典教材：

**Level 0: Naive Interleaved Addressing（最差）**：
```cuda
// stride 从 1 开始翻倍
if (threadIdx.x % (2*stride) == 0)
    sdata[threadIdx.x] += sdata[threadIdx.x + stride];
```
- **问题**：偶数线程工作、奇数线程空闲（warp divergence）。50% 线程浪费
- **性能**：~3% 峰值带宽

**Level 1: Sequential Addressing（消除 divergence）**：
```cuda
// 使用连续线程做归约
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```
- **改进**：前 N/2 个连续线程工作，避免 warp 内 divergence（整个 warp 要么全执行要么全跳过）
- **性能**：~15% 峰值带宽

**Level 2: First Add During Load（加载时归约）**：
```cuda
// 加载时就做第一轮归约：每线程加载两个元素相加
sdata[tid] = input[tid] + input[tid + blockDim.x];
```
- **改进**：将 block 处理的元素范围扩大一倍，无额外 cost（load 时顺便做加法）
- 可扩展为每线程加载 4/8 个元素做归约（grid stride loop）

**Level 3: Warp Shuffle（消除 shared memory 开销）**：
```cuda
// 最后 5 轮（warp 内 32 线程）用 shuffle
for (int offset = 16; offset > 0; offset >>= 1)
    val = val + __shfl_down_sync(0xffffffff, val, offset);
```
- **改进**：warp 内线程天然同步，无需 `__syncthreads`；通过寄存器网络通信，无需 shared memory
- 延迟：shuffle ~5 周期 vs shared memory ~20 周期

**Level 4: 展开最后几轮循环**：
```cuda
// 手动展开 warp 内归约（编译器可能做不好）
if (tid < 32) {
    warpReduce(sdata, tid);  // 无 __syncthreads
}
```
- **改进**：消除最后几轮的循环判断和同步开销

**Level 5: 多 Block 归约策略**：
- 方案 A：Block 内归约 -> 每 Block 结果写到全局数组 -> 启动第二个 kernel 归约
- 方案 B：Block 内归约 -> `atomicAdd` 到全局变量（简单但竞争高）
- 方案 C：Cooperative Groups + grid-level sync（新架构支持）
- 实践中方案 A 最稳定（两个 kernel 的 launch 开销在大数组时可忽略）

**最终性能**：优化后的 reduce kernel 可达 HBM 带宽的 85-95%（因为 reduce 本身就是 memory-bound），此时成为真正的带宽受限——无法进一步优化。

### Q: CUDA中blockDim和blockIdx的含义？

CUDA 的执行模型采用 Grid -> Block -> Thread 三级层次结构，`blockDim` 和 `blockIdx` 是最基本的内置变量：

**blockDim（Block Dimension）**：
- 含义：当前 block 中线程的组织形式（维度大小）
- 类型：`dim3` 结构体，有 `.x`, `.y`, `.z` 三个分量
- 示例：`kernel<<<grid, dim3(256, 1, 1)>>>`，则 `blockDim.x = 256`，`blockDim.y = 1`，`blockDim.z = 1`
- 约束：`blockDim.x * blockDim.y * blockDim.z ≤ 1024`（每 block 最大线程数）
- 每个维度的上限：x ≤ 1024, y ≤ 1024, z ≤ 64

**blockIdx（Block Index）**：
- 含义：当前 block 在 grid 中的位置索引
- 类型：`dim3` 结构体
- 示例：64 个 block 的一维 grid 中，各 block 的 `blockIdx.x` 分别为 0, 1, ..., 63
- 约束：`gridDim.x ≤ 2^31 - 1`（约 21 亿），`gridDim.y/z ≤ 65535`

**全局线程 ID 计算**：
```cuda
// 1D 情况（最常见）
int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D 情况
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int global_tid = row * width + col;
```

**使用场景**：
- `blockDim` 用于计算 block 内线程的相对偏移和循环步长
- `blockIdx` 用于确定 block 负责处理数据的起始位置
- `gridDim`（grid 维度）用于 grid-stride loop：`for (int i = tid; i < n; i += blockDim.x * gridDim.x)`

**设计原则**：
- blockDim 通常选择 128/256/512（32 的倍数，保证完整 warp）
- gridDim 通常设为 `(N + blockDim - 1) / blockDim` 确保覆盖所有数据
- 2D/3D 组织用于矩阵/图像/3D 数据，使索引计算更直观
