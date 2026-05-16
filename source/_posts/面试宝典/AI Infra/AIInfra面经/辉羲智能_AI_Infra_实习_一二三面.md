---
title: '辉羲智能 AI Infra 实习 一二三面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 几种常见卷积算法的优缺点？

卷积是 CNN 的核心操作，不同实现算法各有最佳适用场景：

**1. Im2col + GEMM**：
- **原理**：将每个卷积窗口展开为一列，形成大矩阵，然后调用高度优化的 GEMM 库计算
- **优点**：实现简单；可直接利用 cuBLAS 等高度优化的矩阵乘库；通用性好（任意 kernel size/stride/dilation）
- **缺点**：Im2col 导致内存膨胀 K*K 倍（3x3 卷积膨胀 9 倍），额外的数据重排 overhead
- **适用**：大多数框架的默认实现，通用场景
- **优化**：Implicit GEMM（不显式展开，在 GEMM kernel 内隐式计算索引）消除内存膨胀，cuDNN 使用此方式

**2. Direct Convolution**：
- **原理**：直接按卷积数学定义计算，7 层嵌套循环（batch, C_out, H_out, W_out, C_in, KH, KW）
- **优点**：无额外内存开销；实现最直观；对小 kernel 无变换 overhead
- **缺点**：数据复用率低（同一个输入元素被多次从内存读取）；难以充分利用 SIMD/Tensor Core
- **适用**：1x1 卷积（退化为 GEMM，无展开需要）、Depthwise 卷积（每通道独立计算，im2col 性价比低）
- **性能**：对标准卷积性能远不如 im2col+GEMM，但对 depthwise 是最优选择

**3. Winograd 变换**：
- **原理**：基于 Winograd 最小滤波算法，将卷积转换为频域点乘。F(m,r)：将 r×r 卷积在 m×m 输出 tile 上的 m*r 次乘法减为 (m+r-1)^2 次
- **加速比**：F(2,3) 将 3x3 卷积从 9 次乘法减为 (2+3-1)^2=16 次乘法计算 4 个输出 -> 等效每输出 4 次乘法 vs 原始 9 次 = 2.25x 加速
- **优点**：显著减少 FLOPs，对 3x3 卷积效果最好
- **缺点**：
  - 变换矩阵含大系数，FP16 下数值误差累积严重（精度下降）
  - 变换本身有 overhead（输入变换 + 输出逆变换）
  - 只适用于小 kernel（3x3, 5x5），大 kernel 变换矩阵太大
  - stride > 1 或 dilation > 1 时不适用
- **适用**：ResNet/VGG 等大量 3x3 卷积的网络，FP32 精度要求下

**4. FFT 卷积**：
- **原理**：利用卷积定理——时域卷积 = 频域逐元素相乘。输入和 kernel 做 FFT -> 频域点乘 -> IFFT 回时域
- **优点**：大 kernel 时复杂度从 O(N^2 * K^2) 降为 O(N^2 * log N)，理论加速显著
- **缺点**：
  - FFT 本身计算开销大，小 kernel（3x3/5x5）时 overhead 远超节省的计算
  - 需要 padding 到 2 的幂次浪费内存
  - 频域乘法的数值精度不如直接计算
  - 实现复杂度高
- **适用**：kernel size > 7-11 的场景（音频处理、某些科学计算），在 CV 深度学习中几乎不用

**cuDNN 的自动选择**：调用 `cudnnFindConvolutionForwardAlgorithm` 自动 benchmark 所有可用算法，选择当前 shape 下最快的。不同 batch size/channel/spatial size 的最优算法通常不同。

### Q: 写算子时为什么会发生Bank Conflict？如何解决？

**Bank Conflict 的发生机制**：

GPU Shared Memory 的物理组织：32 个 bank，每个 bank 宽 4 字节，连续 4 字节地址依次分配到 bank 0, 1, ..., 31, 0, 1, ...（地址交错分配）。

**发生条件**：一个 warp（32 线程）的内存事务中，多个线程访问同一 bank 的**不同行**（不同地址）。此时这些访问必须串行化。

**特例（不冲突）**：多线程访问同一 bank 的**同一地址**时触发 broadcast，一次完成。

**常见触发场景**：

1. **转置操作**：矩阵行优先存储到 shared memory 后按列读取
   ```cuda
   __shared__ float smem[32][32];
   smem[threadIdx.x][threadIdx.y] = ...;  // 写无冲突（stride=1 per row）
   val = smem[threadIdx.y][threadIdx.x];   // 读有冲突（列访问 stride=32）
   ```
   列访问时 stride=32，所有线程访问同一 bank（32-way conflict）

2. **特定 stride 访问**：`smem[threadIdx.x * stride]` 中 stride 为 32 的因子时冲突
   - stride=2: 2-way conflict（相邻线程映射到同一 bank）
   - stride=4: 4-way conflict
   - stride=32: 32-way conflict（最坏）

3. **结构体数组**：`struct { float x, y, z, w; } smem[32]`，访问所有元素的 `.x` 字段时 stride=16 bytes=4 banks -> 可能冲突

**解决方案**：

**1. Padding（最常用最简单）**：
```cuda
// 原始：列访问 32-way conflict
__shared__ float smem[32][32];

// 修复：每行多一个 float
__shared__ float smem[32][33];
// 现在列访问 stride=33, 33 mod 32 = 1, 完全无冲突！
```
代价：浪费 3% shared memory（N/(N+1) 利用率）

**2. Swizzle（地址映射）**：
```cuda
// XOR-based swizzle
int physical_col = logical_col ^ (row & 0x1F);
smem[row][physical_col] = data;
```
通过位操作将逻辑地址映射为分散在不同 bank 的物理地址。无额外空间开销，但增加地址计算指令。CUTLASS 中广泛使用。

**3. 调整访问模式**：重新设计算法使相邻线程访问相邻地址。例如矩阵转置：先 coalesced 读入 shared memory（按行写），padding 后再 coalesced 写出（按列读但无冲突）。

**诊断方法**：Nsight Compute -> Memory Workload -> Shared Memory -> Wavefronts/Requests。理想值为 1.0，>1.0 说明存在 bank conflict。

### Q: CPU和GPU架构的区别？

CPU 和 GPU 是针对不同计算模式优化的处理器，架构设计哲学截然不同：

**核心设计哲学**：
- **CPU**：优化**延迟**（Latency-oriented）——让单个任务尽快完成
- **GPU**：优化**吞吐**（Throughput-oriented）——让大量任务同时进行

**架构对比**：

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 4-128（大核心） | 数千-上万（小核心） |
| 时钟频率 | 3-5 GHz | 1-2 GHz |
| 单核能力 | 强（复杂分支预测/乱序执行/投机执行） | 弱（简单 in-order 执行） |
| Cache 占比 | 大（50%+芯片面积给 Cache） | 小（大部分面积给 ALU） |
| 控制逻辑 | 复杂（分支预测/OoO/寄存器重命名） | 简单（SIMT，统一控制） |
| 寄存器文件 | 小（~100 个通用寄存器/核） | 巨大（~65536 个/SM） |
| 内存带宽 | ~100 GB/s (DDR5) | ~2-3 TB/s (HBM3) |
| 适合任务 | 串行/低延迟/分支密集/复杂控制流 | 大规模并行/数据并行/规则计算 |

**为什么 GPU 适合深度学习**：
- 矩阵乘法是高度并行的：C[i,j] = sum(A[i,k] * B[k,j])，每个输出元素独立计算
- 数据规则：张量操作大多是相同操作应用于大量数据（SIMD 友好）
- 计算密集：GEMM 的算术强度 O(N)，可以有效利用大量 ALU
- 带宽需求大：模型参数和中间激活值巨大，需要 HBM 的高带宽

**GPU 不擅长的场景**：
- 高度分支的控制流（warp divergence 降低效率）
- 串行依赖链（如链表遍历/递归）
- 小数据量任务（kernel launch overhead 占主导）
- 延迟敏感的单次操作（GPU 优化吞吐而非单次延迟）

**性能数据对比**（典型值）：
- 矩阵乘（4096x4096）：CPU（512 GFLOPS with AVX-512）vs GPU A100（312 TFLOPS FP16）≈ 600x 差距
- 单次条件分支：CPU ~1ns vs GPU ~100ns（warp divergence）

### Q: Grid、Block、Thread的理解？

CUDA 的三级并行层次结构是 GPU 编程的核心概念：

**Thread（线程）—— 最小执行单位**：
- 每个 thread 执行 kernel 函数的一个实例
- 有自己的寄存器（私有）和 local memory
- 通过 `threadIdx.x/y/z` 标识在 block 内的位置
- **Warp**：32 个连续 thread 组成一个 warp，以 SIMT 方式同步执行同一条指令。warp 是实际的调度和执行单位

**Block（线程块）—— 协作单位**：
- 一组 thread 的集合（最多 1024 个 thread）
- Block 内 thread 可以通过 **shared memory** 协作和通信
- 可以用 `__syncthreads()` 做 block 内同步（barrier）
- 一个 block 映射到一个 SM（Streaming Multiprocessor）上执行
- 通过 `blockIdx.x/y/z` 标识在 grid 中的位置
- Block 之间**无法直接通信**（除了通过 global memory）

**Grid（网格）—— 完整工作空间**：
- 一次 kernel launch 的所有 block 组成 grid
- Grid 可以是 1D/2D/3D 组织（方便映射不同形状的问题）
- `gridDim` 描述 grid 中 block 的数量
- Grid 内所有 block **可以乱序执行**（SM 调度器按需分配）

**映射关系**：
```
Grid (全部工作)
├── Block 0 -> SM_0
│   ├── Warp 0 (thread 0-31)
│   ├── Warp 1 (thread 32-63)
│   └── ...
├── Block 1 -> SM_3 (SM 调度器分配)
├── Block 2 -> SM_1
└── ...
```

**设计原则**：
- Block 大小选择 128/256/512（32 的整数倍保证完整 warp）
- 每个 SM 可同时运行多个 block（受 shared memory 和寄存器限制）
- Block 数量应 >> SM 数量（确保所有 SM 都有工作，隐藏调度间隙）
- Grid 维度组织应匹配数据的逻辑结构（2D grid 处理 2D 数据更直观）

### Q: 写算子时如何最大化利用缓存？

缓存利用是 memory-bound kernel 性能的关键。核心原则是**让数据访问模式匹配缓存层级的容量和替换策略**：

**1. Tiling：让每次迭代的工作集匹配 L1/L2 大小**：
- L1 Cache per SM：128-192 KB（A100），每个 tile 的数据量应 < L1 容量
- L2 Cache shared：40-50 MB（A100），整个 kernel 的活跃工作集应考虑 L2
- GEMM 示例：block tile (128x128) 的 A/B 子矩阵各 128*BK*2=16KB (BK=64, FP16)，可放入 L1

**2. 空间局部性（Spatial Locality）**：
- 连续线程访问连续地址（coalesced access）：一次 L2 cache line（128B）被完整利用
- 非连续访问：cache line 加载 128B 但只用了其中 4B（浪费 97% 带宽）
- 数据布局选择：根据访问模式选 row-major/col-major/interleaved

**3. 时间局部性（Temporal Locality）**：
- 加载数据后尽快多次使用（在被 evict 出 cache 前）
- GEMM K 循环中：A/B tile 加载后被 tile 内所有计算复用
- 反面教材：streaming access（每个数据只用一次）无法利用 cache

**4. 避免 Cache Thrashing**：
- 如果工作集 >> cache 容量，数据频繁被 evict 后重新 load（miss rate 接近 100%）
- 解决：减小 tile 大小使工作集匹配 cache；或使用 `__ldg()` 绕过 L1 走 read-only cache 路径

**5. L2 Cache Residency Control（A100+）**：
- `cudaAccessPolicyWindow` API 可以设置数据的 L2 缓存驻留策略
- 对频繁访问的小数据（如 bias/scale）设置高驻留优先级
- 对 streaming 数据设置 evict-first 策略避免污染 cache

**6. 预取（Prefetch）**：
- 软件预取：在计算当前 tile 时发起下一个 tile 的加载请求
- CP.ASYNC（Ampere+）：异步 global->shared memory 拷贝，不阻塞当前 warp
- Double Buffer：两套 shared memory buffer 交替使用，加载和计算完全重叠

**实际效果**：好的缓存利用可以让 memory-bound kernel 的有效带宽从理论峰值的 30-40% 提升到 80-90%。

### Q: 线程束分歧（Warp Divergence）是什么？

**Warp Divergence** 是 GPU SIMT（Single Instruction, Multiple Threads）执行模型下条件分支导致的效率损失：

**SIMT 执行模型**：
- 一个 warp = 32 个线程，共享一个程序计数器（PC），同时执行**同一条指令**
- 这是 GPU 高吞吐的关键——一次指令调度驱动 32 个线程并行执行

**分歧发生时**：
```cuda
if (threadIdx.x < 16) {
    // Branch A: 线程 0-15 执行
    result = compute_A();
} else {
    // Branch B: 线程 16-31 执行
    result = compute_B();
}
```

GPU 处理方式：
1. 执行 Branch A：线程 0-15 活跃，线程 16-31 被 mask（inactive，空等）
2. 执行 Branch B：线程 16-31 活跃，线程 0-15 被 mask
3. 两个分支**串行执行**，总时间 = time(A) + time(B)

**性能影响**：
- 2-way divergence：效率降为 50%（两个分支串行）
- N-way divergence：效率降为 1/N
- 最坏情况：32 个线程走 32 个不同分支 -> 32x 减速

**优化方法**：

1. **按 warp 对齐分支条件**：确保同一 warp 内线程走相同路径
   ```cuda
   // 好：warp 0 全走 A，warp 1 全走 B
   if (threadIdx.x / 32 < threshold) { ... }
   
   // 差：warp 内一半走 A 一半走 B
   if (threadIdx.x % 2 == 0) { ... }
   ```

2. **数据重排**：将需要相同处理的数据放在连续位置（使连续线程处理同类数据）

3. **无分支替代**：用数学运算替代条件分支
   ```cuda
   // 有分支
   if (x > 0) y = x; else y = 0;
   // 无分支
   y = x * (x > 0);  // 或 y = max(0, x);
   ```

4. **Predication**：短分支编译器自动用 predicated 指令实现（两个分支都计算但选择性写入），避免控制流分歧。但只适用于分支体很短（1-3 条指令）的情况。

**注意**：Volta+ 架构支持 Independent Thread Scheduling，每个线程有独立的 PC 和调用栈，但 divergence 的性能惩罚本质上仍然存在（warp 仍需串行执行不同路径）。

### Q: 手撕：CUDA矩阵乘算子？

（编程题）

### Q: blockDim.x和gridDim.x最大能开多少？

CUDA 硬件对 grid 和 block 的维度有明确限制：

**Block 维度限制**：
- `blockDim.x`：最大 **1024**
- `blockDim.y`：最大 **1024**
- `blockDim.z`：最大 **64**
- **总线程数限制**：`blockDim.x * blockDim.y * blockDim.z ≤ 1024`
- 示例：(1024, 1, 1) 合法，(32, 32, 1) 合法（=1024），(32, 32, 2) 非法（=2048 > 1024）

**Grid 维度限制**：
- `gridDim.x`：最大 **2^31 - 1**（约 21 亿，Compute Capability 3.0+）
- `gridDim.y`：最大 **65535**
- `gridDim.z`：最大 **65535**

**查询方法**：
```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
// prop.maxThreadsPerBlock = 1024
// prop.maxThreadsDim[3] = {1024, 1024, 64}
// prop.maxGridSize[3] = {2147483647, 65535, 65535}
```

**实际设计考虑**：
- blockDim 通常选 128/256/512（经验最优值），不会用到极限 1024（因为 occupancy 受寄存器和 shared memory 限制）
- gridDim.x 的 21 亿上限在实际中几乎不会成为约束
- 真正的限制来自：每 SM 最大 block 数（16-32 blocks/SM）、每 SM 最大 thread 数（2048）、每 SM 寄存器总量（65536）、每 SM shared memory 容量

### Q: 共享内存和Cache的区别？

共享内存（Shared Memory）和 Cache（L1/L2）都是片上快速存储，但管理方式和使用场景完全不同：

| 特性 | Shared Memory | Cache (L1/L2) |
|------|---------------|----------------|
| 管理方式 | **程序员显式控制** | **硬件自动管理** |
| 分配/释放 | 代码中声明，block 生命周期 | 透明，基于访问模式自动填充 |
| 地址控制 | 程序员决定存什么、存在哪 | 硬件决定缓存哪些数据 |
| 延迟 | 确定性 ~20-30 cycles（无 miss） | 不确定（hit ~20 cycles, miss ~400 cycles） |
| 容量 | A100: 最大 164KB/SM（可配置） | L1: 与 shared memory 共享 192KB; L2: 40MB |
| 一致性 | Block 内共享，block 间不可见 | L2 全局可见 |
| 数据搬运 | 需要显式 load/store（代码写搬运逻辑） | 自动（hardware prefetch + demand fetch） |

**Shared Memory 的优势**：
- 延迟可预测（永远不会 miss，因为数据是程序员放进去的）
- 带宽极高（无 tag 查找开销）
- 可以精确控制数据复用模式（完全匹配算法需求）
- 支持 block 内线程间通信（协作计算如 reduction）

**Cache 的优势**：
- 无需修改代码即可获得加速（透明性）
- 自动适应动态访问模式（不需要预先知道数据访问顺序）
- 无需管理数据生命周期和搬运逻辑
- L2 cache 全局共享，跨 block 的数据可以被自动缓存复用

**最佳实践**：
- 对已知会被多次复用的数据（如 GEMM 的 tile）：使用 shared memory 显式管理
- 对偶尔复用或不确定复用模式的数据：依赖 L1/L2 cache 自动缓存
- A100 的 L1/shared memory 共享 192KB pool，可配置比例（如 164KB shared + 28KB L1）

### Q: Tensor Core和CUDA Core的区别？加速矩阵乘谁更快？

**CUDA Core（标量运算单元）**：
- 每个 CUDA Core 每周期执行 **1 次** FMA（Fused Multiply-Add）：a*b+c
- 是通用标量运算单元，可执行任意浮点/整数运算
- A100 有 6912 个 FP32 CUDA Core -> 19.5 TFLOPS FP32

**Tensor Core（矩阵运算单元）**：
- 每个 Tensor Core 每周期执行一次小矩阵乘加：D = A*B + C
  - Volta/Turing: 4x4x4 FP16 乘加
  - Ampere: 各种形状如 16x8x8、16x8x16
  - Hopper: 更大的 warp-group 级矩阵操作
- 专用电路，只能做矩阵乘法（不能做通用计算）
- A100 有 432 个 Tensor Core -> **312 TFLOPS FP16**（是 CUDA Core 的 16x）

**性能对比（A100）**：

| 精度 | CUDA Core | Tensor Core | 加速比 |
|------|-----------|-------------|--------|
| FP32 | 19.5 TFLOPS | 156 TFLOPS (TF32) | 8x |
| FP16 | 39 TFLOPS | 312 TFLOPS | 8x |
| INT8 | — | 624 TOPS | — |
| FP8 (H100) | — | 1979 TFLOPS | — |

**Tensor Core 的使用条件**：
- 输入矩阵维度必须满足对齐要求（如 FP16 需要 M/N/K 为 8 或 16 的倍数）
- 需要特定数据布局（如列优先的 fragment）
- 通过 WMMA API 或 MMA PTX 指令使用
- cuBLAS/cuDNN/CUTLASS 自动利用 Tensor Core

**为什么差距这么大**：
- CUDA Core 做标量 FMA：1 个乘法 + 1 个加法 = 2 FLOPs/cycle/core
- Tensor Core 做矩阵 FMA：如 16x8x8 = 16*8*8*2 = 2048 FLOPs/cycle/TC
- 面积效率：Tensor Core 用专用电路（乘法器阵列 + 加法树）替代通用逻辑，单位面积算力远高于通用 ALU

**实际应用**：GEMM、Attention（FlashAttention 内部的分块矩阵乘）、卷积（im2col 后的 GEMM）。几乎所有 AI 模型的核心计算都可以利用 Tensor Core。

### Q: Softmax算法在深度学习中的应用？

Softmax 在深度学习中无处不在，每个应用场景对实现有不同的性能要求：

**1. Attention 权重计算（最核心应用）**：
```
Attention_weights = softmax(Q @ K^T / sqrt(d_k))
```
- 对 attention score 矩阵的每一行做 softmax（沿 key dimension 归一化）
- 挑战：长序列时矩阵巨大（seq_len^2），FlashAttention 用 Online Softmax 分块计算
- 精度要求：FP32 累加（softmax 对数值精度敏感，FP16 累加可能精度不够）

**2. 分类输出层**：
```
probabilities = softmax(logits)  # logits: [batch, vocab_size]
```
- 将模型的原始输出（logits）转为概率分布
- LLM 的 vocab_size 通常 32K-128K，softmax 计算量不小
- 配合 cross-entropy loss：`loss = -log(softmax(logits)[target])`，实际实现中 softmax + log + NLL 融合为 `log_softmax` + `nll_loss`

**3. MoE 门控（Expert Routing）**：
```
gate_weights = softmax(input @ W_gate)  # [batch*seq, n_experts]
```
- 决定每个 token 发送到哪些专家及其权重
- 通常只有 top-k（如 top-2）专家被激活，所以经常配合 top-k 操作
- 负载均衡 loss 也基于 softmax 输出的分布

**4. 温度采样（LLM 生成）**：
```
sampling_probs = softmax(logits / temperature)
```
- Temperature > 1：分布更平坦 -> 生成更随机
- Temperature < 1：分布更尖锐 -> 生成更确定
- Temperature = 0 时退化为 argmax（贪心解码）

**5. 对比学习（Contrastive Learning）**：
```
similarity_matrix = softmax(embeddings @ embeddings.T / tau)
```
- InfoNCE loss 中对相似度矩阵做 softmax，tau 为温度参数
- CLIP、SimCLR 等模型的核心计算

**6. 注意力门控/软选择**：
- 各种注意力机制中用 softmax 实现"软选择"（soft attention vs hard attention）
- Gumbel-Softmax：在离散选择问题中提供可微分的近似

**实现性能考量**：softmax 是 memory-bound 操作（每元素约 5 FLOPs 但需读写两次），优化重点是 fusion（与前后算子融合）和减少 global memory 遍历次数（online softmax）。

### Q: 手撕：CUDA Softmax算子？

（编程题）
