---
title: 高效CUDA编程速查
date: 2026-03-25 15:00:00
categories:
  - [AI Infra, CUDA编程]
tags: [CUDA, GPU, 性能优化, Tensor Core, Nsight]
---

高效的CUDA程序需要从硬件理解、算法设计、内核实现、存储优化到性能分析的全方位把控。本文提供一个系统化的优化速查逻辑，覆盖从硬件基准到实战调优的完整流程。

<!-- more -->

## 目录

- [1. 硬件性能基准测试](#1-硬件性能基准测试)
- [2. 算法设计](#2-算法设计)
- [3. Kernel设计](#3-Kernel设计)
- [4. 存储优化](#4-存储优化)
- [5. 延迟隐藏](#5-延迟隐藏)
- [6. 指令级优化](#6-指令级优化)
- [7. 性能分析工具链](#7-性能分析工具链)
- [8. 常见误区与最佳实践](#8-常见误区与最佳实践)
- [9. 快速检查清单](#9-快速检查清单)
- [10. 实战优化流程](#10-实战优化流程)

## 1. 硬件性能基准测试

### 1.1 关键硬件指标

**计算能力**

- **CUDA Core vs Tensor Core**：FP64/FP32在CUDA Core执行，TF32/FP16/INT8/FP8在Tensor Core执行，算力差异可达10-20倍
- **混合精度策略**：关键路径用FP32，大规模矩阵用TF32/FP16，推理用INT8/FP4

**并行度参数**

- **SM数量**：调度和资源分配的基本单位（Hopper 132个，Blackwell 148个）
- **Warp规模**：32线程的固定执行单元，是SIMT调度的最小粒度
- **并发限制**：每SM最多64个活跃warp（2048线程）、32个thread block

**存储层次**

```text
Register (最快，~1 cycle)
  ↓
L1/Constant Cache (~28 cycles)
  ↓
Shared Memory (~20-30 cycles，可编程)
  ↓
L2 Cache (~200 cycles)
  ↓
HBM/Global Memory (~400-800 cycles)
```

**关键架构数据 (Hopper H100/Blackwell B100)**

| 资源类型 | Hopper H100 | Blackwell B100 |
|---------|-------------|----------------|
| SM数量 | 132 | 148 |
| FP32算力 | 67 TFLOPS | ~125 TFLOPS |
| FP8算力 | 1979 TFLOPS | 2500 TFLOPS |
| HBM带宽 | 4.8 TB/s | 8 TB/s |
| L2缓存 | 60 MB | 192 MB |
| 每SM寄存器 | 64K (32-bit) | 64K (32-bit) |
| 每SM共享内存 | 228 KB | 228 KB |

### 1.2 性能建模

**Roofline分析**

1. **实测峰值性能**：用矩阵乘法测GEMM实际算力，用带宽测试工具测各级存储吞吐
2. **计算算术强度**：`AI = FLOPs / 字节访问量`
   - AI < 10：访存瓶颈（Memory-bound）
   - 10 < AI < 50：混合瓶颈
   - AI > 50：计算瓶颈（Compute-bound）
3. **绘制roofline曲线**：横轴算术强度，纵轴性能，找到计算/访存平衡点

**工具推荐**

- `nvidia-smi`：GPU利用率、显存占用、温度监控
- `cudaDeviceProp`：查询设备详细参数
- `bandwidthTest`（CUDA Samples）：测试各级存储带宽

## 2. 算法设计

### 2.1 并行分解策略

**数据并行 vs 任务并行**

- **数据并行**（最常见）：同一操作应用于不同数据片段（如矩阵元素级运算）
- **任务并行**：不同操作并发执行（用多stream实现）
- **混合模式**：如Transformer中attention和FFN可部分重叠

**依赖分析**

- 识别可并行的计算路径（DAG分析）
- 处理循环依赖：考虑前缀和、规约等并行算法
- 数据竞争：使用原子操作或分块规约避免冲突

### 2.2 计算密度优化

**算子融合（Kernel Fusion）**

- 将多个逐元素操作合并为单个kernel
- 减少中间结果的global memory往返
- 示例：`(A + B) * C → fused_kernel(A, B, C)`

**Tiling策略**

- 将大问题分解为L2/Shared Memory可容纳的子问题
- 提高数据重用率，降低访存压力
- GEMM典型做法：矩阵分块为64x64或128x128的tile

**算术强度提升技巧**

- 循环融合：减少数据加载次数
- 提前计算：如预计算1/sqrt(x)
- 批处理：增加单次kernel的工作量

## 3. Kernel设计

### 3.1 线程层次配置

**Grid-Block-Thread三层模型**

```cpp
// 典型配置示例
dim3 blockDim(256);  // 每block 256线程 = 8 warps
dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args);
```

**Block Size选择经验**

- **128-256线程**：多数场景的甜点区（4-8个warp）
- **512-1024线程**：计算密集型，但要注意资源限制
- **64线程或更少**：访存密集型，需要更高并发度

**Occupancy计算**

```cpp
Occupancy = active_warps / max_warps_per_SM

影响因素：
1. 寄存器使用量：regs_per_thread × block_size
2. Shared Memory使用量：smem_per_block
3. Block数量限制：max 32 blocks per SM
```

**实用API**

```cpp
// 自动计算最优block size
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                    kernel, dynamicSMemSize, blockSizeLimit);
```

### 3.2 Tensor Core利用

**优先级最高的优化**

- **WMMA (Warp Matrix Multiply-Accumulate)**：手动编程接口
- **CUTLASS**：NVIDIA提供的模板库，易用性好
- **cuBLAS/cuDNN**：调用高度优化的库函数
- **TMA (Tensor Memory Accelerator，Hopper+)**：硬件加速的异步数据搬运

**Tensor Core要求**

- 矩阵维度对齐（通常16的倍数）
- 使用特定数据类型（FP16/BF16/TF32/INT8）
- 内存布局符合要求（行优先/列优先）

### 3.3 避免Warp Divergence

**问题根源**

```cpp
// 坏示例：32个线程可能走不同分支
if (threadIdx.x < threshold) {
    heavy_computation_A();  // 16个线程执行
} else {
    heavy_computation_B();  // 16个线程等待
}
```

**优化策略**

1. **Predication（谓词执行）**

```cpp
// 改为无分支版本
int mask = (threadIdx.x < threshold);
result = mask * compute_A() + (1-mask) * compute_B();
```

2. **数据重排**：将相同分支的数据归并到连续warp
3. **Kernel分离**：将不同分支拆成独立kernel
4. **使用Warp级别的投票函数**：`__ballot_sync()`, `__any_sync()`

### 3.4 Thread Block Cluster (Hopper+)

**新特性优势**

- 多个block间可通过Distributed Shared Memory直接通信
- 减少global memory往返
- 适合需要跨block协作的算法

```cpp
// 声明cluster维度
__global__ void __cluster_dims__(2, 2, 1) kernel() {
    // 使用cluster级别的shared memory
}
```

## 4. 存储优化

### 4.1 Global Memory访问模式

**黄金法则：合并访问（Coalesced Access）**

**理想模式**

```cpp
// 好：连续、对齐的访问
float* data = ...;
int tid = threadIdx.x;
float value = data[tid];  // warp内32线程访问data[0]到data[31]
```

**糟糕模式**

```cpp
// 坏：跨步访问
float value = data[tid * stride];  // stride > 1会导致多次内存事务

// 坏：随机访问
float value = data[random_indices[tid]];

// 坏：反向访问
float value = data[blockDim.x - tid];
```

**优化技术**

- **数据布局转换**：AoS (Array of Structures) → SoA (Structure of Arrays)

```cpp
// AoS (坏)
struct Particle { float x, y, z; };
Particle particles[N];
// 访问x坐标时跨步为3

// SoA (好)
struct Particles {
    float x[N];
    float y[N];
    float z[N];
};
// 访问x坐标时连续
```

- **填充对齐**：确保数据地址对齐到32/128字节边界
- **使用向量类型**：`float4`一次加载4个float

### 4.2 Shared Memory优化

**Bank Conflict避免**

Shared Memory分为32个bank（每个4字节宽），同一warp内多个线程访问同一bank会串行化。

```cpp
// 坏：列访问导致32-way bank conflict
__shared__ float tile[32][32];
float value = tile[threadIdx.x][0];  // 32个线程都访问bank 0

// 好：行访问无冲突
float value = tile[0][threadIdx.x];  // 每个线程访问不同bank
```

**解决方案**

1. **Padding法**

```cpp
__shared__ float tile[32][33];  // 多一列避免固定模式冲突
```

2. **转置法**：先把数据转置到shared memory

```cpp
// 加载时转置
tile[threadIdx.y][threadIdx.x] = global[idx];
__syncthreads();
// 访问时就是连续的了
```

3. **交错存储**：对于步长访问，调整存储模式

**Bank冲突检测**

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
           --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
           ./your_app
```

**大小配置**

```cpp
// 静态分配
__shared__ float smem[256];

// 动态分配
extern __shared__ float smem[];
kernel<<<grid, block, sharedMemBytes>>>();
```

**多Bank利用**

- 使用Volta+的独立L1/Shared Memory分区策略
- `cudaFuncSetAttribute()`调整缓存配置

### 4.3 L2 Cache优化

**L2驻留策略（Hopper+）**

```cpp
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = data;
attr.accessPolicyWindow.num_bytes = size;
attr.accessPolicyWindow.hitRatio = 1.0;  // 尽量保留
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

**跨Block数据复用**

- 设计kernel执行顺序，让相邻block访问相同数据
- L2 Tiling：将工作集调整到L2容量内（H100为60MB，B100为192MB）

### 4.4 常量缓存和纹理缓存

**常量内存（Constant Memory）**

```cpp
__constant__ float coeffs[256];  // 64KB限制

// 主机端设置
cudaMemcpyToSymbol(coeffs, h_coeffs, size);

// 设备端使用
float c = coeffs[idx];  // 广播到warp所有线程
```

**适用场景**：所有线程读取相同数据（广播模式）

**只读缓存（L1/Texture）**

```cpp
// 方法1：__ldg内建函数（Kepler+）
float value = __ldg(&array[idx]);

// 方法2：const __restrict__指针
__global__ void kernel(const float* __restrict__ input) {
    float value = input[idx];  // 编译器可能用只读路径
}
```

### 4.5 寄存器优化

**Register Spilling（寄存器溢出）检测**

```bash
nvcc -Xptxas -v kernel.cu
# 输出示例：
# ptxas info: Used 63 registers, 1024 bytes smem, 0 bytes cmem[0]
# ptxas info: Compiling entry function '_kernel' for 'sm_80'
```

如果看到`stack frame`，说明有寄存器溢出到local memory（慢！）

**减少寄存器压力**

1. **减少活跃变量**

```cpp
// 坏：太多临时变量同时存活
float a = compute1();
float b = compute2();
float c = compute3();
float result = a * b + c;

// 好：尽快释放
float result = compute1() * compute2() + compute3();
```

2. **控制循环展开**

```cpp
#pragma unroll 4  // 限制展开次数
for (int i = 0; i < N; i++) { ... }
```

3. **__launch_bounds__限制**

```cpp
__global__ void __launch_bounds__(256, 4)  // maxThreadsPerBlock=256, minBlocksPerSM=4
kernel() { ... }
```

4. **分段计算**：将大kernel拆分为多个小kernel

**寄存器级别通信**

```cpp
// Warp Shuffle：无需shared memory的warp内通信
float value = __shfl_down_sync(0xffffffff, var, delta);
float sum = __reduce_add_sync(0xffffffff, var);  // Volta+
```

## 5. 延迟隐藏

### 5.1 异步操作

**异步内存拷贝**

```cpp
// 主机到设备异步拷贝
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_data);
cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream);
```

**cp.async指令（Ampere+）**

```cpp
// Global → Shared异步拷贝
#include <cuda/pipeline>
__shared__ float smem[SIZE];
cuda::memcpy_async(smem, &global[idx], cuda::aligned_size_t<16>(SIZE), pipeline);
pipeline.commit();
// ... 做其他计算 ...
pipeline.wait();  // 等待拷贝完成
__syncthreads();
```

**TMA（Tensor Memory Accelerator，Hopper+）**

- 硬件加速的张量搬运
- 支持多维张量的复杂拷贝模式
- warp内广播机制，减少重复访存

### 5.2 多流并发

**Stream优先级**

```cpp
cudaStream_t high_prio_stream, low_prio_stream;
cudaStreamCreateWithPriority(&high_prio_stream, cudaStreamNonBlocking, -1);
cudaStreamCreateWithPriority(&low_prio_stream, cudaStreamNonBlocking, 0);
```

**Pinned Memory（锁页内存）**

```cpp
// 分配
float* h_pinned;
cudaMallocHost(&h_pinned, size);  // 比malloc()快2-3倍传输

// 释放
cudaFreeHost(h_pinned);
```

**流间依赖**

```cpp
cudaEvent_t event;
cudaEventCreate(&event);
kernel1<<<grid, block, 0, stream1>>>();
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0);  // stream2等待stream1
kernel2<<<grid, block, 0, stream2>>>();
```

### 5.3 CUDA Graphs

**优势**

- 减少kernel启动开销（减少CPU端调度时间）
- 整体优化执行路径
- 适合固定拓扑的计算图

**使用方式**

```cpp
// 方法1：显式构建
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);
cudaGraphNode_t kernelNode;
cudaKernelNodeParams params = {...};
cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &params);

// 方法2：捕获模式（更简单）
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel1<<<grid, block, 0, stream>>>();
kernel2<<<grid, block, 0, stream>>>();
cudaStreamEndCapture(stream, &graph);

// 实例化并执行
cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaGraphLaunch(instance, stream);
```

## 6. 指令级优化

### 6.1 数学函数

**快速数学函数**

```cpp
// 慢：标准精度
float y = sinf(x);  // ~100周期

// 快：降低精度
float y = __sinf(x);  // ~20周期

// 编译选项
nvcc -use_fast_math  // 全局启用快速数学
```

**常用替换**

| 标准函数 | 快速函数 | 精度损失 |
|---------|---------|---------|
| `sinf/cosf` | `__sinf/__cosf` | ~2 ulp |
| `expf` | `__expf` | ~2 ulp |
| `logf` | `__logf` | ~2 ulp |
| `powf(x,y)` | `__powf` | 略高 |
| `sqrtf` | `rsqrtf` | 倒数平方根更快 |

### 6.2 算术优化

**FMA指令（Fused Multiply-Add）**

```cpp
float result = __fmaf_rn(a, b, c);  // a*b+c 单指令，精度更高
```

**避免低效操作**

```cpp
// 慢：除法（~20周期）
float result = a / b;

// 快：乘法（~4周期）
float inv_b = 1.0f / b;  // 预计算
float result = a * inv_b;

// 慢：取模
int result = a % b;

// 快：位运算（b为2的幂）
int result = a & (b - 1);
```

**循环展开**

```cpp
#pragma unroll  // 完全展开
for (int i = 0; i < 4; i++) {
    sum += data[i];
}

// 编译后等价于：
sum += data[0];
sum += data[1];
sum += data[2];
sum += data[3];
```

### 6.3 原子操作优化

**Warp级别的规约避免原子操作**

```cpp
// 坏：每个线程都atomicAdd
atomicAdd(&global_sum, local_value);

// 好：先warp内规约
float warp_sum = __reduce_add_sync(0xffffffff, local_value);
if (threadIdx.x % 32 == 0) {
    atomicAdd(&global_sum, warp_sum);  // 减少32倍原子操作
}
```

**使用更快的原子操作（Volta+）**

```cpp
atomicAdd_block();  // block级别，比global更快
atomicAdd_system(); // 跨设备
```

## 7. 性能分析工具链

### 7.1 编译器输出

**寄存器和共享内存使用**

```bash
nvcc -Xptxas -v,-abi=no --resource-usage kernel.cu

# 输出解读：
# registers: 每线程寄存器数（<64为宜）
# shared memory: 每block字节数
# spill loads/stores: 寄存器溢出次数（应为0）
```

**限制寄存器数量**

```bash
nvcc --maxrregcount=32 kernel.cu  # 强制每线程最多32寄存器
```

### 7.2 Profiler工具

**Nsight Compute（Kernel级别详细分析）**

```bash
# 基础profile
ncu --set full -o profile kernel_app

# 关注内存
ncu --metrics l2__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
               l2__t_sectors_pipe_lsu_mem_global_op_st.sum kernel_app

# 关注occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak kernel_app
```

**关键指标**

- `achieved_occupancy`：实际占用率
- `l2_cache_hit_rate`：L2命中率（>70%为佳）
- `global_hit_rate`：全局内存效率
- `shared_efficiency`：共享内存bank冲突（>90%为佳）
- `warp_execution_efficiency`：warp执行效率（分支分化检测）
- `eligible_warps_per_cycle`：可调度warp数

**Nsight Systems（系统级时序分析）**

```bash
nsys profile -o timeline ./app

# 查看：
# - kernel launch开销
# - CPU-GPU数据传输时间
# - 多stream并发情况
# - 主机端瓶颈
```

### 7.3 微基准测试

**带宽测试**

```cpp
__global__ void bandwidth_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float value = data[idx];  // 纯读取
        data[idx] = value;        // 或纯写入
    }
}

// 运行并计时，计算: Bandwidth = N * sizeof(float) / time
```

**Shared Memory延迟测试**

```cpp
__global__ void latency_kernel(float* output) {
    __shared__ float smem[32];
    int start = clock();
    float value = smem[threadIdx.x];  // 单次访问
    int end = clock();
    output[threadIdx.x] = end - start;
}
```

## 8. 常见误区与最佳实践

### 8.1 Occupancy误区

**误区**：Occupancy越高越好

**真相**：

- **Compute-bound任务**：高occupancy帮助有限，因为计算单元已饱和
- **Memory-bound任务**：适度occupancy即可隐藏延迟
- **Tensor Core密集型**：低occupancy（25-50%）也能达到峰值性能

**正确做法**：

1. 优先保证无寄存器/共享内存溢出
2. 测试不同occupancy下的实际性能
3. 计算瓶颈时适当牺牲occupancy换取更多寄存器

### 8.2 Kernel拆分误区

**误区**：把复杂kernel拆成多个小kernel省寄存器

**真相**：

- Launch开销累积（每次几微秒）
- 中间结果写回global memory
- L2缓存利用率下降

**正确做法**：

- 先尝试优化单kernel（循环展开控制、变量生命周期管理）
- 仅在register spilling严重时才拆分
- 使用CUDA Graphs减少多kernel开销

### 8.3 内存优化误区

**误区**：所有局部数组都放寄存器

**真相**：数组超过阈值会溢出到local memory（比shared memory更慢）

**正确做法**：

```cpp
// 小数组：寄存器
float tmp[4];  // OK

// 大数组：shared memory
__shared__ float tile[256];

// 动态大小：考虑拆分或用global memory
```

**误区**：盲目使用`__ldg()`

**真相**：

- Volta后编译器已能自动优化只读访问
- 复杂指针别名场景仍可能有5-30%收益
- 需要profile验证

### 8.4 其他注意事项

**分支预测**

- GPU无分支预测器，所有分支都有开销
- 对热路径的if-else用预测宏优化（CUDA 11+）

**Warp同步**

- `__syncthreads()`只同步block内线程
- Warp内操作隐式同步（但需显式使用`__syncwarp()`保证）
- 跨block同步需要分拆为多个kernel或使用cooperative groups

**数值精度**

- FP16累加容易溢出，关键路径用FP32累加器
- 批归一化等操作保持FP32精度
- 使用Kahan求和算法减少精度损失

## 9. 快速检查清单

### 计算效率

- [ ] 是否使用Tensor Core（WMMA/CUTLASS/cuBLAS）？
- [ ] Achieved occupancy是否合理（>25%）？
- [ ] Warp execution efficiency是否高（>85%）？
- [ ] 是否存在严重的分支分化（检查ballot统计）？
- [ ] 是否使用快速数学函数（`__sinf`, `__expf`等）？
- [ ] 循环展开是否合理（不过度）？

### 访存效率

- [ ] 寄存器是否溢出（`nvcc -Xptxas -v`检查）？
- [ ] Global memory访问是否合并（连续、对齐）？
- [ ] 是否使用Shared Memory缓存重用数据？
- [ ] Shared Memory是否有bank conflict（efficiency >90%）？
- [ ] L2命中率是否理想（>70%）？
- [ ] 数据布局是否为SoA而非AoS？
- [ ] 只读数据是否使用`const __restrict__`？

### 并行度

- [ ] Block size是否合理（128-256为主）？
- [ ] Grid size是否足够（>= SM数量）？
- [ ] 是否充分利用多stream并发？
- [ ] 计算与数据传输是否重叠（异步拷贝）？
- [ ] 是否使用CUDA Graphs减少launch开销？

### 同步开销

- [ ] 是否滥用`__syncthreads()`？
- [ ] 是否使用Warp Shuffle减少block内同步？
- [ ] 原子操作是否先做warp级规约？
- [ ] CPU-GPU同步点是否最小化？

## 10. 实战优化流程

**第一阶段：测量基准**

1. 用Nsight Systems查看整体时序
2. 用Nsight Compute分析最耗时的kernel
3. 确定瓶颈类型（计算/访存/同步）

**第二阶段：针对性优化**

- **计算瓶颈**：Tensor Core、快速数学、FMA
- **访存瓶颈**：Shared Memory、合并访问、L2优化
- **延迟瓶颈**：提高occupancy、异步操作、多流

**第三阶段：迭代验证**

1. 每次改动后重新profile
2. 对比优化前后的关键指标
3. 记录有效和无效的优化（避免重复试错）

**第四阶段：极限优化（可选）**

- 查看SASS汇编（`cuobjdump -sass`）
- 手动调整寄存器分配
- 使用内联PTX
- 考虑混合精度和量化

## 参考资料

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUTLASS](https://github.com/NVIDIA/cutlass) - 高性能线性代数
- [CUB](https://github.com/NVIDIA/cub) - CUDA Unbound并行原语
- [Thrust](https://github.com/NVIDIA/thrust) - 高层次并行算法
