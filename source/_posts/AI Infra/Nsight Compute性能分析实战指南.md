---
title: Nsight Compute性能分析实战指南
date: 2026-03-26 21:30:00
categories:
  - [AI Infra, 性能分析与Benchmark]
tags: [Nsight Compute, 性能分析, CUDA, Kernel优化, Roofline]
---

Nsight Compute 是 NVIDIA 提供的 CUDA Kernel 级深度分析工具，能够采集单个 kernel 的 SM 利用率、内存带宽、指令吞吐、Occupancy 等细粒度硬件指标，并通过 Roofline 模型直观展示优化空间。本文详解 Nsight Compute 的命令行采集、GUI 分析、Roofline 解读、各类 kernel 调优策略，帮助你精准定位并优化 CUDA kernel 的性能瓶颈。

<!-- more -->

## 目录

- [1. 工具定位与分析流程](#1-工具定位与分析流程)
- [2. 安装与环境配置](#2-安装与环境配置)
- [3. 命令行采集（ncu）](#3-命令行采集ncu)
- [4. GUI 界面分析](#4-gui-界面分析)
- [5. 核心指标详解](#5-核心指标详解)
- [6. Roofline 模型](#6-roofline-模型)
- [7. 典型 Kernel 优化场景](#7-典型-kernel-优化场景)
- [8. 深度学习 Kernel 分析](#8-深度学习-kernel-分析)
- [9. 高级用法](#9-高级用法)
- [10. 常见问题与最佳实践](#10-常见问题与最佳实践)
- [检验标准与进阶方向](#检验标准与进阶方向)
- [参考资料](#参考资料)

---

## 1. 工具定位与分析流程

### 1.1 何时使用 Nsight Compute

Nsight Systems 告诉你**哪个 kernel 慢**，Nsight Compute 告诉你这个 kernel **为什么慢**。

```
nsys 发现: volta_fp16_s884gemm 占总 GPU 时间 45%
                ↓
ncu 分析:  该 kernel 的 SM 利用率只有 60%
           内存带宽达到 85%（memory bound）
           shared memory bank conflict 严重
                ↓
优化:     调整 shared memory 布局，减少 bank conflict
                ↓
ncu 验证:  SM 利用率提升到 78%，kernel 耗时减少 25%
```

### 1.2 ncu 的工作原理

Nsight Compute 使用 **Kernel Replay** 机制：对目标 kernel 反复执行多次，每次采集不同的硬件计数器组。

> **白话理解**：ncu 分析一个 kernel 时会让它反复执行多次，每次采不同的"体检指标"——就像体检要分别做验血、B超、心电图，每项检查跑一遍。

这意味着：

- **高开销**：一个 kernel 可能被执行 10-50 次（采集不同 metric set）
- **结果精确**：每组计数器的值都是从完整 kernel 执行中采集的
- **程序要求确定性**：被 replay 的 kernel 每次输入/输出应该一致

```
正常执行:   [kernel] → 1 次
ncu 采集:   [kernel][kernel][kernel]...[kernel] → N 次（N = metric sets 数量）
                                                   ↑ 每次采集不同的计数器
```

---

## 2. 安装与环境配置

### 2.1 安装

```bash
# 随 CUDA Toolkit 安装（通常在这个路径）
ls /usr/local/cuda/bin/ncu

# 检查版本
ncu --version

# 独立安装
# 从 https://developer.nvidia.com/tools-overview 下载
```

### 2.2 权限配置

```bash
# ncu 需要访问 GPU 性能计数器，需要特权
# 方式一：使用 root
sudo ncu ...

# 方式二：设置 GPU 性能计数器权限（推荐）
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

# 方式三：持久化配置
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
    sudo tee /etc/modprobe.d/ncu.conf
```

### 2.3 Docker 环境

```bash
# 需要额外权限
docker run --gpus all \
    --cap-add SYS_ADMIN \
    --security-opt seccomp=unconfined \
    -it nvcr.io/nvidia/pytorch:24.01-py3
```

---

## 3. 命令行采集（ncu）

### 3.1 基础用法

```bash
# 最简单的 profiling（分析所有 kernel）
ncu python my_script.py

# 指定输出文件
ncu -o my_kernel_report python my_script.py

# 只分析前 10 个 kernel
ncu --launch-count 10 -o report python my_script.py
```

### 3.2 核心参数详解

**选择要分析的 Kernel**：

```bash
# 按 kernel 名称过滤（支持正则）
ncu --kernel-name "gemm" python my_script.py

# 按 kernel 名称过滤（精确匹配）
ncu --kernel-name-base demangled \
    --kernel-name "void cutlass::Kernel" \
    python my_script.py

# 跳过前 N 个 kernel（跳过 warmup）
ncu --launch-skip 100 --launch-count 5 -o report python my_script.py
#     跳过前 100 个        只分析 5 个
```

**选择采集的指标集**：

```bash
# 完整采集（所有指标，最慢但最全面）
ncu --set full -o report python my_script.py

# 默认集（平衡速度和信息量，推荐日常使用）
ncu --set default -o report python my_script.py

# Roofline 分析专用
ncu --set roofline -o report python my_script.py

# 只看特定 section
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    -o report python my_script.py
```

可用的 Section（指标分组）：

| Section | 内容 | 用途 |
|---------|------|------|
| `SpeedOfLight` | SM 和内存利用率 vs 理论峰值 | 快速判断 compute/memory bound |
| `ComputeWorkloadAnalysis` | 指令类型分布、SM 活跃率 | 分析计算效率 |
| `MemoryWorkloadAnalysis` | L1/L2/HBM 命中率、带宽利用率 | 分析内存瓶颈 |
| `Occupancy` | 理论/实际 Occupancy | 分析资源使用 |
| `LaunchStatistics` | Grid/Block 维度、寄存器/Shared Memory 用量 | kernel 配置 |
| `WarpStateStatistics` | Warp 停滞原因分布 | 分析 warp 瓶颈 |
| `InstructionStatistics` | 指令吞吐、类型分布 | 分析指令效率 |
| `SourceCounters` | 源码级指标 | 定位到具体代码行 |

**控制 Replay 策略**：

```bash
# 应用级 replay（默认，整个程序重放，最准确）
ncu --replay-mode application -o report python my_script.py

# Kernel 级 replay（只重放目标 kernel，更快但可能有副作用）
ncu --replay-mode kernel -o report python my_script.py
```

### 3.3 常用命令模板

**快速诊断单个 kernel**：

```bash
ncu --kernel-name "my_kernel" \
    --launch-skip 10 \
    --launch-count 1 \
    --set default \
    -o kernel_analysis \
    ./my_cuda_program
```

**GEMM kernel 深度分析**：

```bash
ncu --kernel-name "gemm" \
    --launch-skip 50 \
    --launch-count 3 \
    --set full \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section Occupancy \
    -o gemm_analysis \
    python matmul_benchmark.py
```

**Roofline 分析**：

```bash
ncu --kernel-name "flash_fwd" \
    --launch-skip 20 \
    --launch-count 1 \
    --set roofline \
    -o roofline_analysis \
    python attention_benchmark.py
```

### 3.4 命令行输出

不生成报告文件时，ncu 直接在终端输出结果：

```bash
ncu --kernel-name "vectorAdd" --launch-count 1 ./vector_add
```

输出示例：

```
==PROF== Connected to process 12345
==PROF== Profiling "vectorAdd" - 1: 0%....50%....100%

  vectorAdd(float*, float*, float*, int), Context 1, Stream 7
    Section: GPU Speed Of Light Throughput
    ──────────────────────────────────────
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.37
    Elapsed Cycles                   cycle       15,234
    Memory Throughput                    %        72.45
    DRAM Throughput                      %        72.45
    Duration                      usecond        11.12
    L1/TEX Cache Throughput             %        23.56
    L2 Cache Throughput                 %        45.67
    SM Active Cycles               cycle     12,345.67
    Compute (SM) Throughput             %        15.23

    Section: Memory Workload Analysis
    ──────────────────────────────────
    Memory Throughput   Gbyte/second     234.56
    Mem Busy                      %      72.45
    L2 Hit Rate                   %      45.23
```

---

## 4. GUI 界面分析

### 4.1 打开报告

```bash
# 启动 GUI
ncu-ui

# 直接打开报告
ncu-ui my_kernel_report.ncu-rep
```

### 4.2 界面布局

```
┌──────────────────────────────────────────────────────────────┐
│  菜单栏 / 工具栏                                               │
├──────────────────────────────────────────────────────────────┤
│ ┌─ Kernel 列表 ─────────────────────────────────────────────┐ │
│ │ # │ Kernel Name          │ Duration │ SM% │ Mem% │ Grid  │ │
│ │ 1 │ volta_fp16_gemm...   │ 1.2 ms   │ 82  │ 45   │ ...  │ │
│ │ 2 │ flash_fwd_kernel...  │ 0.8 ms   │ 76  │ 68   │ ...  │ │
│ └───────────────────────────────────────────────────────────┘ │
│ ┌─ 详细分析 ────────────────────────────────────────────────┐ │
│ │                                                           │ │
│ │  [Speed Of Light] [Memory] [Compute] [Occupancy] [Source] │ │
│ │                                                           │ │
│ │  ┌─ Speed Of Light ───────────────────────────────────┐   │ │
│ │  │  Compute (SM) Throughput: ████████████░░░░ 75.2%   │   │ │
│ │  │  Memory Throughput:      ██████░░░░░░░░░░ 38.4%    │   │ │
│ │  │  → This kernel is COMPUTE BOUND                    │   │ │
│ │  └────────────────────────────────────────────────────┘   │ │
│ │                                                           │ │
│ │  ┌─ Roofline Chart ──────────────────────────────────┐   │ │
│ │  │  (图形化 Roofline 模型)                             │   │ │
│ │  └────────────────────────────────────────────────────┘   │ │
│ └───────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 4.3 多 Kernel 对比

选中多个 kernel，ncu-ui 可以**并排对比**各项指标，适合对比优化前后的同一个 kernel、或不同实现方案。

也可以对比两份独立的报告：

```bash
# 分别生成 baseline 和 optimized 的报告
ncu --kernel-name "my_kernel" -o baseline ./program_v1
ncu --kernel-name "my_kernel" -o optimized ./program_v2

# 在 GUI 中 File → Open 同时打开两份报告做对比
```

---

## 5. 核心指标详解

### 5.1 Speed Of Light（SOL）

Speed Of Light 是最重要的汇总指标，直接告诉你 kernel 是 **compute bound** 还是 **memory bound**。

> **白话理解**：SOL 告诉你这个 kernel 离理论极限还有多远——就像跑步成绩和世界纪录的差距。Compute SOL 80% 意味着你的计算效率已经达到了硬件理论峰值的 80%。

```
Compute (SM) Throughput: 实际计算吞吐 / 理论峰值计算吞吐
Memory Throughput:       实际内存带宽 / 理论峰值内存带宽
```

| Compute SOL | Memory SOL | 结论 |
|:-----------:|:----------:|------|
| 高 (>60%) | 低 (<60%) | **Compute Bound** — 计算单元是瓶颈 |
| 低 (<60%) | 高 (>60%) | **Memory Bound** — 内存带宽是瓶颈 |
| 低 | 低 | **Latency Bound** — 既没充分计算也没充分读写，可能 Occupancy 低或有同步等待 |
| 高 | 高 | **接近理论极限** — 优化空间有限 |

### 5.2 Memory Workload Analysis

**内存层次利用率**：

```
L1/TEX Cache:
  Hit Rate: 85%           ← L1 命中率
  Throughput: 1234 GB/s   ← L1 带宽利用

L2 Cache:
  Hit Rate: 62%           ← L2 命中率
  Throughput: 890 GB/s    ← L2 带宽利用

Device Memory (HBM):
  Throughput: 1800 GB/s   ← HBM 带宽利用（接近峰值 ~2TB/s 说明 memory bound）
```

**关键诊断**：

| 指标 | 健康范围 | 问题信号 |
|------|---------|---------|
| L1 Hit Rate | >80% | <50% 可能有 uncoalesced access |
| L2 Hit Rate | 取决于数据量 | 太低说明 working set 超过 L2 |
| HBM Throughput | 取决于是否 memory bound | 接近峰值且 kernel 慢 → 需要减少数据搬运 |

### 5.3 Compute Workload Analysis

```
SM 活跃周期: 实际计算的周期数 / 总周期数
指令发射率:  每周期实际发射的指令数 / 理论峰值

指令类型分布:
  FP32:  45%    ← 单精度浮点
  FP16:  30%    ← 半精度浮点（Tensor Core）
  INT:   15%    ← 整数
  Other: 10%    ← 特殊指令、控制流等
```

如果 FP16/FP32 指令占比低而 INT/Other 占比高，说明有大量非计算指令（地址计算、控制流），需要优化代码逻辑。

### 5.4 Occupancy

**Occupancy = 活跃 Warp 数 / SM 最大支持 Warp 数**

```
理论 Occupancy:  75%   ← 基于寄存器、Shared Memory 用量计算的上限
实际 Occupancy:  62%   ← 实际运行时的平均值
```

| 限制因素 | 含义 | 优化方向 |
|---------|------|---------|
| 寄存器 | 每个线程用了太多寄存器 | 减少局部变量、使用 `__launch_bounds__` |
| Shared Memory | 每个 Block 用了太多 Shared Memory | 减少 Shared Memory 用量或调整 Block 大小 |
| Block 大小 | Block 中线程数不能整除 SM 的 Warp 槽 | 调整 Block 维度为 32 的倍数 |

**注意**：高 Occupancy 不一定等于高性能。有时降低 Occupancy 换取更多寄存器/Shared Memory，反而能提升性能（如 Tiled GEMM）。

### 5.5 Warp State Statistics

Warp 停滞原因分布，直接指出 kernel 瓶颈：

> **白话理解**：Warp Stall 就是线程组被卡住的原因——就像工人停下来等原料（内存等待）、等同事干完上一步（同步等待）、或者工具被占用（计算管线满）。这张表告诉你工人们主要在等什么。

| 停滞原因 | 含义 | 优化方向 |
|---------|------|---------|
| **Stall Long Scoreboard** | 等待长延迟内存操作（HBM/L2） | 优化内存访问模式、提高 Occupancy |
| **Stall Short Scoreboard** | 等待 Shared Memory / L1 操作 | 检查 bank conflict |
| **Stall MIO Throttle** | 内存指令队列满 | 减少内存指令密度 |
| **Stall Math Pipe Throttle** | 计算管线满 | 已经 compute bound，正常 |
| **Stall Barrier** | 等待 `__syncthreads()` | 减少同步次数或优化负载均衡 |
| **Stall Not Selected** | Warp 就绪但未被调度 | Occupancy 高但 SM 调度器忙 |

### 5.6 Source-Level Analysis

如果编译时加了 `-lineinfo`，ncu 可以将指标映射回**源代码行**：

```bash
# 编译时保留行号信息
nvcc -lineinfo -o my_program my_program.cu

# ncu 采集时启用 source section
ncu --section SourceCounters --kernel-name "my_kernel" ./my_program
```

在 GUI 中可以看到每一行源代码的：
- 执行次数
- 内存读写量
- 计算指令数
- 停滞周期

---

## 6. Roofline 模型

### 6.1 Roofline 基础

Roofline 模型用一张图展示 kernel 的性能与理论极限的关系：

> **白话理解**：Roofline 给 kernel 画一张"能力边界图"——屋顶的斜边是内存带宽上限，平顶是算力上限，你的 kernel 在图上的位置决定了它被哪个天花板卡住。如果点落在斜边下方，说明内存带宽是瓶颈；如果落在平顶下方，说明算力是瓶颈。

```
性能                    ┌─── Compute Ceiling (理论峰值算力)
(FLOP/s)               │
    │                  │ ╱
    │                  │╱
    │        Memory   ╱│
    │        Roof    ╱ │
    │              ╱   │
    │            ╱     │
    │          ╱       │
    │        ╱    ●kernel
    │      ╱
    │    ╱
    │  ╱
    │╱
    └──────────────────────────
         计算强度 (FLOP/Byte)
                ↑
           Ridge Point
    (Memory Roof 和 Compute Ceiling 的交点)
```

- **X 轴**：计算强度（Arithmetic Intensity），即 FLOP / 搬运的字节数
- **Y 轴**：性能，即 FLOP/s
- **屋顶**：由内存带宽上限（斜线）和算力上限（水平线）构成

### 6.2 解读 Roofline

**kernel 在 Roofline 上的位置**：

```
位置              含义                     优化方向
────────────────────────────────────────────────────────
靠近 Memory Roof  Memory Bound             减少数据搬运、用缓存、量化
(斜线下方)

靠近 Compute      Compute Bound            用 Tensor Core、优化算法
Ceiling (水平线)

远离两条线        Latency Bound /          提高 Occupancy、减少同步
(左下角)          低效执行

两条线交点附近    平衡点                    两方面都要优化
```

### 6.3 ncu 中的 Roofline

```bash
# 采集 Roofline 数据
ncu --set roofline \
    --kernel-name "my_kernel" \
    --launch-skip 10 \
    --launch-count 1 \
    -o roofline_report \
    ./my_program
```

ncu 会在 GUI 中展示**两个 Roofline**：

1. **FP Roofline**：浮点计算的 Roofline
   - 包含 FP64、FP32、FP16、Tensor Core 等多条 Compute Ceiling
2. **HBM Roofline**：以 HBM 带宽为 Memory Roof

### 6.4 Roofline 实例分析

**案例一：vectorAdd（Memory Bound）**

```
计算强度: 0.25 FLOP/Byte （每读 8 字节做 1 次 FP32 加法）
理论峰值: 受 Memory Roof 限制

Roofline 位置: 紧贴斜线 → Memory Bound → 正常，vectorAdd 本身就是 memory bound
```

**案例二：Naive GEMM（Latency Bound）**

```
计算强度: ~100 FLOP/Byte （理论值高，但实际实现差）
理论峰值: 应该在 Compute Ceiling 附近

Roofline 位置: 远低于 Compute Ceiling → 实现效率低
原因: 全局内存访问多、没用 Shared Memory、没用 Tensor Core
```

**案例二优化：Tiled GEMM**

```
优化后: 使用 Shared Memory tiling + 合适的 block 大小
Roofline 位置: 显著上移，更接近 Compute Ceiling
```

---

## 7. 典型 Kernel 优化场景

### 7.1 Memory Bound Kernel 优化

**ncu 特征**：
- Memory SOL 高（>60%）
- Compute SOL 低（<30%）
- Warp 停滞主要是 `Stall Long Scoreboard`

**优化策略**：

```
1. 合并访存（Coalesced Access）
   ncu 信号: L1 Throughput 低于预期、大量 uncoalesced 访问
   优化: 确保相邻线程访问相邻内存地址

2. 利用 Shared Memory
   ncu 信号: HBM 带宽饱和
   优化: 将重复使用的数据载入 Shared Memory

3. 减少数据搬运
   ncu 信号: 大量 HBM 读写
   优化: kernel 融合、量化（FP16/INT8）

4. 提高 L2 Cache 命中率
   ncu 信号: L2 Hit Rate 低
   优化: 调整数据布局、分 tile 处理
```

**代码示例：合并访存优化**

```c
// Bad: 非合并访存（stride 访问）
__global__ void bad_kernel(float* data, int N, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = data[tid * stride];  // 相邻线程访问不相邻地址
}

// Good: 合并访存
__global__ void good_kernel(float* data, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float val = data[tid];  // 相邻线程访问相邻地址，一次 128B 事务
}
```

### 7.2 Compute Bound Kernel 优化

**ncu 特征**：
- Compute SOL 高（>60%）
- Memory SOL 低
- Warp 停滞主要是 `Stall Math Pipe Throttle`

**优化策略**：

```
1. 使用 Tensor Core
   ncu 信号: FP32 指令占主导
   优化: 使用 FP16/BF16 + wmma / mma 指令

2. 减少冗余计算
   ncu 信号: 指令数多
   优化: 优化算法、预计算常量

3. 指令级并行（ILP）
   ncu 信号: SM 利用率未满
   优化: 展开循环、增加每线程工作量
```

### 7.3 Latency Bound Kernel 优化

**ncu 特征**：
- Compute SOL 和 Memory SOL 都低（<40%）
- Occupancy 可能偏低

**优化策略**：

```
1. 提高 Occupancy
   ncu 信号: 实际 Occupancy << 理论 Occupancy
   优化: 减少寄存器用量、调整 Block 大小

2. 减少同步开销
   ncu 信号: Stall Barrier 占比高
   优化: 减少 __syncthreads() 调用

3. 减少分支分歧
   ncu 信号: 高比例的 predicated-off 指令
   优化: 重组算法减少条件分支
```

### 7.4 Shared Memory Bank Conflict

**ncu 特征**：
- `Stall Short Scoreboard` 占比高
- Shared Memory Throughput 低于预期

```c
// Bank Conflict 示例
__shared__ float smem[32][32];

// Bad: 同一 warp 内的线程访问同一列 → 32-way bank conflict
float val = smem[threadIdx.x][0];  // 所有线程访问 bank 0

// Good: 同一 warp 内的线程访问不同列 → 无冲突
float val = smem[0][threadIdx.x];  // 线程 i 访问 bank i

// Padding 消除冲突
__shared__ float smem[32][33];  // 加 1 列 padding
float val = smem[threadIdx.x][col];  // 交错访问，消除冲突
```

在 ncu 中可以通过 `Memory Workload Analysis` → `Shared Memory` 部分看到：
- `Shared Memory Bank Conflicts`: 冲突次数
- `Shared Memory Wavefronts`: 实际需要的内存事务数（无冲突时应等于请求数）

---

## 8. 深度学习 Kernel 分析

### 8.1 GEMM Kernel 分析

GEMM（矩阵乘法）是深度学习中最核心的 kernel。在 ncu 中分析 GEMM 时关注：

```bash
# 捕获 GEMM kernel
ncu --kernel-name "gemm\|s884\|cutlass\|Gemm" \
    --launch-skip 50 \
    --launch-count 3 \
    --set full \
    -o gemm_report \
    python benchmark_gemm.py
```

**GEMM 性能指标解读**：

| 指标 | 理想值 | 说明 |
|------|-------|------|
| Compute SOL | >70% | GEMM 应该是 compute bound |
| Tensor Core 使用率 | >80% | 如果低说明没用 Tensor Core |
| Achieved Occupancy | >50% | Tiled GEMM 通常 Occupancy 不需要太高 |
| L2 Hit Rate | 取决于矩阵大小 | 大矩阵必然低 |

**为什么 GEMM 的 Compute SOL 不到 100%**：

即使是高度优化的 GEMM，通常也只能达到理论峰值的 70-85%，原因包括：
- Tile 边界的 padding 浪费
- 数据加载与计算的流水线不完美
- Epilogue（结果写回）的开销

### 8.2 Attention Kernel 分析

```bash
# 捕获 FlashAttention kernel
ncu --kernel-name "flash_fwd\|flash_bwd\|fmha" \
    --launch-skip 20 \
    --launch-count 3 \
    --set full \
    -o attention_report \
    python benchmark_attention.py
```

**FlashAttention 的 ncu 特征**：

```
Compute SOL: ~65-75%   ← 比 GEMM 低，因为有 softmax、mask 等非 GEMM 操作
Memory SOL:  ~30-50%   ← FlashAttention 通过 tiling 大幅减少 HBM 访问
Occupancy:   ~50-60%   ← 每个 Block 使用较多 Shared Memory

Warp State:
  Stall Math Pipe Throttle: 高  ← 正常，compute bound
  Stall Long Scoreboard:   中  ← HBM 读取等待
```

### 8.3 Reduction Kernel 分析

```bash
ncu --kernel-name "reduce\|LayerNorm\|Softmax" \
    --launch-skip 20 \
    --launch-count 5 \
    --set default \
    -o reduce_report \
    python benchmark_reduce.py
```

Reduction 类 kernel（LayerNorm、Softmax、BatchNorm）通常是 **memory bound**：

```
典型 ncu 输出:
  Compute SOL: 15-30%   ← 计算简单（加法/除法/exp）
  Memory SOL:  60-80%   ← 瓶颈在读写数据

优化方向:
  - Kernel 融合：把 LayerNorm 和前后的 kernel 融合
  - 向量化读写：使用 float4 读取
  - 减少 kernel 启动次数
```

---

## 9. 高级用法

### 9.1 自定义指标

```bash
# 采集特定的硬件计数器
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_lookup_hit.sum,\
    l1tex__t_bytes_lookup_miss.sum \
    --kernel-name "my_kernel" \
    ./my_program
```

常用 metric 列表：

| Metric | 含义 |
|--------|------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 吞吐利用率 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽利用率 |
| `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | 活跃 Warp 百分比（Occupancy） |
| `smsp__sass_thread_inst_executed_op_fp16_pred_on.sum` | FP16 指令执行数 |
| `smsp__sass_thread_inst_executed_op_fp32_pred_on.sum` | FP32 指令执行数 |
| `l1tex__t_bytes_lookup_hit.sum` | L1 缓存命中字节数 |
| `l2__read_throughput.avg.pct_of_peak_sustained_elapsed` | L2 读带宽利用率 |

### 9.2 Baseline 对比

```bash
# 生成 baseline 报告
ncu --kernel-name "my_kernel" -o baseline ./program_v1

# 生成优化后报告
ncu --kernel-name "my_kernel" -o optimized ./program_v2

# 命令行对比
ncu --import baseline.ncu-rep --import optimized.ncu-rep --page details

# 或在 GUI 中并排打开两份报告
```

### 9.3 与 Python 集成

```python
# 编程式分析 ncu 导出的数据
import sqlite3

# ncu 可以导出为 CSV
# ncu --csv --kernel-name "my_kernel" ./program > metrics.csv

import pandas as pd
df = pd.read_csv("metrics.csv")
print(df[["Kernel Name", "Duration", "SM [%]", "Memory [%]"]])
```

### 9.4 规则检查（Rules）

ncu 内置了一套优化规则（Rules），会自动分析 kernel 并给出优化建议：

```bash
# 启用规则检查
ncu --set full --kernel-name "my_kernel" -o report ./my_program
```

在 GUI 中，规则检查的结果显示在各 Section 的顶部，例如：

```
⚠ Warning: This kernel exhibits low compute throughput and low memory
  throughput. Check the Warp State Statistics section for which state
  is the bottleneck.

💡 Tip: The ratio of peak float (fp32) to double (fp64) performance
  on this device is 64:1. The kernel uses FP64 instructions. Consider
  using FP32 if precision allows.
```

---

## 10. 常见问题与最佳实践

### 10.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `ERR_NVGPUCTRPERM` | 无权限访问 GPU 计数器 | `sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` |
| 程序极度变慢 | Kernel replay 导致 | 使用 `--launch-skip` + `--launch-count` 限制范围 |
| 某些 metric 显示 N/A | 当前 GPU 架构不支持 | 检查 GPU 架构与 metric 兼容性 |
| 报告文件过大 | 采集了太多 kernel | 用 `--kernel-name` 和 `--launch-count` 过滤 |
| Replay 结果不一致 | 程序中有随机性 | 设置随机种子，确保确定性执行 |
| Docker 中无法使用 | 缺少权限 | 添加 `--cap-add SYS_ADMIN --security-opt seccomp=unconfined` |

### 10.2 最佳实践

**1. 先 nsys 后 ncu**

```
永远先用 nsys 找到最耗时的 kernel
→ 再用 ncu 深入分析该 kernel
→ 优化后再用 nsys 验证整体改善
```

**2. 精准定位 Kernel**

```bash
# 不要分析所有 kernel，太慢！
# Bad:
ncu python train.py  # 可能有上万个 kernel

# Good:
ncu --kernel-name "target_kernel" \
    --launch-skip 100 \
    --launch-count 3 \
    python train.py
```

**3. 编译保留调试信息**

```bash
# 保留行号信息（对性能影响极小）
nvcc -lineinfo -O3 -o my_program my_program.cu

# 这样 ncu 可以把指标映射回源代码行
```

**4. 关注瓶颈指标**

```
分析优先级:
1. Speed Of Light → 快速判断 compute/memory bound
2. Warp State Statistics → 找到停滞原因
3. Memory Workload → 如果 memory bound，分析内存层次
4. Compute Workload → 如果 compute bound，分析指令效率
5. Occupancy → 如果 latency bound，分析资源使用
6. Source → 定位到具体代码行
```

**5. Roofline 指导优化方向**

```
Kernel 在 Roofline 的位置       优化策略
──────────────────────────────────────────
远低于 Memory Roof              优化内存访问（合并、缓存）
紧贴 Memory Roof                减少数据搬运（融合、量化）
远低于 Compute Ceiling           提高计算效率（Tensor Core、ILP）
紧贴 Compute Ceiling             已接近极限，考虑算法优化
```

### 10.3 ncu vs nsys 决策表

| 你想知道什么 | 用哪个工具 |
|-------------|----------|
| 哪个 kernel 最耗时 | nsys |
| GPU 是否有空闲气泡 | nsys |
| kernel 是 compute bound 还是 memory bound | ncu |
| kernel 的内存访问是否合并 | ncu |
| kernel 的 Occupancy 受什么限制 | ncu |
| NCCL 通信与计算是否重叠 | nsys |
| 数据传输是否与计算重叠 | nsys |
| Shared Memory 有没有 bank conflict | ncu |
| 某行代码执行了多少条指令 | ncu (source) |

---

## 检验标准与进阶方向

### 自我检验清单

学完本文后，你应该能够做到以下几点：

- [ ] 能使用 `ncu` 命令行对指定 kernel 进行深度分析，生成 `.ncu-rep` 报告
- [ ] 能解读 Speed Of Light 指标，判断 kernel 是 compute bound 还是 memory bound
- [ ] 能在 Roofline 图上定位 kernel 的位置，并判断优化方向
- [ ] 能通过 Warp State Statistics 找到 warp 停滞的主要原因
- [ ] 能通过 Memory Workload Analysis 判断是否存在 uncoalesced access 或 bank conflict
- [ ] 能使用 `--kernel-name` 和 `--launch-skip` 精确采集目标 kernel，避免分析所有 kernel 导致的性能开销
- [ ] 能对比优化前后的 ncu 报告，验证性能改善（命令行或 GUI 对比）
- [ ] 能根据 ncu 的 Rules 自动建议，快速定位 kernel 的优化切入点

### 进阶方向

| 方向 | 内容 | 推荐资源 |
|------|------|---------|
| SASS 级指令分析 | 学习 GPU 汇编（SASS），在 ncu Source 页面分析指令级瓶颈 | [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) |
| 自定义 Section / Rule | 编写 `.section` 和 `.py` 规则文件，定制 ncu 的分析面板和优化建议 | [Nsight Compute Customization Guide](https://docs.nvidia.com/nsight-compute/CustomizationGuide/) |
| Tensor Core 深度分析 | 分析 wmma/mma 指令利用率，判断 Tensor Core 是否被充分使用 | [Tensor Core Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) |
| 多 Kernel 联合优化 | 结合 nsys 的时间线视图和 ncu 的 kernel 级数据，做全局优化决策 | [Nsight Systems + Compute Workflow](https://developer.nvidia.com/nsight-systems) |
| Kernel 自动调优 | 使用 ncu 的 CSV 导出 + 脚本自动化，构建 kernel 参数搜索和性能回归检测 | [Triton Autotuner](https://triton-lang.org/) |

---

## 参考资料

- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA Developer - Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [Roofline Model - Berkeley](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)
- [CUDA C++ Best Practices Guide - Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics)
- [CUTLASS - GitHub (GEMM Optimization Examples)](https://github.com/NVIDIA/cutlass)
- [NVIDIA Developer Blog - Kernel Profiling Guide](https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/)
