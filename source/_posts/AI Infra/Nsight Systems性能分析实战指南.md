---
title: Nsight Systems性能分析实战指南
date: 2026-03-26 21:00:00
categories:
  - [AI Infra, 性能分析与Benchmark]
tags: [Nsight Systems, 性能分析, CUDA, GPU, Profiling]
---

Nsight Systems 是 NVIDIA 提供的系统级性能分析工具，能够从全局视角展示 CPU、GPU、内存、网络的时序关系与交互瓶颈。本文详解 Nsight Systems 的安装配置、命令行采集、GUI 分析、典型场景诊断，帮助你快速定位 CUDA 程序和深度学习训练/推理中的系统级瓶颈。

<!-- more -->

## 目录

- [1. 工具定位与适用场景](#1-工具定位与适用场景)
- [2. 安装与环境配置](#2-安装与环境配置)
- [3. 命令行采集（nsys profile）](#3-命令行采集nsys-profile)
- [4. GUI 界面分析](#4-gui-界面分析)
- [5. 核心概念与视图](#5-核心概念与视图)
- [6. 典型分析场景](#6-典型分析场景)
- [7. 深度学习场景实战](#7-深度学习场景实战)
- [8. 多机多卡与 NCCL 分析](#8-多机多卡与-nccl-分析)
- [9. 高级用法](#9-高级用法)
- [10. 常见问题与最佳实践](#10-常见问题与最佳实践)
- [参考资料](#参考资料)

---

## 1. 工具定位与适用场景

### 1.1 Nsight Systems vs Nsight Compute

| 维度 | Nsight Systems (nsys) | Nsight Compute (ncu) |
|------|:---------------------:|:--------------------:|
| 分析层级 | **系统级**（CPU + GPU + 网络） | **Kernel 级**（单个 CUDA kernel） |
| 分析粒度 | 时序关系、宏观瓶颈 | SM 利用率、内存带宽、指令吞吐 |
| 开销 | 低（<5% 性能影响） | 高（kernel replay，10-100x 慢） |
| 适用阶段 | **先用**，定位哪个阶段/kernel 是瓶颈 | **后用**，深入分析瓶颈 kernel |
| 输出格式 | `.nsys-rep`（时间线报告） | `.ncu-rep`（kernel 详细指标） |

**分析流程**：

```
性能问题
  ↓
Nsight Systems（系统级，找到瓶颈在哪）
  ↓ 确认是某个 kernel 慢
Nsight Compute（Kernel 级，分析为什么慢）
  ↓ 优化 kernel
再用 Nsight Systems 验证整体改善
```

### 1.2 典型使用场景

- 定位 **CPU-GPU 交互瓶颈**：kernel launch 延迟、数据传输阻塞
- 发现 **GPU 空闲气泡**：kernel 之间的间隙、同步等待
- 分析 **内存拷贝与计算重叠**：HtoD / DtoH 传输是否被隐藏
- 检查 **多 GPU 通信**：NCCL 集合通信的耗时和重叠
- 评估 **深度学习训练循环**：前向/反向/优化器/数据加载各阶段耗时

---

## 2. 安装与环境配置

### 2.1 安装方式

**方式一：随 CUDA Toolkit 安装**

CUDA Toolkit 11.0+ 自带 Nsight Systems，安装 CUDA 后即可使用：

```bash
# 检查是否已安装
nsys --version

# 典型路径
ls /usr/local/cuda/bin/nsys
```

**方式二：独立安装（推荐，版本更新）**

从 NVIDIA 官网下载最新版本，独立安装不依赖 CUDA 版本：

```bash
# Ubuntu/Debian
sudo apt install nsight-systems-cli

# 或直接下载 .deb / .run 安装包
# https://developer.nvidia.com/nsight-systems
```

**方式三：Docker 环境**

```bash
# NVIDIA NGC 容器已预装
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3

# 容器内验证
nsys --version
```

### 2.2 环境检查

```bash
# 检查 nsys 版本
nsys --version

# 检查 GPU 驱动
nvidia-smi

# 确认 perf_event_paranoid（Linux，影响 CPU 采样）
cat /proc/sys/kernel/perf_event_paranoid
# 建议设为 1 或 0（需要 root）
sudo sysctl -w kernel.perf_event_paranoid=1
```

---

## 3. 命令行采集（nsys profile）

### 3.1 基础用法

```bash
# 最简单的 profiling
nsys profile python train.py

# 指定输出文件名
nsys profile -o my_report python train.py

# 输出格式和位置
nsys profile -o /tmp/report_%p python train.py
# %p 会被替换为进程 PID
```

### 3.2 核心参数详解

**采集范围控制**：

```bash
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    #        │     │    │     │      │
    #        │     │    │     │      └─ cuBLAS 库调用
    #        │     │    │     └─ cuDNN 库调用
    #        │     │    └─ OS Runtime（pthread、IO 等）
    #        │     └─ NVTX 标注（用户自定义标记）
    #        └─ CUDA API + GPU kernel + 内存拷贝
    python train.py
```

常用 `--trace` 值：

| 值 | 采集内容 |
|----|---------|
| `cuda` | CUDA Runtime/Driver API、kernel 执行、内存传输 |
| `nvtx` | 用户 NVTX 标注（推荐始终开启） |
| `osrt` | OS 运行时（线程、mutex、IO） |
| `cudnn` | cuDNN 库调用 |
| `cublas` | cuBLAS 库调用 |
| `mpi` | MPI 通信 |
| `opengl` | OpenGL API |
| `none` | 不采集任何 trace，仅做统计采样 |

**时间控制**：

```bash
nsys profile \
    --delay=10 \           # 启动后延迟 10 秒开始采集（跳过初始化）
    --duration=30 \        # 采集 30 秒后自动停止
    --kill=sigterm \       # 采集结束后发送信号终止程序
    python train.py
```

**CPU 采样控制**：

```bash
nsys profile \
    --sample=cpu \             # 开启 CPU 采样
    --cpuctxsw=process-tree \  # 采集上下文切换
    --backtrace=dwarf \        # 使用 DWARF 格式回溯（精度高）
    python train.py
```

**GPU 指标采集**：

```bash
nsys profile \
    --gpu-metrics-device=0 \       # 采集 GPU 0 的硬件指标
    --gpu-metrics-frequency=1000 \ # 采样频率 1000 Hz
    python train.py
```

### 3.3 完整的实用命令模板

**深度学习训练 profiling**：

```bash
nsys profile \
    -o train_profile \
    --trace=cuda,nvtx,cudnn,cublas \
    --sample=none \
    --delay=30 \
    --duration=60 \
    --gpu-metrics-device=all \
    python train.py --epochs 5 --batch-size 32
```

**推理服务 profiling**：

```bash
nsys profile \
    -o inference_profile \
    --trace=cuda,nvtx \
    --delay=5 \
    --duration=20 \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct
```

### 3.4 导出统计报告

```bash
# 生成 SQLite 数据库（可用 SQL 查询）
nsys export --type=sqlite my_report.nsys-rep

# 生成文本摘要
nsys stats my_report.nsys-rep

# 只看 CUDA kernel 统计
nsys stats --report cuda_gpu_kern_sum my_report.nsys-rep

# 只看 CUDA API 调用统计
nsys stats --report cuda_api_sum my_report.nsys-rep

# 只看 GPU 内存操作统计
nsys stats --report cuda_gpu_mem_size_sum my_report.nsys-rep
```

`nsys stats` 输出示例：

```
CUDA Kernel Statistics:
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)   Max (ns)   Name
 --------  ---------------  ---------  ---------  ---------  ---------  ---------  ----
    45.2      1,234,567,890       1024  1,205,827  1,198,432    985,231  1,456,789  volta_fp16_s884gemm_...
    23.1        632,456,123       2048    308,812    305,678    285,432    356,789  void at::native::...
    12.4        339,876,543       1024    331,910    328,765    312,456    378,901  void flash::flash_fwd_...
```

---

## 4. GUI 界面分析

### 4.1 打开报告

在本地安装 Nsight Systems GUI（支持 Windows / Linux / macOS），打开 `.nsys-rep` 文件：

```bash
# Linux 启动 GUI
nsys-ui

# 或直接打开报告
nsys-ui my_report.nsys-rep
```

如果在远程服务器上采集，将 `.nsys-rep` 文件拷贝到本地用 GUI 打开：

```bash
scp remote:/tmp/my_report.nsys-rep ./
```

### 4.2 界面布局

GUI 打开后的主要区域：

```
┌─────────────────────────────────────────────────────┐
│  菜单栏 / 工具栏                                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─ Timeline 时间线 ──────────────────────────────┐  │
│  │ [CPU 线程行]  ████░░████████░░░████            │  │
│  │ [CUDA API]   ██░░██████░░░░████████            │  │
│  │ [GPU Kernel] ░░████░░░░████████░░████          │  │
│  │ [GPU Memcpy]     ██         ██                 │  │
│  │ [NVTX]       |-- forward --|-- backward --|    │  │
│  └────────────────────────────────────────────────┘  │
│                                                     │
│  ┌─ 统计面板 / 详情面板 ─────────────────────────────┐  │
│  │ 选中 kernel 的详细信息、调用栈、耗时统计           │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 4.3 关键操作

| 操作 | 快捷键 / 方式 | 说明 |
|------|-------------|------|
| 缩放 | 鼠标滚轮 / Ctrl+滚轮 | 放大/缩小时间线 |
| 平移 | 中键拖拽 / Shift+拖拽 | 左右移动时间线 |
| 选择区域 | 左键拖拽 | 选中一段时间范围，查看统计 |
| 查看详情 | 左键点击 kernel | 显示 kernel 名称、耗时、Grid/Block 维度 |
| 筛选 | 右键 → Filter | 只显示特定线程/kernel |
| 标记 | Ctrl+M | 在时间线上做标记 |

---

## 5. 核心概念与视图

### 5.1 时间线行（Row）

每一行代表一个**活动源**，常见的行：

| 行 | 内容 | 关注点 |
|----|------|--------|
| **CPU Thread** | 每个 CPU 线程的活动 | 线程是否在忙碌、是否有阻塞 |
| **CUDA HW** | GPU 上实际执行的 kernel 和 memcpy | kernel 占用率、气泡 |
| **CUDA API** | CPU 端调用的 CUDA API | 哪些 API 调用耗时长 |
| **NVTX** | 用户标注的逻辑区域 | 前向/反向/数据加载的边界 |
| **GPU Metrics** | SM 活跃率、显存带宽等 | 硬件利用率 |
| **NCCL** | 集合通信操作 | 通信耗时和重叠 |

### 5.2 关键模式识别

**模式一：GPU 气泡（GPU Idle）**

```
CUDA HW: [kernel1]░░░░░░░░░░[kernel2]░░░░░░░[kernel3]
                   ↑ 气泡：GPU 空闲
```

原因：kernel launch 延迟、CPU 计算瓶颈、同步等待。

**模式二：Host-Device 同步阻塞**

```
CPU:     [准备数据][cudaMemcpy][等待GPU...............][处理结果]
GPU:                           [kernel执行]
```

原因：同步拷贝（非异步）阻塞了 CPU。

**模式三：计算与传输重叠良好**

```
Stream 0: [kernel1][kernel2][kernel3]
Stream 1: [HtoD   ][       ][HtoD   ]
          ↑ 传输和计算在不同 Stream 中并行
```

这是理想状态——数据传输被计算隐藏。

**模式四：NCCL 通信成为瓶颈**

```
GPU 0: [compute][AllReduce.........][compute]
GPU 1: [compute][AllReduce.........][compute]
                 ↑ 通信时间占比过大
```

### 5.3 NVTX 标注

NVTX（NVIDIA Tools Extension）是最重要的分析辅助工具，允许你在代码中标注逻辑区域：

```python
import torch
import nvtx  # pip install nvtx

# 方式一：装饰器
@nvtx.annotate("forward_pass", color="blue")
def forward(model, inputs):
    return model(inputs)

# 方式二：上下文管理器
with nvtx.annotate("backward_pass", color="red"):
    loss.backward()

# 方式三：Range（手动开始/结束）
rng = nvtx.start_range("data_loading", color="green")
data = load_batch()
nvtx.end_range(rng)
```

```python
# PyTorch 内置 NVTX 支持
with torch.cuda.nvtx.range("my_region"):
    output = model(input)

# 更细粒度的标注
torch.cuda.nvtx.range_push("layer_1")
x = self.layer1(x)
torch.cuda.nvtx.range_pop()
```

在 Nsight Systems 的时间线上，NVTX 标注显示为带颜色的区域，让你一眼看到每个逻辑阶段的耗时：

```
NVTX:   |-- data_load --|--- forward ---|------- backward -------|-- optimizer --|
CUDA HW:                 [k1][k2][k3]    [k4][k5][k6][k7][k8][k9]  [k10]
```

---

## 6. 典型分析场景

### 6.1 场景一：定位 CPU 瓶颈

**症状**：GPU 利用率低，时间线上大量 GPU 气泡。

**采集**：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=cpu -o cpu_bottleneck python train.py
```

**分析步骤**：

1. 在时间线中查看 `CUDA HW` 行，确认 GPU 是否有大段空闲
2. 查看 `CPU Thread` 行，找到空闲期间 CPU 在做什么
3. 如果 CPU 在执行 Python 代码 → 数据预处理/加载可能是瓶颈
4. 如果 CPU 在等待（mutex/IO）→ 检查数据管道

**常见原因与解决方案**：

| 原因 | 现象 | 解决方案 |
|------|------|---------|
| 数据加载慢 | DataLoader 占用大量时间 | 增加 `num_workers`，使用 `pin_memory=True` |
| Python GIL | CPU 线程交替执行 | 使用多进程替代多线程 |
| 频繁 sync | CPU 等待 GPU 完成 | 减少 `torch.cuda.synchronize()`，使用异步操作 |
| 小 kernel 频繁 launch | launch 开销 > 计算时间 | Kernel 融合，使用 `torch.compile()` |

### 6.2 场景二：内存传输分析

**症状**：数据传输耗时大，阻塞计算。

**采集**：

```bash
nsys profile --trace=cuda --gpu-metrics-device=0 -o memcpy_analysis python infer.py
```

**分析步骤**：

1. 查看 `GPU Memcpy` 行，看 HtoD / DtoH 的频率和大小
2. 检查传输是否与计算重叠（在不同 Stream 上并行）
3. 注意 `cudaMemcpy`（同步）vs `cudaMemcpyAsync`（异步）

**优化方向**：

```python
# Bad：同步拷贝阻塞 CPU
data = data.to("cuda")  # 默认同步

# Better：pinned memory + 异步拷贝
data = data.pin_memory()
data = data.to("cuda", non_blocking=True)

# Best：使用多个 Stream 重叠传输和计算
stream_compute = torch.cuda.Stream()
stream_transfer = torch.cuda.Stream()

with torch.cuda.stream(stream_transfer):
    next_batch = next_batch.to("cuda", non_blocking=True)

with torch.cuda.stream(stream_compute):
    output = model(current_batch)
```

### 6.3 场景三：Kernel Launch 开销

**症状**：大量极短的 kernel，每个只有几微秒，但 launch 间隙很大。

```
CUDA HW: [k][k][k][k][k][k][k][k]...  ← 每个 kernel 很短
CUDA API: [launch][launch][launch]...   ← launch 开销可能 > 计算时间
```

**解决方案**：

- 使用 `torch.compile()` 自动融合小 kernel
- 使用 CUDA Graph 批量 launch：

```python
# CUDA Graph：一次录制，反复回放，消除 launch 开销
g = torch.cuda.CUDAGraph()

# 录制
with torch.cuda.graph(g):
    output = model(static_input)

# 回放（极低 launch 开销）
for batch in dataloader:
    static_input.copy_(batch)
    g.replay()
```

---

## 7. 深度学习场景实战

### 7.1 训练循环 Profiling

```python
import torch
import nvtx

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        # 只 profile 中间几个 step（跳过 warmup）
        if epoch == 0 and i == 5:
            torch.cuda.cudart().cudaProfilerStart()
        if epoch == 0 and i == 15:
            torch.cuda.cudart().cudaProfilerStop()

        with nvtx.annotate(f"step_{i}", color="blue"):
            with nvtx.annotate("data_transfer", color="green"):
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            with nvtx.annotate("forward", color="yellow"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            with nvtx.annotate("backward", color="red"):
                optimizer.zero_grad()
                loss.backward()

            with nvtx.annotate("optimizer", color="purple"):
                optimizer.step()
```

```bash
# 使用 --capture-range=cudaProfilerApi 只采集标记范围
nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,nvtx,cudnn,cublas \
    -o train_detailed \
    python train.py
```

### 7.2 分析训练瓶颈

在 GUI 中打开报告后，重点关注：

**Step 时间分解**：

```
一个完整的训练 Step:
|-- data_transfer (5%) --|--- forward (25%) ---|---- backward (55%) ----|-- optim (15%) --|
                                                                         ↑ 如果 optim 占比
                                                                           意外地大，检查
                                                                           是否有 sync
```

**逐层分析**：

通过 NVTX 标注每一层，可以找到最耗时的层：

```
forward: |-- embed(2%) --|-- attn_0(8%) --|-- ffn_0(6%) --|-- attn_1(8%) --|...
```

### 7.3 PyTorch Profiler 与 Nsight Systems 配合

```python
# PyTorch 自带的 profiler 也能导出 Chrome trace
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs.cuda())
        loss = criterion(outputs, targets.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()
```

PyTorch Profiler 适合快速迭代（不需要 nsys 命令行），Nsight Systems 适合深度分析（更全面的 GPU 视图）。

---

## 8. 多机多卡与 NCCL 分析

### 8.1 多卡 Profiling

```bash
# 使用 torchrun 启动分布式训练时
nsys profile \
    -o ddp_rank%q{RANK} \
    --trace=cuda,nvtx,nccl \
    --gpu-metrics-device=all \
    torchrun --nproc_per_node=4 train_ddp.py
# %q{RANK} 用环境变量区分不同 rank 的报告
```

每个 rank 生成独立报告：`ddp_rank0.nsys-rep`, `ddp_rank1.nsys-rep`, ...

在 GUI 中可以同时打开多个 rank 的报告做对比。

### 8.2 NCCL 通信分析

开启 NCCL trace 后，时间线上会显示集合通信操作：

```
GPU 0 NCCL: [AllReduce 128MB]   [AllGather 64MB]
GPU 1 NCCL: [AllReduce 128MB]   [AllGather 64MB]
GPU 2 NCCL: [AllReduce 128MB]   [AllGather 64MB]
GPU 3 NCCL: [AllReduce 128MB]   [AllGather 64MB]
             ↑ 所有 GPU 同时参与
```

**关注点**：

- 通信耗时占比：通信时间 / (计算 + 通信) 应该 < 20-30%
- 计算与通信重叠：DDP 的 AllReduce 是否与下一层的反向传播重叠
- 通信量：每次 AllReduce 的数据量是否符合预期

### 8.3 NCCL 环境变量辅助

```bash
# 开启 NCCL debug 日志
export NCCL_DEBUG=INFO

# 查看 NCCL 选择的通信算法和拓扑
export NCCL_DEBUG_SUBSYS=GRAPH,TUNING

# 指定通信算法（调试用）
export NCCL_ALGO=Ring   # Ring / Tree / CollnetDirect
```

---

## 9. 高级用法

### 9.1 编程式控制采集范围

```python
# 使用 CUDA Profiler API 精确控制
import ctypes

# 加载 CUDA runtime
_cudart = ctypes.CDLL("libcudart.so")

def start_profiling():
    _cudart.cudaProfilerStart()

def stop_profiling():
    _cudart.cudaProfilerStop()

# 或使用 PyTorch 封装
torch.cuda.cudart().cudaProfilerStart()
# ... 只 profile 这段代码 ...
torch.cuda.cudart().cudaProfilerStop()
```

```bash
# 配合 --capture-range 使用
nsys profile --capture-range=cudaProfilerApi -o precise_profile python train.py
```

### 9.2 多次运行对比

```bash
# 运行 baseline
nsys profile -o baseline python train.py --config baseline.yaml

# 运行优化版本
nsys profile -o optimized python train.py --config optimized.yaml

# 用 stats 对比
nsys stats baseline.nsys-rep > baseline_stats.txt
nsys stats optimized.nsys-rep > optimized_stats.txt
diff baseline_stats.txt optimized_stats.txt
```

### 9.3 导出为其他格式

```bash
# 导出为 JSON（可编程分析）
nsys export --type=json my_report.nsys-rep -o my_report.json

# 导出为 SQLite（SQL 查询）
nsys export --type=sqlite my_report.nsys-rep -o my_report.sqlite

# SQL 查询示例：找到最耗时的 10 个 kernel
sqlite3 my_report.sqlite <<'SQL'
SELECT
    strings.value AS kernel_name,
    COUNT(*) AS count,
    SUM(end - start) AS total_ns,
    AVG(end - start) AS avg_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN StringIds AS strings ON k.demangledName = strings.id
GROUP BY strings.value
ORDER BY total_ns DESC
LIMIT 10;
SQL
```

### 9.4 远程 Profiling

```bash
# 在远程服务器上采集
ssh gpu-server "nsys profile -o /tmp/remote_report python train.py"

# 拷贝到本地分析
scp gpu-server:/tmp/remote_report.nsys-rep ./

# 本地 GUI 打开
nsys-ui remote_report.nsys-rep
```

---

## 10. 常见问题与最佳实践

### 10.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `nsys: command not found` | 未安装或未加 PATH | `export PATH=/usr/local/cuda/bin:$PATH` |
| 报告文件过大（>1GB） | 采集时间过长或 trace 项过多 | 使用 `--delay` + `--duration` 限制范围 |
| 看不到 kernel 名称 | 编译时未保留符号 | 加 `-lineinfo` 编译选项 |
| Permission denied | perf_event_paranoid 限制 | `sudo sysctl -w kernel.perf_event_paranoid=1` |
| GPU 指标缺失 | 驱动版本过低 | 更新到最新驱动 |
| Docker 中无法采集 | 缺少权限 | 添加 `--cap-add SYS_ADMIN --privileged` |

### 10.2 最佳实践

**1. 先宏观后微观**

```
Step 1: nsys profile 全局采集 → 找到最耗时的阶段
Step 2: NVTX 标注细化 → 定位到具体的函数/层
Step 3: ncu 分析瓶颈 kernel → 找到优化方向
Step 4: 优化后再 nsys 验证 → 确认整体改善
```

**2. 跳过 Warmup**

前几个 step 包含 JIT 编译、内存分配等一次性开销，不代表稳态性能：

```bash
nsys profile --delay=30 ...  # 延迟 30 秒跳过初始化
```

**3. 采集足够多的 Step**

至少采集 10-20 个完整的训练 step，确保统计稳定性。

**4. 控制报告大小**

```bash
# 限制采集范围
nsys profile --duration=60 ...                      # 只采 60 秒
nsys profile --capture-range=cudaProfilerApi ...    # 只采标记范围
nsys profile --trace=cuda,nvtx ...                  # 只采必要 trace
```

**5. 使用 NVTX 标注**

没有 NVTX 标注的报告很难理解。至少标注：
- 训练 step 边界
- 前向 / 反向 / 优化器阶段
- 数据加载阶段

### 10.3 性能优化检查清单

```
□ GPU 利用率 > 80%？
  └ 否 → 检查 CPU 瓶颈、数据加载、kernel launch 间隙

□ 数据传输与计算重叠？
  └ 否 → 使用 pin_memory + non_blocking + 多 Stream

□ NCCL 通信与反向传播重叠？
  └ 否 → 检查 DDP bucket 配置

□ 是否有不必要的 GPU 同步？
  └ 是 → 移除多余的 synchronize / item() / cpu()

□ Kernel 是否过于碎片化（大量 <10us 的小 kernel）？
  └ 是 → 使用 torch.compile / CUDA Graph 融合
```

## 参考资料

- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Developer - Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVTX Documentation (C/C++/Python)](https://docs.nvidia.com/nvtx/)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Developer Blog - Nsight Systems for Deep Learning](https://developer.nvidia.com/blog/nvidia-tools-extension-api-nvtx-annotation-tool-for-profiling-code-in-python-and-c-c/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
