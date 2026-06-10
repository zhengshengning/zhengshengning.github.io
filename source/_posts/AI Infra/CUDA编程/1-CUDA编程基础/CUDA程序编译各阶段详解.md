---
title: CUDA程序编译各阶段详解
date: 2026-06-03 15:00:00
mathjax: false
categories:
  - [AI Infra, CUDA编程与算子优化, CUDA编程基础]
tags: [CUDA, nvcc, PTX, 编译, GPU编程]
---

全面解析一个 `.cu` 文件从源代码到可执行二进制的完整编译旅程——预处理、设备代码拆分、PTX 生成、SASS 编译、Host 编译和最终链接，理解每个阶段的输入输出和底层机制。

<!-- more -->

## 📑 目录

- [1. 为什么 CUDA 编译流程如此特殊](#1-为什么-cuda-编译流程如此特殊)
- [2. nvcc 编译器驱动概览](#2-nvcc-编译器驱动概览)
- [3. 编译流程全景图](#3-编译流程全景图)
- [4. 预处理阶段](#4-预处理阶段)
- [5. 设备代码与主机代码分离](#5-设备代码与主机代码分离)
- [6. 设备代码编译：从 CUDA C++ 到 PTX](#6-设备代码编译从-cuda-c-到-ptx)
- [7. PTX 到 SASS：最终机器码生成](#7-ptx-到-sass最终机器码生成)
- [8. 主机代码编译](#8-主机代码编译)
- [9. Fatbinary：多架构打包](#9-fatbinary多架构打包)
- [10. 链接阶段](#10-链接阶段)
- [11. 分离编译与设备链接](#11-分离编译与设备链接)
- [12. JIT 编译：运行时的 PTX 编译](#12-jit-编译运行时的-ptx-编译)
- [13. 编译选项实战参考](#13-编译选项实战参考)
- [14. 常见问题与调试技巧](#14-常见问题与调试技巧)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)

---

## 1. 为什么 CUDA 编译流程如此特殊

传统 C++ 编译器处理的是一种语言、一种目标架构——源代码进去，x86/ARM 机器码出来，一条直路。但 CUDA 编译器面对的是一个**异构编译问题**：同一个 `.cu` 文件里既有跑在 CPU 上的主机代码，又有跑在 GPU 上的设备代码，两者的目标架构完全不同。

这就像一个翻译官同时要把一篇文章翻译成英文和日文——他需要先把文章拆开，识别哪些段落给英文读者、哪些给日文读者，分别翻译后再装订成一本双语书。CUDA 的编译器 `nvcc` 本质上就是这样一个"调度员"，它协调多个子编译器完成这项工作。

CUDA 编译的核心特殊性：

- **双目标架构**：一份源码需要同时生成 x86（Host）和 GPU（Device）两种机器码
- **中间表示 PTX**：设备代码先编译为虚拟指令集 PTX，再根据具体 GPU 架构转为真实机器码 SASS
- **多阶段管线**：源代码经历预处理 → 分离 → 独立编译 → 合并打包 → 链接等多个阶段
- **前向兼容设计**：通过 PTX + JIT 机制支持未来的 GPU 架构

---

## 2. nvcc 编译器驱动概览

`nvcc` 并不是一个传统意义上的编译器，而是一个**编译器驱动（Compiler Driver）**。它本身不直接生成机器码，而是将编译任务分发给不同的底层工具。

### 2.1 nvcc 的角色

| 📊 角色 | 📝 说明 |
|---------|---------|
| 命令行解析 | 接收用户传入的编译选项并分类 |
| 代码分离 | 将 `.cu` 拆分为 Host 代码和 Device 代码 |
| 调度子编译器 | 设备代码交给 `cicc`/`ptxas`，主机代码交给系统 C++ 编译器 |
| 打包 Fatbinary | 将多架构的设备代码打包为胖二进制 |
| 协调链接 | 最终调用系统链接器合并所有目标文件 |

### 2.2 nvcc 依赖的子工具

```
nvcc 编译器驱动
├── cudafe++      — CUDA 前端，分离 Host/Device 代码
├── cicc          — CUDA 设备代码编译器（CUDA C++ → PTX）
├── ptxas         — PTX 汇编器（PTX → SASS 机器码）
├── fatbinary     — 多架构打包工具
├── g++/cl.exe    — 系统 Host C++ 编译器
└── ld/link.exe   — 系统链接器
```

💡 **提示**：可以通过 `nvcc --verbose` 或 `nvcc -dryrun` 查看 nvcc 实际调用的子命令序列，这对理解编译流程非常有帮助。

### 2.3 查看实际编译步骤

```bash
# --verbose 显示每个阶段调用的实际命令
nvcc --verbose -o my_kernel my_kernel.cu

# -dryrun 只打印命令序列但不执行
nvcc -dryrun -o my_kernel my_kernel.cu

# --keep 保留所有中间文件（调试编译问题时非常有用）
nvcc --keep -o my_kernel my_kernel.cu
```

使用 `--keep` 选项后，你会在当前目录看到一系列中间文件：

```
my_kernel.cpp1.ii          # 预处理后的设备代码
my_kernel.cpp4.ii          # 预处理后的主机代码
my_kernel.ptx              # 设备代码的 PTX 中间表示
my_kernel.sm_86.cubin      # 特定架构的 SASS 二进制
my_kernel.fatbin            # 打包的 Fatbinary
my_kernel.cudafe1.cpp      # 为主机编译器准备的桩代码
```

---

## 3. 编译流程全景图

一个 `.cu` 文件的完整编译路径：

{% mermaid graph TD %}
    A[".cu 源文件"] --> B["预处理（cpp）"]
    B --> C["cudafe++ 前端分离"]
    C --> D["设备代码（.gpu）"]
    C --> E["主机代码（.cpp）"]
    D --> F["cicc 编译器"]
    F --> G["PTX 代码（.ptx）"]
    G --> H["ptxas 汇编器"]
    H --> I["SASS/cubin（.cubin）"]
    I --> J["fatbinary 打包"]
    G --> J
    J --> K["Fatbinary（.fatbin）"]
    K --> L["嵌入到主机目标文件"]
    E --> M["Host C++ 编译器（g++/cl）"]
    L --> M
    M --> N["主机目标文件（.o）"]
    N --> O["链接器（ld）"]
    O --> P["最终可执行文件"]
{% endmermaid %}

⚠️ **注意**：这张图展示的是最简单的单文件编译路径。实际项目中涉及多文件、分离编译、设备链接等更复杂的情况，将在后续章节详述。

---

## 4. 预处理阶段

### 4.1 预处理做了什么

预处理是编译的第一步，和标准 C++ 预处理完全一致：

- **头文件展开**：`#include` 指令替换为文件内容
- **宏替换**：`#define` 定义的宏全部展开
- **条件编译**：根据 `#ifdef`/`#ifndef` 选择保留的代码段
- **行号标记**：插入 `#line` 指令用于后续报错定位

### 4.2 CUDA 特有的预定义宏

nvcc 在预处理阶段会注入一系列特殊宏，让代码能区分编译上下文：

| 宏名称 | 📝 含义 |
|--------|---------|
| `__CUDACC__` | 表示当前文件正被 nvcc 处理 |
| `__CUDA_ARCH__` | 设备编译时定义，值为主版本号×100 + 次版本号×10（如计算能力 8.6 对应 860） |
| `__NVCC__` | nvcc 编译器标识 |
| `__CUDACC_VER_MAJOR__` | nvcc 主版本号 |
| `__CUDACC_VER_MINOR__` | nvcc 次版本号 |

```cpp
// 利用 __CUDA_ARCH__ 实现架构特定优化
__device__ float fast_exp(float x) {
#if __CUDA_ARCH__ >= 750
    // Turing 及以上架构使用快速数学函数
    return __expf(x);
#else
    // 旧架构使用标准实现
    return expf(x);
#endif
}
```

📌 **关键点**：`__CUDA_ARCH__` 只在设备代码编译阶段定义。如果在主机代码中检查它，值始终为未定义。这是因为同一份源码会被编译两次（一次给 Host、一次给 Device），每次编译的宏环境不同。

### 4.3 预处理阶段的输出

```bash
# 只运行预处理，查看展开后的代码
nvcc -E my_kernel.cu -o my_kernel.ii

# 用 --keep 保留中间文件时，预处理结果对应 .cpp1.ii 和 .cpp4.ii
```

---

## 5. 设备代码与主机代码分离

### 5.1 分离的必要性

一个典型的 `.cu` 文件混合了两种代码：

```cpp
#include <stdio.h>

// 设备代码：在 GPU 上执行
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 主机代码：在 CPU 上执行
int main() {
    int n = 1024;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    add_kernel<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
```

`cudafe++`（CUDA 前端）负责将这份代码拆分为两部分，分别送往不同的编译器。

### 5.2 cudafe++ 的工作机制

`cudafe++` 根据函数修饰符判断代码归属：

| 修饰符 | 归属 | 处理方式 |
|--------|------|---------|
| `__global__` | Device | 提取为设备代码 |
| `__device__` | Device | 提取为设备代码 |
| `__host__` | Host | 保留在主机代码中 |
| `__host__ __device__` | 两者 | 在两份代码中各保留一份 |
| 无修饰符 | Host | 默认为主机代码 |

### 5.3 Kernel 启动语法的转换

三尖括号 `<<<...>>>` 不是合法的 C++ 语法，`cudafe++` 会将其转换为运行时 API 调用：

```cpp
// 原始 CUDA 语法
add_kernel<<<grid, block, shared_mem, stream>>>(a, b, c, n);

// cudafe++ 转换后的等效主机代码（简化版）
cudaConfigureCall(grid, block, shared_mem, stream);
cudaSetupArgument(&a, sizeof(a), 0);
cudaSetupArgument(&b, sizeof(b), 8);
cudaSetupArgument(&c, sizeof(c), 16);
cudaSetupArgument(&n, sizeof(n), 24);
cudaLaunch(add_kernel);
```

💡 **提示**：从 CUDA 9.0+ 开始，实际使用的是更高效的 `cudaLaunchKernel()` 统一入口，上面的分步调用是为了说明原理。

### 5.4 设备代码桩函数

分离后，设备代码被提取走了，但主机代码仍然需要"知道" Kernel 的存在（用于 launch）。`cudafe++` 会在主机代码中生成一个**桩函数（stub）**，它不包含 Kernel 的实际逻辑，只用于触发 Kernel 启动的运行时 API 调用。

---

## 6. 设备代码编译：从 CUDA C++ 到 PTX

### 6.1 什么是 PTX

PTX（Parallel Thread Execution）是 NVIDIA 定义的一种**虚拟指令集架构（Virtual ISA）**。它在编译流程中的地位类似于 LLVM IR 或 Java 字节码——是一种与具体硬件解耦的中间表示。

为什么需要 PTX 这一层？想象你在写一封信，如果直接用方言写，换个地方的人就看不懂；但如果先写成普通话（PTX），到了哪个地方都能"翻译"成当地方言（SASS）。PTX 的存在使得同一份编译产物可以在不同代际的 GPU 上运行。

### 6.2 cicc 编译器的工作

`cicc` 是 NVIDIA 的设备代码编译器，负责将 CUDA C++ 转为 PTX。它的主要工作包括：

- **语法和语义分析**：检查设备代码的合法性
- **优化**：循环展开、常量折叠、死代码消除、内联等
- **寄存器分配**：为虚拟寄存器分配逻辑编号
- **PTX 生成**：输出文本格式的 PTX 指令

### 6.3 PTX 代码示例

```bash
# 生成 PTX 文件
nvcc -ptx -arch=sm_86 my_kernel.cu -o my_kernel.ptx
```

生成的 PTX 代码大致如下：

```
.version 8.0
.target sm_86
.address_size 64

.visible .entry add_kernel(
    .param .u64 add_kernel_param_0,  // float* a
    .param .u64 add_kernel_param_1,  // float* b
    .param .u64 add_kernel_param_2,  // float* c
    .param .u32 add_kernel_param_3   // int n
)
{
    .reg .pred   %p<2>;
    .reg .f32    %f<4>;
    .reg .b32    %r<6>;
    .reg .b64    %rd<8>;

    // 计算全局线程索引: idx = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32      %r1, %ctaid.x;         // blockIdx.x
    mov.u32      %r2, %ntid.x;          // blockDim.x
    mov.u32      %r3, %tid.x;           // threadIdx.x
    mad.lo.s32   %r4, %r1, %r2, %r3;   // idx = r1 * r2 + r3

    // 边界检查: if (idx < n)
    ld.param.u32 %r5, [add_kernel_param_3];
    setp.ge.s32  %p1, %r4, %r5;
    @%p1 bra     EXIT;

    // 加载 a[idx] 和 b[idx]
    ld.param.u64 %rd1, [add_kernel_param_0];
    ld.param.u64 %rd2, [add_kernel_param_1];
    cvt.s64.s32  %rd3, %r4;
    shl.b64      %rd4, %rd3, 2;         // idx * 4 (float 字节偏移)
    add.s64      %rd5, %rd1, %rd4;
    add.s64      %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];          // a[idx]
    ld.global.f32 %f2, [%rd6];          // b[idx]

    // c[idx] = a[idx] + b[idx]
    add.f32      %f3, %f1, %f2;
    ld.param.u64 %rd7, [add_kernel_param_2];
    add.s64      %rd7, %rd7, %rd4;
    st.global.f32 [%rd7], %f3;

EXIT:
    ret;
}
```

### 6.4 PTX 关键特征

| ✅ 特征 | 📝 说明 |
|---------|---------|
| 无限虚拟寄存器 | 使用 `%r<N>`、`%f<N>` 等虚拟寄存器，不受物理限制 |
| 强类型指令 | 每条指令带类型后缀，如 `add.f32`、`ld.global.f32` |
| 显式并行语义 | 内置 `%tid`、`%ctaid`、`%ntid` 等线程索引寄存器 |
| 目标无关 | 同一份 PTX 可编译为不同架构的 SASS |
| 文本格式 | 人类可读，便于调试和分析 |

### 6.5 PTX 虚拟架构与真实架构

nvcc 使用两个维度描述目标：

- **虚拟架构（`compute_XX`）**：决定 PTX 使用的指令集特性
- **真实架构（`sm_XX`）**：决定最终 SASS 的目标 GPU

```bash
# -arch 指定虚拟架构（PTX 能力），-code 指定真实架构（SASS 目标）
nvcc -arch=compute_86 -code=sm_86 my_kernel.cu

# 简写形式：-arch=sm_86 等价于 -arch=compute_86 -code=sm_86
nvcc -arch=sm_86 my_kernel.cu
```

📌 **关键点**：虚拟架构必须 ≤ 真实架构。你不能用 `compute_90` 的 PTX 特性去编译 `sm_86` 的机器码——这就像用 Python 3.12 的语法特性去编译 Python 3.8 一样不合理。

---

## 7. PTX 到 SASS：最终机器码生成

### 7.1 什么是 SASS

SASS 是 GPU 实际执行的**原生机器指令**（NVIDIA 未公开 SASS 缩写的官方含义，社区常见的解释有 Streaming ASSembly 等）。它是特定于某一代 GPU 架构的二进制码，不同架构之间不兼容。

PTX 到 SASS 的关系，可以类比 Java 字节码到具体 CPU 的机器码：PTX 是平台无关的中间表示，SASS 是跑在特定硬件上的最终指令。

### 7.2 ptxas 汇编器

`ptxas` 负责将 PTX 编译为 SASS，它的工作远不止简单的"翻译"：

- **物理寄存器分配**：将 PTX 的无限虚拟寄存器映射到有限的物理寄存器（每个 SM 有 65536 个 32 位寄存器）
- **指令调度**：优化指令顺序以隐藏延迟、避免流水线停顿
- **Bank 冲突优化**：调整 Shared Memory 访问模式
- **指令选择**：将 PTX 指令映射为具体架构的 SASS 指令（可能是一对一，也可能是一对多）
- **资源统计**：计算每个 Kernel 使用的寄存器数量、Shared Memory 大小等

```bash
# 单独调用 ptxas（通常由 nvcc 自动调用）
ptxas -arch=sm_86 my_kernel.ptx -o my_kernel.cubin

# 查看资源使用情况
ptxas -arch=sm_86 -v my_kernel.ptx
# 输出示例：
# ptxas info: Used 16 registers, 0 bytes shared memory
# ptxas info: Function properties for add_kernel
#     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

### 7.3 查看 SASS 指令

```bash
# 从 cubin 反汇编 SASS
cuobjdump -sass my_kernel.cubin

# 从可执行文件反汇编
cuobjdump -sass my_program

# 使用 nvdisasm 获取更详细的信息
nvdisasm my_kernel.cubin
```

SASS 指令示例（Ampere 架构）：

```
/*0000*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;  // 加载 blockIdx.x
/*0010*/  S2R R0, SR_TID.X ;                         // 读取 threadIdx.x
/*0020*/  IMAD R0, R1, c[0x0][0x0], R0 ;             // idx = blockIdx.x * blockDim.x + threadIdx.x
/*0030*/  ISETP.GE.AND P0, PT, R0, c[0x0][0x170], PT ; // if (idx >= n)
/*0040*/  @P0 EXIT ;                                  // 条件退出
/*0050*/  LDG.E R2, [R4] ;                           // 从 Global Memory 加载 a[idx]
/*0060*/  LDG.E R3, [R6] ;                           // 从 Global Memory 加载 b[idx]
/*0070*/  FADD R2, R2, R3 ;                          // a[idx] + b[idx]
/*0080*/  STG.E [R8], R2 ;                           // 存储到 c[idx]
```

### 7.4 PTX 与 SASS 的对比

| 📊 维度 | PTX | SASS |
|---------|-----|------|
| 层次 | 虚拟指令集 | 原生机器码 |
| 可移植性 | 跨 GPU 代际 | 仅限特定架构 |
| 寄存器 | 无限虚拟寄存器 | 有限物理寄存器 |
| 可读性 | 文本格式，人类可读 | 二进制格式（可反汇编） |
| 调度 | 无指令调度信息 | 包含调度控制位 |
| 格式 | `.ptx` 文本文件 | `.cubin` 二进制文件 |

---

## 8. 主机代码编译

### 8.1 主机编译器的选择

nvcc 将主机代码编译工作**委托给系统 C++ 编译器**：

| 📊 平台 | 默认 Host 编译器 | 切换方式 |
|---------|-----------------|---------|
| Linux | `g++` | `nvcc -ccbin /usr/bin/g++-12` |
| macOS（旧版） | `clang++` | `nvcc -ccbin /usr/bin/clang++` |
| Windows | `cl.exe`（MSVC） | `nvcc -ccbin "C:\Program Files\..."` |

```bash
# 指定主机编译器路径
nvcc -ccbin /usr/bin/g++-12 -o my_program my_kernel.cu

# 向主机编译器传递额外选项
nvcc -Xcompiler "-O3,-Wall,-march=native" my_kernel.cu
```

### 8.2 主机代码中包含什么

经过 `cudafe++` 处理后，主机编译器看到的代码包括：

- **原始的主机函数**：`main()`、其他 Host 函数
- **Kernel 桩函数**：用于触发 `cudaLaunchKernel` 的桩代码
- **设备代码注册**：`__cudaRegisterFunction` 等调用，将 Kernel 符号与 Fatbinary 中的设备代码关联
- **Fatbinary 嵌入**：设备代码以数据段的形式嵌入主机目标文件

### 8.3 `-Xcompiler` 与 `-Xlinker`

nvcc 提供机制将选项透传给底层工具：

```bash
# -Xcompiler：透传给 Host 编译器
nvcc -Xcompiler "-fPIC,-shared" -o libkernel.so kernel.cu

# -Xlinker：透传给链接器
nvcc -Xlinker "-rpath,/usr/local/cuda/lib64" -o my_program main.cu

# -Xptxas：透传给 ptxas 汇编器
nvcc -Xptxas "-v,-dlcm=ca" -o my_program kernel.cu
```

---

## 9. Fatbinary：多架构打包

### 9.1 什么是 Fatbinary

实际项目通常需要支持多种 GPU 架构（V100 是 sm_70、A100 是 sm_80、RTX 4090 是 sm_89…）。Fatbinary（胖二进制）就是把多种架构的设备代码打包到一起的容器格式。

类比理解：Fatbinary 就像一张多语言菜单，不管来的客人说什么语言，都能找到自己能看懂的那一版。运行时 CUDA Driver 会从 Fatbinary 中挑选与当前 GPU 匹配的那份代码来执行。

### 9.2 `-gencode` 选项

`-gencode` 是控制 Fatbinary 内容的核心选项：

```bash
# 为多种架构编译，生成同时包含 SASS 和 PTX 的 Fatbinary
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     -o my_program my_kernel.cu
```

每个 `-gencode` 指定一对（虚拟架构, 目标码）：

- `code=sm_XX`：生成该架构的 SASS（二进制，直接执行，无需 JIT）
- `code=compute_XX`：生成该架构的 PTX（文本，需 JIT 编译，用于前向兼容）

⚠️ **注意**：最后一个 `-gencode` 通常设置 `code=compute_XX` 以包含 PTX，确保在比所有 SASS 目标都新的 GPU 上仍能通过 JIT 运行。如果 Fatbinary 中没有匹配的 SASS 且没有兼容的 PTX，程序会在新 GPU 上启动失败。

### 9.3 Fatbinary 的运行时选择逻辑

```
程序启动 → CUDA Driver 检测当前 GPU 架构
         ↓
    Fatbinary 中有精确匹配的 SASS？
         ├── 是 → 直接加载执行（最快）
         └── 否 → Fatbinary 中有兼容的 PTX？
                    ├── 是 → JIT 编译为 SASS 后执行
                    └── 否 → 报错：no kernel image available
```

### 9.4 查看 Fatbinary 内容

```bash
# 列出可执行文件/库中包含的所有设备代码
cuobjdump -lelf my_program

# 输出示例：
# ELF file 1: my_kernel.sm_70.cubin
# ELF file 2: my_kernel.sm_80.cubin
# ELF file 3: my_kernel.sm_86.cubin
# PTX file 1: my_kernel.compute_90.ptx

# 提取特定架构的 cubin
cuobjdump -xelf sm_86 my_program
```

---

## 10. 链接阶段

### 10.1 链接的职责

链接是编译流程的最后一步，将所有目标文件（`.o`/`.obj`）合并为最终的可执行文件或共享库：

- **符号解析**：将函数调用与函数定义匹配
- **地址重定位**：修正代码中的地址引用
- **CUDA 运行时库链接**：链接 `libcudart`（运行时 API）和其他 CUDA 库
- **设备代码注册**：确保全局初始化代码正确注册所有 Kernel

### 10.2 链接 CUDA 库

```bash
# 显式链接常用 CUDA 库
nvcc -o my_program main.o kernel.o -lcudart -lcublas -lcurand

# 动态链接（默认）
nvcc -o my_program main.cu -lcublas

# 静态链接 CUDA 运行时
nvcc -o my_program main.cu --cudart static
```

常用 CUDA 库的链接选项：

| 📦 库 | 链接选项 | 用途 |
|--------|---------|------|
| CUDA Runtime | `-lcudart` | 运行时 API（默认已链接） |
| cuBLAS | `-lcublas` | 线性代数 |
| cuDNN | `-lcudnn` | 深度学习原语 |
| cuFFT | `-lcufft` | 快速傅里叶变换 |
| cuRAND | `-lcurand` | 随机数生成 |
| cuSPARSE | `-lcusparse` | 稀疏矩阵运算 |

### 10.3 常见链接错误

```bash
# 错误：undefined reference to `cudaMalloc`
# 原因：没有链接 CUDA 运行时
# 解决：使用 nvcc 链接，或手动添加 -lcudart -L/usr/local/cuda/lib64

# 错误：undefined reference to `__cudaRegisterFatBinary`
# 原因：用 g++ 而不是 nvcc 做最终链接
# 解决：用 nvcc 做链接步骤，或者链接 -lcudart_static
```

---

## 11. 分离编译与设备链接

### 11.1 传统模式的限制

默认情况下，nvcc 采用**整程序编译（Whole Program Compilation）**模式：每个 `.cu` 文件独立完成设备代码编译，设备函数只能在同一编译单元内调用。

```cpp
// helper.cu
__device__ float helper_func(float x) { return x * x; }

// main.cu
__global__ void kernel(float* data) {
    // 编译错误！helper_func 不在当前编译单元
    data[threadIdx.x] = helper_func(data[threadIdx.x]);
}
```

### 11.2 分离编译模式

加上 `-dc`（device compilation）标志开启分离编译，允许设备代码跨文件引用：

```bash
# 步骤 1：分别编译各 .cu 文件（-dc = device code compilation）
nvcc -dc -arch=sm_86 helper.cu -o helper.o
nvcc -dc -arch=sm_86 main.cu -o main.o

# 步骤 2：设备链接（将设备代码合并）
nvcc -dlink -arch=sm_86 helper.o main.o -o device_link.o

# 步骤 3：最终链接
nvcc -arch=sm_86 helper.o main.o device_link.o -o my_program
```

### 11.3 设备链接阶段做了什么

设备链接（`-dlink`）是分离编译独有的阶段：

- **设备符号解析**：将跨文件的 `__device__` 函数调用与定义匹配
- **重定位设备代码**：修正设备代码中的跨文件地址引用
- **生成统一 Fatbinary**：将所有编译单元的设备代码合并打包

{% mermaid graph LR %}
    A["helper.o（含设备代码）"] --> D["设备链接器（nvlink）"]
    B["main.o（含设备代码）"] --> D
    C["utils.o（含设备代码）"] --> D
    D --> E["device_link.o（统一设备码）"]
    A --> F["系统链接器（ld）"]
    B --> F
    C --> F
    E --> F
    F --> G["最终可执行文件"]
{% endmermaid %}

### 11.4 使用 `extern __device__` 声明

分离编译模式下，跨文件引用设备函数需要声明：

```cpp
// helper.h — 设备函数声明
#pragma once
extern __device__ float helper_func(float x);

// helper.cu — 设备函数定义
#include "helper.h"
__device__ float helper_func(float x) { return x * x; }

// main.cu — 使用设备函数
#include "helper.h"
__global__ void kernel(float* data) {
    data[threadIdx.x] = helper_func(data[threadIdx.x]);
}
```

💡 **提示**：分离编译虽然灵活，但有轻微的性能代价——编译器无法跨文件内联设备函数。对性能关键路径上的小函数，考虑将其定义放在头文件中用 `__forceinline__ __device__` 修饰。

---

## 12. JIT 编译：运行时的 PTX 编译

### 12.1 JIT 编译机制

当程序在 Fatbinary 中找不到当前 GPU 架构的 SASS，但找到了兼容的 PTX 时，CUDA Driver 会**在运行时**将 PTX 编译为当前架构的 SASS。这就是 JIT（Just-In-Time）编译。

就好比你买了一本附带"世界语"原文的多语言书——如果书里没有你母语的翻译，翻译官可以现场从世界语翻译给你。代价是首次执行有额外的编译延迟。

### 12.2 JIT 缓存

JIT 编译的结果会被 CUDA Driver 缓存到磁盘，后续运行不会重复编译：

```bash
# 默认缓存目录
# Linux: ~/.nv/ComputeCache/
# Windows: %APPDATA%/NVIDIA/ComputeCache/

# 通过环境变量控制 JIT 缓存
export CUDA_CACHE_DISABLE=0       # 0=启用缓存（默认），1=禁用
export CUDA_CACHE_MAXSIZE=268435456  # 缓存大小上限（字节），默认 256MB
export CUDA_CACHE_PATH=/tmp/cuda_cache  # 自定义缓存目录
```

### 12.3 JIT 编译的优缺点

| ✅ 优点 | ❌ 缺点 |
|---------|---------|
| 前向兼容：新 GPU 无需重新编译程序 | 首次运行有编译延迟（可达数秒） |
| 减小二进制体积（只发布 PTX） | JIT 编译器优化可能不如离线编译充分 |
| 支持运行时特化 | 需要分发 PTX（暴露中间表示） |

### 12.4 何时依赖 JIT vs 离线编译

| 📊 场景 | ✅ 推荐策略 |
|---------|------------|
| 发布给已知 GPU 型号的集群 | 只编译对应 SASS，无需 PTX |
| 发布给终端用户（GPU 型号未知） | 编译主流架构的 SASS + 最新 PTX |
| 开发调试阶段 | 只编译当前开发机 GPU 的 SASS |
| 库发布（如 cuDNN、cuBLAS） | 全架构 SASS + PTX |

---

## 13. 编译选项实战参考

### 13.1 常用 nvcc 选项速查

| 选项 | 📝 作用 |
|------|---------|
| `-arch=sm_XX` | 指定目标架构（简写形式） |
| `-gencode arch=compute_XX,code=sm_XX` | 精确指定虚拟/真实架构对 |
| `-O0` / `-O2` / `-O3` | 优化级别（设备代码） |
| `-g` | 生成调试信息 |
| `-G` | 生成设备调试信息（严重影响性能，仅调试用） |
| `-lineinfo` | 保留行号信息（轻量，推荐用于 Profiling） |
| `-maxrregcount=N` | 限制每线程最大寄存器数 |
| `-use_fast_math` | 启用快速数学函数（牺牲精度换速度） |
| `-ptx` | 只生成 PTX，不编译为 SASS |
| `-cubin` | 生成 cubin 文件 |
| `-dc` | 启用分离编译 |
| `-dlink` | 执行设备链接 |
| `-shared` | 生成共享库 |
| `-std=c++17` | 指定 C++ 标准版本 |
| `--keep` | 保留所有中间文件 |
| `--verbose` | 打印详细编译过程 |
| `-Xcompiler` | 透传选项给 Host 编译器 |
| `-Xptxas` | 透传选项给 ptxas |
| `--threads N` | 多线程并行编译（加速多 gencode） |

### 13.2 典型编译命令示例

```bash
# 开发调试：快速编译，当前架构，带调试信息
nvcc -arch=sm_86 -g -lineinfo -O0 -o debug_app kernel.cu

# 性能分析：带行号映射，用于 Nsight Compute
nvcc -arch=sm_86 -O3 -lineinfo -o profile_app kernel.cu

# 发布版本：多架构 + PTX 前向兼容
nvcc -O3 \
     -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_90,code=compute_90 \
     --threads 4 \
     -o release_app kernel.cu

# 限制寄存器使用以提升 Occupancy
nvcc -arch=sm_86 -maxrregcount=32 -O3 -o opt_app kernel.cu

# 生成共享库
nvcc -arch=sm_86 -O3 -shared -Xcompiler -fPIC -o libkernel.so kernel.cu
```

### 13.3 CMake 中的 CUDA 编译配置

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyGPUApp LANGUAGES CXX CUDA)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90")

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(my_app main.cu kernel.cu)

# 设置编译选项
target_compile_options(my_app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        --threads 4
    >
)

# 分离编译支持
set_target_properties(my_app PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接 CUDA 库
target_link_libraries(my_app PRIVATE
    CUDA::cublas
    CUDA::curand
)
```

---

## 14. 常见问题与调试技巧

### 14.1 架构不匹配问题

```
CUDA error: no kernel image is available for execution on the device
```

原因：Fatbinary 中没有当前 GPU 架构的 SASS，也没有可 JIT 的 PTX。

解决方案：

```bash
# 查看当前 GPU 的计算能力
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# 确保编译时包含对应架构
nvcc -gencode arch=compute_89,code=sm_89 ...  # 例如 RTX 4090

# 或者包含 PTX 以支持任意新架构
nvcc -gencode arch=compute_86,code=compute_86 ...
```

### 14.2 编译性能优化

多架构编译会显著增加编译时间（每个 gencode 是一次完整编译）：

```bash
# 开发时只编译当前 GPU 架构
nvcc -arch=sm_86 ...   # 比多 gencode 快 5-10 倍

# 使用并行编译
nvcc --threads 8 ...   # 多 gencode 并行编译

# CMake 中设置（仅 CI/Release 时编译全架构）
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_ARCHITECTURES "native")  # 只编译本机架构
else()
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90")
endif()
```

### 14.3 `-G` 对性能的影响

⚠️ **注意**：`-G` 选项（设备调试）会**完全禁用设备代码优化**，性能可能下降 10-100 倍。只在需要 cuda-gdb 单步调试时使用。性能分析请用 `-lineinfo` 替代。

| 📊 选项 | 用途 | 性能影响 |
|---------|------|---------|
| `-g` | Host 调试信息 | 无影响 |
| `-G` | Device 调试信息 | 严重降低（10-100x） |
| `-lineinfo` | 保留行号映射 | 极小（< 1\%） |

### 14.4 中间文件分析工作流

当遇到编译相关的疑难问题时，可以利用中间文件逐步定位：

```bash
# 1. 保留所有中间文件
nvcc --keep -arch=sm_86 problem_kernel.cu -o problem

# 2. 检查预处理结果（宏展开是否正确）
less problem_kernel.cpp1.ii

# 3. 检查 PTX（设备代码逻辑是否正确）
less problem_kernel.ptx

# 4. 查看 SASS 资源使用
cuobjdump -res-usage problem

# 5. 反汇编 SASS 检查指令级细节
cuobjdump -sass problem
```

---

## 📝 总结

CUDA 程序的编译是一个精心设计的多阶段流水线，核心设计哲学是**异构编译 + 前向兼容**：

1. **预处理**：标准 C++ 预处理 + CUDA 特有宏注入
2. **代码分离**：`cudafe++` 将 Host/Device 代码拆分，转换 `<<<>>>` 语法
3. **设备编译**：`cicc` 将 CUDA C++ 编译为 PTX 虚拟指令
4. **SASS 生成**：`ptxas` 将 PTX 编译为特定架构的原生机器码
5. **Fatbinary 打包**：将多架构 SASS + PTX 打包为统一容器
6. **主机编译**：系统 C++ 编译器编译主机代码，嵌入 Fatbinary
7. **链接**：合并所有目标文件，链接 CUDA 运行时库

理解这个流程让你能够：精确控制编译目标以优化二进制体积和兼容性，定位编译错误到具体阶段，合理使用 PTX/JIT 实现前向兼容，以及通过分析中间产物深入理解程序行为。

---

## 🎯 自我检验清单

- 能说出 nvcc 编译 `.cu` 文件经历的主要阶段及各阶段输出产物
- 能区分 PTX（虚拟指令集）和 SASS（原生机器码）的定位与关系
- 能解释 `-arch=compute_XX` 和 `-code=sm_XX` 的区别及组合规则
- 能使用 `-gencode` 为多种 GPU 架构编译 Fatbinary
- 能使用 `--keep` 和 `--verbose` 诊断编译问题
- 能配置分离编译（`-dc`/`-dlink`）实现跨文件设备函数调用
- 能解释 JIT 编译的触发条件、缓存机制和适用场景
- 能根据开发/调试/发布不同场景选择合适的编译选项组合
- 能在 CMake 项目中正确配置 CUDA 编译参数
- 能读懂 PTX 代码的基本结构并关联到原始 CUDA C++ 源码

---

## 📚 参考资料

- [NVIDIA CUDA Compiler Driver NVCC Documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [NVIDIA PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA CUDA C++ Programming Guide - Compilation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)
- [NVIDIA cuobjdump Documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump)
- [CMake CUDA Support Documentation](https://cmake.org/cmake/help/latest/manual/cmake-compile-features.7.html#cuda-standards)
