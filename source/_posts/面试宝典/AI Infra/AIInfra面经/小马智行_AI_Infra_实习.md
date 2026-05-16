---
title: '小马智行 AI Infra 实习'
date: 2026-04-16 12:08:48
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: 什么是CUDA Graph？为什么会用到更多显存？推理的什么阶段更适合CUDA Graph？

**CUDA Graph是什么？**

CUDA Graph将一系列GPU操作（kernel launch、内存拷贝等）**录制为一个图结构**，后续可以一次性提交整个图执行，消除CPU端逐个launch的开销。

```cuda
// 录制阶段（一次性）
cudaStreamBeginCapture(stream);
kernelA<<<grid, block, 0, stream>>>(...);
kernelB<<<grid, block, 0, stream>>>(...);
kernelC<<<grid, block, 0, stream>>>(...);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph);

// 执行阶段（反复调用）
for (int step = 0; step < num_steps; step++) {
    cudaGraphLaunch(graphExec, stream);  // 一次launch执行A+B+C
    // CPU端开销从 3×5us 降为 1×5us
}
```

**为什么会用到更多显存？**

| 原因 | 解释 |
|------|------|
| 静态内存分配 | Graph录制时确定所有中间buffer地址，运行时不能动态复用 |
| 无法做内存池化 | 普通执行可以在kernel间复用同一buffer（生命周期不重叠），Graph中所有buffer必须同时存在 |
| 多版本共存 | 如果为不同shape录制多个Graph，每个Graph独立分配内存 |
| 输入/输出buffer固定 | Graph的输入输出地址在录制时绑定，切换请求时需要额外copy |

**典型额外显存开销：** 10-30%（取决于中间tensor数量和大小）

**推理哪个阶段更适合CUDA Graph？**

| 阶段 | 适合程度 | 原因 |
|------|---------|------|
| **Decode** | 非常适合 | 每步计算模式固定（shape不变：batch×1×hidden），kernel序列完全相同 |
| Prefill | 不太适合 | 序列长度变化大，每个请求的shape不同，无法用固定Graph |
| Prefill(固定bucket) | 部分适合 | 如果pad到固定长度(512/1024/2048)，可以为每个bucket录制一个Graph |

**vLLM/SGLang中的CUDA Graph使用：**
- 为不同batch_size录制多个Graph（如bs=1,2,4,8,16,...,256）
- Decode时根据当前batch大小选择最接近的Graph执行
- 输入数据通过`cudaGraphExecKernelNodeSetParams`更新（避免重新录制）

### Q: 讲讲跨Block的通信方式和Warp原语？

**跨Block通信——Block间没有共享内存，需要通过全局机制：**

**1. 全局内存 + 原子操作（最常用）：**
```cuda
__device__ int block_counter = 0;  // 全局计数器

__global__ void kernel(float* partial_sums, float* result) {
    // 每个block计算自己的partial_sum
    __shared__ float sdata[256];
    // ... reduce within block ...
    
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sdata[0];  // 写入全局内存
        __threadfence();  // 确保写入对其他block可见
        
        // 最后一个完成的block做最终reduce
        int old = atomicAdd(&block_counter, 1);
        if (old == gridDim.x - 1) {
            // 我是最后一个block，做最终规约
            float total = 0;
            for (int i = 0; i < gridDim.x; i++)
                total += partial_sums[i];
            *result = total;
        }
    }
}
```

**2. Cooperative Groups（CUDA 9+）：**
```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    cg::grid_group grid = cg::this_grid();
    // ... 每个block完成自己的工作 ...
    grid.sync();  // 全GPU所有block同步！（需要所有block同时驻留在SM上）
    // ... 所有block的结果现在对所有人可见 ...
}
// launch时需要用cooperative launch API
cudaLaunchCooperativeKernel(kernel, gridDim, blockDim, args);
```
限制：所有block必须同时驻留在GPU上（block数 × 每block资源 ≤ GPU总资源）

**Warp原语（Warp-level Primitives）：**

| 原语 | 功能 | 典型用途 |
|------|------|---------|
| `__shfl_sync(mask, val, lane)` | 广播：获取指定lane的值 | 分发参数 |
| `__shfl_down_sync(mask, val, delta)` | 向下偏移 | Reduce(规约) |
| `__shfl_up_sync(mask, val, delta)` | 向上偏移 | Scan(前缀和) |
| `__shfl_xor_sync(mask, val, mask)` | 异或交换 | Butterfly reduce |
| `__ballot_sync(mask, predicate)` | 32线程投票→32bit结果 | 条件统计/压缩 |
| `__any_sync(mask, predicate)` | 任一线程为true则true | 条件检查 |
| `__all_sync(mask, predicate)` | 所有线程true才true | 条件检查 |
| `__match_any_sync(mask, val)` | 值相等的线程组成子集 | 数据分组 |
| `__popc(__ballot_sync(...))` | 计算满足条件的线程数 | 计数 |

**为什么Warp原语重要？**
- 无需共享内存（零额外存储）
- 无需`__syncthreads()`（warp内硬件同步）
- 延迟极低（1-2 cycles vs 共享内存~28 cycles）
- 是实现高效reduce/scan/broadcast的基础

### Q: NV芯片PTX机器模型的认识？

PTX（Parallel Thread Execution）是NVIDIA的**虚拟ISA**，介于CUDA C++和实际硬件SASS指令之间。

**PTX的设计理念：**
```
CUDA C++ → (nvcc前端) → PTX (虚拟ISA, 设备无关)
                          → (ptxas后端) → SASS (实际机器码, 设备相关)
```

**PTX机器模型的核心抽象：**

| 概念 | PTX模型 | 对应硬件 |
|------|---------|---------|
| CTA (Cooperative Thread Array) | 协作线程组 | Thread Block |
| Warp | 32线程的执行单元 | SIMT调度单位 |
| Thread | 最小执行单元 | CUDA Core上执行 |
| 寄存器(.reg) | 无限虚拟寄存器 | 物理寄存器(由ptxas分配) |
| 共享内存(.shared) | Block级快速存储 | SM的SRAM |
| 全局内存(.global) | 全设备可见 | HBM |
| 局部内存(.local) | 线程私有溢出空间 | 实际位于global(经L1/L2缓存) |
| 常量内存(.const) | 只读全局 | 专用cache路径 |

**PTX的关键特性：**
- **无限寄存器假设**：程序员使用任意数量的虚拟寄存器，由ptxas做物理寄存器分配和溢出(spill)决策
- **设备无关**：同一PTX可在不同GPU架构上运行（JIT编译为对应SASS）
- **显式内存模型**：`membar.gl`(全局fence)、`membar.cta`(CTA内fence)保证内存可见性
- **同步语义**：`bar.sync`(CTA级同步)、warp内隐式同步

**PTX在实际开发中的用途：**
- 查看编译器生成的指令（`nvcc -ptx`或`cuobjdump --dump-ptx`）
- 使用内联PTX汇编访问新硬件特性（如cp.async、mma指令）
- 性能调优时分析指令级行为

### Q: CUDA代码的编译流程？

**完整编译流程：**

```
.cu文件(host+device混合代码)
    ↓ nvcc前端(cudafe++)
    ├── Host代码 → .cpp → gcc/clang → .o (普通C++编译)
    └── Device代码 → .ptx (虚拟ISA)
                      ↓ ptxas
                      .cubin/.sass (实际机器码, 特定架构如sm_80)
    ↓ fatbinary打包
    .fatbin (可含多个架构的SASS + PTX备用)
    ↓ 链接
    可执行文件(host代码 + 嵌入的fatbinary)
```

**关键编译选项：**
```bash
# 指定目标架构
nvcc -gencode arch=compute_80,code=sm_80  # A100的SASS
nvcc -gencode arch=compute_80,code=compute_80  # A100的PTX(可JIT到未来架构)

# 常用组合（兼容多代GPU）
nvcc -gencode arch=compute_70,code=sm_70 \   # V100
     -gencode arch=compute_80,code=sm_80 \   # A100
     -gencode arch=compute_90,code=sm_90     # H100

# 调试/优化选项
nvcc -G          # 设备代码调试信息(禁用优化)
nvcc -lineinfo   # 保留行号信息(不影响优化)
nvcc --maxrregcount=128  # 限制每线程最大寄存器数
nvcc -Xptxas -v  # 显示寄存器/共享内存使用情况
```

**Fatbinary机制：**
- 一个可执行文件中嵌入多个架构版本
- 运行时CUDA driver选择匹配当前GPU的SASS版本
- 如果没有精确匹配的SASS，使用PTX做JIT编译（首次运行较慢，结果会被缓存）

**JIT编译缓存：**
- 缓存位置：`~/.nv/ComputeCache/`
- 首次运行某PTX在新架构上可能慢数秒（JIT编译）
- 后续运行直接使用缓存的编译结果

### Q: MLIR是什么？为什么要设计MLIR？

**MLIR（Multi-Level Intermediate Representation）** 是LLVM的子项目，提供一个**可扩展的编译器基础设施框架**。

**设计动机——解决"IR孤岛"问题：**

```
传统编译器生态（各自为战）：
  TensorFlow: XLA HLO
  PyTorch: TorchScript IR, FX Graph
  TVM: Relay + TIR
  ONNX: ONNX Graph
  每个框架重新发明IR、优化Pass、代码生成 → 巨大的重复工程

MLIR的愿景（统一基础设施）：
  所有框架 → MLIR Dialects → 共享优化基础设施 → 多硬件后端
```

**为什么需要"多层"IR？**

传统LLVM IR的问题：
- 只有一层IR（LLVM IR），从高层语言一步降到这一层时**信息大量丢失**
- 例如：一个矩阵乘在LLVM IR中变成了嵌套循环 → 编译器无法识别这是矩阵乘 → 无法调用Tensor Core
- 高层优化（如算子融合、tiling策略）无法在LLVM IR上做

MLIR的解决方案：
- 支持**任意数量的IR层级**（通过Dialect机制）
- 每一层保留该层需要的语义信息
- 在最合适的层级做对应的优化

**Dialect生态：**
```
高层:  tf(TensorFlow ops), torch(PyTorch ops), tosa(标准化ML ops)
中层:  linalg(线性代数), tensor(张量), memref(内存引用)
低层:  scf(控制流), affine(仿射循环), gpu(GPU映射)
底层:  llvm(对接LLVM), nvvm(NVIDIA PTX), spirv(Vulkan/OpenCL)
```

**在AI Infra中的应用：**
- **IREE**：基于MLIR的端到端ML编译器（Google）
- **Triton**：Triton IR是一种MLIR dialect
- **XLA/StableHLO**：Google的ML编译IR，基于MLIR
- **torch-mlir**：PyTorch模型导入MLIR的桥梁
- **各NPU厂商**：定义自己的后端Dialect适配专有硬件

### Q: 手撕：归并排序？

（编程题）
