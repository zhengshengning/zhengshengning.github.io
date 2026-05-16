---
title: '快手 AI Infra 校招 (3)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: AI 编译器的核心工作是什么？

AI 编译器负责将深度学习模型从框架级表示（如 PyTorch 计算图）转换为高效的底层硬件代码，填补"模型描述"与"硬件执行"之间的 gap：

**1. 图优化（Graph-level Optimization）**：
- **算子融合**：将 Linear+Bias+GELU 等连续操作合并为一个 kernel，减少 3→1 次 kernel launch + 中间 tensor HBM 读写
- **常量折叠**：编译期计算所有固定输入的子图
- **死代码消除**：移除不影响输出的计算
- **Layout 转换**：选择对硬件最优的内存布局（如 NHWC vs NCHW），消除运行时 transpose

**2. 算子调度/Tiling（Operator Scheduling）**：
- 确定每个算子内部的**循环顺序**（如矩阵乘的 M/N/K 维度遍历顺序）
- 确定 **tile 大小**（分块粒度影响数据复用和并行度）
- 确定**并行维度**（哪些循环并行到 GPU block/thread）
- 确定**内存层级映射**（哪些数据放 register/shared memory/global memory）

**3. 内存规划（Memory Planning）**：
- 分析所有 tensor 的生命周期，复用已释放的 buffer
- 优化数据搬运调度（如 DMA 预取与计算重叠）
- 减少总显存峰值

**4. 代码生成（Code Generation）**：
- 将优化后的 IR 降级为目标硬件代码：CUDA PTX/SASS、HIP、OpenCL、NPU 指令
- 处理硬件特定的约束（对齐、bank conflict 等）

**5. 自动调优（Auto-Tuning）**：
- 搜索最优的 tile size、循环顺序、并行策略等参数配置
- TVM AutoScheduler/Ansor：基于代价模型的搜索
- Triton：编译器自动选择最优配置

**代表项目对比**：
| 项目 | 特点 | 适用场景 |
|---|---|---|
| TVM | 完整编译栈，支持自动调优 | 跨平台部署 |
| Triton | Python DSL，block级编程 | 快速 GPU kernel 开发 |
| XLA | Google 生态，JAX 后端 | TPU + 大规模训练 |
| torch.compile (Inductor) | PyTorch 原生，动态图编译 | PyTorch 用户透明加速 |

### Q: 异构平台优化（NVIDIA/AMD）的挑战？

**核心挑战**：

**1. 编程模型差异**：
- CUDA（NVIDIA）vs HIP/ROCm（AMD），API 相似度 ~90%（hipify 可自动转换）
- 但底层硬件行为差异大，同样的代码性能可能差很多

**2. 硬件架构差异**：
| 维度 | NVIDIA | AMD |
|---|---|---|
| 执行单位 | Warp (32 threads) | Wavefront (64 threads) |
| 矩阵加速 | Tensor Core (WMMA/MMA) | Matrix Core (MFMA) |
| Shared Memory | 最大 164KB/SM (A100) | 最大 64KB/CU |
| L2 Cache | 40-50 MB | 8-16 MB |
| HBM 带宽 | 3.35 TB/s (H100) | 5.3 TB/s (MI300X) |

**3. 工具链成熟度差距巨大**：
- NVIDIA：Nsight Compute（详细 kernel 分析）+ Nsight Systems（系统级）+ cuDNN/cuBLAS
- AMD：rocProf（功能较弱）+ MIOpen（覆盖不全）+ hipBLAS
- 调试和性能分析效率差 3-5 倍

**4. Kernel 性能不可直接移植**：
- Warp 32 vs Wavefront 64 导致 reduce/shuffle 逻辑不同
- Shared Memory 大小差异影响 tiling 策略
- 不同的 cache 行为需要不同的预取策略

**解决方案**：
- **编译器抽象**：Triton/TVM 做硬件抽象，一份代码生成不同后端
- **HIP 兼容层**：hipify-clang 自动翻译 CUDA→HIP（大部分 API 一一对应）
- **性能调优**：热点 kernel 需要针对目标架构重新调优参数（tile size、unroll factor）
- **统一框架**：PyTorch 的 dispatcher 机制支持多后端透明切换

### Q: Triton 与 CUDA 的区别？

| 维度 | CUDA | Triton |
|---|---|---|
| 抽象级别 | **线程级**（需关心 thread/warp/block） | **Block 级**（以 tile 为单位编程） |
| 编程语言 | C/C++ + PTX | Python |
| 内存管理 | 手动：需显式管理 shared memory 分配、地址计算、同步 | **自动**：编译器处理 tiling、shared memory、register |
| Bank Conflict | 手动避免（padding/swizzle） | 编译器自动处理 |
| Tensor Core | 手动调用 WMMA/MMA 指令 | 声明式 `tl.dot(a, b)`，编译器自动生成 MMA |
| 可移植性 | 仅 NVIDIA | NVIDIA + AMD（通过不同 backend） |
| 性能上限 | 极致（可直接写 PTX） | 通常达 CUDA 手写的 **80-95%** |
| 开发效率 | 低（100-500 行实现一个 kernel） | 高（20-50 行实现同等功能） |
| 调试 | compute-sanitizer, printf | print, 但调试工具较少 |
| JIT 编译 | 需要预编译或运行时 NVRTC | 天然 JIT，运行时编译 |
| 适用场景 | 极致性能优化、复杂算子 | 快速原型、中等复杂算子、研究 |

**Triton 的编程范式**：
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # 类似 blockIdx
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 一次处理 BLOCK 个元素
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)  # 向量化加载
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**何时选 Triton vs CUDA**：
- 追求开发速度、原型验证 → Triton
- 需要 95%+ 峰值性能、复杂数据流控制 → CUDA
- FlashAttention 最初用 CUDA 实现，后来有 Triton 版本（性能约 90%）
