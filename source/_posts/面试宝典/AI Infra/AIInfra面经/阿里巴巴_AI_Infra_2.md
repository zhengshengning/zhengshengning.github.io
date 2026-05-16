---
title: '阿里巴巴 AI Infra (2)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: TVM Relay调度原语和TIR调度原语的区别？

这两层调度原语分别作用于**不同的抽象层次**：

**Relay调度原语（图级别）**：
- **操作对象**：计算图中的算子节点和子图
- **关注点**：算子间的数据流关系、整体执行策略
- **核心原语**：
  - `FuseOps`：算子融合——将满足融合条件的算子合并为一个kernel。融合规则分四种模式：
    - Injective（element-wise）：可与任何算子融合
    - Reduction：只能作为融合子图的根节点
    - Complex（如Conv）：不能融合在一起
    - Opaque：不参与融合
  - `ConvertLayout`：数据布局转换（NCHW→NHWC→NCHW4c等），匹配硬件最优内存布局
  - `FoldConstant`：常量折叠——在编译时计算只依赖常量的子图
  - `EliminateCommonSubexpr`：公共子表达式消除
  - `SimplifyInference`：将BN替换为scale+bias等推理时简化
- **本质**：减少kernel数量、减少数据搬运、减少冗余计算

**TIR调度原语（算子级别）**：
- **操作对象**：单个kernel内部的循环结构和内存访问
- **关注点**：如何将计算高效映射到硬件执行单元
- **核心原语**：

```python
# 循环变换
s[C].split(i, factor=32)       # 将循环i拆分为外层和内层（tile的基础）
s[C].fuse(i, j)                # 融合两个循环维度
s[C].tile(i, j, 32, 32)       # 二维分块（对缓存友好）
s[C].reorder(j_o, i_o, j_i, i_i)  # 改变循环嵌套顺序

# 并行化与线程映射
s[C].parallel(i_o)             # CPU多核并行
s[C].vectorize(i_i)           # SIMD向量化
s[C].bind(i_o, te.thread_axis("blockIdx.x"))   # GPU block映射
s[C].bind(i_i, te.thread_axis("threadIdx.x"))  # GPU thread映射

# 内存层次
AA = s.cache_read(A, "shared", [C])    # 加载到共享内存
AL = s.cache_read(AA, "local", [C])    # 加载到寄存器
s[C].set_scope("local")                # 输出先存寄存器

# 其他
s[C].unroll(i_i)              # 循环展开
s[C].compute_at(s[output], j_o)  # 控制计算位置
s[C].pragma(i, "unroll_explicit", True)
```

**两者的关系与区别**：

| 维度 | Relay调度 | TIR调度 |
|------|----------|---------|
| 抽象层次 | 图级别（算子间） | 算子级别（循环内） |
| 优化目标 | 减少kernel数量/数据搬运 | 单kernel内性能最大化 |
| 影响范围 | 整个计算图结构 | 单个kernel的生成代码 |
| 自动化程度 | 规则驱动（启发式） | 搜索驱动（Auto-Tuning） |
| 硬件相关性 | 弱（通用规则） | 强（tile size等依赖硬件参数） |

### Q: MLIR和TVM的区别？

**TVM — 端到端深度学习编译器**：
- 完整的产品：输入模型（ONNX/TF/PyTorch）→ 输出可执行代码（CUDA/ARM/x86）
- 固定的IR层次：Relay（图级IR）→ TIR（张量级IR）→ 目标代码
- 内置Auto-Tuning：AutoTVM/Ansor/Meta-Schedule
- 内置runtime：支持模型部署
- **开箱即用**：给一个模型就能编译优化运行

**MLIR — 通用编译器基础设施框架**：
- 不是编译器，是**构建编译器的工具箱**
- 核心创新：**Dialect**机制——可定义任意层次的IR表示
  ```
  高层: TensorFlow Dialect, Torch Dialect
  中层: Linalg Dialect, Affine Dialect
  低层: GPU Dialect, LLVM Dialect, SPIRV Dialect
  ```
- 提供：IR定义框架、Pass管理、Pattern Rewriting引擎、代码生成基础设施
- **需要自行开发**：定义Dialect、编写transformation pass、连接前后端

**核心区别总结**：

| 维度 | TVM | MLIR |
|------|-----|------|
| 定位 | DL编译器产品 | 编译器框架 |
| 完整度 | 开箱即用 | 需要大量开发 |
| 灵活性 | 受限于固定IR层次 | 任意Dialect可扩展 |
| 学习曲线 | 较低（用户视角） | 较高（开发者视角） |
| 优化范围 | DL模型 | 任意领域 |
| 生态 | 独立社区 | LLVM社区支持 |

**实际联系**：
- TVM Unity（新版本）开始基于MLIR思想重构，引入类似Dialect的Relax IR
- 许多硬件厂商用MLIR构建自己的编译栈，然后接入TVM的前端或Auto-Tuning
- Google的IREE项目是用MLIR构建的端到端ML编译器（类似TVM定位但基于MLIR）

**何时选择哪个？**
- 需要快速将模型部署到目标设备 → TVM
- 需要为新硬件构建完整编译栈 → MLIR
- 需要极致的编译时优化定制 → MLIR
- 需要Auto-Tuning能力 → TVM（或在MLIR上自建）

### Q: 如何提高单个算子的性能？GEMM/Conv op优化有哪些方法？

**GEMM（通用矩阵乘法）优化 — 从naive到极致**：

**1. Tiling（分块计算）— 最核心的优化**
```
目标：让数据重用发生在更快的存储层次
典型分块层次：
  Thread Block Tile: 128x128 (适配L2 Cache / Shared Memory)
  Warp Tile: 64x32 (适配Warp级别的mma指令)
  Thread Tile: 8x8 (适配寄存器)
```
- K维度也需要分块：每次加载K_tile大小的A/B到SMEM，计算后再加载下一块
- 分块大小的选择直接决定性能（需要tuning）

**2. 内存访问优化**
- **向量化加载**：使用`float4`/`LDS.128`一次加载16字节（4个FP32），减少内存事务数4倍
- **双缓冲（Double Buffering）**：
  ```
  while (has_next_tile):
    compute(buffer[current])    # 计算当前tile
    load(buffer[1-current])     # 同时预取下一tile到另一buffer
    swap(current)
  ```
  实现计算与数据加载完全overlap
- **Swizzle/Padding消除Bank Conflict**：共享内存有32个bank，当多线程访问同一bank时串行化。通过地址变换（XOR swizzle）或padding一列避免conflict

**3. Tensor Core利用**
- A100 FP16 Tensor Core: 312 TFLOPS vs FP32 CUDA Core: 19.5 TFLOPS（**16倍差距**）
- 使用`wmma::mma_sync`或`mma.sync`PTX指令
- 约束：矩阵分片必须满足特定shape（如m16n8k16）
- CUTLASS库提供了Tensor Core GEMM的完整实现参考

**4. Software Pipelining**
- 多级流水线：将global load → shared memory store → shared to register → compute拆分为多个阶段
- 用异步拷贝指令（`cp.async`）实现全局内存到共享内存的非阻塞传输

**性能参考（A100, FP16, M=N=K=4096）**：
- Naive实现: ~2 TFLOPS (< 1% 峰值)
- 分块+SMEM: ~50 TFLOPS (~16%)
- +向量化+双缓冲: ~150 TFLOPS (~48%)
- +Tensor Core: ~280 TFLOPS (~90%)
- cuBLAS: ~300 TFLOPS (~96%)

**Conv（卷积）优化方法**：

**1. Im2Col + GEMM**
- 将卷积展开为矩阵乘：输入按滑窗展开为大矩阵，卷积核展开为另一矩阵
- 优点：复用高度优化的GEMM实现
- 缺点：Im2Col需要额外O(C_in * K² * H_out * W_out)显存

**2. 隐式GEMM（Implicit GEMM）**
- 不显式做Im2Col，而是在GEMM计算过程中动态计算元素地址映射
- 省去Im2Col的额外显存分配和数据拷贝
- cuDNN的卷积实际多采用隐式GEMM

**3. Winograd变换**
- 对3x3卷积：将乘法次数减少为原来的 4/9 ≈ 2.25倍加速
- 原理：F(m,r) Winograd用 `(m+r-1)²` 次乘法替代 `m² * r²` 次（m=输出tile, r=卷积核大小）
- 代价：需要额外的变换计算、数值精度略有下降
- 适用：小卷积核（3x3），大feature map

**4. FFT卷积**
- 将空域卷积转为频域逐点乘：`F(x*w) = F(x) · F(w)`
- 适用于大卷积核（7x7+），小卷积核时FFT开销不划算

**5. 特殊形态优化**
- **1x1 Conv**：退化为GEMM（无需Im2Col），是最容易优化的卷积
- **Depthwise Conv**：每个通道独立卷积，是memory-bound操作，需要专用kernel（与标准Conv优化策略不同）
- **Grouped Conv**：Group GEMM实现，CUTLASS支持
