---
title: '上海AI实验室 AI Infra 实习 二面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 算子融合（Op Fusion）有哪些方式？

算子融合是深度学习编译器中最关键的优化，通过将多个算子合并为单个kernel执行，减少中间tensor的显存读写和kernel launch开销。

**融合方式分类：**

**1. Element-wise融合（最常见，收益最直接）：**
- 连续逐元素操作合并：如 ReLU + Add + Mul 合并为一个kernel
- 每个输出元素只依赖对应位置的输入，数据流是1:1映射
- 中间结果完全留在寄存器中，**减少N-1次HBM读写**（N个逐元素op融合后只需1次读+1次写）
- 典型收益：3个连续element-wise操作融合后速度提升3-5倍（从memory-bound变为compute-bound）

**2. Reduce融合（Element-wise + Reduce）：**
- 如LayerNorm中：先计算x-mean（element-wise），再计算variance（reduce），再normalize
- 融合后在一次kernel中完成所有步骤，避免中间tensor写回HBM
- BatchNorm的mean/var计算也属于此类融合

**3. 复合算子融合（大粒度融合）：**
- Conv + BN + ReLU：推理时BN退化为线性变换，吸收到Conv的weight和bias中
- MatMul + Bias + GELU/SiLU：Transformer MLP层的核心pattern
- 这类融合需要框架/编译器识别特定pattern

**4. 注意力融合（FlashAttention类）：**
- 将 QK^T + Scale + Mask + Softmax + Dropout + V矩阵乘 全部在SRAM中完成
- 消除中间N×N attention矩阵对HBM的读写
- IO量从O(N^2)降到O(N^2·d/M)，是最具代表性的融合优化

**5. 计算通信融合（分布式优化）：**
- GEMM输出分块与AllReduce通信重叠
- 一部分GEMM结果就绪后立即开始该部分的通信，不等所有计算完成
- 减少通信等待时间（latency hiding）

**框架实现比较：**

| 框架/编译器 | 融合方式 | 特点 |
|------------|----------|------|
| TensorRT | 静态图Pattern Matching | 预定义融合规则，覆盖全面 |
| XLA | HLO层面融合 | 自动发现element-wise融合机会 |
| NVFuser/torch.compile | JIT融合 | 运行时动态生成融合kernel |
| TVM/Ansor | 自动调度 | 搜索最优融合+调度策略 |
| 手动(cuDNN/CUTLASS) | 库级别 | 预实现常见融合组合 |

---

### Q: MindSpore的特点是什么？

MindSpore是华为开源的全场景深度学习框架，主要定位是与华为昇腾(Ascend)生态深度配合。

**核心特点：**

**1. 统一架构支持端边云全场景：**
- 端侧：MindSpore Lite（手机/IoT设备部署）
- 边缘：支持Atlas系列边缘设备
- 云端：Ascend 910集群训练
- 一套代码多端部署，降低开发成本

**2. 图算融合（Graph-Kernel Fusion）：**
- 自动将计算图中的子图识别并融合为高效kernel
- 不同于手动pattern matching，使用polyhedral模型自动生成融合kernel
- 对Ascend NPU的特殊指令集(CUBE/Vector)深度适配

**3. 自动并行：**
- 用户只需给少量算子标注并行策略（如shard(weight, (2, 4))表示按2x4切分）
- 框架自动推导其他算子的切分方式和插入通信算子
- 支持数据并行+模型并行+流水线并行的混合策略自动搜索

**4. 动静统一（PIJit）：**
- 动态图模式开发（类PyTorch eager mode），静态图模式执行（性能优化）
- PIJit技术将Python动态图代码JIT编译为静态图
- 类似PyTorch的torch.compile思路

**5. 函数式微分：**
- 支持函数式自动微分（类JAX风格）
- 支持高阶导数（Hessian矩阵计算等）
- `mindspore.grad(fn)`直接获取梯度函数

**与PyTorch/TensorFlow对比：**

| 维度 | MindSpore | PyTorch | TensorFlow |
|------|-----------|---------|-----------|
| 硬件生态 | Ascend优先 | NVIDIA GPU主流 | 通用 |
| 编程范式 | 动静统一 | 动态图为主 | 静态图(1.x)/动态图(2.x) |
| 自动并行 | 原生支持 | 需FSDP/Megatron | 需手动 |
| 生态成熟度 | 发展中 | 最活跃 | 大但增速放缓 |

---

### Q: TVM了解吗？

TVM是端到端的深度学习编译器框架，核心理念是将高层模型表示**编译优化**为特定硬件上的高效代码，而非依赖手写库。

**整体架构：**

```
ML Model (ONNX/PyTorch/TF)
        ↓ Import
    Relay IR (高层图优化)
    - 算子融合、常量折叠、布局变换
        ↓ Lower
    TE/TIR (张量表达/低层IR)
    - 循环变换、分块、向量化
        ↓ CodeGen
    目标代码 (CUDA/LLVM/Metal/OpenCL)
```

**核心设计哲学——计算与调度分离：**
```python
# 计算定义（WHAT to compute）
C = te.compute((M, N), lambda i, j: 
    te.sum(A[i, k] * B[k, j], axis=k))

# 调度定义（HOW to compute）
s = te.create_schedule(C.op)
xo, xi = s[C].split(C.op.axis[0], factor=32)  # 分块
s[C].bind(xo, te.thread_axis("blockIdx.x"))   # 映射到GPU线程
s[C].vectorize(xi)                             # 向量化
```

**自动调度搜索（AutoTVM / Ansor/Meta-Schedule）：**
- AutoTVM：用户定义搜索空间模板，通过随机搜索+XGBoost cost model找最优配置
- Ansor(Auto-Scheduler)：无需模板，自动生成搜索空间并搜索
- Meta-Schedule(TVM Unity)：统一框架，支持更灵活的搜索策略
- 搜索数千种配置组合（tile size、unroll factor、并行策略等）

**TVM的优势与局限：**
- 优势：多后端支持（一次优化，多硬件部署）、自动搜索替代手工调优
- 局限：搜索时间长（数小时）、动态shape支持有限、对新硬件适配需要定义target

---

### Q: 如何优化一个GPU kernel？

GPU kernel优化需要系统性方法，首先确定瓶颈类型，再针对性优化。

**Step 1: Profile确定瓶颈类型**

使用Nsight Compute分析kernel，关注：
- Memory Throughput SOL%（实际带宽/理论带宽）→ 高则memory-bound
- Compute Throughput SOL%（实际算力/理论算力）→ 高则compute-bound
- 两者都低 → latency-bound（并行度不足）

**Step 2: 针对不同瓶颈的优化策略**

**Memory-bound优化（最常见）：**
| 优化手段 | 原理 | 预期收益 |
|----------|------|---------|
| 合并访问(Coalesced) | 连续线程访问连续地址，合并为一次事务 | 最高32倍 |
| 向量化加载(float4) | 一次load 128bit，减少指令数 | 2-4倍 |
| 共享内存缓存 | 将重复读取的数据放入SRAM | 取决于复用率 |
| 避免Bank Conflict | Padding/Swizzle消除共享内存串行化 | 消除瓶颈 |
| 减少冗余读写 | 算子融合、寄存器缓存中间结果 | 显著 |

**Compute-bound优化：**
- 利用Tensor Core（FP16/INT8 GEMM）
- 提高ILP：循环展开、独立指令交错
- 减少warp分支发散（同一warp内走不同路径会串行化）
- 使用特殊函数单元(SFU)：__fmaf_rn, __expf等

**Latency-bound优化：**
- 增大并行度：更多active warp隐藏访存延迟
- 合适的block size：太小则SM资源浪费，太大则占用过多共享内存/寄存器
- 异步操作：cp.async异步加载数据

**Step 3: 迭代优化**

每次修改后重新profile验证效果，因为优化一个瓶颈可能暴露另一个瓶颈（如优化访存后可能变为compute-bound）。

---

### Q: 手撕：判断字符串中除前两个数字外，其余数字都由前两个数字相加组成？

（编程题）
