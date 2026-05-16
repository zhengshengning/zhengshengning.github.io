---
title: '易控智驾 AI Infra 二面'
date: 2026-04-16 12:20:33
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 如何设计一个推理任务调度器？

推理调度器是高性能推理服务的核心组件，需要在延迟、吞吐、公平性之间取得平衡：

**核心模块设计**：

**1. 请求队列管理**：
- 维护多级优先级队列（如VIP用户 > 普通用户 > 后台任务）。
- 支持到达时间排序（FIFO）和SLA deadline排序（EDF, Earliest Deadline First）。
- 请求元数据：输入长度、预估输出长度、优先级、到达时间、deadline。

**2. 批处理策略（核心）**：
- **Continuous Batching**：每次decode迭代后检查——完成的请求移出，等待队列中的新请求填入。消除padding浪费。
- **Prefill-Decode混合调度**：权衡在当前batch中插入prefill请求的时机——prefill计算密集可能增加decode延迟。策略：设置prefill与decode的时间预算比例。
- **动态batch size**：根据当前显存使用情况动态调整batch上限。

**3. 资源分配与跟踪**：
- 维护GPU显存使用状态（主要是KV Cache block占用情况）。
- 根据请求的输入长度+预估输出长度，预估所需KV Cache block数。
- 显存不足时触发排队或抢占机制。

**4. 抢占机制（Preemption）**：
- 当高优先级请求到来但显存不足时，需要抢占低优先级请求。
- **Swap策略**：将被抢占请求的KV Cache异步拷贝到CPU内存，恢复时拷回。
- **Recompute策略**：直接丢弃被抢占请求的KV Cache，恢复时从头重新计算prefill。
- 选择依据：KV Cache大小 vs prefill重计算成本。

**5. 负载均衡（多实例场景）**：
- 按当前各实例的请求数/显存使用率/预估等待时间分发。
- 考虑请求亲和性：相同prefix的请求路由到同一实例（利用Prefix Cache）。
- 健康检查和故障转移。

**6. SLA保障**：
- 实时监控每个请求的已等待时间和预估完成时间。
- 即将超时的请求提升优先级（deadline-aware scheduling）。
- 提供降级策略：接近超时时截断生成输出。

**关键性能指标**：
- 调度延迟：<100us（不能成为瓶颈，通常用C++实现调度器核心逻辑）。
- 吞吐量：在满足P99延迟约束下最大化tokens/s。
- 尾延迟：P99 TTFT和P99 TPOT满足SLA。

---

### Q: NPU的计算过程是怎样的？

以华为昇腾达芬奇架构（Ascend 910B/310）为例，详解NPU的计算流程：

**硬件架构（AI Core）**：

```
┌──────────────────────────────────────┐
│            AI Core                    │
│  ┌─────────┐  ┌────────┐  ┌───────┐ │
│  │  Scalar │  │ Vector │  │  Cube │ │
│  │  单元    │  │  单元   │  │  单元  │ │
│  └─────────┘  └────────┘  └───────┘ │
│        ↕           ↕          ↕       │
│  ┌────────────────────────────────┐  │
│  │     L1 Buffer (Local Memory)   │  │
│  └────────────────────────────────┘  │
│        ↕  MTE (Memory Transfer Engine)│
│  ┌────────────────────────────────┐  │
│  │          L2 Cache              │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
              ↕ HBM
```

**各计算单元职责**：
- **Cube单元**：执行矩阵乘法（16×16 matmul），支持FP16/INT8，类似NVIDIA的Tensor Core。专门处理GEMM密集计算。
- **Vector单元**：执行逐元素向量运算——激活函数（SiLU/GeLU）、归一化（LayerNorm）、逐元素加减乘除。类似CUDA Core的向量操作。
- **Scalar单元**：执行标量运算、地址计算、流程控制（循环/分支），相当于控制处理器。

**计算流水线（三级流水）**：
```
Stage 1 (搬入): MTE从HBM搬运数据到L1 Buffer (DMA操作，不占计算资源)
Stage 2 (计算): Cube/Vector在L1 Buffer上计算
Stage 3 (搬出): MTE将结果从L1 Buffer写回HBM
```

三个stage可以overlap执行——当前batch计算时，下一batch的数据已在搬入，上一batch的结果在搬出。

**编程模型（Ascend C）**：
```cpp
// 类似CUDA的显式内存管理，但数据搬运是核心
__aicore__ void kernel() {
    // 1. 声明本地buffer
    LocalTensor<float16> inputLocal;
    // 2. 数据搬入 (类似CUDA的shared memory加载)
    DataCopy(inputLocal, globalInput, dataSize);
    // 3. 计算
    Cube(outputLocal, inputA, inputB);  // 矩阵乘
    Adds(outputLocal, inputLocal, bias);  // 向量加
    // 4. 数据搬出
    DataCopy(globalOutput, outputLocal, dataSize);
}
```

**与NVIDIA GPU的核心区别**：
- GPU有cache自动管理（L1/L2可自动缓存），NPU需要程序员**完全显式管理**数据在L1 Buffer和HBM之间的搬运。
- GPU的并行粒度是Warp（32线程SIMT），NPU的并行粒度是向量/矩阵操作。
- NPU的编程思维更接近"数据流"——先规划好数据搬运路径，再安排计算。

---

### Q: CUDA有哪些常见优化方法？

CUDA优化方法按瓶颈类型分类，需要先通过Profiling确定是memory-bound还是compute-bound：

**1. 内存优化（Memory-bound kernel的首选）**：
- **合并访存（Coalesced Access）**：warp内线程访问连续128字节地址，单次事务完成。Stride访问或随机访问会导致多次事务，带宽浪费严重。
- **共享内存**：将数据从HBM(2TB/s)加载到Shared Memory(19TB/s)进行多次复用。典型场景：GEMM的tile、reduce的中间结果。
- **向量化读写（float4）**：128位单次传输，减少4倍事务数，要求16字节对齐。
- **避免Bank Conflict**：shared memory 32个bank，同warp线程访问同bank不同地址会串行。解决方法：padding。

**2. 计算优化（Compute-bound kernel的首选）**：
- **减少Warp Divergence**：同warp线程走不同分支会串行执行。优化：数据重排使相邻线程条件相同、用位运算替代分支。
- **Tensor Core利用**：FP16/BF16/INT8矩阵乘加速8-16倍。通过WMMA API或cuBLAS自动使用。
- **指令级并行（ILP）**：循环展开让独立指令可以在不同功能单元上并行执行。

**3. 并行度优化**：
- **提高Occupancy**：让更多warp同时驻留SM以隐藏延迟。权衡：有时降低occupancy但增加每线程可用资源（寄存器）反而更快。
- **合理的Block/Grid设置**：Block为32倍数，Grid中block数为SM数（如A100有108个SM）的整数倍。

**4. 延迟隐藏**：
- **Double Buffering**：计算buffer A数据时异步加载数据到buffer B（cp.async指令），无等待切换。
- **多Stream并行**：不同stream的kernel和memcpy可以并行执行（如通信与计算overlap）。

**5. 算子融合**：
- 将多个小kernel合并为一个：减少kernel launch开销（~5us/次）+ 消除中间tensor的HBM读写。
- 典型：LayerNorm+Add融合、QKV投影融合、GEMM+Bias+Activation epilogue融合。

**6. 异步执行**：
- CUDA Stream：不同stream的操作可以并行（H2D拷贝 + Kernel执行 + D2H拷贝）。
- cp.async：Ampere+架构的异步内存拷贝，DMA搬运不占用计算流水线。

---

### Q: 自动驾驶场景对高性能计算有哪些特殊要求？

自动驾驶的HPC需求与数据中心AI推理有本质区别，核心在于**实时性、确定性、可靠性**三大硬约束：

**1. 严格的实时性约束**：
- 感知pipeline（目标检测+语义分割+3D重建）：端到端 **<100ms**。
- 预测模块（轨迹预测）：**<50ms**。
- 规划决策：**<30ms**。
- 超时不是性能问题——是**安全问题**。100ms延迟在120km/h车速下对应3.3m的反应距离。

**2. 确定性（Deterministic）**：
- 运行时间抖动必须极小，**最坏情况时间（WCET）**必须可预测。
- GPU的非确定性来源：warp调度随机性、cache行为、动态频率调节。
- 解决：固定GPU频率（禁用DVFS）、预分配内存、避免动态行为。
- 不使用CUDA的动态并行（dynamic parallelism）、避免runtime compilation。

**3. 功耗约束**：
- 车载平台总功耗有限：NVIDIA Orin 60W TDP，整个域控制器通常<200W。
- 对比：数据中心H100 TDP 700W。
- 结果：算法选择必须考虑FLOPs/Watt（能效比），不能简单堆算力。
- 优化：INT8/INT4量化、模型剪枝、低功耗模式（感知到低风险场景时降频）。

**4. 多任务并行与资源隔离**：
- 同时运行：6-12路摄像头感知 + LiDAR点云处理 + 多传感器融合 + 预测 + 规划 + 建图。
- 要求：任务间资源隔离（一个任务不能饿死其他任务）、优先级调度（制动控制 > 地图更新）。
- 技术：NVIDIA MPS/MIG（Multi-Instance GPU）、CUDA Stream优先级、时间片轮转。

**5. 异构计算协同**：
- CPU：控制逻辑、数据预处理、不规则计算。
- GPU：DNN推理、大规模并行计算。
- NPU/DLA：固定模型的高能效推理。
- DSP：信号处理（雷达数据）。
- 挑战：数据在异构设备间搬运的延迟和同步。

**6. 可靠性与冗余**：
- 单点故障不可接受——需要冗余计算路径（如双GPU互相校验）。
- ECC内存保护、锁步执行（Lockstep）模式。
- 符合ISO 26262功能安全标准（ASIL-B/D级别）。
