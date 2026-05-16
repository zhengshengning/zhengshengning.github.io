---
title: '蔚来 AI Infra 实习 二面'
date: 2026-04-16 12:14:53
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: CUDA优化的常用方法有哪些？

CUDA优化是一个系统工程，需要从**访存、计算、并行度**三个维度综合优化。优化前必须先通过Profiling确定瓶颈类型。

**1. 访存优化（大多数kernel的首要瓶颈）：**

| 技术 | 原理 | 收益 |
|------|------|------|
| 合并访问(Coalesced Access) | 连续线程访问连续地址，合并为一次128B事务 | 最高32倍(vs完全分散) |
| 向量化加载(float4/int4) | 一条指令加载128bit | 减少load指令数，提高带宽利用2-4x |
| 共享内存缓存 | 热数据加载到SRAM，多次复用 | 取决于复用次数（GEMM中可达tile_size倍） |
| 避免Bank Conflict | Padding/Swizzle错开bank映射 | 消除N-way串行化 |
| 减少全局内存访问 | 算子融合、寄存器缓存中间结果 | 与融合的算子数成比例 |

**2. 计算优化：**
- **Tensor Core**：使用WMMA/MMA指令，FP16矩阵乘吞吐比CUDA Core高数倍
- **循环展开**：`#pragma unroll`消除循环开销，提高ILP（指令级并行）
- **减少分支发散**：同一warp内不同线程走不同分支会串行化
- **使用快速数学函数**：`__expf()`替代`expf()`（精度略低但更快）
- **Warp Shuffle**：warp内数据交换不需要共享内存（~1 cycle vs ~28 cycles）

**3. 并行度优化：**
- **选择合适Block Size**：通常128/256/512，通过occupancy calculator确定
- **提高Occupancy**：更多active warp → 更好的延迟隐藏（但不是越高越好）
- **减少同步点**：去掉不必要的`__syncthreads()`
- **双缓冲/异步拷贝**：`cp.async`异步加载下一轮数据与计算重叠

**4. 系统级优化：**
- **CUDA Graph**：录制kernel序列一次提交，消除CPU端反复launch开销
- **算子融合**：减少kernel数量和中间tensor的HBM读写
- **计算通信Overlap**：NCCL通信与kernel计算同时进行
- **多Stream并发**：独立kernel分配到不同Stream并行执行

**优化决策流程：**
```
Profile(Nsight Compute) → 判断瓶颈类型
├── Memory-bound: 优化访存模式 → 再次profile
├── Compute-bound: 利用特殊硬件/提高ILP → 再次profile  
└── Latency-bound: 增大并行度 → 再次profile
```

### Q: 手撕：岛屿数量（DFS/BFS解法）？

（编程题）
