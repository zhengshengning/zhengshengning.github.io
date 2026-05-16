---
title: '快手 AI Infra 校招 一面'
date: 2026-04-16 12:08:24
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 如何通过 Profiling 定位性能瓶颈？

Profiling 是性能优化的第一步，"不测量就不优化"。需要从系统级到 kernel 级逐步深入：

**工具链**：
- **Nsight Systems（nsys）**：系统级时间线分析，看全局 kernel 调度、CPU-GPU 交互、通信时间
- **Nsight Compute（NCU/ncu）**：单 kernel 深度分析，提供 Roofline、SOL、warp stall 原因等

**分析流程**：

**Step 1：Nsight Systems 全局扫描**
- 看时间线中的热点 kernel（占比最大的 kernel）
- 看 kernel 间的 gap（是否有 CPU-GPU 同步等待、kernel launch 开销）
- 看通信与计算的重叠程度（DP 场景）

**Step 2：NCU 深入分析热点 kernel**

关键指标：

| 指标 | 含义 | 判断标准 |
|---|---|---|
| SM Throughput SOL | 实际计算 vs 理论峰值 | >80% 说明 compute-bound |
| Memory Throughput SOL | 实际带宽 vs 硬件峰值 | >80% 说明 memory-bound |
| Warp Stall - Memory | 因等待内存而 stall 的比例 | 高 → memory latency 瓶颈 |
| Warp Stall - Execution | 因执行依赖而 stall | 高 → 指令级并行不足 |
| Occupancy | 活跃 warp 数 / SM 最大 warp 数 | 低 → 可能寄存器/shared memory 用太多 |
| Tensor Core Utilization | Tensor Core 使用率 | 矩阵乘应该接近 100% |

**Step 3：Roofline 分析**
- NCU 直接提供 Roofline 图
- 算子落在 memory-bound 区域 → 优化访存（合并、向量化、tiling）
- 算子落在 compute-bound 区域 → 优化计算（Tensor Core、减少冗余）
- 算子远离 roofline → 实现效率低，有很大优化空间

**PP 训练特有指标**：
- 流水线气泡率 = (P-1)/(P-1+M)，其中 P=stage 数，M=micro-batch 数
- 气泡率应 <10%（如 P=8, M≥64）

---

### Q: Matmul 分块策略？

矩阵乘 C[M,N] = A[M,K] × B[K,N] 的高效 GPU 实现需要多级分块：

**三级 Tiling 架构**：

```
Grid Level:   每个 Block 负责 C 的 BM×BN tile
Block Level:  Block 内 Warp 划分，每个 Warp 负责 WM×WN
Thread Level: 每个线程负责 TM×TN 个输出元素（存在寄存器中）
```

**Block Tiling（Global → Shared Memory）**：
- 每个 thread block 负责计算 C 的一个 BM×BN 的 tile
- 沿 K 维循环，每次加载 A[BM×BK] 和 B[BK×BN] 到 shared memory
- BM, BN, BK 的选择决定了数据复用率和 shared memory 使用量

**Thread Tiling（Shared → Register）**：
- 每个线程从 shared memory 读取 TM 个 A 元素和 TN 个 B 元素
- 在寄存器中计算 TM×TN 个输出元素的部分积
- 寄存器带宽无限 → 计算访存比最高

**K 维分块**：
- 外层循环遍历 K 维的 ceil(K/BK) 个 tile
- 每个 tile：load → sync → compute → sync → 下一个 tile
- 双缓冲：load tile[i+1] 的同时 compute tile[i]

**典型参数选择**：
```
BM = BN = 128 (block tile)
BK = 8-32 (K维步长，越大数据复用越好但shared memory用量越大)
TM = TN = 4-8 (thread tile)

Shared Memory = (BM×BK + BK×BN) × sizeof(half) × 2(双缓冲)
例：(128×32 + 32×128) × 2 × 2 = 32 KB/block
```

**分块大小选择的权衡**：
- Tile 越大 → 数据复用率越高（每个 A/B 元素参与更多乘法）
- Tile 越大 → Shared Memory 用量越大 → 每 SM 能驻留的 block 越少（occupancy 下降）
- 最优点通常在 occupancy 50-75% 时达到（通过 benchmark 确定）

---

### Q: 手撕：n! 中尾随 0 的个数？

（编程题）

---

### Q: 手撕：买卖股票的最大利润？

（编程题）
