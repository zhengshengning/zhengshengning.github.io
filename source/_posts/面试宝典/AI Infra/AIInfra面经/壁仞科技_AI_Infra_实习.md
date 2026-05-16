---
title: '壁仞科技 AI Infra 实习'
date: 2026-04-16 12:19:48
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: GEMM中如果K值远大于M或N，会对计算产生什么影响？

GEMM 计算 C[M,N] = A[M,K] * B[K,N]，当 K >> M 或 K >> N 时：

**1. 寄存器压力增大**：
- 每个线程的 M_tile * N_tile 累加器需要持续存在于寄存器中，K 维度的循环次数增多意味着累加器需要更长时间占用寄存器
- 如果 M_tile * N_tile 过大导致寄存器溢出到 Local Memory，性能剧降（~100x 慢）

**2. 数值精度问题**：
- FP16 累加时，K 次加法的舍入误差累积可能导致精度严重下降
- 解决：使用 FP32 累加器（Tensor Core 天然支持 FP16 输入 FP32 累加）
- 极端情况（K>10000）可能需要分段累加后合并

**3. 数据复用率分析**：
- 算术强度 = 2*M*N*K / (M*K + K*N + M*N) * sizeof(dtype)
- 当 K >> M,N 时，算术强度 ≈ 2*M*N / (M+N)，与 K 无关
- 但每个 K-tile 迭代的 tile 大小受 shared memory 限制

**4. 解决方案——Split-K GEMM**：
- 将 K 维度切分为多段，每段独立计算部分矩阵乘
- 各段的部分结果最后通过 reduction 合并
- 好处：增加并行度（更多 block），解决 M*N 很小时 block 数不足的问题
- 代价：额外的 reduction kernel 和全局内存写入

---

### Q: CUDA调试工具有哪些？

GPU 程序调试的工具链（按场景分类）：

1. **cuda-gdb**：NVIDIA 的 GPU 调试器。支持在 kernel 中设置断点、单步执行、查看线程状态、检查 shared/global memory 值。使用 `-G -g` 编译开启调试信息。注意：调试模式下 kernel 性能大幅下降且 register 分配不同。

2. **compute-sanitizer**（替代旧版 cuda-memcheck）：
   - `--tool memcheck`：检测全局/共享内存越界访问、未对齐访问
   - `--tool racecheck`：检测 shared memory 上的数据竞争（race condition）
   - `--tool initcheck`：检测未初始化的 GPU 内存读取
   - 运行时检测，有 2-10x 性能开销

3. **Nsight Compute**：不是调试器而是性能分析器，但能帮助理解 kernel 行为。显示每条 warp 的执行统计。

4. **printf 调试**：kernel 内 `printf` 最简单直接。注意：printf 有缓冲区大小限制（默认 1MB），大量输出可能丢失；性能影响大；输出顺序不保证。

5. **assert()**：CUDA kernel 内可使用 `assert(condition)`，条件不满足时中止 kernel 执行并报告位置。编译时加 `-DNDEBUG` 可禁用。
