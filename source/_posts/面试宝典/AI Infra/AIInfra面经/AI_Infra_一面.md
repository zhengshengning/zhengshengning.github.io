---
title: 'AI Infra 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试,综合面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 当训练推理卡规模倍增时，最容易产生瓶颈的位置是什么？原因和优化方案？

**瓶颈位置**：网络通信（卡间/机间通信）。

**原因**：卡数增加使得集合通信量线性增长（如AllReduce），而网络带宽不会线性扩展；机间通信延迟远高于机内NVLink；通信拓扑复杂度增加导致调度困难；故障概率随规模线性增长。

**优化方案**：
- 通信拓扑优化：NVLink/NVSwitch（机内）+ RDMA/InfiniBand（机间）。
- 通信与计算重叠（overlap）：梯度分桶异步AllReduce。
- 减少通信量：梯度压缩、混合精度通信、ZeRO分片策略。
- 并行策略优化：TP放机内（通信密集）、PP/DP跨机（通信稀疏）。
- 容错机制：checkpoint、弹性训练。

---

### Q: 请介绍Roofline模型，如何判断性能已达到计算瓶颈？

Roofline模型以算术强度（Arithmetic Intensity = FLOPs/Bytes）为x轴，可达性能（FLOPS）为y轴，画出一条由内存带宽和计算能力决定的"屋顶线"。

- 屋顶线拐点 = 峰值算力 / 峰值带宽（单位：FLOPs/Byte）。
- 若算子的算术强度 < 拐点，则受带宽限制（memory-bound）。
- 若算子的算术强度 > 拐点，则受计算限制（compute-bound）。

判断是否达到计算瓶颈：实际性能接近峰值算力（如达到80%+），且进一步优化访存无法带来性能提升。

---

### Q: C++中数组越界写导致其他数据结构被写坏，保留了coredump文件，如何排查？

1. 用`gdb`加载coredump：`gdb <binary> <core>`，查看崩溃调用栈（bt）。
2. 检查被损坏的变量/数据结构的地址，对比正常地址范围。
3. 使用`watchpoint`（如果可复现）监控被写坏的地址。
4. 检查栈帧中数组变量的大小和实际访问索引。
5. 使用AddressSanitizer（-fsanitize=address）复现，可精确定位越界写的位置。
6. 使用Valgrind的memcheck工具检测内存错误。
7. 审查coredump中堆栈相邻变量的内存布局，判断越界方向和偏移量。

---

### Q: 请介绍Flash Attention？

Flash Attention通过IO感知（IO-aware）的方式实现精确注意力计算，核心思想是避免将完整的N*N注意力矩阵写入HBM。

**关键技术**：
- **Tiling**：将Q、K、V分块加载到SRAM（共享内存）中计算，避免HBM上的大矩阵。
- **Online Softmax**：分块计算softmax，通过维护running max和running sum实现数值等价。
- **重计算（Recomputation）**：反向传播时不存储注意力矩阵，而是重新计算，用计算换显存。

效果：将注意力计算的IO复杂度从O(N²)降至O(N²d/M)（M为SRAM大小），显存从O(N²)降至O(N)。

---

### Q: GEMM一定是计算瓶颈算子吗？如果要优化它，思路是什么？

**不一定是计算瓶颈**。当矩阵规模小、batch小时，GEMM可能变为访存密集型（如decode阶段的小batch GEMV）。算术强度随矩阵维度变化。

**优化思路**：
- **Tiling策略**：设计合理的tile大小适配SM的共享内存和寄存器。
- **数据复用**：通过共享内存tiling提高数据复用率。
- **向量化访存**：使用128-bit load/store（float4/LDS.128）。
- **流水线**：双缓冲（double buffer）实现访存与计算重叠。
- **利用Tensor Core**：使用wmma/mma指令利用硬件矩阵计算单元。
- **减少Bank Conflict**：共享内存padding或swizzle。

---

### Q: 性能优化的定位和瓶颈检测有什么方法？

- **Profiling工具**：NVIDIA Nsight Compute（kernel级分析）、Nsight Systems（系统级时间线）。
- **关键指标**：SM利用率、内存带宽利用率、Warp Stall原因分析。
- **Roofline分析**：判断是compute-bound还是memory-bound。
- **Timeline分析**：检查kernel间空隙、CPU-GPU同步等待、通信开销。
- **对比理论峰值**：实际FLOPS vs 峰值FLOPS，实际带宽 vs 峰值带宽。

---

### Q: 手撕：手写包含GQA的Attention模块实现？

（编程题）
