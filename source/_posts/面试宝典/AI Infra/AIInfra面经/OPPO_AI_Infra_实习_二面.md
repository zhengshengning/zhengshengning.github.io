---
title: 'OPPO AI Infra 实习 二面'
date: 2026-04-16 12:10:22
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: NHWC vs NCHW数据布局在训练和推理中怎么选？

- **NCHW**：通道优先，适合GPU训练（cuDNN默认优化格式，卷积计算高效）。
- **NHWC**：通道最后，适合推理部署（Tensor Core对NHWC有优化、TensorRT推荐、移动端更高效）。

选择依据：训练通常用NCHW（框架默认）；推理看硬件——NVIDIA Tensor Core/Intel VNNI对NHWC有优化，可能更快。需要实际profile确认。

### Q: 何时应该关闭Shared Memory？

- Bank Conflict严重且难以消除时（如访问模式高度不规则）。
- 数据复用率很低，只读一次就不再使用，此时L2 Cache足够。
- Shared Memory占用过多导致occupancy严重下降（活跃warp太少无法隐藏延迟）。
- 数据量很小已经被L1/L2缓存很好地服务。

此时直接访问全局内存（依赖L2 Cache）可能反而更快。

### Q: 特定Shape导致使用Shared Memory时结果异常如何排查？

1. 检查shared memory是否越界（某些shape导致index计算超出分配大小）。
2. 检查`__syncthreads()`位置是否正确（条件分支中的sync导致死锁）。
3. 检查是否存在Bank Conflict导致的读写冲突（特定shape恰好触发）。
4. 用compute-sanitizer检测共享内存越界和竞争。
5. 对比有无shared memory版本的输出差异，定位具体位置。
6. 打印中间结果验证tile边界处理逻辑。

### Q: Thread/Warp/Block/SM/Grid的映射关系？

- **Thread**：最小执行单元。
- **Warp（32个Thread）**：SM的调度和执行粒度，SIMT方式执行。
- **Block**：多个Warp组成，Block内线程共享Shared Memory和同步。一个Block只能在一个SM上执行。
- **SM**：可同时驻留多个Block（受资源限制），通过切换Warp隐藏延迟。
- **Grid**：所有Block的集合，对应一次kernel launch。Block被硬件调度器分配到各SM。

### Q: 如何确定最优线程数？

- **Occupancy导向**：使用`cudaOccupancyMaxPotentialBlockSize`或Occupancy Calculator，在寄存器/共享内存约束下找最大occupancy的block size。
- **经验法则**：通常128-512线程/block。太少（<64）无法隐藏延迟，太多（>1024）可能因寄存器压力降低occupancy。
- **实际profile**：不同kernel最优值不同，必须实测对比。考虑算法结构（如reduction需2的幂次）。

### Q: CUDA Stream的使用前提是什么？

Stream实现并发的前提条件：
- 不同stream中的操作之间**无数据依赖**（不访问相同内存区域或有正确同步）。
- 使用**pinned memory**（cudaMallocHost）才能实现真正的异步传输。
- GPU硬件支持**并发执行**（有独立的copy engine和compute engine）。
- 避免使用default stream（会触发隐式同步）。

### Q: 算子融合的决策条件？什么场景适合融合？

**适合融合的场景**：
- 连续的memory-bound算子（如bias+activation+dropout），融合后减少中间结果的全局内存读写。
- 生产者-消费者关系且中间数据不被他人使用。
- GEMM + epilogue（bias/activation/residual add）。

**不适合融合的场景**：
- 融合后寄存器/共享内存超限导致occupancy骤降。
- 两个算子最优parallelism配置差异大。
- 中间结果需要被多个后续算子使用。
