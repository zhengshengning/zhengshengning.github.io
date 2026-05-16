---
title: '阿里巴巴 云 AI Infra 实习 二面 (1)'
date: 2026-04-16 12:18:43
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 推理/训练中遇到显存异常和长稳问题如何排查？

**排查思路**：
1. 拉日志对照计算图逐条定位链路。
2. 检查图优化后的数据流——shape tensor是否被错误处理（如host tensor被当作device tensor传递）。
3. 定位是否有host/device内存语义混乱（读取到错误地址导致维度异常）。

**修复方式**：在图优化链路中显式约束shape信息的内存路径（确保host memory），避免优化pass将shape tensor错误地放到device memory上。

### Q: 算子性能优化的工程思路是什么？

以一个加法类算子为例，核心不在算子本身复杂度，而在工程角度：
- **Buffer复用**：检查输入buffer的生命周期和依赖关系，若无其他依赖可直接复用输入作为输出buffer，省去malloc/free。
- **减少内存管理开销**：省去额外内存分配和管理的时间。
- 优化效果例：21us → 12us（43%提升），极限约6-7us（受shape规整程度和调度开销影响）。

### Q: AWQ量化（W4A16）的实现细节？

**实现要点**：
- 权重量化为INT4，激活保持FP16。
- Linear层需改造：存储INT4权重+scale+zero_point。
- 执行时反量化INT4→FP16后做矩阵乘。
- **访存优化**：向量化加载（一次读多个INT4权重打包的int32），减少内存事务。
- **正确性验证**：与非量化版本对比，确认精度在可接受范围（perplexity增加<0.5）。

### Q: 多卡协同和卡间通信的技术栈？

- **NVLink**：机内GPU间高带宽互联（如NVLink 4.0 900GB/s双向），适合TP等通信密集场景。
- **NVSwitch**：全互联交换机，任意GPU间等带宽通信。
- **RDMA/InfiniBand**：机间高速网络（400Gbps+），绕过CPU直接访问远端内存。
- **NCCL**：NVIDIA集合通信库，自动选择最优通信路径。
- **层次化设计**：机内NVLink做TP（频繁通信），机间IB做DP/PP（通信量少）。

### Q: 算子融合和图优化有哪些工作？

**算子融合**：将连续的memory-bound算子合并为一个kernel（如bias+activation+residual add），减少中间结果的全局内存读写。GEMM epilogue融合后处理操作。

**图优化**：常量折叠、死代码消除、布局转换、内存规划（buffer复用/生命周期分析）、子图替换（pattern matching将低效子图替换为高效实现）。

### Q: 如何看待用AI Agent辅助Infra开发？

Agent可以辅助：代码审查（发现常见错误模式）、性能优化建议（根据profile结果推荐优化策略）、Boilerplate代码生成（如CUDA kernel模板）、问题排查（分析日志定位异常）。但核心架构决策、极致性能优化、底层微架构利用仍需人类专家。Agent更适合提效而非替代。
