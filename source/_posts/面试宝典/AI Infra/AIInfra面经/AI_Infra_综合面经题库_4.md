---
title: 'AI Infra 综合面经题库 (4)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试,综合面经]
tags: [AIInfra, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 如果出现NCCL timeout，一般怎么定位和解决？

**定位步骤**：
1. 设置`NCCL_DEBUG=INFO`查看详细日志，确认卡在哪个操作。
2. 检查是否有stragglers（用`NCCL_DEBUG_SUBSYS=COLL`定位慢节点）。
3. 检查网络连通性：`nvidia-smi topo -m`查看拓扑，`ib_write_bw`测试RDMA带宽。
4. 检查GPU状态：`nvidia-smi`看是否有ECC错误、温度降频。
5. 检查是否有死锁：某些rank未参与集合通信。

**解决方案**：增大`NCCL_TIMEOUT`；修复网络问题；替换故障卡；检查代码中rank不一致的分支逻辑。

### Q: RL中MOE模型的优化有哪些？

- **专家负载均衡**：RL训练中策略变化大导致路由不稳定，需要动态调节auxiliary loss或使用无辅助损失的bias方法。
- **通信优化**：MoE的AllToAll通信量大，可通过专家并行分组、减少activated experts数量来优化。
- **显存优化**：非激活专家可offload到CPU/NVMe。
- **计算优化**：Group Gemm合并不同专家的小batch计算。
- **采样效率**：MoE推理快可加速RL中的采样阶段。

### Q: 大模型推理有哪些优化方法？

- **KV Cache**：缓存已计算的Key/Value避免重复计算。
- **量化**：W4A16/W8A8降低访存和计算开销。
- **Continuous Batching**：动态batch提高GPU利用率。
- **Flash Attention/Flash Decoding**：IO优化的注意力计算。
- **Speculative Decoding**：小模型草稿+大模型验证加速生成。
- **算子融合**：减少kernel launch和中间存储。
- **PagedAttention**：虚拟内存管理KV Cache减少碎片。

### Q: 有没有做过kernel级别的优化？比如用CuTe DSL或手写CUDA做算子融合？

kernel级优化常见方式：
- **手写CUDA kernel**：直接编写fused kernel，如fused attention、fused add+layernorm+dropout。
- **CuTe/CUTLASS**：利用其Layout抽象和MMA原语快速开发高性能GEMM和attention kernel。
- **Triton**：Python DSL快速开发自定义kernel。

融合优化的关键：识别连续的memory-bound算子，将它们合并为一个kernel减少全局内存读写次数。

### Q: 做kernel fusion时倾向于用什么方式？

选择取决于场景：
- **简单融合**（elementwise + reduce）：手写CUDA或Triton，开发效率高。
- **GEMM相关融合**：用CUTLASS/CuTe，利用其Epilogue机制将后处理融入GEMM。
- **复杂融合**（如Flash Attention）：手写CUDA + 共享内存管理。
- **图级别自动融合**：TVM/XLA/TensorRT的算子编排优化。

### Q: 有没有做fusion结果性能反而下降的情况？原因是什么？

**可能的原因**：
- **Occupancy下降**：融合后kernel使用更多寄存器/共享内存，导致并发warp数减少，延迟隐藏能力下降。
- **Tile size不适配**：融合后不同操作的最优tile大小不同，妥协导致两者都不高效。
- **失去异步并行**：原本两个独立kernel可以用不同stream并发，融合后变成串行。
- **指令cache压力**：kernel过大导致指令cache miss增加。

### Q: Hopper架构中Warp Specialization是什么？底层怎么实现的？

Warp Specialization是将不同warp分配不同职责（生产者warp负责数据加载，消费者warp负责计算），通过异步屏障（mbarrier）协调。

**底层实现**：
- 生产者warp发起TMA异步加载到共享内存，完成后signal mbarrier。
- 消费者warp wait mbarrier确认数据就绪后开始计算。
- 双/多缓冲实现流水线：消费者处理当前buffer时，生产者加载下一个buffer。
- 通过`setmaxnreg`指令为不同warp group分配不同寄存器数量（计算warp需要更多寄存器）。

### Q: 如何用Agent生成CUDA内核？

- 提供详细的规格说明（输入输出shape、数据类型、性能要求）。
- 迭代优化：先生成正确性版本，再通过profile反馈指导Agent优化。
- 给Agent提供参考实现和硬件规格（SM数量、共享内存大小等）。
- 让Agent生成测试用例验证正确性，对比baseline检查性能。
- 利用Agent做代码review和bug分析。
