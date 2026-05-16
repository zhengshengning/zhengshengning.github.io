---
title: '阿里巴巴 AI Infra (1)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 如果出现NCCL timeout，一般怎么定位和解决？

**定位**：设置`NCCL_DEBUG=INFO`查看通信日志 → 确认卡在哪个rank的哪个操作 → 检查网络（ib_write_bw测带宽、nvidia-smi topo看拓扑） → 检查GPU状态（ECC错误、降频） → 排查代码中是否有rank不一致的分支。

**解决**：增大NCCL_TIMEOUT、修复网络/硬件问题、替换故障卡、确保所有rank参与相同的集合通信。

---

### Q: RL中MoE的优化有哪些做法？

- AllToAll通信优化：分组通信、overlap计算与通信。
- 负载均衡：RL训练中路由不稳定，需动态调节bias或使用Expert Choice。
- 显存优化：非激活专家offload。
- 计算优化：Group Gemm批量处理不同专家的小batch。
- 容量因子调节：RL策略变化大时适当放大专家容量。

---

### Q: 大模型推理有哪些优化方法？

KV Cache、量化（W4A16/W8A8）、Continuous Batching、Flash Attention/Flash Decoding、Speculative Decoding、算子融合、PagedAttention、CUDA Graph、Prefix Caching。

---

### Q: 有没有做过kernel级别的优化？比如用CuTe DSL或手写CUDA做算子融合？

常见方式：手写CUDA实现fused kernel（如fused attention、add+layernorm+dropout）；用CUTLASS/CuTe利用其Layout抽象和Epilogue机制；用Triton快速开发。关键是识别连续memory-bound算子并合并，减少全局内存读写。

---

### Q: 做kernel fusion时倾向于用什么方式？

视场景而定：简单elementwise融合用Triton或手写CUDA；GEMM相关融合用CUTLASS Epilogue；复杂融合（如Flash Attention）手写CUDA + 精细内存管理。图级别自动融合用XLA/TensorRT。

---

### Q: 做fusion性能反而下降的原因是什么？

- Occupancy下降：融合后资源（寄存器/共享内存）使用增加，活跃warp减少。
- Tile size不适配：不同操作的最优切分方式冲突。
- 丧失并行机会：原本可多stream并发的独立kernel融合后串行化。
- 指令cache压力：kernel体积过大。

---

### Q: Hopper架构中Warp Specialization是什么？底层怎么实现？

将不同warp分配不同角色（生产者warp做数据加载，消费者warp做计算），通过mbarrier异步屏障协调。生产者发起TMA异步加载到共享内存并signal mbarrier，消费者wait数据就绪后计算。双缓冲实现流水线，通过setmaxnreg为不同warp group分配不同寄存器数量。

---

### Q: 如何用Agent生成CUDA内核？

提供详细规格（shape、dtype、性能目标）→ 让Agent生成初版 → profile反馈指导迭代优化 → 生成测试验证正确性 → 与baseline对比性能。关键是给Agent足够的硬件上下文和参考实现。

---

### Q: 如果去掉Warp Specialization只保留tile和shared memory优化，大概会损失在哪？

- **流水线效率下降**：无法实现计算与数据加载的overlap，每次计算前必须等待数据加载完成。
- **Tensor Core利用率降低**：计算warp需要分出资源处理地址计算和加载指令。
- **延迟隐藏能力减弱**：所有warp都做相同工作，无法利用生产者warp的空闲计算周期。
- 预计性能损失20-40%（取决于算子的计算/访存比）。

---

### Q: 怎么判断MoE模型是真的学到了分工，还是只是把Dense模型拆开了？

- **专家特化分析**：检查不同专家处理的token是否有语义聚类（如某些专家偏好数学/代码/多语言）。
- **路由熵**：路由分布应适度集中（不是均匀随机），说明有选择性。
- **专家消融**：移除单个专家看对特定任务影响是否不均匀。
- **对比实验**：与相同激活参数的Dense模型对比，MoE若显著更好说明专家分工有效。

---

### Q: RL + MoE中reward把routing学坏（所有token都走某几个expert）怎么处理？

- 添加路由entropy正则化，鼓励多样化路由。
- 使用Expert Choice路由，强制均匀分配。
- 将路由和RL策略解耦：冻结路由网络或用较小学习率更新。
- 限制单步路由变化幅度（类似PPO的clip思想）。
- 使用无辅助损失的bias方法替代可学习路由。
