---
title: 'AI Infra 综合面经题库 (7)'
date: 2026-04-16 11:46:30
categories:
  - [求职面试,综合面经]
tags: [AIInfra, 推理优化, 训练优化, 面经]
---


<!-- more -->

---

### Q: 大模型训练和推理如何加速？

**训练加速**：混合精度训练（FP16/BF16）、数据并行+模型并行+流水线并行、梯度累积减少通信频率、ZeRO优化器状态分片、Flash Attention、编译优化（torch.compile）、通信与计算重叠。

**推理加速**：KV Cache、量化（INT8/INT4）、Flash Attention/Flash Decoding、Speculative Decoding、Continuous Batching、算子融合、CUDA Graph减少launch开销、PagedAttention。

### Q: 是否了解DeepSpeed？

DeepSpeed是微软开发的分布式训练优化库，核心功能：
- **ZeRO**（Zero Redundancy Optimizer）：Stage 1/2/3分别切分优化器状态/梯度/参数。
- **ZeRO-Offload/Infinity**：将数据卸载到CPU/NVMe。
- **混合精度训练**：FP16/BF16训练支持。
- **Pipeline Parallelism**：流水线并行实现。
- **DeepSpeed-Chat**：RLHF训练框架。
- **通信优化**：梯度压缩、分桶AllReduce。

### Q: 对多机多卡分布式训练的了解？

分布式训练主要并行策略：
- **数据并行（DP/DDP）**：每卡持有完整模型，数据分片，梯度AllReduce同步。
- **模型并行（TP）**：将层内参数切分到多卡，计算后AllReduce/AllGather。
- **流水线并行（PP）**：将模型层间切分到多卡，micro-batch流水线执行。
- **专家并行（EP）**：MoE模型中不同专家分布在不同卡上。
- **序列并行（SP）**：长序列切分到多卡处理。

通常组合使用，机内TP+SP（通信密集用NVLink），机间PP+DP（通信少用IB）。

### Q: 怎么提高训练效率（时间和资源角度）？

**时间角度**：混合精度减少计算量、增大batch size减少迭代次数、通信与计算重叠、减少数据加载瓶颈（prefetch、多worker）、使用更高效的优化器。

**资源角度**：ZeRO减少显存冗余、梯度checkpointing以时间换空间、offload到CPU/NVMe、模型压缩/蒸馏用更小模型达到效果、选择合适的并行策略最大化硬件利用。

### Q: 训练速度很慢时该从哪些方面排查？

1. **GPU利用率**：`nvidia-smi`检查GPU利用率是否饱满。
2. **数据加载**：是否IO瓶颈（CPU预处理慢、磁盘读取慢），增加DataLoader workers、使用prefetch。
3. **通信瓶颈**：Nsight Systems查看通信占比，考虑通信overlap或换更快互联。
4. **显存不足频繁swap**：检查是否触发了OOM保护或swap。
5. **同步等待**：多卡负载不均导致等待stragglers。
6. **batch size太小**：GPU计算未饱和。
7. **编译优化**：是否使用了torch.compile/CUDA Graph。

### Q: 多机多卡训练中的通信问题有哪些？

- **通信带宽瓶颈**：机间带宽远低于机内NVLink，梯度同步成为瓶颈。
- **通信延迟**：大规模集群AllReduce延迟累积。
- **负载不均**：不同节点计算速度差异导致同步等待。
- **网络抖动**：RDMA/TCP网络偶发延迟导致timeout。
- **拓扑感知**：未充分利用网络拓扑导致链路拥塞。

解决：梯度累积减少通信频率、异步通信overlap、梯度压缩、层级AllReduce、拓扑感知调度。

### Q: 梯度累积的原理是什么？

梯度累积将一个大batch拆分为多个micro-batch，每个micro-batch前向反向后梯度累加到梯度buffer中，不立即更新参数。累积N个micro-batch后，用累积的梯度做一次参数更新。等效于使用N倍大的batch size，但峰值显存只需一个micro-batch的量。通信也只在参数更新时进行一次AllReduce，频率降为1/N。

### Q: 了解哪些模型的量化方法和加速方法？

**量化方法**：
- PTQ：GPTQ（权重逐层量化，Hessian补偿）、AWQ（保护重要通道）、SmoothQuant（平衡权重和激活的量化难度）。
- QAT：量化感知训练，精度最好但成本高。

**加速方法**：Flash Attention、CUDA Graph、Speculative Decoding、Continuous Batching、PagedAttention、算子融合、TensorRT/vLLM推理引擎。

### Q: 训练时loss震荡可能是什么原因？

- **学习率过大**：参数更新幅度过大导致在最优点附近震荡。
- **batch size太小**：梯度估计方差大，导致更新方向不稳定。
- **数据噪声**：训练数据质量差或标签有误。
- **数值不稳定**：梯度爆炸/消失，特别是FP16训练中。
- **数据分布不均**：不同类别数据比例失调。
- **学习率调度问题**：warmup不充分或衰减策略不当。
- **模型架构问题**：缺少残差连接或归一化层。
