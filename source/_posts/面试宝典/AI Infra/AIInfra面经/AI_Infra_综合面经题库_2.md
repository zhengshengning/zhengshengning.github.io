---
title: 'AI Infra 综合面经题库 (2)'
date: 2026-04-16 12:07:08
categories:
  - [求职面试,综合面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: 手撕：快速排序？

（编程题）

---

### Q: 手撕：寻找两个正序数组的中位数？

（编程题）

---

### Q: 手撕：下一个排列？

（编程题）

---

### Q: 手撕：二叉树中的最大路径和？

（编程题）

---

### Q: 手撕：Path Sum III？

（编程题）

---

### Q: 手撕：给定若干点的数轴坐标数组和固定数量的等长线段，求线段最短长度覆盖所有点？

（编程题）

---

### Q: 手撕：前K个高频字符串（词频一样时按字典序升序）？

（编程题）

---

### Q: 手撕：字符串右旋拼接后求第N个位置的字符？

（编程题）

---

### Q: 手撕：两根手指在26字母键盘上敲出字符串的最少移动距离？

（编程题）

---

### Q: 手撕：用PyTorch实现Multi-Head Attention？

（编程题）

---

### Q: 手撕：实现Flash Attention v1？

（编程题）

---

### Q: 手撕：Flow Matching Model采样的伪代码？

（编程题）

---

### Q: Flow Matching模型预测的是什么？怎么理解conditional velocity？

Flow Matching模型预测的是从噪声到数据的速度场（velocity field）`v(x_t, t)`，即在时间t时数据点应该移动的方向和速度。

Conditional velocity（conditioned on data sample x0）是指在已知目标数据点x0的条件下，连接噪声x1到x0的最优传输路径上的瞬时速度。训练时对每个样本x0，条件速度是确定性的直线路径：`v_t(x|x0) = x0 - x1`（对于线性插值路径）。模型学习的是对所有x0的边际速度场的回归。

---

### Q: 如何计算Qwen-VL中的time shift？

Qwen-VL中time shift用于多模态位置编码，将视觉token的时间位置信息编码到旋转位置编码（RoPE）中。对于视频输入，每帧的token分配相同的temporal position id，不同帧之间按帧序号递增。图片视为单帧（temporal_id=0）。通过将temporal维度信息融入position_ids，使模型感知视频时序关系。

---

### Q: 介绍Flash Attention的原理和实现思路？

**核心原理**：通过tiling将注意力计算分块，在SRAM（共享内存）中完成计算，避免在HBM上存储N*N的注意力矩阵。

**实现思路**：
1. 将Q分成块，外层遍历Q块。
2. 内层遍历KV块，每次加载一块K和V到共享内存。
3. 计算局部attention score，通过online softmax维护running max和sum实现数值稳定的分块softmax。
4. 累积加权V得到输出，只需O(N)显存。

---

### Q: GPU matrix transpose使用shared memory有什么好处？

全局内存按行存储，直接转置时写操作会变成非合并访问（strided write）。使用shared memory的好处：先将一个tile按合并方式从全局内存读入shared memory，再从shared memory按转置方式合并写回全局内存。这样读写全局内存都是合并访问（coalesced），充分利用带宽。代价是需要注意shared memory的bank conflict（通过padding解决）。

---

### Q: CPU按列遍历行优先矩阵为什么比按行遍历慢？具体是哪个性能指标变差？

按行遍历利用空间局部性，连续元素在同一Cache Line内，命中率高。按列遍历跨行访问（stride = 列数 * 元素大小），每次访问可能加载新的Cache Line但只用其中一个元素。

具体变差的指标是**Cache Miss Rate**（L1/L2 cache miss rate显著上升），导致实际内存带宽利用率低、访存延迟增大。对于大矩阵，还可能导致TLB miss增加。

---

### Q: Weight-only量化有哪些？实现weight-only量化CUDA kernel时如何优化访存？是否了解Marlin kernel？

**Weight-only量化**：GPTQ（逐列/逐组量化，Hessian感知）、AWQ（激活感知权重量化，保护重要通道）、SqueezeLLM。

**CUDA kernel访存优化**：
- 向量化加载：用int4/int32一次加载多个低精度权重，减少内存事务数。
- 权重紧密打包：4-bit权重按8个一组打包为int32存储。
- 反量化与GEMM融合：加载后立即反量化并参与计算，避免中间存储。

**Marlin kernel**：DeepSpeed的高效4-bit量化GEMM kernel，通过精心设计的内存布局（将权重按适合Tensor Core的格式重排）、异步拷贝流水线、分组反量化等技术，在A100上接近FP16 GEMM的80%+性能。

---

### Q: Megatron SP（Sequence Parallelism）的实现方式？

Megatron SP在Tensor Parallelism基础上，将TP不切分的部分（如LayerNorm、Dropout）沿序列维度切分到不同GPU上。

实现方式：TP中AllReduce分解为Reduce-Scatter + All-Gather。在TP区域的输入输出处，分别做all-gather（收集完整序列给TP切分的线性层）和reduce-scatter（将TP结果归约并沿序列维度散发）。非TP区域（LN、Dropout）每个GPU只处理局部序列片段，减少激活显存占用约TP倍。

---

### Q: DeepSpeed ZeRO Stage 1和Stage 2的通信量区别？论文和代码实现有无gap？

- **Stage 1**：只切分优化器状态。通信量同标准数据并行，每步一次AllReduce梯度（2倍模型大小）。
- **Stage 2**：切分优化器状态 + 梯度。使用Reduce-Scatter替代AllReduce收集梯度，通信量与Stage 1相同（Reduce-Scatter通信量 = AllReduce的一半，但Stage 2每个rank只需要自己负责参数的梯度）。

**论文vs代码的gap**：论文声称通信量不增加，但实际实现中Stage 2可能额外需要一次All-Gather来广播更新后的参数（因为每个rank只更新部分参数），总通信量仍为2倍模型大小，与DDP相同。

---

### Q: 多GPU通信时NVSHMEM和NVLink的区别？

- **NVLink**：硬件互联技术，提供GPU间高带宽低延迟的物理连接通道（如NVLink 4.0每方向100GB/s）。是通信的物理层。
- **NVSHMEM**：编程模型/通信库，基于OpenSHMEM标准，提供GPU间直接内存访问（one-sided通信）。kernel内线程可直接put/get远端GPU内存，无需退回host。可运行在NVLink或PCIe/网络上。

NVSHMEM适合细粒度、不规则的GPU间通信（如稀疏AllToAll），而NCCL更适合批量集合通信。
