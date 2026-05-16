---
title: 'AI Infra 综合面经题库 (1)'
date: 2026-04-16 12:06:20
categories:
  - [求职面试,综合面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: 手撕：实现内存池（支持类似new Foo[]和delete[]功能）？

（编程题）

---

### Q: C++如何比较两个float是否相等？

不能直接用`==`比较，因为浮点数存在精度误差。正确做法是判断差值的绝对值是否小于一个极小值（epsilon）：`fabs(a - b) < epsilon`。对于不同量级的数，应使用相对误差：`fabs(a - b) / max(fabs(a), fabs(b)) < epsilon`。也可使用ULP（Units in the Last Place）比较。

---

### Q: 手撕：实现LRU Cache？

（编程题）

---

### Q: 手撕：岛屿个数？

（编程题）

---

### Q: 手撕：二叉树的层序遍历？

（编程题）

---

### Q: 手撕：Hamming Weight？

（编程题）

---

### Q: 手撕：K-Coverage Intervals？

（编程题）

---

### Q: 手撕：用PyTorch实现LoRA Adapter？

（编程题）

---

### Q: 手撕：CUDA实现支持torch broadcast的4D tensor elementwise mul？

（编程题）

---

### Q: 手撕：CUDA计算 A(1,256) * B(256,128) * C(128,256)？

（编程题）

---

### Q: 手撕：CUDA实现Embedding Sparse Feature Pooling？

（编程题）

---

### Q: LLM的知识蒸馏放在预训练阶段做是否合适？

可以但需权衡。预训练蒸馏的优点是学生模型从一开始就学习教师模型的表示能力，数据利用更充分。缺点是：教师模型需要在线推理产生logits，训练成本高（2倍前向）；预训练数据量大导致蒸馏开销巨大；早期阶段学生能力太弱，可能无法有效学习教师的soft label。实践中更常见的是在预训练后用蒸馏做模型压缩，或者在预训练中只用部分数据做蒸馏。

---

### Q: Hopper架构TMA的优点、调用方式，是否需要经过L1？

**TMA（Tensor Memory Accelerator）优点**：
- 异步拷贝全局内存到共享内存，不占用计算线程。
- 支持多维tensor的复杂地址计算（自动处理stride、padding）。
- 减少地址计算指令，只需一个线程发起即可为整个threadblock加载数据。

**调用方式**：通过`cp.async.bulk`指令或CUTLASS/CuTe的TMA descriptor API。只需一个线程发起TMA请求。

**L1**：TMA绕过L1直接从全局内存加载到共享内存，不经过L1 cache。

---

### Q: Flash Attention v2为什么外层对Q循环？Flash Decoding的combine kernel耗时占比大约是多少？

**v2外层对Q循环**：v1外层遍历KV内层遍历Q，需要对中间结果做全局归约。v2改为外层遍历Q，每个block负责一个Q块的完整计算，避免跨block归约，提高并行度，且每个block只需写一次输出。

**Flash Decoding combine kernel耗时**：combine kernel占比通常很小（约5-10%），因为它只做简单的加权归约（各split的partial结果合并），主要耗时在split attention的计算部分。

---

### Q: Mooncake的KV-Cache Centric PD分离架构是什么？

Mooncake将推理服务分为Prefill节点和Decode节点，核心是以KV Cache为中心的调度：Prefill节点计算KV Cache后通过高速网络传输给Decode节点，而非让同一GPU同时承担Prefill和Decode。优势在于Prefill是计算密集型可用高算力卡，Decode是访存密集型可用大显存卡，资源利用更高效。通过分布式KV Cache池化管理实现灵活调度。

---

### Q: DiT的推理框架设计思路和LLM有什么异同？

**相同点**：都需要kernel优化、显存管理、batch调度。

**不同点**：
- DiT是固定步数的迭代去噪，计算图每步相同；LLM是自回归逐token生成，长度不确定。
- DiT无需KV Cache，但需要多步iteration间的中间状态管理。
- DiT的batch内序列长度固定（图像patch数）；LLM需要处理变长序列。
- DiT更适合静态图优化和CUDA Graph；LLM需要动态调度（continuous batching）。

---

### Q: 分析MLA decode的计算访存比，与seqlen和batch size是否相关？

MLA（Multi-head Latent Attention）decode阶段将KV压缩为低秩latent向量，计算过程为：先将latent解压为K和V，再计算注意力。

计算访存比与batch size相关：batch越大，相同KV Cache被复用次数越多，计算访存比提升。与seqlen的关系：seqlen越长KV Cache加载量线性增大，但计算量也线性增大，所以计算访存比主要取决于head_dim和latent_dim的比值及batch size。

---

### Q: Diffusion model的训练和推理步骤？推理num_inference_steps为40时，为什么训练timesteps仍设成1000？

**训练**：对干净数据加不同程度噪声（timestep从1到1000），模型学习在每个噪声级别预测噪声/速度。**推理**：从纯噪声出发，通过调度器（scheduler）选取子集步骤逐步去噪。

训练设1000步是为了让模型学习更精细的噪声分布，覆盖从微小噪声到纯噪声的完整连续谱。推理时通过DDIM/DPM-Solver等高效采样器，只需跳步采样40步即可获得好质量，因为这些采样器能在较少步数内很好地近似连续去噪过程。

---

### Q: 什么是dLLM？如何看待它和AR（自回归）的区别？

dLLM（Discrete Diffusion LLM）是基于离散扩散过程的语言模型，如MDLM、SEDD。训练时对token序列加离散噪声（如mask），模型学习去噪恢复原始token。

**与AR区别**：
- AR从左到右逐token生成，dLLM并行生成所有token后迭代精化。
- dLLM可并行解码提高生成速度，但质量通常略低于AR。
- AR有严格的因果约束，dLLM可利用双向上下文。
- dLLM更容易控制生成（如infilling、editing）。

---

### Q: torch.repeat和torch.expand的区别？

- **expand**：只能扩展size为1的维度，不分配新内存（返回原tensor的view，共享存储），扩展后的tensor不连续。
- **repeat**：可以沿任意维度重复，会分配新内存拷贝数据，返回新的连续tensor。

使用建议：只需要广播效果时用expand（零内存开销），需要真实数据副本时用repeat。

---

### Q: torchrun的启动参数有哪些？如何在Linux上批量kill包含torchrun的进程？

**torchrun参数**：`--nproc_per_node`（每节点进程数）、`--nnodes`（节点数）、`--node_rank`（节点编号）、`--master_addr`（主节点地址）、`--master_port`（主节点端口）、`--rdzv_backend`（rendezvous后端）。

**批量kill**：`pkill -f torchrun` 或 `ps aux | grep torchrun | grep -v grep | awk '{print $2}' | xargs kill -9`。
