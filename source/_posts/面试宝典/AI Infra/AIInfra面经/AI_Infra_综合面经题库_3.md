---
title: 'AI Infra 综合面经题库 (3)'
date: 2026-04-16 12:07:16
categories:
  - [求职面试,综合面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: 给定训练所需的Tokens，怎么估计模型训练所需的完整时间？

估算公式：`训练时间 = 6 * N * D / (GPU数 * 单卡FLOPS * MFU)`

其中N=模型参数量，D=训练token数，6是前向+反向的系数（前向2N FLOPS/token，反向4N FLOPS/token），MFU是模型算力利用率（Model FLOPs Utilization，通常30-50%）。还需考虑通信开销、故障恢复时间等。

---

### Q: Prefill和Decode阶段各有什么优化技术？

**Prefill优化**（计算密集型）：
- Flash Attention减少显存和IO。
- Tensor Parallelism提高并行度。
- 长序列分块处理（chunked prefill）。
- Prefix caching复用重复前缀的KV Cache。

**Decode优化**（访存密集型）：
- KV Cache量化减少访存量。
- Speculative Decoding（投机解码）减少解码步数。
- Continuous Batching提高GPU利用率。
- PagedAttention减少内存碎片。
- Multi-Query/Grouped-Query Attention减少KV Cache大小。

---

### Q: 什么是Two-batch overlap？什么场景下是负优化？

Two-batch overlap是将一个batch的计算和另一个batch的通信（如AllReduce）同时进行，利用计算和通信的硬件独立性实现重叠。

**负优化场景**：
- 单个batch已经能充分占满GPU计算资源，overlap不会带来额外收益反而增加调度开销。
- 通信量很小，通信本身不是瓶颈。
- 显存紧张时，两个batch同时在GPU上增加显存压力。
- 通信和计算共享带宽资源时（如PCIe），overlap导致相互争抢反而都变慢。

---

### Q: Megatron-LM中通信优化怎么做？

- **TP通信与计算重叠**：将AllReduce分解为Reduce-Scatter + All-Gather，与计算流水线化。
- **Sequence Parallelism**：减少激活显存，将非TP部分沿序列切分。
- **PP通信优化**：1F1B调度减少bubble、interleaved schedule。
- **异步通信**：使用独立的通信stream与计算overlap。
- **分层通信**：机内NVLink做TP，机间IB做DP/PP。
- **梯度累积**：减少通信频率。

---

### Q: 多机PD分离有KV Cache transfer开销，为什么还要做PD分离？

- **资源异构适配**：Prefill是计算密集型（用高算力卡），Decode是访存密集型（用大带宽/大显存卡），分离后各自用最适合的硬件。
- **调度灵活性**：Prefill和Decode负载波动独立，分离后可独立扩缩容。
- **消除干扰**：混合部署时Prefill的长计算会阻塞Decode的低延迟要求。
- **总体效率**：虽有传输开销，但资源利用率和吞吐量的提升远超传输成本（NVLink/RDMA可达100+GB/s）。

---

### Q: Muon和AdamW的pretrain和posttrain为什么不能混用？

Muon（Momentum Orthogonalization）优化器通过正交化动量更新来优化模型，其更新方向和大小与AdamW有根本性差异。混用导致问题：
- 两种优化器对参数空间的"最优区域"假设不同，pretrain用一种到达的参数状态可能不适合另一种继续优化。
- Muon维护的动量状态与AdamW的一阶/二阶矩估计不兼容，切换时优化器状态无法继承。
- 梯度统计分布在切换点突变，可能导致训练不稳定。

---

### Q: 如何看待跨SM的PD分离和AF（Attention-FFN）分离？

跨SM分离将SM划分为不同功能组：如部分SM做Attention（访存密集），部分做FFN（计算密集），类似warp specialization的思想扩展到SM级别。

优势：不同workload对硬件资源需求不同，分离后可以为各部分配置最优的资源（如共享内存分配、occupancy设置）。劣势：增加SM间通信开销、负载均衡困难、实现复杂度高。目前更多是学术探索，实际中Flash Attention + 大batch的pipeline重叠更实用。

---

### Q: CUDA的global memory和shared memory访存分别需要注意什么？

**Global Memory**：
- 合并访存（coalesced access）：同一Warp内线程访问连续地址，一次事务完成。
- 对齐访问：起始地址对齐到128字节。
- 避免partition camping（早期架构）。

**Shared Memory**：
- 避免Bank Conflict：32个bank，4字节为一个bank宽度。
- 适当padding消除conflict。
- 注意shared memory容量限制对occupancy的影响。

---

### Q: DeepSeek-V3有哪些优化点？

- **MLA（Multi-head Latent Attention）**：将KV Cache压缩为低维latent表示，大幅减少推理时KV Cache大小。
- **DeepSeekMoE**：细粒度专家 + 共享专家设计，提高专家利用率。
- **FP8训练**：混合精度训练进一步降低显存和加速计算。
- **Multi-Token Prediction（MTP）**：训练辅助目标，提升模型质量，推理时可用于speculative decoding。
- **DualPipe**：优化的流水线并行策略，减少bubble。
- **无辅助损失的负载均衡**：通过bias调节替代auxiliary loss。

---

### Q: DeepSeek-DSA、NSA和MoBA的区别？

- **DSA（DeepSeek Sparse Attention）**：训练时使用的稀疏注意力，基于预设的稀疏模式减少计算量。
- **NSA（Native Sparse Attention）**：硬件原生支持的稀疏注意力实现，利用block-sparse等模式在Tensor Core上高效执行。
- **MoBA（Mixture of Block Attention）**：将注意力计算分为多个block，通过路由机制选择每个query关注哪些key block，类似MoE思想应用于注意力。

核心区别在于稀疏模式的确定方式：DSA是预定义模式，NSA是硬件适配模式，MoBA是动态学习的路由模式。

---

### Q: NCCL中的通信原语有哪些？AllReduce参数更新一次需要几次通信？

**通信原语**：AllReduce、AllGather、ReduceScatter、Broadcast、Reduce、AllToAll、Send/Recv。

**AllReduce通信次数**：Ring AllReduce分为两个阶段：Reduce-Scatter（N-1步）+ All-Gather（N-1步），总通信量为2*(N-1)/N倍数据量（N为GPU数）。对于一次参数更新，只需要一次AllReduce操作（内含多步通信），总通信数据量约2倍模型大小。

---

### Q: 小数据量场景用NVSHMEM直接读取其他GPU数据做本地reduce，相比Ring AllReduce有什么好处？

- **延迟更低**：NVSHMEM是one-sided通信，kernel内线程直接发起远端读取，无需集合通信的同步协调开销。
- **适合小数据**：Ring AllReduce的启动开销（launch latency）对小数据占比大，NVSHMEM细粒度通信更高效。
- **编程灵活**：可以在kernel内部按需通信，不需要退出kernel调用NCCL API。
- **减少同步点**：不需要等待所有GPU都就绪才开始通信。

对大数据量，Ring AllReduce的带宽利用更优（接近理论带宽峰值）。

---

### Q: 训练时如何设计超长序列下的并行？

- **Sequence Parallelism（SP）**：将序列切分到不同GPU，非TP层各GPU处理局部序列。
- **Ring Attention / Context Parallelism**：将KV序列分到不同GPU，Q序列块循环访问所有KV块。
- **Ulysses**：对attention的head维度做AllToAll，将序列并行转为head并行。
- **结合使用**：TP + SP + Context Parallelism多维并行。
- **梯度Checkpointing**：减少激活显存。
- **Flash Attention**：减少单步的显存占用。

---

### Q: 将Ampere架构的算子适配到Hopper架构，会对哪些地方进行升级改造？

- **TMA（Tensor Memory Accelerator）**：用TMA替代手写的全局内存加载逻辑，异步预取到共享内存。
- **Warp Specialization**：将生产者（数据加载）和消费者（计算）分配到不同warp group，流水线化。
- **Cluster**：利用Thread Block Cluster实现跨block的共享内存访问（Distributed Shared Memory）。
- **FP8 Tensor Core**：将FP16计算升级为FP8利用更高的Tensor Core吞吐。
- **异步屏障**：用`cp.async.bulk`和`mbarrier`替代传统同步方式。
