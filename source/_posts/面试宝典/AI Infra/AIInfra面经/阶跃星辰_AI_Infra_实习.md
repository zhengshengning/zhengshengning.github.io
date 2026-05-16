---
title: '阶跃星辰 AI Infra 实习'
date: 2026-04-16 12:14:00
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: 序列并行的实现和通信开销？如何继续优化？

**序列并行（Sequence Parallelism, SP）** 是对张量并行（TP）的补充优化，将 TP 区域之外的操作（LayerNorm、Dropout）也并行化：

**背景问题**：在标准 TP 中，Attention 和 FFN 内部做了张量切分，但 LayerNorm 和 Dropout 仍然在每卡上对**完整序列**做冗余计算。这意味着这些操作的激活值每卡都完整保存——浪费显存。

**SP 的实现**：
- 在 TP 组（如 8 卡）内，将 LayerNorm/Dropout 的序列维度均分到各卡
- 每卡只处理 seq_len/TP_size 长度的序列片段
- TP 区域内的 AllReduce 被拆分为：
  - TP 计算前：AllGather（收集完整序列用于 Attention/FFN 输入）
  - TP 计算后：ReduceScatter（将结果分片回各卡）

**通信模式变化**：
```
标准 TP：  LayerNorm(全) -> AllReduce -> Attention(分) -> AllReduce -> FFN(分) -> AllReduce
序列并行： LayerNorm(分) -> AllGather -> Attention(分) -> ReduceScatter -> LayerNorm(分) -> AllGather -> FFN(分) -> ReduceScatter
```

**通信开销分析**：
- 标准 TP：每层 2 次 AllReduce，通信量 = 2 * 2 * hidden * seq * batch（AllReduce = 2x data）
- SP：每层 2 次 AllGather + 2 次 ReduceScatter = 等价于 2 次 AllReduce
- **通信量相同**，但激活显存从 seq*hidden/卡 降为 (seq/TP)*hidden/卡（节省约 (TP-1)/TP 的激活显存）

**进一步优化手段**：

**1. 通信-计算重叠（最关键）**：
- AllGather 可以分块执行：收到前几块就开始计算，同时继续接收后续块
- ReduceScatter 同理：计算完一块就发送，同时计算下一块
- 实现：将 GEMM 按 K 维度分块，每块计算与对应块的通信重叠

**2. 减少通信量**：
- FP16 通信代替 FP32（如果训练精度允许）
- 梯度压缩：对激活值做有损压缩后传输（如 1-bit Adam 的思路）
- 选择性通信：只传输变化显著的部分（梯度稀疏化）

**3. 拓扑感知调度**：
- 利用 NVLink 全互联拓扑，选择通信效率最高的 AllGather/ReduceScatter 实现
- NCCL 的 ring/tree/recursive halving-doubling 选择根据数据量和节点数自动调优
- 多级并行时避免不同维度的通信竞争同一物理链路

**4. Ulysses Sequence Parallelism**：
- 不同于 Megatron-SP 的切分方式，Ulysses 在 attention 维度切分 head（类似 TP 切 head）
- 通信量可能更少（只需 All-to-All 而非 AllGather+ReduceScatter）

### Q: 训练稳定性和容错方面的工作有哪些？

大规模训练（千卡/万卡级别）面临的稳定性和容错挑战是工程化的核心问题：

**训练稳定性**：

**1. 梯度裁剪（Gradient Clipping）**：
- `max_norm = 1.0`：将梯度的全局范数裁剪到不超过 1.0
- 防止梯度爆炸导致参数更新过大 -> loss spike -> 训练 diverge
- 几乎所有大模型训练都启用（LLaMA/GPT/PaLM 均用 max_norm=1.0）

**2. Loss Spike 检测与处理**：
- 监控 loss 的 moving average，当偏离均值超过 3σ 时标记为 spike
- 处理策略：(1) 跳过异常数据（bad data skip）；(2) 回滚到上一个 checkpoint；(3) 降低学习率重新训练该 step
- 根因分析：通常由异常数据样本（极长/极短/乱码）引起

**3. FP32 累加**：
- 混合精度训练中关键 reduction（如 LayerNorm 的均值/方差、softmax 的 sum）使用 FP32 累加
- 防止 FP16/BF16 精度不足导致的数值不稳定（尤其在 long sequence 上累加时）

**4. 权重衰减与正则化**：
- Weight decay = 0.1（大模型常用值），防止参数过大
- Embedding 层通常不使用 weight decay（不是正常的可学习参数分布）

**5. 学习率调度**：
- Warmup + Cosine Decay 是标配
- Warmup 防止训练初期参数剧烈变化；Cosine Decay 后期平滑降低避免 overshoot
- 某些方案（如 WSD）在 Decay 结束后加 Stable 阶段

---

**容错机制**：

**1. Checkpoint 保存/恢复**：
- 每 100-500 步保存一次（平衡恢复粒度和 IO 开销）
- 异步保存避免阻塞训练（GPU -> CPU pinned memory -> 异步写存储）
- 保留最近 3-5 个 checkpoint（防止最新 checkpoint 损坏）
- 分布式 checkpoint：每卡只保存自己的参数分片，并行 IO

**2. 弹性训练（Elastic Training）**：
- 节点故障后自动将其踢出训练集群
- 剩余节点重新组织并行策略（如 64 卡变 60 卡，重新分配 DP 组）
- 故障恢复后可以重新加入
- 框架支持：PyTorch Elastic（torchelastic）、DeepSpeed Elastic

**3. 故障检测与预防**：
- Heartbeat 机制：定期检测各节点/GPU 是否存活
- NVLink/PCIe 错误监控：ECC 错误累积超阈值主动迁移
- GPU 温度/功耗监控：异常升高时提前告警
- NCCL 超时检测：通信操作超时说明某节点可能卡死

**4. 冗余计算**：
- 关键操作（如 gradient checksum）在多节点执行，结果比对验证
- Megatron-LM 的 deterministic mode：确保重放可重现

**5. 数据容错**：
- 训练数据 pipeline 需要 exactly-once 语义（避免重启后重复/遗漏数据）
- 记录每步消费的 data offset，恢复时从断点继续
- 数据 shuffle 的随机种子固定且可重现

**千卡规模的可靠性数据**：1000 GPU 训练中，平均每 2-4 天遇到一次硬件故障。好的容错系统可以将恢复时间从"人工干预数小时"降为"自动恢复 5-10 分钟"。

### Q: Infra领域正在做的前沿技术/方向？

AI Infra 是一个快速演进的领域，当前的前沿方向包括：

**1. 超长上下文高效推理（100K-1M+ tokens）**：
- 挑战：KV Cache 显存随 seq_len 线性增长（128K token 的 KV Cache 比模型权重还大）
- 方向：Attention 稀疏化（H2O/StreamingLLM 动态淘汰不重要的 KV）、KV Cache 压缩（量化/低秩近似）、分层注意力（近 token 精细注意力 + 远 token 粗略注意力）
- Ring Attention / Context Parallelism：跨设备分布 KV Cache 解决单卡放不下的问题

**2. MoE 模型的高效训练和推理**：
- 训练挑战：All-to-All 通信瓶颈、专家负载不均（某些专家总是被选中）
- 推理挑战：专家权重太大无法全部加载到 GPU（需要 offload + prefetch）
- 方向：动态专家加载（根据输入预测需要哪些专家）、专家并行+数据并行联合优化

**3. 异构计算调度（CPU+GPU+NPU+加速器）**：
- 当前各计算单元各自为政，缺乏统一的编程模型和调度框架
- 方向：自动将不同算子调度到最适合的硬件（如 softmax 在 CPU 上可能更快于小 GPU）
- 挑战：跨设备数据传输延迟、统一内存管理、负载动态平衡

**4. 通信-计算重叠的极致优化**：
- 目标：通信完全隐藏在计算时间内（零通信开销）
- 方向：细粒度流水线（将 GEMM 分块与 AllReduce 分块交替执行）、硬件级通信加速（GDR, SHARP）
- Hopper 的 TMA + warp specialization 使 GEMM 和通信可在 SM 内同时执行

**5. 自动并行策略搜索**：
- 手动配置 3D 并行参数（TP/PP/DP/SP 大小、micro-batch 数、重计算策略）极其复杂
- 方向：基于成本模型自动搜索最优并行策略（如 Alpa、Galvatron）
- 目标：给定模型 + 硬件拓扑 + 显存约束，自动输出最优配置

**6. 在线学习与推理联合系统**：
- 打破"离线训练 + 在线推理"的分离范式
- 方向：推理中收集用户反馈 -> 实时在线学习更新模型 -> 模型持续进化
- 挑战：训练和推理共享 GPU 资源的调度、模型一致性、版本管理

**7. 推理编译优化（AI Compiler）**：
- 方向：将 PyTorch 模型自动编译为极致优化的 kernel（如 torch.compile/Inductor）
- 挑战：动态 shape 支持、编译时间 vs 运行时性能权衡、跨平台代码生成

### Q: DMA和RDMA是什么？

**DMA（Direct Memory Access）**和 **RDMA（Remote DMA）**是高性能数据传输的核心技术：

**DMA（直接内存访问）**：

- **原理**：外设（如 GPU、网卡、磁盘控制器）可以直接读写主存，无需 CPU 参与每个字节的传输
- **工作流程**：
  1. CPU 配置 DMA 控制器：设置源地址、目标地址、传输长度
  2. DMA 控制器自主完成数据搬运（CPU 可以同时做其他事）
  3. 传输完成后 DMA 控制器中断通知 CPU

- **典型应用**：
  - GPU 数据传输：`cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)` 底层使用 DMA
  - NVMe SSD 读写：磁盘控制器 DMA 直接写入用户态 buffer
  - 网卡收发数据包：数据从网卡 DMA 到内核 buffer

- **优势**：CPU 释放出来做计算而非忙于数据搬运；传输带宽可达总线极限（PCIe 4.0: 32 GB/s）

**RDMA（远程直接内存访问）**：

- **原理**：一台主机可以直接读写远程主机的内存，完全绕过远程主机的 CPU 和 OS 内核
- **零拷贝路径**：应用 buffer -> 网卡 -> 网络 -> 远程网卡 -> 远程应用 buffer（无内核参与、无额外 buffer copy）

- **关键特性**：
  - **Kernel Bypass**：数据不经过 OS 网络协议栈（消除内核态开销）
  - **Zero-copy**：数据直接从用户态 buffer 发送到远程用户态 buffer
  - **CPU offload**：传输过程不占用 CPU 周期

- **性能数据**：
  - 延迟：~1-2 微秒（vs TCP/IP 的 ~50-100 微秒）
  - 带宽：InfiniBand HDR 200 Gbps / NDR 400 Gbps
  - CPU 占用：接近 0%（vs TCP 的 30-50% CPU 用于协议处理）

- **协议实现**：
  - **InfiniBand**：专用 RDMA 网络，性能最好，需要专用交换机
  - **RoCE v2（RDMA over Converged Ethernet）**：在普通以太网上实现 RDMA，部署成本低
  - **iWARP**：TCP/IP 上的 RDMA，兼容性最好但性能最差

- **在分布式训练中的应用**：
  - NCCL 底层使用 GPUDirect RDMA：GPU A 的显存 -> 网卡 -> 网络 -> 远程网卡 -> GPU B 的显存
  - 绕过两端的 CPU 和主存，延迟从 ~10us 降到 ~2us
  - AllReduce/AllGather 等集合通信的高效实现依赖 RDMA

### Q: 为什么cudaMemcpy时要锁住内存（Page-locked/Pinned Memory）？

**问题背景**：OS 使用虚拟内存管理，物理内存页可能被换出（swap）到磁盘。GPU 的 DMA 引擎需要稳定的物理地址来完成数据传输。

**Pageable Memory（普通内存）的问题**：
```
malloc() 分配的内存 -> 虚拟地址
虚拟地址对应的物理页可能：
1. 不在物理内存中（被 swap 到磁盘）-> Page Fault
2. 物理位置被 OS 重新映射 -> 物理地址变化
```

如果 DMA 正在传输过程中物理页被 swap 出去或重新映射，会导致数据损坏或访问无效地址。

**Pinned Memory（锁页内存）的解决方式**：
- `cudaMallocHost()` 或 `cudaHostAlloc()` 分配锁页内存
- OS 保证该内存页**永远驻留在物理内存中**，不会被换出
- 物理地址固定不变，GPU DMA 引擎可以安全直接访问

**Pageable 内存的 cudaMemcpy 实际过程**：
```
1. CUDA 驱动先分配一个 pinned staging buffer
2. CPU 将数据从 pageable buffer 复制到 pinned staging buffer（memcpy）
3. DMA 引擎从 pinned staging buffer 传输到 GPU 显存
```
多了一次额外的 CPU 端内存拷贝，带宽降低约 50%。

**Pinned 内存的 cudaMemcpy**：
```
1. DMA 引擎直接从 pinned buffer 传输到 GPU 显存（一次传输）
```

**性能对比**：
| 场景 | 带宽（典型值） |
|------|-------------|
| Pageable -> GPU | ~6-8 GB/s |
| Pinned -> GPU | ~12-16 GB/s (PCIe 3.0 x16) |
| Pinned -> GPU (PCIe 4.0) | ~25-28 GB/s |

**Pinned Memory 的代价**：
- 占用物理内存不释放（减少 OS 可用内存）
- 分配速度慢（需要内核调用锁定页面，vs malloc 纯用户态）
- 过度使用可能导致系统内存不足（其他进程可用物理内存减少）

**最佳实践**：
- 大块频繁传输的数据使用 pinned memory
- 小块或一次性数据使用 pageable memory（避免分配开销）
- CUDA 异步传输（`cudaMemcpyAsync`）**必须**使用 pinned memory（否则退化为同步传输）
- 使用 CUDA Streams + Pinned Memory 实现计算与传输重叠

### Q: 进程和线程的区别？

进程和线程是操作系统中两个层次的执行抽象：

**进程（Process）—— 资源分配的基本单位**：
- 拥有独立的虚拟地址空间（4GB on 32-bit, 128TB on 64-bit Linux）
- 独立的资源：页表、文件描述符表、信号处理表、堆/栈
- 进程间隔离：一个进程无法直接访问另一个进程的内存（安全性保证）
- 创建开销大：`fork()` 需要复制页表（COW 优化后仍需分配页表结构）
- 上下文切换慢：切换页表 + flush TLB（~1000-5000 cycles）

**线程（Thread）—— CPU 调度的基本单位**：
- 共享进程的地址空间（代码段/数据段/堆）
- 私有资源：栈（默认 8MB）、寄存器集、线程 ID
- 通信方便：直接读写共享变量（但需要同步机制防止 data race）
- 创建开销小：只需分配栈和 TCB（Thread Control Block）
- 上下文切换快：只需保存/恢复寄存器和栈指针（~100-500 cycles）

**详细对比**：

| 维度 | 进程 | 线程 |
|------|------|------|
| 地址空间 | 独立 | 共享 |
| 创建时间 | ~1ms（fork） | ~10-50us（pthread_create） |
| 切换时间 | ~5-20us | ~1-5us |
| 通信方式 | IPC（管道/共享内存/socket/信号） | 直接读写共享变量 |
| 崩溃影响 | 不影响其他进程 | 整个进程终止 |
| 同步需求 | 通常不需要（隔离性好） | 必须（mutex/条件变量/atomic） |
| 安全性 | 高（隔离） | 低（共享内存可能数据竞争） |

**在 AI Infra 中的应用**：
- **多进程**：分布式训练中每个 GPU 一个进程（PyTorch DDP），通过 NCCL 通信
- **多线程**：推理服务内部的请求处理（共享模型权重避免多份副本），CUDA kernel 内的"线程"
- **选择依据**：需要隔离性/安全性选进程，需要高效共享/低切换开销选线程

### Q: 手撕：最多可以完整看多少个电视节目（区间调度/贪心）？

（编程题）

### Q: 手撕：实现LRU Cache？

（编程题）
