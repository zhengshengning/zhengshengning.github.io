---
title: 'B站 AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: GPU算子优化的通用方法论是什么？如何通过profiling判断算子是memory-bound还是compute-bound？

**系统化优化方法论**：先Profile定性→确定瓶颈类型→针对性优化→再Profile验证，循环迭代。

**Step 1：Profiling与瓶颈判定**

使用 **Nsight Compute** 分析kernel的关键指标：
- **算术强度（Arithmetic Intensity）**：FLOPs / Bytes accessed，即每加载一字节数据能做多少次计算
- **Roofline模型**：将kernel的算术强度标注在Roofline图上
  - 如果kernel落在**memory bandwidth屋顶**下方→Memory-bound
  - 如果kernel落在**compute屋顶**下方→Compute-bound

**核心指标对照表**：

| 指标 | Memory-bound特征 | Compute-bound特征 |
|------|-----------------|------------------|
| SM Occupancy | 可能较高但性能仍差 | 通常较高 |
| Memory Throughput | 接近理论峰值（>80%） | 较低（<50%） |
| Compute Throughput | 较低（<50%） | 接近理论峰值 |
| L1/L2 Cache Hit Rate | 低 | 影响不大 |
| Warp Stall原因 | Long Scoreboard（等待内存） | 计算单元busy |

**典型数值参考（A100）**：
- HBM带宽：2039 GB/s
- FP16 Tensor Core峰值：312 TFLOPS
- 分界点算术强度：312T / 2039G ≈ 153 FLOPs/Byte

**Step 2：针对Memory-bound优化**
1. **Coalesced Access**：确保同一Warp内线程访问连续128B对齐地址，一次内存事务满足所有线程
2. **使用共享内存（SMEM）**：将重复访问的数据cache到SMEM（带宽约19 TB/s，vs HBM 2TB/s）
3. **减少全局内存事务**：向量化加载（float4/LDS.128），一次加载16字节
4. **数据重用**：通过Tiling增加数据局部性，减少HBM访问次数
5. **减少bank conflict**：SMEM访问时做padding或swizzle

**Step 3：针对Compute-bound优化**
1. **Tensor Core利用**：使用`wmma`/`mma.sync`指令，FP16吞吐是FP32的16倍
2. **减少不必要的计算**：去除冗余计算、利用数学恒等式简化
3. **减少Warp Divergence**：避免同一Warp内线程走不同分支
4. **指令级并行（ILP）**：循环展开让编译器调度更多独立指令

**Step 4：通用优化**
- 调整block size（通常128~256）平衡occupancy和寄存器压力
- 使用CUDA Graph减少kernel launch开销（对小kernel特别有效）
- 算子融合（Kernel Fusion）减少中间结果写回HBM
- Pipeline：计算与内存访问overlap（双缓冲/多级流水线）

---

### Q: 分布式通信原语AllReduce、AllGather、AllToAll的语义区别？

**AllReduce（全局归约）**：
- **语义**：每个rank贡献一份等大小的数据，对所有rank的数据做elementwise归约（如求和、取最大值），**所有rank获得相同的归约结果**
- **通信量**：Ring实现下每GPU发送 `2*(N-1)/N * M ≈ 2M`
- **典型用途**：数据并行中的梯度同步（每个GPU计算局部梯度→AllReduce求平均→所有GPU获得一致的全局梯度）
- **实现**：通常分解为 Reduce-Scatter + All-Gather

**AllGather（全收集）**：
- **语义**：每个rank贡献一块数据（可以不同大小），**所有rank获得所有数据块的拼接**
- **通信量**：每GPU发送 `(N-1)/N * M`（Ring实现），接收相同量
- **典型用途**：
  - 张量并行中收集分片的权重/激活（如FSDP的前向传播收集完整参数）
  - Sequence Parallelism中收集分片的激活
- **特点**：输出大小是输入的N倍

**AllToAll（全交换）**：
- **语义**：每个rank有N块数据，第i块发送给第i个rank。每个rank**向每个其他rank发送不同的数据块**，最终收到来自所有rank的对应块
- **通信量**：每GPU发送 `(N-1)/N * M`
- **典型用途**：
  - MoE的Expert Parallelism：将token按路由结果dispatch到对应专家所在的GPU
  - 矩阵转置式的数据重分布
- **特点**：是最一般化的通信模式，AllGather/ReduceScatter都可视为AllToAll的特例

**三者对比**：

| 原语 | 输入(每GPU) | 输出(每GPU) | 通信模式 |
|------|------------|------------|---------|
| AllReduce | M | M（归约结果） | 聚合+广播 |
| AllGather | M/N | M（完整拼接） | 广播 |
| AllToAll | M | M（重新分布） | 点对点交换 |
| ReduceScatter | M | M/N（部分归约） | 聚合+分散 |

**关系**：AllReduce = ReduceScatter + AllGather。理解这个分解很重要——Megatron的Sequence Parallelism正是将AllReduce拆开，分别与不同的计算overlap来隐藏通信延迟。

---

### Q: 手撕：CUDA实现large array reduce sum（shared memory归约 + 分层kernel设计）？

（编程题）

---

### Q: 进程、线程和协程的概念及区别？

**进程（Process）**：
- OS资源分配的基本单位，拥有独立的虚拟地址空间、文件描述符表、信号处理等
- 进程间隔离性强（一个进程崩溃不影响其他进程）
- **切换开销大**：需要切换页表（TLB flush）、保存/恢复完整CPU状态，典型延迟~1-10us
- 进程间通信需要IPC机制（pipe/socket/共享内存）
- 典型内存开销：数MB（页表、内核栈等）

**线程（Thread）**：
- OS调度的基本单位，同一进程内的多个线程共享地址空间、文件描述符
- **切换开销中等**：需要内核态切换（系统调用）、保存/恢复线程上下文（寄存器、栈指针），典型延迟~1-10us（但无需TLB flush）
- 共享地址空间使通信简单（直接读写共享变量），但需要同步机制（mutex/semaphore）
- 典型栈大小：1-8MB（Linux默认8MB）
- 受限于OS调度开销，数千~数万线程时性能下降

**协程（Coroutine）**：
- 用户态的轻量级执行单元，由程序自身调度（非抢占式）
- **切换开销极小**：只需保存/恢复少量寄存器和栈指针，典型延迟~100ns
- OS不感知协程的存在，不涉及系统调用
- 典型栈大小：2-8KB（Go goroutine）或无栈（Python async）
- 适合高并发IO场景（可以轻松创建百万级协程）
- **缺点**：单个协程阻塞会导致整个线程阻塞；不能利用多核（除非多线程+协程）

**综合对比**：

| 特性 | 进程 | 线程 | 协程 |
|------|------|------|------|
| 调度者 | OS内核 | OS内核 | 用户程序 |
| 切换开销 | ~1-10us | ~1-10us | ~100ns |
| 内存开销 | 数MB | 1-8MB栈 | 2-8KB |
| 并发量级 | 数百~数千 | 数千~数万 | 数十万~数百万 |
| 是否共享地址空间 | 否 | 是 | 是 |
| 利用多核 | 天然支持 | 支持（无GIL的语言） | 需配合多线程 |
| 典型应用 | 独立服务、安全隔离 | 并行计算、GUI | 高并发IO、异步编程 |

---

### Q: CPU调度粒度是怎样的？进程级和线程级的公平性如何？

**现代Linux的调度单位是线程（task_struct）**：

Linux内核中进程和线程在调度层面没有本质区别——都是`task_struct`。进程只是恰好不与其他task共享地址空间的线程。调度器（CFS）对所有task一视同仁。

**CFS（Completely Fair Scheduler）的公平性模型**：
- CFS维护一棵红黑树，按`vruntime`（虚拟运行时间）排序
- `vruntime`增长速度与权重（nice值）成反比：nice=0的线程vruntime正常增长，nice=-20的增长最慢（获得更多CPU）
- 每次调度选择vruntime最小的task执行
- **默认对线程公平**：每个线程获得相等的CPU时间

**对进程公平性的影响**：
- 一个有100个线程的进程（如Java应用）vs 一个单线程进程：前者获得约100倍的CPU时间
- 这在很多场景下是不公平的

**解决进程级公平性的手段**：
1. **cgroup（Control Groups）**：将属于同一进程的所有线程放入一个cgroup，对cgroup设定CPU份额。例如 `cpu.shares=1024` 意味着该cgroup获得1024/(总shares)比例的CPU
2. **nice值**：调整线程优先级（范围-20到19），nice值越大优先级越低
3. **CFS Group Scheduling**：`CONFIG_FAIR_GROUP_SCHED`启用后，先在组间公平分配，再在组内线程间公平分配
4. **cpuset**：绑定进程到特定CPU核心，物理隔离
5. **实时调度策略**：`SCHED_FIFO`/`SCHED_RR`用于需要严格时间保证的任务

**AI Infra中的实践**：
- 训练框架通常将数据加载线程和计算线程分配到不同CPU核（`taskset`/`numactl`）
- NCCL通信线程可能需要高优先级以避免通信延迟抖动
- 使用`OMP_NUM_THREADS`控制PyTorch CPU算子的线程数，避免过度订阅
