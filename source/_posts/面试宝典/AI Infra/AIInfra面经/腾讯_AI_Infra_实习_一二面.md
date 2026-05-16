---
title: '腾讯 AI Infra 实习 一二面'
date: 2026-04-16 12:18:23
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 介绍一下MPI？

MPI（Message Passing Interface）是分布式并行计算的**标准通信接口规范**，定义了进程间消息传递的API。

**核心概念：**

```c
MPI_Init(&argc, &argv);           // 初始化MPI环境
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 当前进程编号
MPI_Comm_size(MPI_COMM_WORLD, &size);  // 总进程数
// ... 通信操作 ...
MPI_Finalize();                   // 清理
```

**通信类型：**

| 类型 | 操作 | 描述 | 复杂度 |
|------|------|------|--------|
| 点对点 | Send/Recv | 两个进程间直接传递 | O(n) |
| 广播 | Bcast | 一对多，一个进程发给所有 | O(n·log p) |
| 散射 | Scatter | 将数据均分发给各进程 | O(n) |
| 聚集 | Gather | 各进程数据汇总到一个进程 | O(n) |
| 规约 | Reduce | 汇总并做运算(sum/max等) | O(n·log p) |
| 全规约 | AllReduce | Reduce后结果广播给所有 | O(n·(p-1)/p) Ring |
| 全交换 | AlltoAll | 每个进程向每个进程发不同数据 | O(n·p) |

**编程模型——SPMD（Single Program Multiple Data）：**
- 所有进程运行同一份程序，通过rank区分行为
- 深度学习中：NCCL是NVIDIA针对GPU优化的集合通信库，接口类似MPI但性能更高
- OpenMPI/MPICH是CPU上的主流实现

**在深度学习中的应用：**
- 数据并行梯度同步 → AllReduce
- 张量并行中间结果同步 → AllReduce/AllGather
- MoE专家并行token路由 → AlltoAll
- 流水线并行层间传递 → Send/Recv

### Q: 将一个无序数组插入一个有序数组，保持有序，无序数组长度小于有序数组？

（编程题）

### Q: 了解多线程吗？

多线程是在同一进程内并发执行多个执行流，**共享进程的地址空间和资源**。

**C++11多线程核心组件：**

```cpp
// 线程创建
std::thread t([](int id) { /* work */ }, thread_id);
t.join();  // 等待完成

// 互斥锁保护共享数据
std::mutex mtx;
{
    std::lock_guard<std::mutex> lock(mtx);  // RAII加锁
    shared_data++;  // 临界区
}  // 自动解锁

// 条件变量（生产者-消费者）
std::condition_variable cv;
cv.wait(lock, []{ return !queue.empty(); });  // 等待条件
cv.notify_one();  // 唤醒一个等待者

// 原子操作（无锁）
std::atomic<int> counter{0};
counter.fetch_add(1, std::memory_order_relaxed);  // 原子递增
```

**多线程关键问题与解决方案：**

| 问题 | 描述 | 解决方案 |
|------|------|---------|
| 数据竞争 | 多线程同时读写共享数据 | mutex/atomic/线程局部存储 |
| 死锁 | 多个线程互相等待对方的锁 | 锁顺序一致/std::scoped_lock/超时锁 |
| 优先级反转 | 低优先级线程持锁阻塞高优先级 | 优先级继承协议 |
| 伪共享 | 不同线程访问同一cache line的不同变量 | alignas(64)对齐/padding |
| 线程爆炸 | 创建过多线程超出系统限制 | 线程池复用 |

**与CUDA多线程的区别：**
- CPU线程：重量级（每个有独立栈,MB级），数量有限（数十~数百），上下文切换开销大
- CUDA线程：轻量级（主要状态是寄存器），数量巨大（数万~百万），切换几乎零开销
- CPU线程靠时间片切换实现并发；CUDA线程靠硬件同时调度实现真并行

### Q: 操作系统中死锁是什么？产生条件和解决方法？

**死锁定义：** 多个进程/线程各自持有某些资源，同时等待其他进程/线程持有的资源，形成环形等待，所有进程永久阻塞。

**经典示例：**
```cpp
// 线程A                  // 线程B
mutex1.lock();           mutex2.lock();
mutex2.lock(); // 等B    mutex1.lock(); // 等A
// 互相等待，永远不能继续
```

**四个必要条件（缺一不可）：**

| 条件 | 含义 | 破坏方法 |
|------|------|---------|
| 互斥 | 资源同一时刻只能一个进程使用 | 使资源可共享（如读写锁） |
| 持有并等待 | 持有资源同时请求新资源 | 一次性申请所有需要的资源 |
| 不可抢占 | 已获取的资源不能被强制释放 | 允许系统强制回收资源 |
| 循环等待 | 进程间形成环形等待链 | **资源有序分配**（最实用） |

**解决策略：**

1. **预防（Prevention）**——破坏四个条件之一：
   - 有序资源分配：所有线程按固定顺序获取锁（如按锁地址排序）
   - C++: `std::scoped_lock(mtx1, mtx2)` 自动处理锁顺序

2. **避免（Avoidance）**——动态检查是否安全：
   - 银行家算法：每次分配前检查是否存在安全序列
   - 实际系统中开销太大，很少使用

3. **检测+恢复（Detection & Recovery）**：
   - 构建资源分配图，检测环的存在
   - 恢复：终止死锁进程、回滚到checkpoint、抢占资源

4. **实际工程最佳实践：**
   - 锁超时：`try_lock_for()`超时后放弃并重试
   - 层次化锁：定义锁的level，只允许从低level向高level获取
   - 减少锁持有时间：临界区最小化

### Q: 介绍一下UDP和TCP的区别？

**核心对比：**

| 特性 | TCP | UDP |
|------|-----|-----|
| 连接性 | 面向连接（三次握手） | 无连接 |
| 可靠性 | 可靠（确认+重传） | 不可靠（best effort） |
| 有序性 | 保证顺序（序列号） | 不保证顺序 |
| 流量控制 | 滑动窗口 | 无 |
| 拥塞控制 | 有（慢启动/拥塞避免） | 无 |
| 头部开销 | 20-60字节 | 8字节 |
| 传输方式 | 字节流 | 数据报(datagram) |
| 一对多 | 不支持 | 支持广播/组播 |
| 适用场景 | 文件传输、HTTP、数据库 | 实时音视频、DNS、游戏 |

**为什么UDP更快？**
- 无需建连（省去三次握手的RTT）
- 无确认/重传机制（不等待ACK）
- 头部更小（8B vs 20B+）
- 无拥塞控制（不会主动降速）
- 无需维护连接状态（服务器可支持更多客户端）

**实际选择建议：**
- 需要可靠传输 → TCP（或QUIC = UDP + 自实现可靠性）
- 实时性优先（允许丢包） → UDP
- RDMA通信 → 基于UDP的InfiniBand/RoCE协议

### Q: TCP怎么保证可靠传输？

TCP通过**多层机制协同**保证可靠传输：

**1. 序列号(Sequence Number) + 确认号(ACK)：**
- 每个字节都有唯一序列号
- 接收方通过ACK告诉发送方"我已成功接收到seq X之前的所有数据"
- 保证数据不丢失、不重复、有序

**2. 超时重传(Retransmission)：**
- 发送数据后启动重传定时器
- 超时未收到ACK则重发该段数据
- RTO(重传超时)根据RTT动态计算：RTO = SRTT + 4×RTTVAR
- 快重传：收到3个重复ACK立即重传（不等超时）

**3. 滑动窗口(Sliding Window)——流量控制：**
- 接收方通告自己的接收窗口大小(rwnd)
- 发送方发送量不超过min(cwnd, rwnd)
- 防止发送方淹没接收方

**4. 拥塞控制(Congestion Control)：**
```
慢启动: cwnd从1开始指数增长（每RTT翻倍）
拥塞避免: cwnd达到ssthresh后线性增长（每RTT+1）
快重传: 3个dup ACK → 立即重传
快恢复: cwnd减半 → 线性增长（不回到慢启动）
```

**5. 校验和(Checksum)：**
- 对TCP头部+数据+伪头部计算16位校验和
- 检测传输过程中的比特错误
- 错误的报文直接丢弃（等待重传）

### Q: C++ STL中除了优先队列还有什么可以实现堆？

**STL堆操作函数（可在任意RandomAccessIterator上操作）：**

```cpp
vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

std::make_heap(v.begin(), v.end());     // 建堆 O(n)
// v现在是max-heap: [9, 6, 4, 1, 5, 3, 2, 1]

std::push_heap(v.begin(), v.end());     // 假设末尾已有新元素，上浮 O(log n)
std::pop_heap(v.begin(), v.end());      // 堆顶移到末尾，剩余重建堆 O(log n)

// 堆排序
std::sort_heap(v.begin(), v.end());     // 反复pop_heap实现排序 O(n log n)
```

**与priority_queue的关系：**
- `priority_queue`本质就是`vector` + 这些heap算法的封装
- 区别：priority_queue是adapter（限制操作），直接用heap函数更灵活（可以遍历、修改任意元素）

**其他可实现堆/有序结构的方式：**

| 方式 | 底层 | 特点 |
|------|------|------|
| priority_queue | vector + heap算法 | 默认max-heap，只能访问top |
| make_heap系列 | 任意随机访问容器 | 灵活，可自定义比较函数 |
| set/multiset | 红黑树 | 有序，支持任意位置删除，O(log n) |
| Boost.Heap | 多种堆实现 | Fibonacci/Binomial/d-ary heap |

### Q: 什么是POD类型？

POD（Plain Old Data）是C++中与C语言兼容的简单数据类型，满足两个条件：**平凡(Trivial)** + **标准布局(Standard-layout)**。

**平凡类型(Trivial)要求：**
- 有编译器生成的默认构造、析构、拷贝、移动函数
- 没有虚函数和虚基类
- 可以安全地用`memcpy`复制、`memset`初始化

**标准布局(Standard-layout)要求：**
- 所有非静态成员具有相同的访问控制(全public或全private)
- 没有虚函数/虚基类
- 基类和第一个成员类型不同（避免空基类优化歧义）
- 只有一个类（自身或基类）有非静态成员

**POD类型的特权操作：**
```cpp
struct PodType { int x; float y; char z; };  // 是POD

// 可以做的事：
memcpy(&pod1, &pod2, sizeof(PodType));  // 按位复制（安全）
fwrite(&pod, sizeof(pod), 1, file);     // 直接序列化到文件
// 与C代码互操作（ABI兼容）
// 用于CUDA __constant__内存和结构体传递
```

**C++20后的推荐判断方式：**
```cpp
static_assert(std::is_trivial_v<MyType>);         // 平凡性
static_assert(std::is_standard_layout_v<MyType>); // 标准布局
static_assert(std::is_trivially_copyable_v<MyType>); // 可memcpy
```

### Q: CUDA相关概念：Shuffle、Warp Reduce、SM？

**Warp Shuffle——warp内线程间直接交换寄存器数据：**

```cuda
// __shfl_sync: 广播（指定lane的值发给所有线程）
float val = __shfl_sync(0xffffffff, src_val, src_lane);

// __shfl_down_sync: 向下偏移（常用于reduce）
float val = __shfl_down_sync(0xffffffff, my_val, offset);
// lane i 收到 lane (i+offset) 的值

// __shfl_xor_sync: 异或交换（butterfly pattern）
float val = __shfl_xor_sync(0xffffffff, my_val, mask);
```

**为什么Shuffle比共享内存更优？**
- 无需额外内存空间（直接操作寄存器）
- 无需`__syncthreads()`（warp内硬件同步，lock-step执行）
- 延迟极低：1-2个时钟周期 vs 共享内存~28个时钟周期
- 不存在bank conflict问题

**Warp Reduce实现（5步完成32个元素的规约）：**
```cuda
__device__ float warpReduceSum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);  // 合并16对
    val += __shfl_down_sync(0xffffffff, val, 8);   // 合并8对
    val += __shfl_down_sync(0xffffffff, val, 4);   // 合并4对
    val += __shfl_down_sync(0xffffffff, val, 2);   // 合并2对
    val += __shfl_down_sync(0xffffffff, val, 1);   // 合并1对
    return val;  // lane 0持有最终结果
}
```

**SM（Streaming Multiprocessor）——GPU的基本计算单元：**
- 包含CUDA Core(FP32计算)、Tensor Core(矩阵乘加)、寄存器堆、共享内存、Warp调度器
- 多个warp在SM上**时分复用**：一个warp等待数据时，调度器切换到另一个warp执行
- Block被映射到SM上执行，一个SM可同时驻留多个Block（受资源限制）
- A100有108个SM，H100有132个SM

### Q: 什么是TVM和Halide？

**核心共同理念——计算与调度分离(Compute-Schedule Separation)：**

```
算法描述(WHAT) — 不变的数学定义
    × 
调度策略(HOW) — 可搜索的执行方式
    = 
高效代码
```

**Halide（图像处理DSL/编译器）：**
```cpp
// 算法定义（永远不变）
Func blur_x, blur_y;
blur_x(x, y) = (input(x-1,y) + input(x,y) + input(x+1,y)) / 3;
blur_y(x, y) = (blur_x(x,y-1) + blur_x(x,y) + blur_x(x,y+1)) / 3;

// 调度（独立调整，不影响正确性）
blur_y.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);
blur_x.compute_at(blur_y, x).vectorize(x, 8);
```
- 专为图像处理管道设计
- 调度语言表达力强：tile/vectorize/parallel/compute_at/store_at
- 支持CPU(SIMD)/GPU(CUDA)/DSP等多后端

**TVM（深度学习编译器框架）：**
- 借鉴Halide的计算/调度分离思想
- 增加了自动调度搜索（AutoTVM/Ansor）
- 支持深度学习特有的优化（算子融合、量化、内存规划）
- 更广泛的硬件支持（GPU/NPU/FPGA等）

**TVM vs Halide对比：**

| 维度 | Halide | TVM |
|------|--------|-----|
| 定位 | 图像处理 | 深度学习 |
| 调度方式 | 手动编写Schedule | 自动搜索(Ansor) |
| 前端 | C++ DSL | Python(Relay/TE) |
| 算子融合 | 有限 | 自动图级融合 |
| 模型导入 | 无 | ONNX/PyTorch/TF |

### Q: AoS和SoA数据布局的区别？

**AoS (Array of Structures) vs SoA (Structure of Arrays)：**

```cpp
// AoS: 一个粒子的所有属性连续存储
struct Particle { float x, y, z, vx, vy, vz; };
Particle particles[N];  
// 内存: [x0,y0,z0,vx0,vy0,vz0, x1,y1,z1,vx1,vy1,vz1, ...]

// SoA: 同一属性的所有粒子连续存储
struct Particles { 
    float x[N], y[N], z[N]; 
    float vx[N], vy[N], vz[N]; 
};
// 内存: [x0,x1,x2,...,xN, y0,y1,y2,...,yN, ...]
```

**为什么SoA对GPU/SIMD更友好？**

| 维度 | AoS | SoA |
|------|-----|-----|
| 向量化 | 加载x0后下一个x1隔5个float | x连续存放，SIMD一次加载4/8/16个x |
| GPU合并访问 | 32线程各要x，地址间隔stride=6 | 32线程各要x，地址连续→一次128B事务 |
| Cache利用 | 加载一个cache line含无关数据(y,z,v...) | cache line全是需要的x数据 |
| 带宽利用率 | 如果只用x，浪费5/6带宽 | 100%有效带宽 |

**实际选择建议：**
- GPU编程：**强烈推荐SoA**（合并访存是性能关键）
- CPU + 需要完整对象操作：AoS更直观（如遍历粒子做完整模拟）
- 混合方案(AoSoA)：每个"结构"包含SIMD宽度个元素，兼顾两者优势

```cpp
// AoSoA (Array of Structure of Arrays) - 混合布局
struct ParticleBlock {  // 每个block包含8个粒子(SIMD宽度)
    float x[8], y[8], z[8];
    float vx[8], vy[8], vz[8];
};
ParticleBlock blocks[N/8];
// 块内SoA→向量化友好，块间AoS→局部性好
```
