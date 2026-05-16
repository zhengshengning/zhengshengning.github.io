---
title: '小光子 AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: CUDA算子怎么优化？

CUDA算子优化是一个**迭代式工程过程**，核心方法论是"Profile → 定位瓶颈 → 针对性优化 → 验证 → 循环"。

**Step 1: Profiling确定瓶颈类型**

使用Nsight Compute分析kernel的关键指标：
```bash
ncu --set full --target-processes all ./my_app
```

| 指标 | Memory-bound | Compute-bound | Latency-bound |
|------|-------------|---------------|---------------|
| Memory SOL% | >60% | <30% | <30% |
| Compute SOL% | <30% | >60% | <30% |
| Achieved Occupancy | 可能高 | 可能高 | 低 |
| 主要stall原因 | Wait(等待数据) | 无明显stall | MIO Throttle/指令不足 |

**Step 2: 针对不同瓶颈的优化策略**

**Memory-bound优化（最常见）：**
```cuda
// 优化前: 每个线程加载1个float, 32线程32次事务
float val = input[tid];

// 优化后: 向量化加载, 32线程8次事务(4x效率)
float4 val4 = reinterpret_cast<float4*>(input)[tid];
```
- 合并访问(Coalesced Access): 确保连续线程访问连续地址
- 向量化(float4): 一次load 128bit, 减少load指令数
- 共享内存: 将重复读取的数据缓存到SRAM (~19TB/s vs HBM 2TB/s)
- 避免Bank Conflict: padding或swizzle错开共享内存bank

**Compute-bound优化：**
- Tensor Core: 使用wmma/mma指令加速矩阵乘(FP16吞吐312 TFLOPS on A100)
- 循环展开: `#pragma unroll` 提高ILP(指令级并行)
- Warp Shuffle: 替代共享内存做warp内通信(1-2 cycles vs ~28 cycles)
- 减少分支发散: 确保同一warp内所有线程走相同路径

**Latency-bound优化：**
- 增大block size → 更多active warp → 更好的延迟隐藏
- 异步操作: `cp.async` 预取下一轮数据
- 减少同步点: 去掉不必要的`__syncthreads()`

**Step 3: 系统级优化**
- CUDA Graph: 录制kernel序列，消除重复launch开销(~5-10us/launch)
- 算子融合: 多个小kernel合并为一个，减少中间tensor的HBM读写
- 计算通信Overlap: NCCL通信与kernel计算在不同stream并行

### Q: Socket编程的过程？

**TCP Socket编程完整流程：**

```
Server端:                          Client端:
socket() → fd                      socket() → fd
    ↓                                  ↓
bind(IP, Port)                     connect(Server IP, Port)
    ↓                                  ↓
listen(backlog)                    send()/recv()
    ↓                                  ↓
accept() → new_fd [阻塞等待连接]    close()
    ↓
recv()/send() [在new_fd上通信]
    ↓
close()
```

**各系统调用的作用：**

| 调用 | 作用 | 关键参数 |
|------|------|---------|
| socket() | 创建socket文件描述符 | AF_INET(IPv4), SOCK_STREAM(TCP) |
| bind() | 绑定本地IP和端口号 | 端口需>1024(非root) |
| listen() | 标记为被动socket,设置等待队列 | backlog: 等待连接队列长度(通常128) |
| accept() | 接受新连接,返回新fd | 阻塞直到有客户端连接 |
| connect() | 主动发起连接(触发三次握手) | 指定目标IP和端口 |
| send()/recv() | 发送/接收数据 | 注意返回值可能<请求字节数(需循环) |
| close() | 关闭连接(触发四次挥手) | - |

**高性能服务器的IO模型：**

| 模型 | 原理 | 适用场景 |
|------|------|---------|
| 多线程(thread-per-connection) | 每个连接一个线程 | 连接数少(<1000) |
| IO多路复用(select/poll) | 单线程监听多个fd | 连接数中等 |
| epoll(Linux)/kqueue(macOS) | 事件驱动,O(1)就绪通知 | 高并发(>10K连接) |
| io_uring(Linux 5.1+) | 异步IO,零拷贝 | 最高性能 |

**epoll核心思想：**
```c
int epfd = epoll_create1(0);
epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &event);  // 注册感兴趣的fd
int n = epoll_wait(epfd, events, MAX_EVENTS, timeout);  // 只返回就绪的fd
// 不需要遍历所有fd，O(就绪数)而非O(总fd数)
```

### Q: Linux内核虚拟地址空间布局？

**64位Linux进程虚拟地址空间（从低地址到高地址）：**

```
0x0000000000000000
┌────────────────────────┐
│  用户空间 (低48位可用)   │  0x0 ~ 0x7FFF_FFFF_FFFF (128TB)
│  ┌──────────────────┐  │
│  │ 代码段(.text)     │  │  程序指令(可执行+只读)
│  │ 数据段(.data/.bss)│  │  已初始化/未初始化全局变量
│  │ 堆(Heap) ↓       │  │  brk/mmap动态分配，向高地址增长
│  │       ...         │  │
│  │ 内存映射区(mmap) ↓ │  │  动态库(.so)、大块malloc、文件映射
│  │       ...         │  │  
│  │ 栈(Stack) ↑       │  │  向低地址增长，默认8MB
│  └──────────────────┘  │
├────────────────────────┤  0x7FFF_FFFF_FFFF
│  Canonical Hole        │  不可寻址的地址空间
├────────────────────────┤  0xFFFF_8000_0000_0000
│  内核空间              │  0xFFFF8000... ~ 0xFFFFFFFF...
│  ┌──────────────────┐  │
│  │ 直接映射区         │  │  物理内存线性映射(page_offset_base)
│  │ vmalloc区         │  │  虚拟连续但物理不连续的分配
│  │ 内核代码           │  │  内核的.text/.data
│  │ 固定映射区         │  │  编译时确定的特殊映射
│  └──────────────────┘  │
└────────────────────────┘  0xFFFFFFFFFFFFFFFF
```

**关键设计原则：**
- 用户态不能直接访问内核空间（权限位保护，ring3 vs ring0）
- 所有进程共享同一份内核映射（切换进程时内核页表不变，节省TLB）
- 堆向上增长，栈向下增长 → 中间有大量未映射空间供两者扩展
- ASLR(地址空间随机化): 栈/堆/mmap起始地址每次运行随机化（安全性）

**GPU显存没有虚拟内存？**
- 传统CUDA: 全局内存地址是物理地址(通过驱动分配)
- CUDA Unified Memory: 提供虚拟地址抽象，页面按需在CPU↔GPU迁移
- vLLM的PagedAttention: 在应用层实现了类似虚拟内存的分页管理

### Q: 零拷贝（Zero-Copy）技术？

**问题背景——传统文件发送需要4次数据拷贝：**

```
读文件发网络的传统路径:
磁盘 → [DMA] → 内核缓冲区 → [CPU] → 用户缓冲区 
     → [CPU] → Socket内核缓冲区 → [DMA] → 网卡

4次拷贝 + 4次上下文切换(read+write各2次)
```

**零拷贝方案对比：**

| 方案 | 拷贝次数 | CPU参与 | 适用场景 |
|------|---------|---------|---------|
| 传统read+write | 4次 | 2次CPU拷贝 | - |
| mmap+write | 3次 | 1次CPU拷贝 | 需要修改数据再发送 |
| sendfile() | 2次(DMA only) | 0次CPU拷贝 | 静态文件传输(如Nginx) |
| splice/tee | 0次(管道间) | 0次 | 管道间数据传输 |
| DMA Scatter/Gather | 2次(DMA only) | 0次 | 网卡支持SG-DMA |

**sendfile()原理：**
```c
// 内核中直接将文件页面DMA到网卡，不经过用户空间
sendfile(socket_fd, file_fd, &offset, count);
// 磁盘→[DMA]→内核缓冲→[DMA直接到网卡] (仅2次DMA拷贝)
```

**CUDA中的"零拷贝"概念：**

```cuda
// Pinned Memory (锁页内存)
cudaMallocHost(&h_ptr, size);  // 物理上固定，不会被swap
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);  // DMA传输，CPU不参与

// Mapped Pinned Memory (GPU直接访问CPU内存)
cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
cudaHostGetDevicePointer(&d_ptr, ptr, 0);
// GPU通过PCIe直接读写CPU内存，无需显式cudaMemcpy
// 适合少量数据或一次性访问（避免PCIe带宽成为瓶颈）

// Unified Memory (统一内存)
cudaMallocManaged(&ptr, size);
// CPU和GPU共用同一虚拟地址，页面按需自动迁移
// 简化编程但可能有页面迁移开销
```

**零拷贝的本质：** 减少数据在不同存储/地址空间之间的不必要复制，让数据就近被处理或直接传输到目的地。
