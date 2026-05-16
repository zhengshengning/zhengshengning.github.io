---
title: '腾讯 AI Infra 实习 一面 (1)'
date: 2026-04-16 12:06:39
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Chunked Prefill是什么？它为了解决什么问题？

Chunked Prefill将长序列的prefill阶段分成多个固定大小的chunk（如512或1024 token）分批处理，而不是一次性计算整个序列。

**解决的核心问题：**

**1. Prefill阻塞Decode（Head-of-Line Blocking）：**
```
传统方式：
  [========长prefill(5000token)========] [decode] [decode] [decode]
                                          ↑ 被阻塞等待，延迟飙升

Chunked Prefill：
  [chunk1][decode][chunk2][decode][chunk3][decode][chunk4][decode]
                   ↑ 穿插执行，decode延迟平稳
```
- 长序列prefill可能占用GPU数百毫秒（如5000 token的prefill约100-500ms）
- 期间所有decode请求被阻塞，导致已在生成的用户体验到卡顿
- 分chunk后每个chunk只占10-50ms，decode请求的等待时间大幅缩短

**2. 显存峰值过高：**
- Attention计算的临时空间与序列长度相关（非FlashAttention时为O(n^2)）
- FlashAttention下虽然O(1)临时空间，但整个序列的KV-Cache一次性分配仍占用大量显存
- 分chunk后每次只需分配一个chunk大小的临时空间

**实际效果（SarathiServe论文数据）：**
- P99 TPOT（Time Per Output Token）降低50-80%
- 代价：总prefill时间略增（分chunk间有调度开销）
- 整体吞吐基本不受影响

**实现细节：**
- Chunk之间的KV-Cache需要持久化（前一个chunk计算的KV存入cache）
- 后续chunk的Attention需要关注前面所有chunk的KV（cross-chunk attention）
- 调度器在每个iteration判断：当前step执行prefill chunk还是decode

### Q: Reduce-Scatter和All-to-All通信的区别？

**Reduce-Scatter：规约 + 分发**

```
输入: 每个rank持有完整数据（如完整梯度）
操作: 先规约(如sum)，再将结果等分发给各rank

Rank 0: [A0, A1, A2, A3]    →  规约后: Rank 0得到 sum([A0,B0,C0,D0])
Rank 1: [B0, B1, B2, B3]    →          Rank 1得到 sum([A1,B1,C1,D1])
Rank 2: [C0, C1, C2, C3]    →          Rank 2得到 sum([A2,B2,C2,D2])
Rank 3: [D0, D1, D2, D3]    →          Rank 3得到 sum([A3,B3,C3,D3])

通信量: (n-1)/n × data_size（每rank发送和接收）
```

**All-to-All：全交换**

```
输入: 每个rank持有不同数据要发给不同目标
操作: rank i 的第j块发给rank j

Rank 0: [A0, A1, A2, A3] → Rank 0收到[A0, B0, C0, D0]
Rank 1: [B0, B1, B2, B3] → Rank 1收到[A1, B1, C1, D1]
Rank 2: [C0, C1, C2, C3] → Rank 2收到[A2, B2, C2, D2]
Rank 3: [D0, D1, D2, D3] → Rank 3收到[A3, B3, C3, D3]

通信量: (n-1)/n × data_size（转置操作）
```

**应用场景对比：**

| 操作 | 应用场景 | 特点 |
|------|---------|------|
| Reduce-Scatter | ZeRO梯度分片、TP行切后的输出分片 | 有规约操作(sum/avg) |
| All-to-All | MoE的token dispatch/combine、序列并行 | 纯数据重排，无计算 |
| AllReduce | DDP梯度同步 | = Reduce-Scatter + AllGather |
| AllGather | ZeRO3参数收集、TP列切后拼接 | 每rank持有部分，收集为完整 |

**MoE中All-to-All的具体使用：**
```
Expert Parallelism中，每个rank负责不同expert：
1. Forward All-to-All: 各rank将本地token按路由结果发给对应expert所在rank
2. Expert计算
3. Backward All-to-All: 将expert输出发回token所属的原始rank
```

### Q: 怎么减少Launch Kernel的开销？

每次kernel launch涉及CPU端的驱动调用和GPU端的任务分发，有**固定开销约5-10微秒/次**。当kernel本身很小（如element-wise操作处理少量数据）时，launch开销占比可能超过50%。

**主要优化方法：**

**1. CUDA Graph（最有效，decode场景标配）：**
```cuda
// 录制阶段（一次性）
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_A<<<...>>>();
kernel_B<<<...>>>();
kernel_C<<<...>>>();
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// 执行阶段（反复调用，单次launch开销替代多次）
cudaGraphLaunch(graphExec, stream);  // 一次launch执行整个图
```
- 将多次launch合并为一次graph launch
- CPU端开销从N×5us降为1×5us
- 限制：graph内的kernel参数和shape不能变化

**2. 算子融合（从根本上减少kernel数量）：**
- 将ReLU+Add+Mul融合为一个kernel → 3次launch变1次
- FlashAttention将多个步骤融合 → 显著减少launch次数
- NVFuser/torch.compile自动做element-wise融合

**3. Persistent Kernel（长驻kernel）：**
```cuda
__global__ void persistent_kernel(WorkQueue* queue) {
    while (true) {
        Task task = queue->dequeue();  // 等待新任务
        if (task.type == EXIT) break;
        process(task);                 // 处理任务
    }
}
```
- Kernel不退出，循环等待新任务
- 适合任务频繁到达的场景（如流式处理）

**4. 其他方法：**
- 减少不必要的`cudaDeviceSynchronize()`（同步会强制等待所有kernel完成）
- 使用多Stream并发launch（CPU可以连续提交而不阻塞）
- 增大单个kernel的工作量（少量大kernel优于大量小kernel）

### Q: CUDA编程中Bank Conflict是什么？怎么解决？

**什么是Bank Conflict？**

GPU共享内存被组织为**32个bank**，每个bank宽4字节。连续的4字节地址映射到连续的bank：
```
地址 0-3   → Bank 0
地址 4-7   → Bank 1
...
地址 124-127 → Bank 31
地址 128-131 → Bank 0 (循环)
```

当同一warp中的**多个线程同时访问同一bank的不同地址**时，这些访问被**串行化**（N-way conflict → N倍延迟）。

**典型冲突示例：**
```cuda
__shared__ float s[32][32];
// 列访问: s[threadIdx.x][col]
// 列方向stride=32×4=128字节，正好跨32个bank回到同一bank
// → 32路bank conflict！所有线程访问Bank 0
```

**解决方案：**

**方法1: Padding（最简单有效）：**
```cuda
__shared__ float s[32][33];  // 多加1列
// 现在列方向stride=33×4=132字节，错开1个bank
// 线程0访问Bank 0, 线程1访问Bank 1, ... 无冲突！
```
代价：浪费少量共享内存（32×4=128字节/矩阵）

**方法2: Swizzle（地址重映射）：**
```cuda
// 通过XOR运算重映射行列索引，使访问模式无冲突
int new_col = col ^ (row % 32);  // 示例swizzle策略
```
无额外内存开销，但代码复杂度增加。Cutlass库大量使用此技术。

**方法3: 向量化访问：**
```cuda
// 使用float4一次读4个连续元素，跨越4个bank
float4 val = reinterpret_cast<float4*>(&s[row][col])[0];
// 每个线程占用4个连续bank，减少冲突概率
```

**判断是否有bank conflict：** Nsight Compute中查看`Shared Memory Bank Conflicts`指标。

### Q: 场景题：大集群中节点内有NVLink，节点间部分机器有RDMA，如何设计分布式推理方案？

**设计思路——感知网络拓扑，匹配通信需求与互联能力：**

**核心原则：** 通信密集的并行方式用高带宽互联，通信稀疏的用低带宽互联。

```
┌─── Node 1 (NVLink互联) ───┐    ┌─── Node 2 (NVLink互联) ───┐
│ GPU0 ←NVLink→ GPU1         │    │ GPU4 ←NVLink→ GPU5         │
│ GPU2 ←NVLink→ GPU3         │    │ GPU6 ←NVLink→ GPU7         │
│ (TP=4, 层内切分)           │    │ (TP=4, 下一个PP stage)     │
└──────────── RDMA ──────────┘    └────────────────────────────┘
                    ↕ PP (只传激活值，通信量小)
```

**具体方案：**

| 层级 | 并行方式 | 互联 | 原因 |
|------|---------|------|------|
| 节点内(8卡) | 张量并行(TP=8) | NVLink(600GB/s) | TP每层需要AllReduce，通信频繁且数据量大 |
| RDMA节点间 | 流水线并行(PP) | RDMA(~50GB/s) | PP只传激活值(batch_size×hidden_size)，通信量小 |
| 无RDMA节点 | 数据并行/不参与 | TCP | 不适合参与同一推理任务 |

**详细设计考量：**

1. **TP组划分**：同一TP组必须在NVLink互联的GPU上
   - A100 NVLink带宽600GB/s，AllReduce一个14GB的hidden state只需~25ms
   - 如果用PCIe(32GB/s)做TP，延迟增加20倍，不可接受

2. **PP调度**：使用RDMA连接的节点组成PP pipeline
   - 每次传递的数据量 = micro_batch_size × seq_len × hidden_size × 2bytes(FP16)
   - 典型值：1×2048×4096×2 = 16MB，RDMA传输约0.3ms，可接受

3. **容错设计**：
   - RDMA链路故障：将该节点的PP stage迁移到备用节点
   - 单GPU故障：该TP组整体不可用，需要reconfigure

4. **调度器实现**：
   - 维护集群拓扑图（NVLink/RDMA/TCP连接关系）
   - 请求路由时做亲和性调度：同一推理任务的TP组绑定到NVLink节点
   - 监控链路健康状态，动态调整路由

### Q: 手撕：K个一组翻转链表？

（编程题）
