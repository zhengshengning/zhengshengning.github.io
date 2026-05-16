---
title: '科大讯飞 AI Infra 校招 一面'
date: 2026-04-16 12:11:58
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 面经]
---


<!-- more -->

---

### Q: BF16 和 FP16 的区别？为什么大模型训练用 BF16？

| 属性 | FP16 | BF16 |
|---|---|---|
| 符号位 | 1 | 1 |
| 指数位 | 5 | 8 |
| 尾数位 | 10 | 7 |
| 动态范围 | ±6.5×10⁴ | ±3.4×10³⁸（与 FP32 相同） |
| 精度（最小精度差） | ~0.001 | ~0.01 |
| 特殊值 | 有 Inf/NaN | 有 Inf/NaN |

**为什么大模型训练用 BF16**：

1. **动态范围是关键**：训练中梯度和 loss 值可能非常大（>65504 溢出 FP16 的上限）或非常小（<6e-8 下溢为 0），BF16 的 8 位指数与 FP32 动态范围一致（±3.4×10³⁸），完全避免溢出问题

2. **无需 Loss Scaling**：FP16 训练必须配合 loss scaling（放大 loss 防止梯度下溢，更新前再缩小）。BF16 动态范围足够，可以直接 drop-in 替换 FP32，无需额外处理

3. **精度损失可接受**：虽然 BF16 尾数只有 7 位（精度比 FP16 低），但大模型训练中重要的是梯度方向而非绝对精度。实验表明 BF16 训练精度与 FP32 几乎一致

4. **硬件支持**：A100+ 的 Tensor Core 原生支持 BF16，吞吐与 FP16 相同

**典型混合精度策略**：
- 主权重（master weight）：FP32（Adam 优化器状态中）
- 前向/反向计算：BF16
- 梯度 AllReduce：BF16
- 这样只需 2 bytes/param 的通信和计算，但保持 FP32 级别的训练稳定性

### Q: 大端和小端的区别？主流架构用什么？

**大端（Big-Endian）**：高字节在低地址。数据在内存中的顺序与人类阅读顺序一致。
```
0x12345678 → 内存低地址到高地址：[12][34][56][78]
```

**小端（Little-Endian）**：低字节在低地址。便于硬件实现（从低地址开始读就是最低有效位）。
```
0x12345678 → 内存低地址到高地址：[78][56][34][12]
```

**主流架构的字节序**：
- **x86/x64（Intel/AMD）**：小端——PC 和服务器的绝对主流
- **ARM**：默认小端（Bi-endian，可配置为大端，但实际几乎都用小端）
- **RISC-V**：小端
- **网络协议（TCP/IP）**：大端（网络字节序），因此跨网络传输时需要 `htonl()`/`ntohl()` 转换
- **POWER（IBM）**：大端（旧版），Power9+ 支持小端

**为什么需要关注**：
- 网络编程中必须处理字节序转换
- 序列化/反序列化二进制数据时需要一致
- GPU 通信中（NCCL/RDMA）也需注意，不过 NVIDIA GPU 与 x86 一样使用小端

### Q: 如果设计一个 CPU，从哪几个部分考虑？

设计 CPU 需要从微架构的多个维度进行权衡：

**1. 指令集架构（ISA）**：
- RISC vs CISC：RISC 指令定长、译码简单、利于流水线（ARM/RISC-V）；CISC 指令变长但代码密度高（x86）
- 指令格式、寻址模式、寄存器数量（通用寄存器越多减少访存）

**2. 流水线设计**：
- 经典五级：取指（IF）→ 译码（ID）→ 执行（EX）→ 访存（MEM）→ 写回（WB）
- 现代高性能 CPU 流水线 12-20 级（更深 = 更高频率，但分支惩罚更大）
- 数据冒险处理：Forwarding/Bypassing

**3. 分支预测**：
- 现代 CPU 95%+ 预测准确率（TAGE 预测器、Neural Branch Predictor）
- BTB（Branch Target Buffer）缓存跳转目标地址
- 预测失败惩罚 = 流水线深度 × cycle（深流水线代价更大）

**4. 缓存层次**：
- L1（32-64 KB，1-4 cycles）→ L2（256KB-1MB，10-20 cycles）→ L3（数十 MB，30-50 cycles）
- 关联度（way）、替换策略（LRU/随机）、一致性协议（MESI/MOESI）

**5. 乱序执行（Out-of-Order）**：
- Tomasulo 算法 + 保留站（Reservation Station）实现指令动态调度
- ROB（Reorder Buffer）保证指令顺序退休
- 允许独立指令提前执行，隐藏 cache miss 等延迟

**6. 多核/超标量**：
- 超标量：每 cycle 发射多条指令（如 4-wide、6-wide）
- 多核：多个独立执行核心，共享 L3 和内存
- SMT（超线程）：一个物理核虚拟为 2 个逻辑核，共享执行资源

### Q: 用户态和内核态的区别？如何切换？

**本质区别**——CPU 权限级别不同：

- **用户态（Ring 3）**：程序运行在受限权限下，不能直接访问硬件（I/O 端口、MMU）、不能执行特权指令（如关中断 `cli`、修改页表 `mov cr3`）、只能访问自己的虚拟地址空间

- **内核态（Ring 0）**：操作系统内核运行，拥有完全权限，可访问所有物理内存、操控硬件、管理进程调度

**切换方式（用户态 → 内核态）**：

1. **系统调用（Syscall）**——主动：
   - 用户程序通过 `syscall` 指令（x64）或 `int 0x80`（x86）发起
   - 如 `read()`、`write()`、`mmap()` 等
   - CPU 切换到 Ring 0，跳转到内核的 syscall handler

2. **中断（Interrupt）**——被动：
   - 硬件中断：时钟中断（触发调度）、网卡/磁盘 I/O 完成通知
   - CPU 自动保存上下文并跳转到中断服务程序

3. **异常（Exception）**——被动：
   - 缺页异常（Page Fault）：访问未映射的虚拟地址
   - 除零错误、非法指令、段错误等
   - CPU 陷入内核处理异常

**切换过程开销**：
- 保存用户态寄存器上下文（压栈）
- 切换栈指针到内核栈
- 刷新 TLB（如果需要切换页表）
- 典型开销：数百纳秒到数微秒
- 频繁 syscall 是性能瓶颈——现代方案：io_uring（减少 syscall 次数）、vDSO（纯用户态读时钟）

### Q: NPU 开发的难点和策略？

**难点分析**：

1. **软件生态不成熟**：
   - CUDA 经过 15+ 年迭代，有 cuDNN/cuBLAS/NCCL 等完善的库；NPU 的等效库覆盖不足
   - 调试工具不完善：缺少类似 compute-sanitizer/Nsight Compute 的成熟工具
   - 文档和社区资源少

2. **算子库覆盖不足**：
   - LLM 中的长尾算子（RoPE、SwiGLU、各种 Attention 变体）需要手写适配
   - 不同 NPU 的编程模型差异大（华为 Ascend C、寒武纪 BANG C、摩尔线程 MUSA）

3. **内存模型差异**：
   - NPU 的片上存储层次与 GPU 不同（如华为的 AI Core 有 L0A/L0B/L1/UB 多级 buffer）
   - 数据搬运需要显式编程（类似 DMA），不像 GPU 有硬件 cache

4. **性能调优困难**：
   - Profiling 工具不完善，很多行为是黑盒
   - 性能计数器不如 NVIDIA 丰富
   - Roofline 分析缺少精确的硬件参数

**应对策略**：

1. **算子迁移**：优先对齐 CUDA 算子语义（保持输入输出一致），逐步迁移验证
2. **利用厂商高性能库**：如华为 CANN（包含 nn 算子库、图优化引擎）
3. **图层面优化**：最大化算子融合减少数据搬运——在 NPU 上搬运开销比 GPU 更显著
4. **编译器路线**：用 TVM/Triton-like 编译器生成 kernel，减少手写工作
5. **紧密协作**：与硬件团队获取底层性能信息、micro-benchmark 结果

### Q: Softmax 优化中如何解决负载不均衡？

Softmax 计算 `softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))` 涉及 reduce（求 max、求 sum）和 elementwise（exp、div）操作，负载不均衡主要出现在以下场景：

**不均衡来源**：
- 不同 token 的实际序列长度不同（padding 场景）
- Attention Softmax 中不同 Q token 对应不同 K 长度（因果 mask 下越靠前的 token K 越少）
- Batch 内不同样本序列长度差异大

**解决方案**：

1. **动态分配**：按实际有效长度分配计算任务，对 padding 部分不做计算。实现上通过传入 `valid_length` 数组，每个线程只处理有效范围

2. **分块处理（Tiling）**：将长序列切分为固定大小 block（如 256），每个 thread block 处理一个 block。这样即使序列长度不同，每个 block 的工作量一致

3. **Online Softmax**：分块计算时维护 running max/sum，避免全局归约的同步等待：
   ```
   m_new = max(m_old, block_max)
   l_new = l_old * exp(m_old - m_new) + block_sum * exp(block_max - m_new)
   ```
   
4. **Warp 级归约**：利用 `__shfl_down_sync` 在 warp（32线程）内做 reduce，避免 shared memory 同步的开销。对于短序列（≤32 元素），一个 warp 即可完成

5. **Persistent Kernel**：对于 batch 中序列长度差异大的情况，使用 persistent kernel + 任务队列，每个 block 处理完当前任务后从队列取下一个，天然负载均衡

### Q: Tensor Parallel 切分的是什么？涉及哪些通信？

**切分内容**：TP 切分的是模型**线性层的权重矩阵**，将大矩阵按行或列切分到多个 GPU 上，每张卡执行部分矩阵乘法。

**具体切法（以 LLaMA 的 Transformer MLP 为例）**：

```
输入 x: [batch, seq, hidden]

第一层（gate_proj/up_proj）—— 列切分（Column Parallel）：
  W_gate: [hidden, ffn_dim] → 切为 [hidden, ffn_dim/P]，每卡存一列块
  每卡计算：y_partial = x @ W_gate_shard  → [batch, seq, ffn_dim/P]
  无需通信！每卡独立计算部分输出特征

第二层（down_proj）—— 行切分（Row Parallel）：
  W_down: [ffn_dim, hidden] → 切为 [ffn_dim/P, hidden]，每卡存一行块
  每卡计算：z_partial = y_partial @ W_down_shard  → [batch, seq, hidden]（部分和）
  需要 AllReduce！将各卡的部分和求和得到最终输出
```

**通信操作总结**：

| 操作 | 通信类型 | 通信量 | 触发位置 |
|---|---|---|---|
| Column Parallel → Row Parallel | 无通信 | 0 | 中间无需同步 |
| Row Parallel 输出 | AllReduce | 2×batch×seq×hidden×dtype | 每层 MLP 结束 |
| Attention O_proj 输出 | AllReduce | 2×batch×seq×hidden×dtype | 每层 Attention 结束 |

每个 Transformer 层的前向需要 **2 次 AllReduce**（Attention 和 MLP 各一次）。

**优化——Sequence Parallel**：
- 用 ReduceScatter + AllGather 替代 AllReduce
- 将 LayerNorm/Dropout 沿序列维度分片（每卡处理部分 token 的 norm）
- 通信量不变，但可以与计算重叠，且 norm 计算也被分摊
