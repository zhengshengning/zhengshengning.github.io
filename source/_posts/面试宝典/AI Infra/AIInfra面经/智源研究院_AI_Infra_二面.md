---
title: '智源研究院 AI Infra 二面'
date: 2026-04-16 12:21:01
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 推理框架的设计要素：计算图、运行图、内存管理？

推理框架的核心设计可以分为三个层次，从逻辑描述到物理执行再到资源管理：

**1. 计算图（Logic Graph / IR）**：

描述模型的**算子拓扑和数据依赖关系**，与硬件无关的纯逻辑表示：
- 节点 = 算子（如Conv、MatMul、Softmax）。
- 边 = Tensor数据流（带shape/dtype信息）。
- 来源：ONNX、TorchScript、SavedModel等模型格式导入。
- 特点：不关心执行顺序、内存分配、硬件映射。

**2. 运行图（Execution Graph / Physical Plan）**：

从计算图经过一系列优化pass后生成的**面向具体硬件的执行计划**：
- **算子融合**：识别可融合的pattern（Conv+BN+ReLU），生成单一kernel。
- **内存规划**：分析tensor生命周期，规划buffer复用方案。
- **调度顺序**：拓扑排序基础上，考虑并行执行机会（多stream/多硬件单元）。
- **Kernel选择**：为每个op选择最优实现（如cuDNN的不同conv算法auto-tune）。
- **图切分**：多设备场景下决定哪些op在哪个设备执行。

**3. 内存管理（Memory Management）**：

内存管理是推理框架性能的关键，好的管理能将内存使用降低2-5倍：

- **静态规划（编译时）**：
  - Liveness Analysis：分析每个tensor的创建点和最后使用点。
  - Buffer复用：生命周期不重叠的tensor可以共享同一块内存。
  - 最优复用是NP-hard问题，通常用启发式（First-Fit、Best-Fit）。
  
- **动态分配（运行时）**：
  - 内存池（Memory Pool）：预分配大块内存，内部快速分配/回收。避免频繁的cudaMalloc（开销~100us）。
  - BFC Allocator（TensorFlow）：Best-Fit with Coalescing。
  
- **Inplace优化**：
  - 输入输出复用同一buffer（如ReLU: output可以覆盖input）。
  - 需要确认input后续不再被使用（no other consumers）。
  
- **内存对齐**：
  - GPU: 128字节对齐（满足coalesced access需求）。
  - Tensor Core: 某些格式需要256字节或更大对齐。

**实际框架示例**：TensorRT在build阶段完成所有优化生成运行图 + 静态内存规划，runtime阶段仅执行已优化的计划，零动态开销。

### Q: 动态图、静态图、动态Shape的区别？

这三个概念容易混淆，它们描述了计算图在不同维度上的"动态性"：

| 维度 | 静态图 | 动态图 | 动态Shape |
|------|-------|-------|----------|
| 图结构 | 运行前固定 | 运行时构建 | 运行前固定 |
| Tensor Shape | 编译时已知 | 运行时确定 | 运行时确定 |
| 控制流 | 图中特殊节点 | Python原生 | 图中静态表示 |
| 优化能力 | 最强（全局优化） | 最弱（逐op） | 中等（部分优化） |
| 代表 | TensorRT, XLA | PyTorch eager | torch.compile + dynamic |

**静态图**：一切在编译时确定——图拓扑、tensor形状、内存分配方案。可以做极致优化（TensorRT引擎），但不灵活。

**动态图**：逐op执行，每步都可能根据数据做不同的计算（如NLP中不同batch的序列长度不同、条件分支依赖中间结果）。灵活但每个op都有独立的launch/分配开销。

**动态Shape（关键区别）**：图结构固定（哪些op、连接关系不变），但tensor的维度值运行时才知道。编译器需要处理"未知维度"：
- **符号Shape推导**：用符号变量表示，建立约束（如"输出batch_size == 输入batch_size"）。
- **Bucket编译**：对常见shape区间各预编译一个优化版本，运行时匹配最近的bucket。
- **Shape Guard**：torch.compile的做法——首次trace记录shape，后续运行如果shape变了就重新编译。
- **保守代码生成**：用动态循环边界，牺牲部分优化（如不能做固定的loop unroll）。

### Q: 常见的图优化技术有哪些？

图优化是推理引擎（TensorRT/torch.compile/TVM）性能提升的核心来源，通常以**pass**的形式逐步应用：

**1. 算子融合（Operator Fusion）—— 最有效的优化**：
- **Element-wise融合**：将连续的逐元素操作（Add+Mul+ReLU）合并为单个kernel。减少kernel launch和中间tensor的HBM读写。
- **Pattern融合**：Conv+BN+ReLU → 单个kernel。BN参数在推理时可以数学合并到Conv的weight和bias中。
- **Epilogue融合**：将bias add、activation、scale等操作融合到GEMM/Conv的输出阶段（epilogue），不生成独立kernel。
- 效果：对memory-bound的小算子链可加速3-10倍。

**2. 常量折叠（Constant Folding）**：
- 预计算仅依赖常量输入的子图，将结果作为常量embed到图中。
- 典型：Embedding层的shape推导、BN的running_mean/var → 折叠为scale/bias。

**3. 死代码消除（Dead Code Elimination, DCE）**：
- 移除输出未被任何后续节点使用的算子。
- 常见场景：模型导出时可能包含训练专用节点（如dropout/BN统计更新）。

**4. 公共子表达式消除（Common Subexpression Elimination, CSE）**：
- 识别完全相同的计算子图（相同op + 相同输入），复用其结果。
- 示例：多个注意力头可能共享相同的QKV投影计算。

**5. Layout优化**：
- 选择最适合硬件的数据格式：
  - NVIDIA GPU Tensor Core偏好NHWC（HWC连续便于16字节对齐的channel load）。
  - CPU SIMD偏好NCHW（channel连续便于向量化）。
- 在图中插入最少的transpose操作，使关键算子在最优layout下运行。

**6. 内存规划（Memory Planning）**：
- Liveness分析确定每个tensor的生死区间。
- 分配算法找到最小总内存使得所有生命周期不重叠的tensor可以共享buffer。
- 效果：减少50-70%的中间内存占用。

**7. 算子替换（Op Substitution）**：
- 用等价但更高效的实现替代：如将大的depthwise conv拆分为更小的kernel组合。
- 或将自定义op映射到高度优化的库函数（如cuDNN的特定conv算法）。

### Q: Warp之间如何通信？

Warp间通信的方式取决于Warp是否在同一Block内：

**同一Block内的Warp通信**：

**1. 共享内存（Shared Memory）—— 最通用**：
```cuda
__shared__ float buffer[256];
// Warp 0 写数据
buffer[threadIdx.x] = my_data;
__syncthreads();  // 必须同步！确保所有warp写完
// Warp 1 读数据
float other_data = buffer[other_idx];
```
- 需要`__syncthreads()`保证可见性和顺序。
- 延迟约20-30 cycles，带宽约19TB/s/SM。
- 注意Bank Conflict。

**2. Warp Shuffle（仅限同一Warp内部）**：
```cuda
// 同一Warp内线程间直接寄存器交换，无需共享内存
float val = __shfl_down_sync(0xFFFFFFFF, my_val, offset);
// 效果：获取当前lane + offset位置线程的my_val值
```
- 延迟仅1-2 cycles（寄存器级别）。
- 变体：`__shfl_sync`(直接指定源lane)、`__shfl_up_sync`(向上偏移)、`__shfl_xor_sync`(按XOR模式交换，常用于butterfly reduce)。
- **限制**：只能在32个线程（同一Warp）间交换。

**3. 协作组（Cooperative Groups, CUDA 9+）**：
```cuda
auto block = cooperative_groups::this_thread_block();
auto tile32 = cooperative_groups::tiled_partition<32>(block);
tile32.sync();  // 只同步32线程的tile
```
- 提供灵活的同步粒度：tile(8/16/32) / block / grid级。
- Grid级同步需要cooperative launch（所有block在所有SM上同时执行）。

**不同Block的Warp通信**：

- **只能通过全局内存 + 原子操作**，且无硬件同步保证。
- 典型模式：Block完成计算后原子写结果到全局内存，其他Block读取。
- 需要`__threadfence()` 保证写入对其他Block可见。
- 无法保证Block间的执行顺序（GPU不保证Block调度顺序）。
- 替代方案：多次kernel launch（自然同步点）。

### Q: CUDA Reduce如何实现？

Reduce（规约）是GPU上的经典算法优化案例，优化层次从naive到极致依次为：

**Level 0: Naive（交错寻址，大量divergence）**：
```cuda
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0)  // 严重warp divergence!
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```
问题：tid%2判断导致每次迭代都有一半线程inactive。

**Level 1: 顺序寻址（减少divergence）**：
```cuda
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s)  // 前半线程活跃，divergence集中在最后几步
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```
改进：活跃线程是连续的前半部分，divergence只在warp边界处。

**Level 2: 首次加载时规约**：
```cuda
// 每线程加载2个元素，做第一次加法
sdata[tid] = input[tid] + input[tid + blockDim.x];
__syncthreads();
// 后续正常reduce
```
Block可处理2倍数据量，或减少一半Block数。

**Level 3: Warp内展开（消除最后几步的sync开销）**：
```cuda
// 当剩余元素<=32时（一个warp），不需要__syncthreads()
if (tid < 32) {
    sdata[tid] += sdata[tid + 32];  // warp内同步执行
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
```

**Level 4: Warp Shuffle（完全避免shared memory）**：
```cuda
float val = thread_data;
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
// lane 0得到warp内32个值的sum
```
无共享内存开销、无bank conflict、无sync开销。

**Level 5: 多级规约（大数据）**：
- Block内reduce → Block间：原子操作（atomicAdd全局结果）或两轮kernel。
- Grid-stride loop：每线程循环处理多个元素，单kernel覆盖任意大数据。

### Q: CUDA Softmax实现：Warp处理与Block处理的区别？

Softmax需要三步：求max（数值稳定）、求exp(x-max)之和、归一化。核心是两次reduce操作。

**Warp级Softmax（适合行长<=32或较短的情况）**：
- 一个Warp（32线程）处理一行的softmax。
- 用`__shfl_xor_sync`做butterfly reduce求max和sum（5步=log2(32)步）。
- 无需共享内存，无需`__syncthreads()`。
- 延迟极低（warp shuffle在寄存器级别交换）。
- **限制**：行长最多32（每线程一个元素）。如果行长>32，每线程处理多个元素（串行reduce后再warp shuffle合并）。

```cuda
// Warp-level softmax sketch (row_len <= 32)
float val = (lane_id < row_len) ? input[row * row_len + lane_id] : -INFINITY;
float max_val = warp_reduce_max(val);   // shfl_xor butterfly
float exp_val = expf(val - max_val);
float sum = warp_reduce_sum(exp_val);    // shfl_xor butterfly
float result = exp_val / sum;
```

**Block级Softmax（适合长行）**：
- 一个Block处理一行（或一个Block处理多行）。
- 用共享内存做Block-wide reduce（需要`__syncthreads()`）。
- 支持任意行长（每线程grid-stride处理多个元素后block reduce合并）。
- 开销：共享内存+多次sync。

**选择策略**：

| 行长 | 推荐方案 | 原因 |
|------|---------|------|
| <=32 | 1 Warp / 行 | 纯shuffle，零开销 |
| 33-1024 | 1 Block / 行 | Block内shared memory reduce |
| >1024 | Online Softmax / 多Block | 分块计算，online更新统计量 |

**实际框架（如FlashAttention）中**：Softmax与注意力计算融合，使用Online Softmax在SRAM中逐block处理，避免两次遍历完整的N长度行。

### Q: Block/Grid设置为什么会影响算子速度？

Block和Grid大小的选择通过以下机制直接影响GPU的硬件利用率和执行效率：

**Block大小影响**：

1. **Occupancy（占用率）**：
   - Block太大：每Block需要的寄存器和shared memory多，SM只能驻留少量Block → 活跃Warp少 → 延迟隐藏差。
   - Block太小：即使SM驻留多个Block，总Warp数可能仍不够。
   - 最佳点：通常128-256线程/Block，需用Occupancy Calculator确认。

2. **资源分配**：每Block可用的shared memory = SM总量 / 同时驻留Block数。Block越多，每Block可用shared memory越少。

3. **同步开销**：`__syncthreads()`等待Block内所有线程到达barrier。Block越大，等待最慢线程的概率越高。

4. **Warp对齐**：Block线程数必须是32的倍数，否则尾部不足一个Warp的线程浪费硬件资源。

**Grid大小影响**：

1. **波次效应（Wave Effect）**：
   - 假设GPU有108个SM，每SM驻留2个Block → 一个"wave"=216个Block可以并行。
   - 如果Grid有217个Block → 需要2个wave，第二个wave只有1个Block，107个SM空闲！
   - **应确保Grid中Block数是wave容量的整数倍**，或远大于（尾部效应可忽略）。

2. **负载均衡**：Block数太少可能无法覆盖所有SM，部分SM空闲。

3. **调度开销**：Block数极多（百万级）时，Grid调度器有少量开销，通常可忽略。

**实践建议**：
- Block大小：128或256（满足大多数kernel的resource约束）。
- Grid大小：确保至少有2-4个wave（即Block数 >= 4 × SM数 × 驻留Block数/SM）。
- 对于逐元素kernel：Grid大小 = ceil(N / BlockSize)，通常自然满足。
- 对于GEMM：Block数由矩阵大小和tile大小决定，需确保覆盖所有SM。

### Q: CUDA的计算模型（执行模型）是什么？

CUDA采用SIMT（Single Instruction Multiple Threads）执行模型，本质是一种**层次化的大规模并行编程抽象**：

**线程层次结构（软件侧）**：
```
Grid (整个kernel的所有线程)
├── Block 0 (线程协作单位, 可共享shared memory/同步)
│   ├── Warp 0 (32 threads, 锁步执行)
│   ├── Warp 1
│   └── ...
├── Block 1
└── ...
```

**硬件映射关系**：
- Grid → 整个GPU（由GPC调度器分发Block）。
- Block → SM（一个Block完整运行在一个SM上，不迁移）。
- Warp → SM内的Warp调度器（每SM有4个warp调度器，每cycle可各发射1条指令）。
- Thread → CUDA Core / Tensor Core / Load-Store Unit。

**内存层次结构**：
```
寄存器 (线程私有, ~1 cycle, 最快)
  → Shared Memory (Block共享, ~20-30 cycles, 程序员管理)
    → L1 Cache (SM私有, ~30 cycles, 硬件管理)
      → L2 Cache (全局共享, ~200 cycles)
        → HBM/Global Memory (全局, ~400 cycles, 最慢)
```

**执行特点**：
- **Warp内锁步**：同一Warp的32线程执行相同指令（SIMT），divergence时串行化。
- **Warp间独立**：不同Warp由调度器独立调度，执行顺序不确定。
- **Block间无通信保证**：不同Block可能在不同时间执行，全局内存+原子操作是唯一通信方式。
- **延迟隐藏靠TLP**：GPU不像CPU用深流水线/乱序执行隐藏延迟，而是靠大量Warp快速切换——一个Warp stall时立即切换到另一个就绪Warp。

### Q: FlashAttention V1和V2的区别？

**FlashAttention V1（核心创新）**：

- **核心思想**：分块（tiling）+ online softmax，避免N×N注意力矩阵在HBM中的存储。
- **内存**：O(N^2) → O(N)，使得长序列训练成为可能。
- **速度**：比PyTorch标准实现快2-4倍（减少HBM访问）。
- **前向**：外层循环在KV block上（固定一个Q block，遍历所有KV block计算该Q的输出）。
- **反向**：不存储中间P矩阵，只存softmax统计量(m, l)。反向时重计算P再求梯度。

**FlashAttention V2的改进（速度翻倍）**：

**改进1：减少非矩阵乘FLOPs**

V1中online softmax的rescale操作（乘以修正因子）占用了大量非matmul计算。V2重新组织计算公式，将rescale延迟到最后统一执行，减少非matmul运算约25%。由于Tensor Core做matmul远快于CUDA Core做标量运算，减少非matmul FLOPs对整体提升显著。

**改进2：改进并行度（循环顺序交换）**

V1外层循环在KV block上：对每个Q block，遍历所有KV。这意味着序列长度维度的并行性在内层。

V2外层循环在Q block上：每个thread block处理一个Q block（遍历所有KV）。优势：
- Q block数 = seq_len / block_size，直接在grid维度并行，更容易填满GPU。
- 当batch_size × num_heads不够大（如长序列场景）时，V2通过序列维度的并行性额外占满SM。

**改进3：Warp分工优化**

V1：Block内4个Warp各自独立计算QKV partial attention，最后reduce合并。Warp间有通信开销。

V2：4个Warp分别处理不同的KV block（split-K方式），每个Warp独立完成自己负责的KV block的完整计算，最后在shared memory中合并。减少了warp间的频繁通信。

**改进4：反向优化**

更高效的分块和并行化，减少shared memory争用。

**性能数据对比**（A100, seq_len=2048, head_dim=128）：
- V1: ~125 TFLOPS
- V2: ~230 TFLOPS（接近A100 FP16峰值312T的73%）
- V2比V1快约1.7-2倍。
