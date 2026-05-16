---
title: 'OPPO 云 AI Infra 实习 一面'
date: 2026-04-16 12:11:02
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 如何保证精度转换时的数值稳定性？

混合精度训练中，FP16/BF16的有限表示范围（FP16: ±65504，最小正次正规数5.96e-8）使得精度转换成为关键问题。以下是系统化的稳定性保障方法：

**1. 损失缩放（Loss Scaling）**
- **问题**：FP16反向传播中，小梯度值（<2⁻²⁴≈5.96e-8）会下溢为0，导致参数无法更新
- **方案**：将loss乘以一个大的scale factor（如2¹⁶=65536），使梯度值放大到FP16可表示范围
- 更新前将梯度除以scale factor恢复真实值（此时已在FP32中操作）

**2. 动态损失缩放（Dynamic Loss Scaling）**
- 初始scale factor设为较大值（如2²⁴）
- 如果梯度出现inf/nan：scale factor减半，跳过本次更新
- 如果连续N步（如2000步）无inf/nan：scale factor加倍
- PyTorch的`torch.cuda.amp.GradScaler`自动实现此逻辑
- 好的训练过程中，scale factor应稳定在一个合理范围内

**3. 关键计算保持高精度**

| 计算环节 | 精度要求 | 原因 |
|---------|---------|------|
| Softmax中的max减法 | FP32 | 防止exp()上溢（e^大数=inf） |
| 累加器（GEMM内部） | FP32 | 大量小数累加精度损失严重 |
| LayerNorm/BN统计量 | FP32 | 均值/方差计算需要高精度 |
| 权重更新（optimizer state） | FP32 | 学习率*梯度可能很小 |
| 损失计算 | FP32 | cross-entropy中log()对小概率敏感 |

**4. Kahan Summation（补偿求和）**
```python
# 标准求和：大量小数累加时低位信息丢失
# Kahan补偿：
sum = 0.0; c = 0.0  # c是补偿变量
for x in values:
    y = x - c        # 补偿上一步的误差
    t = sum + y
    c = (t - sum) - y  # 记录丢失的低位
    sum = t
```
- 将O(n)的舍入误差降为O(1)，对FP16下的大规模reduction特别重要

**5. 混合精度策略的整体设计**

```
权重 Master Copy (FP32) ←--更新-- Optimizer States (FP32)
         |                              ↑
         | cast                         | cast + unscale
         ↓                              |
权重 Working Copy (FP16/BF16) → Forward/Backward → 梯度 (FP16/BF16)
```

**FP16 vs BF16选择**：
- **FP16**：精度高（10位尾数），范围小（±65504），需要loss scaling
- **BF16**：精度低（7位尾数），范围大（与FP32相同，±3.4e38），**不需要loss scaling**
- A100+/H100推荐BF16（硬件原生支持，训练更稳定）；V100只支持FP16

**6. 其他数值稳定性技巧**
- **梯度裁剪（Gradient Clipping）**：`clip_grad_norm_(params, max_norm=1.0)` 防止梯度爆炸
- **权重衰减（Weight Decay）**：防止权重过大超出FP16范围
- **初始化策略**：Xavier/Kaiming初始化确保初始激活值在合理范围
- **残差连接**：维持梯度在合理量级

### Q: Warp/Block划分策略与SIMT架构的关系？

**SIMT（Single Instruction Multiple Threads）核心约束**：
GPU硬件要求同一Warp（32线程）在同一时刻执行**相同的指令**（但操作不同的数据）。这个硬件约束直接决定了Block/Warp的划分策略。

**Warp级别的考量**：

**1. 分支发散（Warp Divergence）**
- 如果Warp内线程走不同的if/else分支，硬件会**串行执行**所有分支（mask掉不走该分支的线程）
- 最坏情况：Warp利用率降为1/32
- **策略**：让同一Warp内的线程尽量走相同分支
  - 按warp对齐数据分区：`if (threadIdx.x / 32 < threshold)` 好于 `if (threadIdx.x % 32 < threshold)`
  - 用数学运算替代条件分支：`result = a * mask + b * (1-mask)`

**2. 合并内存访问（Coalesced Memory Access）**
- 同一Warp内32个线程在同一指令周期访问全局内存时，硬件会尝试合并为最少的内存事务
- **最优**：线程i访问地址 `base + i * sizeof(element)`（连续128字节对齐）→ 1次事务
- **最差**：线程随机访问 → 32次事务（32倍带宽浪费）
- **策略**：数据布局和线程映射使得相邻线程访问相邻内存

**Block级别的考量**：

**3. Block大小选择（blockDim）**
- 必须是32的倍数（否则最后一个Warp有空闲线程浪费资源）
- 常用选择：128、256、512
- **权衡**：
  - Block越大→每个block可用的寄存器/共享内存越少（资源均分）
  - Block越小→可能无法隐藏内存延迟（需要足够的warp做切换）

**4. Occupancy（SM占用率）**
- `Occupancy = 活跃Warp数 / SM最大Warp数`
- 受限于三个资源：寄存器数、共享内存大小、最大block/warp数
- 高occupancy不保证高性能（可能导致cache thrashing），但太低会导致延迟隐藏不足
- **典型目标**：50%~75% occupancy，用Nsight Compute的Occupancy Calculator分析

**5. 共享内存与线程协作**
- Block内线程可通过共享内存通信（`__shared__`），Block间不能
- Block大小决定了tile大小→影响数据重用率
- `__syncthreads()`只同步Block内线程，越大的Block同步开销越大

**实际设计原则**：
```
// 2D block划分示例（适用于矩阵运算）
dim3 block(32, 8);  // 32列对齐内存访问，8行提供并行度
dim3 grid((N+31)/32, (M+7)/8);

// 线程到数据的映射
int col = blockIdx.x * 32 + threadIdx.x;  // Warp内连续访问列
int row = blockIdx.y * 8 + threadIdx.y;
```

### Q: 如何快速上手全新的大模型推理框架？

一个系统化的学习路径，从宏观到微观逐步深入：

**Phase 1：架构理解（1-2天）**
- 阅读官方文档/论文了解**设计哲学**和**整体架构**
- 识别核心子系统：请求调度器（Scheduler）、内存管理器（KV Cache Manager）、执行引擎（Model Runner）、采样器（Sampler）
- 画架构图：数据流向、模块依赖关系

**Phase 2：运行与观测（1天）**
- 跑通最简单的推理示例（单请求、单卡）
- 用`nsys profile`/`py-spy`看端到端执行时间线
- 观察关键指标：TTFT（Time To First Token）、TPS（Tokens Per Second）、GPU利用率
- 开启debug日志看内部状态变化

**Phase 3：关键路径追踪（2-3天）**
- 追踪一个请求从API入口到输出token的完整代码路径
- 重点关注：
  - Tokenize → 放入等待队列 → 调度决策 → Prefill/Decode → KV Cache分配 → Kernel执行 → 采样 → 返回
- 在关键节点打断点或加print理解状态流转

**Phase 4：核心模块深入（持续）**
- **Attention实现**：用的FlashAttention v1/v2/v3？是否支持PagedAttention？MQA/GQA的特殊处理？
- **KV Cache管理**：分配策略（Paged vs Contiguous）、回收策略、prefix sharing实现
- **调度器**：如何决定batch组成？Prefill vs Decode的优先级？是否支持chunked prefill？抢占策略？
- **并行策略**：TP如何实现？PP是否支持？PD分离？

**Phase 5：对比学习**
- 与已熟悉的框架对比关键设计差异：

| 设计维度 | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|-------------|
| KV Cache | PagedAttention | RadixAttention | 预分配 |
| 调度 | 简单FCFS | 基于prefix的共享 | In-flight Batching |
| 前端 | Python | Python + DSL | C++ |
| 重点优化 | 通用性 | 前缀共享/结构化生成 | 极致性能 |

**Phase 6：修改实验（验证理解）**
- 尝试做小改动：修改调度策略（如改变batch_size上限）、添加一个新的采样参数、修改KV Cache回收阈值
- 通过改动的效果验证自己对系统行为的理解
- 贡献小的bug fix或feature是最好的学习方式

### Q: 场景题：如何优化一个云端CV模型推理服务（单卡吞吐/延时）？

**延迟优化（降低P99延迟）**：

**模型层面**：
- **TensorRT编译优化**：算子融合（Conv+BN+ReLU→一个kernel）、FP16/INT8量化、kernel auto-tuning
  - 典型加速：ResNet-50 FP32→TensorRT FP16可获得3-5倍加速
- **ONNX Runtime优化**：graph optimization level设为ORT_ENABLE_ALL，开启parallel execution
- **选择性量化**：INT8量化计算密集层（Conv/FC），保持精度敏感层（首尾层）为FP16

**前后处理优化**：
- **GPU预处理**：将Resize/Normalize/Pad等操作移到GPU执行（DALI库/自定义CUDA kernel）
  - CPU预处理可能占总延迟的20-40%，移到GPU可减少数据传输和CPU瓶颈
- **CUDA JPEG/PNG Decode**：nvJPEG硬件解码，避免CPU解码瓶颈
- **零拷贝（Zero-Copy）**：使用pinned memory + async memcpy实现CPU-GPU传输与计算overlap

**系统层面**：
- 减少CPU-GPU同步点：使用CUDA Stream异步执行，避免不必要的`cudaDeviceSynchronize()`
- CUDA Graph：将推理pipeline固化为Graph一次launch，减少kernel launch开销（对小模型尤其有效，可减少~0.5ms）
- 模型预热：服务启动时做dummy推理，避免首次调用的JIT编译/内存分配延迟（冷启动可能增加数百ms）

**吞吐优化（提高QPS）**：

**Batching策略**：
- **Dynamic Batching**：在短时间窗口（如5-10ms）内攒批，增大batch size
  - GPU利用率随batch size增大而提高（利用GPU并行度），但延迟也增加
  - 找到延迟SLA下的最大batch size
- **多Stream并发**：为不同batch分配不同CUDA stream，overlap不同请求的计算
  - 注意：同一GPU上多stream并发不一定能提升吞吐（如果GPU已满载）

**资源效率**：
- **INT8量化**：计算量和显存减半→同样的延迟约束下batch可以更大
  - INT8 Tensor Core吞吐是FP16的2倍（A100: 624 vs 312 TOPS）
- **模型蒸馏/剪枝**：用更小更快的模型，权衡精度和速度
- **多模型复用GPU**：低负载模型共享GPU（MPS/时分复用）

**服务架构层面**：
- **异步推理Pipeline**：请求接收→预处理→推理→后处理各阶段流水线化（三级pipeline可提高吞吐3倍）
- **负载均衡**：多GPU/多实例间按请求复杂度路由（大图片→空闲GPU）
- **自动扩缩容**：根据QPS指标动态调整实例数（K8s HPA + GPU metrics）
- **结果缓存**：相似输入的推理结果缓存（适用于重复请求多的场景）
