---
title: '蚂蚁 AI Infra 实习 一面 (1)'
date: 2026-04-16 12:16:19
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 如何判断一个算子要不要优化？优化的具体思路？

**判断是否值得优化**——投入产出比分析：

1. **是否为耗时热点**：通过 Nsight Systems profiling 确认。经验法则：占总推理时间 > 5-10% 的 kernel 值得投入优化
2. **是否为高频算子**：如果该算子在整网中出现 N 次（如每层一个 LayerNorm = 80 次），单点优化 10% = 整网优化 N×10%
3. **是否有优化空间**：用 NCU 的 SOL（Speed of Light）判断当前实现离硬件上限有多远。如果已达 80%+ SOL 则收益有限

**优化的系统化思路**：

```
Step 1: Profiling 确认瓶颈类型
    ↓
Step 2: 选择对应策略
    ├─ Memory-bound → 减少数据搬运
    │   ├─ 合并访存（确保 warp 连续访问）
    │   ├─ 向量化（float4 加载，4x 事务数减少）
    │   ├─ Shared Memory tiling（减少重复 global 读取）
    │   └─ 算子融合（消除中间 tensor HBM 读写）
    ├─ Compute-bound → 提高计算效率
    │   ├─ Tensor Core（MMA 指令，16x 吞吐）
    │   ├─ 指令级并行（独立指令并行发射）
    │   └─ 减少冗余计算
    └─ Latency-bound → 减少 kernel 开销
        ├─ 算子融合（减少 kernel launch 次数）
        └─ Persistent kernel（一个 kernel 处理多个任务）
    ↓
Step 3: 实施 + 正确性验证 + 性能回归
```

### Q: 如何推进一个还没量化的模型做量化？

**完整的量化落地流程**：

**Phase 1：分析与规划**
1. 确定模型大小和目标硬件（决定可用方案范围）
2. 确定性能目标：需要多少吞吐？延迟 SLA？
3. 确定精度底线：PPL 增加不超过多少？关键 benchmark 下降不超过多少？

**Phase 2：方案选择**

| 目标 | 推荐方案 | 理由 |
|---|---|---|
| 极致速度，精度容忍 | W4A16 (AWQ/GPTQ) | 权重减 4x，decode 加速 2-3x |
| 均衡速度+精度 | W8A8 (SmoothQuant) | INT8 Tensor Core 加速，精度损失小 |
| Hopper 硬件 | FP8 (Transformer Engine) | 原生支持，自动管理 scale |
| 极致压缩（端侧） | W4A8 或 W2 (QuIP#) | 模型极小但精度损失较大 |

**Phase 3：执行量化**
1. 准备校准数据：选取有代表性的 128-512 条数据
2. 运行量化工具：`AutoGPTQ`、`AutoAWQ`、`llm-compressor`
3. 生成量化模型文件

**Phase 4：评估与调优**
1. 精度评估：PPL + 下游 benchmark（MMLU、GSM8K、HumanEval）
2. 性能测试：实际吞吐（tokens/s）和延迟（TTFT/TPOT）
3. 如果精度不达标：尝试增大 group_size、混合精度（敏感层保持高精度）、换更好的量化方法

### Q: AWQ 量化后为什么整体还能加速（明明多了反量化步骤）？

这是一个核心认知——**LLM Decode 阶段是 memory-bound，不是 compute-bound**：

**原因分析**：

Decode 阶段每步只处理 1 个 token（或少量 token），实际执行的是 GEMV（矩阵向量乘）：
```
输出 [1, hidden] = 输入 [1, hidden] × 权重 [hidden, hidden]
```

算术强度 AI = 2×hidden / (hidden×hidden×bytes + hidden×bytes) ≈ 2/hidden ≈ 0.0005
远远小于硬件拐点（~156 for A100），完全是 **memory-bound**！

**关键推理**：
1. **性能瓶颈 = 权重读取带宽**：每步需要从 HBM 读取全部模型权重
2. **INT4 权重减少 4x 数据量**：FP16 每个权重 2 bytes → INT4 每个权重 0.5 bytes
3. **读取时间减少 4x** → 理论加速 4x
4. **反量化开销极小**：dequant（INT4→FP16）是在寄存器内做的简单乘加运算（compute），而算子本身是 memory-bound，compute 资源是闲置的

**打个比方**：你在等货车（内存带宽）送货，货物减少了 4x 所以等待时间减少 4x。虽然收到货后需要额外 1 秒拆包（dequant），但这 1 秒的"计算"完全被等待带宽的时间覆盖。

**融合实现**：AWQ/GPTQ 的推理 kernel 将 dequant 融合在 GEMM 内部——读入 INT4 数据后立即在寄存器做 dequant 再计算，不产生额外的 HBM 访问。

### Q: 为什么不直接用 Nsight/NCU 而要自己写 timing 工具？

**NCU 的局限**：
- **开销巨大**：NCU replay 模式下 kernel 执行速度慢 10-100x，一次完整 profiling 可能需要数十分钟
- **粒度太细**：分析单个 kernel 的每条指令利用率，信息量过大
- **干扰执行**：改变了真实的调度行为（replay + barrier），结果可能与实际运行有偏差
- **不适合日常开发**：每次代码修改后做完整 NCU profiling 太重

**自写 timing 工具的优势**：
- **开销极低**：CUDA event 计时开销 < 1μs，不影响正常执行
- **嵌入开发流程**：每次构建都可以运行，秒级反馈
- **自定义维度**：按算子类型汇总、按层汇总、按请求汇总
- **粗粒度定位**：快速找到耗时 Top-5 的算子类别
- **适合 A/B 对比**：快速比较优化前后的性能变化

**最佳实践——两者结合**：
1. 日常开发用自写 timing 工具快速定位热点（"哪类算子最慢？"）
2. 对热点算子用 NCU 深入分析（"它为什么慢？瓶颈在哪？"）
3. 优化后再用 timing 工具确认整网收益

### Q: 静态图和动态图的区别？为什么要转静态图做优化？

**动态图（Define-by-Run）**——PyTorch 的 eager mode：
- 每次 forward 即时执行每个操作，不构建全局图
- 优点：Python 原生控制流、调试方便（pdb 可断点）、灵活
- 缺点：**无法看到全局结构**，不能做跨 kernel 优化

**静态图（Define-and-Run）**——TorchScript/TensorRT/XLA：
- 先定义/捕获完整计算图，编译优化后再执行
- 优点：全局优化、高效执行、可部署
- 缺点：不灵活、动态控制流需要特殊处理

**为什么推理优化需要静态图**：

| 优化类型 | 为什么需要全局图 |
|---|---|
| 算子融合 | 需要知道两个 kernel 是生产者-消费者关系 |
| 内存规划 | 需要知道所有 tensor 的生命周期才能复用 |
| Layout 优化 | 需要全局分析最优的数据布局 |
| 常量折叠 | 需要识别不依赖输入的子图 |
| 死代码消除 | 需要全局数据流分析 |
| Kernel 调度 | 需要知道执行顺序才能优化 stream 分配 |

**PyTorch 的方案**：
- `torch.compile`（Inductor）：JIT 捕获动态图为静态图 → TorchInductor 编译优化 → 生成高效 Triton/CUDA kernel
- 实际加速通常 1.5-3x（主要来自融合和 Triton kernel）

### Q: vLLM 的核心机制？

vLLM 的核心设计目标：**最大化单位显存下的并发请求数和吞吐量**。

六大核心机制：

1. **PagedAttention**：KV Cache 分页管理（16 tokens/block），消除碎片，利用率 ~100%
2. **Continuous Batching**：每个 iteration 动态调整 batch（加入完成的请求 slot 给新请求），吞吐 2-10x
3. **Prefix Caching**：相同前缀自动共享 KV Cache block（如 system prompt），避免重复 Prefill
4. **Scheduler + Preemption**：
   - 显存不足时按 FCFS 优先级抢占
   - Swap mode：KV Cache 暂存 CPU，恢复时加载回
   - Recompute mode：释放 KV Cache，恢复时重新 Prefill
5. **KV Cache 量化**：支持 FP8 KV Cache，显存减半
6. **Speculative Decoding**：集成投机解码，大小模型协同加速

### Q: 手撕：复制带随机指针的链表？

（编程题）
