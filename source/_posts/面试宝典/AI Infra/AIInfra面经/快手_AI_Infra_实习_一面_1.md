---
title: '快手 AI Infra 实习 一面 (1)'
date: 2026-04-16 12:06:53
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 介绍模型量化和 FlashAttention，实际优化效果如何？

**模型量化**：

将模型权重/激活从高精度（FP32/FP16）压缩到低精度（INT8/INT4/FP8），核心收益来自两方面：

1. **减少显存占用**：8B 模型 FP16 需 16GB，INT4 只需 4GB
2. **减少内存带宽消耗**：LLM Decode 阶段是 memory-bound（每个 token 需读取全部权重），带宽减少直接转化为加速

**常见方案和效果**：

| 方案 | 类型 | 加速比 | 精度损失 |
|---|---|---|---|
| W8A8 (SmoothQuant) | PTQ | 1.5-2x | PPL +0.1-0.3 |
| W4A16 (GPTQ/AWQ) | PTQ | 2-3x | PPL +0.3-1.0 |
| FP8 (E4M3) | PTQ | ~2x (Hopper) | PPL +0.05-0.1 |
| QAT | 训练时 | 类似 PTQ | 损失更小 |

**为什么量化能加速 LLM**：Decode 阶段每 token 只做 batch=1 的 GEMV，算术强度极低（< 1 FLOP/Byte），完全 memory-bound。INT4 权重减少 4x 数据量 = 理论 4x 加速（实际 2-3x，因为有 dequant 开销）。

---

**FlashAttention**：

通过 tiling 将 Attention 计算分块在 GPU SRAM（~20MB）中完成，核心创新是 Online Softmax 实现精确分块计算。

**实际效果**：
- **训练端**：前向+反向加速 1.5-3x（seq_len 越长加速越明显）
- **推理端**：Prefill 阶段加速 2-4x（大 batch/长序列场景）
- **显存节省**：O(N²) → O(N)，seq_len=4096 时单层 attention 节省约 32MB/head
- **Decode 阶段**：加速不明显（已经是 seq=1 的 GEMV，不存在 N×N 矩阵）

**FlashAttention 的局限**：
- 对短序列加速不大（overhead 抵消收益）
- 需要 head_dim 为特定值（64/128/256）
- 不支持所有 attention mask 模式（需要特殊适配）

---

### Q: 介绍 TensorRT，底层如何加速？

TensorRT 是 NVIDIA 的推理优化引擎，将训练好的模型转换为高效的 inference engine。底层加速机制：

**1. 图优化（Graph Optimization）**：
- **层融合**：将 Conv+BN+ReLU 融合为一个 kernel（减少 3 次 kernel launch + 2 次中间 tensor 读写）
- **常量折叠**：编译期计算固定输入的子图
- **Dead Code Elimination**：移除无用计算分支
- **Transpose/Reshape 消除**：通过调整 kernel 内部 layout 消除显式 transpose

**2. 量化（Quantization）**：
- 支持 INT8/FP8 量化，利用校准数据（calibration dataset）确定每层的 scale
- 校准策略：MinMax、Entropy（KL 散度最小化）、Percentile
- 量化感知：自动识别对量化敏感的层保持高精度

**3. Kernel Auto-Tuning**：
- 对每个层/融合后的子图，**枚举所有可能的 kernel 实现**并实际运行 benchmark
- 针对具体硬件（SM 数量、cache 大小）和输入形状选择最优 kernel
- 这就是为什么 TensorRT build engine 很慢（在做穷举搜索）

**4. 内存优化**：
- Tensor 复用：生命周期不重叠的 tensor 共享同一块显存
- 显存池化：减少 cudaMalloc/cudaFree 的调用
- Workspace 管理：为临时计算分配高效的工作区

**5. 精度混合（Mixed Precision）**：
- 不同层根据敏感度使用不同精度
- 对精度敏感的层保持 FP16/FP32，不敏感的层用 INT8
- 自动寻找精度-性能的最优 pareto 点

**6. 动态 Shape 支持**：
- 通过 Optimization Profile 定义 min/opt/max 输入尺寸
- 为最常见的 opt 尺寸做最深度优化
- 运行时根据实际输入在已优化的 kernel 中选择

---

### Q: 介绍 vLLM 框架和 PagedAttention？

**vLLM** 是面向 LLM 推理的高吞吐引擎，核心设计目标是最大化单 GPU 上能同时服务的请求数：

**核心特性**：
- **PagedAttention**：KV Cache 分页管理，将显存利用率从 20-40% 提升到接近 100%
- **Continuous Batching**：每个 forward iteration 动态加入/移除请求（比 static batching 吞吐高 2-10x）
- **Prefix Caching**：相同 system prompt 的请求自动共享 KV Cache
- **Scheduler + Preemption**：显存不足时将低优先级请求的 KV Cache swap 到 CPU

**PagedAttention 详解**：

传统 KV Cache 管理为每个请求预分配 max_seq_len 的连续显存——导致 60-80% 的显存浪费（内部碎片 + 外部碎片）。

PagedAttention 的设计：
```
物理显存被划分为固定大小 Block（如 16 tokens 的 KV）
每个请求维护 Block Table（逻辑 block → 物理 block 映射）

逻辑序列: [token_0...token_15][token_16...token_31][token_32...]
                    ↓                    ↓                 ↓
Block Table:    phys_block_7       phys_block_2       phys_block_15
```

**PagedAttention 的关键优势**：
- **无碎片**：按需分配，用多少占多少。不需要预留 max_seq_len 的连续空间
- **按需增长**：序列每生成 16 个 token 才分配一个新 block
- **Copy-on-Write 共享**：Beam search 中多个 beam 共享公共前缀的 block，只在产生分歧时复制
- **利用率**：论文报告从 20-40% → 接近 100%，意味着同等显存可以服务 2-4x 更多的并发请求

**Attention Kernel 适配**：原始 attention kernel 假设 KV 是连续存储的；PagedAttention kernel 需要根据 block table 做间接寻址，获取分散的 KV block 做计算。性能开销约 3-5%，但整体吞吐因 batch 增大而大幅提升。

---

### Q: Qwen 模型占多少内存？如何部署？

以 Qwen-7B（FP16）为例进行估算：

**显存占用**：
- 模型参数：7B × 2 bytes = **14 GB**
- KV Cache（batch=1, seq=2048, 32层, 32 heads, 128 dim）：2×32×32×128×2048×2 ≈ 1 GB
- 激活值 + 框架开销：~2-4 GB
- 总计约 **18-20 GB**

**常见部署方式**：

| 场景 | 方案 | 显存需求 | 适合硬件 |
|---|---|---|---|
| 低资源部署 | INT4 量化 (GPTQ/AWQ) | ~5 GB | RTX 3060/4060 |
| 标准推理 | FP16 + vLLM | ~20 GB | RTX 4090/A10 |
| 高吞吐生产 | FP8 + TensorRT-LLM | ~10 GB | H100/L40S |
| 超大模型(72B) | TP=4 + FP16 | ~40 GB/卡 | 4×A100 |

**部署最佳实践**：
- **单卡 7B**：vLLM + AWQ INT4 量化，单张 RTX 4090 可跑，吞吐约 30-50 tokens/s
- **生产环境**：vLLM/TensorRT-LLM + Continuous Batching，关注 QPS 和 P99 延迟
- **多卡大模型**：Tensor Parallel 切分（如 72B 用 4-8 卡 TP）

---

### Q: PyTorch 核心基础功能是什么？如何对 GPU 进行管理？

**PyTorch 核心功能**：

1. **动态计算图（Define-by-Run）**：每次 forward 都重新构建图，支持 Python 原生控制流（if/for），调试直观
2. **自动微分（Autograd）**：记录前向操作形成 DAG，反向时自动计算梯度。`tensor.backward()` 触发
3. **Tensor 运算**：N-维数组 + 丰富的数学操作，API 类似 NumPy 但支持 GPU 加速
4. **nn.Module**：模块化网络构建，支持嵌套、参数管理、序列化
5. **多后端**：CPU/CUDA/MPS/XPU，通过 dispatcher 统一路由

**GPU 管理机制**：

**设备管理**：
```python
torch.cuda.set_device(0)            # 设置默认设备
torch.cuda.device_count()           # 可用 GPU 数
torch.cuda.current_device()         # 当前设备
```

**显存管理（Caching Allocator）**：
- PyTorch 使用**缓存分配器**：首次 cudaMalloc 预分配大块显存（如 512MB），后续 tensor 分配从缓存中取 block
- 好处：避免频繁 cudaMalloc/cudaFree（每次 ~1ms），分配降至 ~1μs
- `torch.cuda.memory_allocated()`：当前被 tensor 占用的显存
- `torch.cuda.memory_reserved()`：被 PyTorch 缓存占用的总显存（包含未使用的缓存 block）
- `torch.cuda.empty_cache()`：释放未被 tensor 占用的缓存（不释放活跃 tensor 的显存）
- OOM 时 PyTorch 会自动尝试 empty_cache 并重试分配

**CUDA Stream 管理**：
```python
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    # 在自定义 stream 上异步执行
    output = model(input)
torch.cuda.current_stream().wait_stream(stream)  # 同步
```

---

### Q: 模型训练和推理在资源消耗上有什么区别？训练有哪些性能优化手段？

**资源消耗对比**：

| 维度 | 训练 | 推理 |
|---|---|---|
| 显存 | 参数 + 梯度 + 优化器状态 + 激活值 ≈ 16-20x 参数量 | 参数 + KV Cache + 少量激活 ≈ 2-4x 参数量 |
| 计算量 | 前向 + 反向 ≈ 3x 前向 FLOPs | 仅前向，Decode 时每步只算 1 token |
| 通信 | 梯度 AllReduce（多卡必需）| TP 的 AllReduce（可选）|
| 带宽需求 | 相对均衡（大 batch GEMM 是 compute-bound）| Decode 强 memory-bound |

**具体对比**（8B 模型，FP16）：
- 训练每卡显存：参数 16GB + 梯度 16GB + Adam 48GB + 激活 ~20GB = **~100 GB**（需要 ZeRO 分片）
- 推理每卡显存：参数 16GB + KV Cache 1-4GB + 开销 2GB = **~20 GB**

**训练性能优化手段**：

1. **混合精度训练（AMP）**：前向反向用 BF16/FP16，主权重用 FP32。计算量减半 + 显存减半 + 带宽减半
2. **梯度累积**：小 batch_size 多步累积梯度再更新，模拟大 batch 但不增加显存
3. **ZeRO 显存优化**：分片优化器状态（ZeRO-1）→ 梯度（ZeRO-2）→ 参数（ZeRO-3），极大突破单卡显存限制
4. **3D 并行**：DP（数据）+ TP（张量）+ PP（流水线），大规模集群的标准方案
5. **梯度 Checkpoint（激活重算）**：不保存中间激活，反向时重新前向计算。时间增加 ~33%，显存可节省 60-70%
6. **FlashAttention**：Attention 显存从 O(N²) 降至 O(N)，同时加速 1.5-3x
7. **计算通信重叠**：反向传播时边算梯度边 AllReduce，通信完全隐藏在计算背后

---

### Q: GPU 基础的物理执行单元是什么？

GPU 的基本物理执行单元是 **SM（Streaming Multiprocessor）**。以 A100 为例（108 个 SM），每个 SM 是一个独立的计算核心，包含：

**计算资源**：
- **CUDA Core**（FP32/INT32）：64 个/SM，执行标量浮点/整数运算
- **Tensor Core**（第三代）：4 个/SM，执行 4×4×4 的矩阵乘加运算（FP16/BF16/TF32/INT8）
- **FP64 Core**：32 个/SM（HPC 用）
- **SFU**（Special Function Unit）：16 个/SM，执行 sin/cos/exp/sqrt 等超越函数

**存储资源**：
- **寄存器文件**：65536 个 32-bit 寄存器/SM（总 256KB/SM）
- **Shared Memory/L1 Cache**：192KB/SM（可配置 shared:L1 比例）
- **L2 Cache**：40MB（全 GPU 共享）

**调度单元**：
- **Warp Scheduler**：4 个/SM，每 cycle 各选一个就绪 warp 发射指令
- **Dispatch Unit**：将指令分发到相应执行单元

**执行层次**（从小到大）：
```
Thread（标量）→ Warp（32 线程，SIMT 执行）→ Thread Block（多个 Warp，共享 Shared Memory）→ Grid（所有 Block）
```

**关键特性**：
- Warp 是最小调度单位（32 线程锁步执行同一指令）
- 一个 SM 可同时驻留多个 Block（受寄存器/shared memory 限制）
- 通过大量 Warp 切换隐藏内存延迟（零开销上下文切换）

---

### Q: 手撕：将有序数组转化为平衡二叉搜索树？

（编程题）
