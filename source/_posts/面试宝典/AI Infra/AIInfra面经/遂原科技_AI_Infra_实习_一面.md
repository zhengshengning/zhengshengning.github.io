---
title: '遂原科技 AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 量化策略选择：为什么选INT8量化？A100和H100对不同量化的支持？量化模型权重还是KV-Cache？Scale如何选择？

**为什么选INT8量化？**

选择INT8的核心考量是**硬件支持 × 精度损失 × 加速比**的综合平衡：
- A100 Tensor Core INT8吞吐为624 TOPS，是FP16(312 TFLOPS)的**2倍**
- INT8数据量是FP16的一半，memory-bound场景下带宽利用率翻倍
- 大多数模型INT8量化后精度损失<1%（配合合适的量化方案）
- INT8量化工具链成熟（TensorRT、CUTLASS原生支持）

**A100 vs H100 精度支持对比：**

| 精度 | A100峰值算力 | H100峰值算力 | 说明 |
|------|-------------|-------------|------|
| FP32 | 19.5 TFLOPS | 67 TFLOPS | 标准训练精度 |
| TF32 | 156 TFLOPS | 989 TFLOPS | A100默认GEMM精度 |
| FP16/BF16 | 312 TFLOPS | 1979 TFLOPS | 混合精度训练主流 |
| INT8 | 624 TOPS | 3958 TOPS | 推理加速 |
| FP8 (E4M3/E5M2) | 不支持 | 3958 TFLOPS | H100新增，训练+推理 |

**FP8 vs INT8：**
- FP8动态范围更大（E4M3: ±448 vs INT8: [-128,127]），对outlier更友好
- FP8不需要zero-point，量化/反量化更简单
- H100上FP8和INT8吞吐相同，但FP8精度通常更好

**量化权重 vs 量化KV-Cache——不同场景不同策略：**

| 量化对象 | 主要收益 | 适用场景 |
|----------|---------|---------|
| 权重量化(Weight-Only) | 减少模型显存占用和加载带宽 | Decode阶段(memory-bound)显著加速 |
| 激活+权重量化(W8A8) | 计算也能用INT8 Tensor Core | Prefill阶段(compute-bound)加速 |
| KV-Cache量化 | 减少长序列KV存储和读取带宽 | 长上下文推理，减少OOM |

**Scale选择方法：**

| 粒度 | Scale维度 | 精度 | 开销 | 适用 |
|------|-----------|------|------|------|
| Per-tensor | 1个scale | 低 | 最小 | 分布均匀的tensor |
| Per-channel | [N]（每输出通道） | 中 | 小 | 权重量化主流 |
| Per-token | [M]（每token） | 中 | 小 | 激活值量化 |
| Per-group | [K/g, N]（每g个元素） | 高 | 中等 | GPTQ/AWQ使用(g=128) |

Scale确定方法：
- **AbsMax**：scale = max(|x|) / 127，简单但对outlier敏感
- **Percentile**：取99.99%分位数作为max，截断极端outlier
- **KL散度校准**：TensorRT方法，搜索使量化后分布与原始分布KL散度最小的阈值
- **学习scale(LSQ)**：QAT中将scale作为可学习参数

### Q: Triton算子实现逻辑和分块策略？

Triton是OpenAI开发的GPU编程语言，以**block（数据块）**为基本计算粒度，介于CUDA（线程级）和高层框架（算子级）之间。

**Triton编程模型核心概念：**

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 确定当前program处理的数据范围
    pid = tl.program_id(0)              # 类似CUDA的blockIdx
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 数据偏移
    mask = offsets < n_elements         # 边界保护
    
    # 2. 从HBM加载数据到SRAM（共享内存/寄存器）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 3. 在SRAM中完成计算
    result = x + y
    
    # 4. 写回HBM
    tl.store(out_ptr + offsets, result, mask=mask)
```

**Triton vs CUDA对比：**

| 特性 | Triton | CUDA |
|------|--------|------|
| 编程粒度 | Block（一组数据） | Thread（单个线程） |
| 共享内存 | 自动管理 | 手动__shared__ |
| 合并访存 | 自动优化 | 手动确保coalesced |
| Warp调度 | 编译器处理 | 需理解warp行为 |
| 代码量 | CUDA的1/3~1/5 | 完整控制，代码多 |
| 性能上限 | 接近手写CUDA(90%+) | 理论上限(100%) |

**分块策略考量：**

1. **BLOCK_SIZE选择影响：**
   - 太小：并行度不足，SM利用率低
   - 太大：寄存器/共享内存溢出，降低occupancy
   - 经验值：BLOCK_SIZE = 512~2048（element-wise），64~128（矩阵维度tiling）

2. **Auto-tuning机制：**
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}),
    ],
    key=['M', 'N', 'K']  # 根据问题规模选择最优配置
)
def matmul_kernel(...):
    ...
```

3. **L2 Cache命中率优化：**
   - Program的启动顺序影响L2 cache重用
   - Swizzle/reorder program_id使相邻program访问相邻数据

### Q: Decode阶段是compute bound还是memory bound？KV-Cache量化提升的是什么？

**Decode阶段是典型的memory-bound。**

**原因分析：**

Decode时每次只生成1个token，batch大小通常较小。以一次Attention计算为例：
- 数据量：需要读取整个KV-Cache，大小为 seq_len × d_head × 2（K和V）× num_layers
- 计算量：新token的Q与所有K做点积 = seq_len × d_head次FMA

**算术强度（Arithmetic Intensity）计算：**
```
FLOPs = 2 × seq_len × d_head（Q·K^T的计算量）
Bytes = seq_len × d_head × 2 bytes（读取KV，FP16）+ d_head × 2 bytes（读Q）
AI = FLOPs / Bytes ≈ 2 / 2 = 1 FLOP/byte

A100 Roofline拐点 = 312 TFLOPS / 2 TB/s ≈ 156 FLOPs/byte
```

实际AI(~1)远小于Roofline拐点(~156)，确认为**严重的memory-bound**。

**KV-Cache量化提升的本质——降低带宽需求：**

| 量化精度 | 每token KV大小 | 读取带宽需求 | 相对加速 |
|----------|---------------|-------------|---------|
| FP16 | 2 bytes/element | 基准 | 1x |
| INT8 | 1 byte/element | 减半 | ~2x |
| INT4 | 0.5 bytes/element | 1/4 | ~4x |

**KV-Cache量化的额外好处：**
- 显存占用减少 → 可以支持更长的上下文或更大的batch
- 更大的batch → 提高算术强度 → 更好的GPU利用率
- 更多请求可以同时服务 → 提高系统吞吐

### Q: A100的理论带宽上限？

**A100各级内存/互联带宽：**

| 层级 | 带宽 | 大小 | 延迟 |
|------|------|------|------|
| 寄存器 | - | 每SM 256KB(65536×32bit) | ~1 cycle |
| 共享内存/L1 | ~19 TB/s (全芯片) | 192KB/SM | ~5ns(~28 cycles) |
| L2 Cache | ~6-8 TB/s | 40MB | ~50ns |
| HBM2e (SXM版) | **2039 GB/s** | 80GB | ~400ns |
| HBM2e (PCIe版) | 1555 GB/s | 40GB | ~400ns |
| NVLink 3.0 (单卡双向) | 600 GB/s | - | ~1us |
| PCIe 4.0 x16 | 32 GB/s (单向) | - | ~1us |

**实际可达带宽 vs 理论带宽：**
- HBM实际可达：理论值的80-90%（~1600-1800 GB/s）
- 影响因素：访存模式（合并vs分散）、TLB miss、ECC开销
- Benchmark工具：`bandwidthTest`(CUDA samples)、`ncu --set full`中的Memory Throughput

**与H100对比：**

| 指标 | A100 SXM | H100 SXM |
|------|----------|----------|
| HBM带宽 | 2039 GB/s (HBM2e) | 3350 GB/s (HBM3) |
| L2 Cache | 40MB | 50MB |
| NVLink | 600 GB/s (12x NVLink3) | 900 GB/s (18x NVLink4) |
| SM数量 | 108 | 132 |

### Q: 有没有想过用CUDA开发算子？为什么使用Triton？

**选择Triton而非CUDA的核心考量——开发效率与性能的权衡：**

| 维度 | Triton | CUDA |
|------|--------|------|
| 代码量 | FlashAttention ~200行 | FlashAttention ~2000行 |
| 开发周期 | 数天 | 数周~数月 |
| 调试难度 | Python级调试+print | printf/Nsight Compute |
| 性能达成 | 手写CUDA的85-95% | 理论上限100% |
| 自动优化 | 共享内存、合并访存、warp调度 | 全手动 |
| Auto-tuning | 内置grid search | 需自己实现 |
| 可维护性 | 高（Python代码可读性好） | 低（大量底层细节） |
| 与PyTorch集成 | `torch.compile`原生支持 | 需要C++/pybind |

**CUDA仍然必要的场景：**
- 需要使用Tensor Core的复杂数据布局（如Cutlass的warp-level MMA）
- 需要极致的寄存器/共享内存控制（如FlashAttention的最新版本）
- 需要异步拷贝(cp.async)等新硬件特性
- 性能要求达到硬件极限（如cuBLAS-level GEMM）
- 需要利用PTX/SASS级别指令

**Triton的性能实例：**
- Triton实现的FlashAttention达到手写CUDA版本90-95%的性能
- Triton实现的FusedAttention已被PyTorch torch.compile采用
- 对于element-wise和reduce类kernel，Triton通常能达到理论峰值带宽

### Q: 有没有做过profile？测出来性能指标如何？后续优化思路？

**常用Profiling工具：**

| 工具 | 用途 | 关键输出 |
|------|------|---------|
| Nsight Systems | 全局时间线 | kernel耗时、CPU-GPU同步、通信等待 |
| Nsight Compute | 单kernel深度分析 | SOL%、occupancy、stall原因、roofline |
| torch.profiler | PyTorch级别 | 算子耗时排名、GPU利用率 |
| nvprof/ncu CLI | 命令行profiling | 快速获取关键指标 |

**Nsight Compute关键指标解读：**

```
Compute (SM) Throughput:     75% SOL  → 计算利用率
Memory Throughput:           90% SOL  → 带宽利用率 → memory-bound!
Achieved Occupancy:          45%      → 较好（不一定越高越好）
Warp Stall Reasons:
  - Wait: 30%               → 等待数据从内存返回
  - Instruction Fetch: 5%   → 指令cache miss
  - Short Scoreboard: 10%   → 等待前序指令完成
```

**后续优化决策树：**

```
Memory-bound (Memory SOL% >> Compute SOL%):
  ├── 检查合并访问率 → 不合并则调整数据布局
  ├── 检查向量化程度 → 用float4/LDS.128
  ├── 检查数据复用 → 增大tile/使用共享内存
  └── 检查冗余读写 → 算子融合

Compute-bound (Compute SOL% >> Memory SOL%):
  ├── 检查Tensor Core使用 → 使用wmma/mma指令
  ├── 检查ILP → 循环展开、指令重排
  └── 检查分支发散 → 消除warp divergence

Latency-bound (两者都低):
  ├── 增大block size → 更多active warp
  ├── 减少同步点 → 去掉不必要的__syncthreads
  └── 使用异步操作 → cp.async预取
```

**优化目标参考：**
- Memory-bound kernel → 目标达到HBM带宽的80%+（~1600 GB/s on A100）
- Compute-bound kernel → 目标达到算力的70%+（~220 TFLOPS FP16 on A100）
