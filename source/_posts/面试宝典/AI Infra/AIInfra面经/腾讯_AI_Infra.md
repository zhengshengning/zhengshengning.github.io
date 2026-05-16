---
title: '腾讯 AI Infra'
date: 2026-04-16 12:08:32
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: FP4量化是什么？

**FP4量化**是用4bit浮点数表示模型权重的极端压缩方案。相比INT4（定点），FP4保留了浮点数的**指数-尾数结构**，提供更好的动态范围。

**FP4格式分类：**

| 格式 | 结构 | 可表示值 | 动态范围 | 应用 |
|------|------|---------|---------|------|
| E2M1 | 1sign + 2exp + 1mantissa | 16个值 | ±6 | 通用FP4 |
| E3M0 | 1sign + 3exp + 0mantissa | 16个值 | ±240（宽范围） | 大范围权重 |
| NF4 (NormalFloat4) | 非均匀量化 | 16个值 | 自适应 | QLoRA |

**NF4的特殊设计（QLoRA使用）：**

```
假设权重服从正态分布 N(0, σ²)
NF4的16个量化点选择使得：
- 每个量化区间内的概率密度相等（等概率量化）
- 信息论最优：最小化量化后的信息损失

具体值(归一化后): [-1.0, -0.6962, -0.5251, -0.3949, 
                   -0.2844, -0.1848, -0.0911, 0.0,
                    0.0796, 0.1609, 0.2461, 0.3379,
                    0.4407, 0.5626, 0.7230, 1.0]
                    
对比INT4均匀量化: [-8, -7, -6, ..., 6, 7] 等间距
NF4在0附近更密集(权重集中区域精度更高)
```

**FP4 vs INT4对比：**

| 维度 | FP4 (E2M1) | INT4 | NF4 |
|------|-----------|------|-----|
| 动态范围 | ±6（浮点） | [-8, 7]（定点） | 自适应 |
| 精度分布 | 接近0处更密 | 均匀 | 正态分布最优 |
| 硬件支持 | Blackwell架构 | A100+ Tensor Core | 软件解码 |
| 典型精度损失 | 中等 | 中等 | 最小（对正态分布权重） |
| 计算方式 | 需反量化到FP16再计算 | INT4 Tensor Core直接计算 | 查表反量化 |

**FP4的使用方式：**

```python
# QLoRA中的NF4量化(bitsandbytes库)
import bitsandbytes as bnb
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # 使用NF4格式
    bnb_4bit_compute_dtype=torch.float16,  # 计算时反量化到FP16
    bnb_4bit_use_double_quant=True     # 对scale再量化(节省更多内存)
)
# 模型体积: 70B参数 × 4bit / 8 = ~35GB (原FP16需140GB)
```

**FP4量化的局限性：**
- 精度损失较大：需要配合LoRA等微调恢复精度
- 计算效率依赖反量化：当前硬件多需先解码到FP16再计算（Blackwell原生支持FP4）
- Group-wise scale开销：每64-128个权重需一个FP16 scale值，额外存储~0.5bit/weight

### Q: 量化矩阵乘的维度和流程？

**量化GEMM的完整计算流程：**

对于标准矩阵乘 `Y = X × W` 其中 X=[M, K], W=[K, N], Y=[M, N]：

```
完整流程:
                    ┌─────────────────────────────┐
                    │  离线(PTQ)/训练时(QAT)       │
                    │  W_fp16 → W_int8 + scale_w  │
                    └─────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────┐
│  在线推理:                                                │
│  1. X_fp16 → 在线量化 → X_int8 + scale_x              │
│  2. Y_int32 = X_int8 × W_int8    (INT8 Tensor Core)    │
│  3. Y_fp16 = Y_int32 × scale_x × scale_w  (反量化)     │
└─────────────────────────────────────────────────────────┘
```

**不同量化粒度的Scale维度：**

| 粒度 | Scale维度 | 含义 | 精度 | 计算开销 |
|------|----------|------|------|---------|
| Per-tensor | [1] | 整个张量一个scale | 最差 | 最低 |
| Per-channel(列) | [N] | W的每列一个scale | 好 | 低 |
| Per-token(行) | [M] | X的每行一个scale | 好 | 低 |
| Per-group | [K/G, N] | 每G个元素一组 | 最好 | 较高 |

**Per-group量化详解（最常用）：**

```
W = [K, N], group_size = 128
scale_w 维度: [K/128, N]  每128个连续元素共享一个scale

量化: W_int8[i][j] = round(W[i][j] / scale_w[i//128][j])
反量化: W_approx[i][j] = W_int8[i][j] * scale_w[i//128][j]

对于GEMM Y = X × W:
- 需要在K维度的每128个元素处"分段"计算：
  Y = Σ(X[:, g*128:(g+1)*128] × W_int8[g*128:(g+1)*128, :]) × scale_w[g, :]
```

**W8A8 GEMM（SmoothQuant方式）：**

```
X: [M, K] per-token量化 → scale_x: [M, 1]
W: [K, N] per-channel量化 → scale_w: [1, N]

Y_int32 = matmul_int8(X_int8, W_int8)   // [M, N] INT8 Tensor Core
Y_fp16 = Y_int32 * (scale_x @ scale_w)  // 逐元素乘以scale
       = Y_int32 * scale_x * scale_w^T   // [M, N] * [M,1] * [1,N] 广播
```

**Weight-Only量化GEMM（INT4/INT8权重，FP16激活）：**

```
X: [M, K] FP16 (不量化)
W: [K, N] INT4 → 需要反量化

方法1 - Dequant + FP16 GEMM:
  W_fp16 = dequantize(W_int4, scale_w)  // 反量化到FP16
  Y = X_fp16 × W_fp16                    // FP16 GEMM

方法2 - 融合kernel(如GPTQ/AWQ用的Marlin kernel):
  在GEMM kernel内部，每次从全局内存加载W_int4时
  在寄存器中反量化到FP16，然后做FP16 FMA
  好处: 减少显存带宽(只读INT4)，计算在寄存器中完成
```

**Decode阶段为什么Weight-Only量化有效：**

```
Decode: batch_size=1, Y = x × W, x=[1,K], W=[K,N]
- 计算量: 2KN FLOPs
- 读取权重: KN × bytes_per_weight
- 算术强度 = 2KN / (KN × bpw) = 2/bpw

FP16: 2/2 = 1 FLOP/Byte → 极度memory-bound
INT8: 2/1 = 2 FLOP/Byte → 仍然memory-bound但带宽需求减半
INT4: 2/0.5 = 4 FLOP/Byte → 接近compute-bound

结论: Weight-Only量化直接减少权重读取带宽，decode速度几乎线性提升
```

### Q: 硬件怎么做量化的？量化硬件需要什么？

**Tensor Core的量化计算架构：**

```
              INT8 Tensor Core (A100)
┌──────────────────────────────────────────┐
│  A矩阵(INT8)    B矩阵(INT8)              │
│  [m×k tile]     [k×n tile]               │
│       ↓              ↓                    │
│  ┌─────────────────────────────┐         │
│  │  INT8 × INT8 乘法器阵列      │         │
│  │  (低精度乘法，面积小能效高)    │         │
│  └─────────────┬───────────────┘         │
│                ↓                          │
│  ┌─────────────────────────────┐         │
│  │  INT32 累加器                 │         │
│  │  (高精度累加，避免溢出)        │         │
│  └─────────────┬───────────────┘         │
│                ↓                          │
│  ┌─────────────────────────────┐         │
│  │  反量化单元(可选)             │         │
│  │  INT32 × scale → FP16/FP32  │         │
│  └─────────────────────────────┘         │
└──────────────────────────────────────────┘
```

**各代GPU的量化硬件支持：**

| GPU架构 | 支持精度 | 峰值算力(TOPS/TFLOPS) | 关键特性 |
|--------|---------|---------------------|---------|
| V100 (Volta) | FP16 | 125 TFLOPS(FP16) | 首代Tensor Core |
| A100 (Ampere) | FP16/BF16/INT8/INT4/TF32 | 624 TOPS(INT8) | 支持稀疏(2:4) |
| H100 (Hopper) | 上述 + FP8(E4M3/E5M2) | 3958 TOPS(FP8) | Transformer Engine |
| B200 (Blackwell) | 上述 + FP4 | ~10000 TOPS(FP4) | 原生FP4支持 |

**硬件量化需要的关键组件：**

| 组件 | 功能 | 设计考虑 |
|------|------|---------|
| 低精度乘法器 | INT8×INT8→INT16结果 | 面积小(INT8乘法器=1/4 FP16面积) |
| 高精度累加器 | 防止多次累加溢出 | INT32或FP32，比乘法器宽 |
| 量化单元(Q) | FP16→INT8的在线量化 | 计算scale、round、clamp |
| 反量化单元(DQ) | INT32→FP16/FP32 | 乘以scale，可融合到后处理 |
| Scale缓存 | 存储group/channel的scale | 快速读取，减少带宽开销 |
| 混合精度数据通路 | 不同精度间的数据流 | 避免不必要的精度转换 |

**H100 Transformer Engine的FP8工作流：**

```
Transformer Engine自动管理FP8精度:
1. 每层维护一个amax历史(记录张量最大值)
2. 基于amax动态计算scale: scale = FP8_MAX / amax
3. 前向: 
   - 将激活/权重缩放到FP8范围
   - FP8 Tensor Core计算(E4M3格式)
   - FP32累加
   - 反量化输出为BF16
4. 反向:
   - 梯度用FP8(E5M2格式,更大动态范围)
   - 权重更新保持FP32 master weight

关键: Delayed Scaling
  - 不是每个tensor实时计算scale(需要reduction开销)
  - 用上一步的amax估计当前步的scale
  - 几乎零额外开销
```

**为什么硬件量化能提升性能？**
- **带宽节省**：INT8权重只需FP16一半的HBM带宽
- **计算密度**：相同芯片面积容纳4倍INT8乘法器(相比FP16)
- **能效**：INT8乘法能耗约为FP16的1/4
- **实际加速**：A100上INT8 GEMM比FP16快约2倍(带宽翻倍+算力翻倍)

### Q: 怎么分析系统瓶颈？

**系统性能分析的方法论——从全局到细节：**

**Level 1: 端到端Profiling（Nsight Systems）**

```bash
nsys profile --trace=cuda,nvtx,osrt -o report ./my_app
```

```
时间线视图分析:
─────────────────────────────────────────────────
CPU Thread: [kernel_launch][kernel_launch][      idle     ][kernel_launch]
GPU Stream: [      ][kernel_A][kernel_B][    gap    ][kernel_C]
NCCL:       [                   AllReduce                     ]
─────────────────────────────────────────────────
                                   ^
                       发现: GPU idle等待NCCL通信
                       瓶颈: 通信未与计算overlap
```

**Nsight Systems关注的指标：**

| 指标 | 正常范围 | 异常含义 |
|------|---------|---------|
| GPU Active Time | >90% | 低于70%说明CPU launch或通信有gap |
| Kernel间隔 | <5us | 大间隔说明CPU端有瓶颈 |
| CUDA API耗时 | 很小 | 大量cudaMalloc/cudaSync说明设计问题 |
| CPU-GPU同步 | 尽量少 | 频繁同步破坏流水线 |

**Level 2: 单Kernel分析（Nsight Compute）**

```bash
ncu --set full --target-processes all -o kernel_report ./my_app
```

**Roofline模型判定瓶颈：**

```
算术强度(Arithmetic Intensity) = FLOPs / Bytes_accessed

A100 Roofline:
  峰值算力: 19.5 TFLOPS (FP32)
  峰值带宽: 2039 GB/s (HBM)
  平衡点: 19.5T / 2039G ≈ 9.6 FLOP/Byte

  kernel的AI < 9.6 → Memory-bound (优化访存)
  kernel的AI > 9.6 → Compute-bound (优化计算)
```

**Nsight Compute关键指标解读：**

| 指标 | Memory-bound | Compute-bound | Latency-bound |
|------|-------------|---------------|---------------|
| SM SOL% (Compute) | <30% | >60% | <30% |
| Memory SOL% | >60% | <30% | <30% |
| Achieved Occupancy | 可能高或低 | 通常高 | 低 |
| Warp Stall原因 | Wait(等内存) | 无明显stall | MIO/Barrier |
| L1 Hit Rate | 低(数据未复用) | 不关键 | - |
| 带宽利用率 | 接近峰值 | 远低峰值 | 远低峰值 |

**Level 3: 分布式系统瓶颈**

```
多卡/多节点分析:
1. 通信与计算比例:
   通信时间 / 计算时间 > 1 → 通信瓶颈
   
2. 通信带宽利用率:
   实际带宽 / 理论带宽(NVLink 900GB/s, IB 400Gb/s)
   低于50%可能是消息粒度太小或拓扑不优

3. 负载均衡:
   各rank的kernel执行时间差异
   差异大 → 数据不均或模型分割不均

4. Pipeline bubble:
   bubble_ratio = (p-1) / (p-1+m)
   m(micro-batch数)太少 → bubble大
```

**常见瓶颈及对应解决方案：**

| 瓶颈现象 | 诊断方法 | 解决方案 |
|---------|---------|---------|
| GPU利用率低 | nsys看gap | CUDA Graph/增大batch |
| 单kernel慢(memory) | ncu看带宽SOL | 向量化/共享内存/融合 |
| 单kernel慢(compute) | ncu看算力SOL | Tensor Core/循环展开 |
| 通信阻塞 | nsys看NCCL时间 | Overlap/梯度分桶/TP优化 |
| CPU瓶颈 | nsys看CPU占用 | 异步调度/减少Python开销 |
| 显存不足 | nvidia-smi | 量化/梯度检查点/offload |

### Q: FlashAttention有什么用？矩阵维度推导？多头多batch怎么并行？

**FlashAttention解决的核心问题：**

标准Attention的IO瓶颈：
```
标准实现: Q[N,d] × K^T[d,N] → S[N,N] → softmax → P[N,N] × V[N,d] → O[N,d]
                                  ↑
                        N×N矩阵必须写入HBM再读回
                        seq_len=4096时: 4096² × 2bytes = 32MB / head
                        多头: 32 heads × 32MB = 1GB (仅attention score!)
```

FlashAttention通过**分块计算+Online Softmax**，让中间矩阵S/P从不完整地落地HBM：

```
FlashAttention IO:
  读取: Q(Nd) + K(Nd) + V(Nd) = 3Nd bytes
  写出: O(Nd) bytes
  总IO: O(Nd) —— 与标准的O(Nd + N²)相比，消除了N²项！
  
当N=4096, d=128时:
  标准: 3×4096×128 + 4096² = 1.5M + 16.7M ≈ 18.2M 元素
  Flash: 3×4096×128 + 4096×128 = 2M 元素
  IO减少: ~9x
```

**完整维度推导：**

```
输入:
  Q: [batch_size, num_heads, seq_len, head_dim] = [B, H, N, d]
  K: [batch_size, num_kv_heads, seq_len, head_dim] = [B, H_kv, N, d]
  V: [batch_size, num_kv_heads, seq_len, head_dim] = [B, H_kv, N, d]

中间结果(逻辑上，不实际存储):
  S = Q × K^T: [B, H, N, N]     ← 这是瓶颈！O(N²)显存
  P = softmax(S / √d): [B, H, N, N]

输出:
  O = P × V: [B, H, N, d]

FlashAttention分块:
  Q分块: 每块 B_r × d (B_r ≈ min(M/(4d), d), M为SRAM大小)
  K/V分块: 每块 B_c × d (B_c ≈ min(M/(4d), d))
  块内S: B_r × B_c (在SRAM中计算，不写HBM)
  
  A100: M = 192KB, d = 128
  B_r = B_c ≈ 192KB / (4 × 128 × 2bytes) ≈ 192 tokens
```

**多头多Batch的并行策略：**

```
FlashAttention-1 并行:
  Grid维度: (batch_size × num_heads, num_blocks_N)
  每个thread block处理一个(batch, head)中的一个Q块
  → B×H个独立的attention计算完全并行

FlashAttention-2 改进并行:
  外层循环改为遍历Q块（而非KV块）
  Grid: (batch_size × num_heads, ceil(N/B_r))
  好处: 每个thread block的输出是O的一部分（无需原子操作累加）
  
  Outer loop: for each Q block (并行, 每个分配一个thread block)
    Inner loop: for each KV block (串行, 在SRAM中累加)
      load K_j, V_j to SRAM
      compute S_ij = Q_i × K_j^T (in SRAM)
      online softmax update
      update O_i accumulator
    write O_i to HBM (一次写出)

FlashAttention-3 (H100优化):
  利用H100的异步warpgroup:
  - Warpgroup 0: 做WGMMA计算(Tensor Core)
  - Warpgroup 1: 做TMA加载下一轮数据
  → 计算和数据加载完全overlap
```

**实际性能（A100, FP16, causal）：**

| seq_len | 标准Attention | FlashAttention-2 | 加速比 |
|---------|-------------|------------------|--------|
| 512 | 0.5ms | 0.2ms | 2.5x |
| 2048 | 8ms | 2ms | 4x |
| 8192 | 128ms | 15ms | 8.5x |
| 32768 | OOM | 120ms | - |

### Q: FlashAttention中K的维度包不包含当前Q的那个token？

**对于Causal Attention（自回归LLM），位置i的Q可以attend到位置0到i的所有K/V，即包含自身位置。**

**Causal Mask的含义：**

```
Q位置: 0  1  2  3  4
K位置: 0  1  2  3  4

Mask矩阵(1=可见, 0=不可见):
     K0 K1 K2 K3 K4
Q0 [  1  0  0  0  0 ]   位置0只能看到自己
Q1 [  1  1  0  0  0 ]   位置1看到0和自己
Q2 [  1  1  1  0  0 ]   位置2看到0,1和自己
Q3 [  1  1  1  1  0 ]   ...
Q4 [  1  1  1  1  1 ]   位置4看到所有

未来位置(mask=0)的score设为-inf → softmax后权重为0
```

**FlashAttention中Causal Mask的高效实现：**

```
分块后，Q和KV各自分为块:
Q blocks: [Q0, Q1, Q2, ...]  每块B_r个token
K blocks: [K0, K1, K2, ...]  每块B_c个token

对于Q块i和K块j的交互:
- j > i (K块完全在Q块"右边"): 整块skip，不计算
- j < i (K块完全在Q块"左边"): 整块有效，正常计算
- j == i (对角线块): 需要逐元素应用causal mask

┌────┬────┬────┬────┐
│ △  │ 0  │ 0  │ 0  │  △ = 对角线块(需要mask)
│ ■  │ △  │ 0  │ 0  │  ■ = 完整计算块
│ ■  │ ■  │ △  │ 0  │  0 = 跳过(不计算)
│ ■  │ ■  │ ■  │ △  │
└────┴────┴────┴────┘

优化: 跳过约50%的块 → causal时FlashAttention约快2x
```

**Prefill vs Decode的区别：**
- **Prefill**：一次处理整个序列，Q和K长度相同（N×N的causal矩阵）
- **Decode**：Q只有1个token（当前步），K包含所有历史token+当前token，不需要causal mask（因为Q就是最后一个位置，可以看到所有K）
