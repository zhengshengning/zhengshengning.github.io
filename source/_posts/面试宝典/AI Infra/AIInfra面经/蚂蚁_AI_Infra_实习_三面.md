---
title: '蚂蚁 AI Infra 实习 三面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: W4A16 量化是什么？量化流程是怎样的？

**W4A16 含义**：权重（Weight）量化为 INT4，激活（Activation）保持 FP16 不量化。这是当前 LLM 推理最流行的量化配置。

**为什么 W4A16 流行**：
- LLM Decode 阶段是 memory-bound（每步读取全部权重，但只计算 1 个 token）
- 权重从 FP16（2B）→ INT4（0.5B），带宽需求减少 4x → 理论加速 4x
- 激活保持高精度，避免累积误差

**量化流程**：
1. **选择粒度**：per-channel 或 per-group（group_size=128 最常见）
   - per-group：每 128 个权重共享一个 scale+zero_point
   - 粒度越细精度越好，但 scale 存储开销越大
2. **统计分布**：用校准数据运行模型，统计每组权重的 min/max
3. **计算量化参数**：
   ```
   scale = (max - min) / (2⁴ - 1)  # = (max-min) / 15
   zero_point = round(-min / scale)
   ```
4. **量化**：`q = clamp(round(w / scale + zero_point), 0, 15)`
5. **存储**：INT4 权重 + FP16 scale/zero_point（每 128 个权重存一组）

**推理时的执行**：
```
load INT4 权重 (0.5B/element) → 寄存器内 dequant 为 FP16 → 与 FP16 激活做 GEMM
```
dequant 与 GEMM 融合在同一个 kernel 中，不产生额外 HBM 读写。

**精度评估**：
- PPL 增加 0.3-1.0（取决于模型和方法）
- 下游任务（MMLU/GSM8K）通常下降 1-3%
- AWQ/GPTQ 是主流 W4 量化方法，精度接近 FP16

### Q: INT8 和 FP8 量化方案有什么区别？

| 维度 | INT8 | FP8 (E4M3/E5M2) |
|---|---|---|
| 量化方式 | **均匀量化**（等间距 bin） | **非均匀量化**（对数间距） |
| 参数 | scale + zero_point（对称量化无 zp） | 仅 scale |
| 数据分布适配 | 均匀分布最优 | 正态/对数正态分布更优 |
| 硬件支持 | 所有主流 GPU（Turing+）| Hopper（H100）原生支持 |
| Tensor Core | INT8 IMMA（A100: 624 TOPS） | FP8 TC（H100: 1978 TFLOPS） |
| 典型精度损失 | PPL +0.1-0.3 | PPL +0.05-0.15 |
| 管理复杂度 | 需管理 zero_point | 无 zero_point，更简单 |

**为什么 FP8 适合训练**：
- 梯度分布跨度大（某些层梯度很大，某些很小），FP8 的浮点表示自然适配
- E5M2 动态范围达 ±57344，不容易溢出
- INT8 的 [-128, 127] 范围对梯度可能不够

**为什么 INT8 仍然重要**：
- 硬件覆盖广（A100/T4/V100 都支持 INT8）
- SmoothQuant 的 W8A8 配置可以利用 INT8 Tensor Core 做整数 GEMM，吞吐高

### Q: KV Cache 为什么需要？怎么计算大小？

**为什么需要 KV Cache**：

自回归生成中，生成第 t 个 token 时，Attention 需要用当前 token 的 Q 与所有历史 token（0 到 t-1）的 K/V 做计算：
```
Attention_t = softmax(q_t × [k_0, k_1, ..., k_{t-1}]^T) × [v_0, v_1, ..., v_{t-1}]
```

**不缓存的后果**：
- 每生成一个新 token，需要对所有历史 token 重新执行 Embedding + 前 N 层的前向得到 K/V
- 生成长度 T 的序列：总计算量 ∝ O(T²)（每步重算历史所有 token）

**缓存后**：
- 已计算过的 K/V 保存在显存中
- 每步只需计算新 token 的 K/V，追加到缓存
- 总计算量 ∝ O(T)（每步只算 1 个 token）

**大小计算公式**：
```
KV Cache = 2(K和V) × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes
```

**实际计算示例**：

| 模型 | 配置 | KV Cache |
|---|---|---|
| LLaMA-8B, FP16, bs=1, seq=2048 | 2×32×8×128×2048×1×2 | **256 MB** |
| LLaMA-8B, FP16, bs=32, seq=4096 | 2×32×8×128×4096×32×2 | **16 GB** |
| LLaMA-70B, FP16, bs=1, seq=4096 | 2×80×8×128×4096×1×2 | **1 GB** |
| LLaMA-70B, FP8, bs=16, seq=4096 | 2×80×8×128×4096×16×1 | **8 GB** |

可见大 batch + 长序列时 KV Cache 很快占满显存——这就是 PagedAttention 和 KV Cache 压缩的必要性。

### Q: Attention 为什么比权重更难量化？

**四个核心原因**：

1. **动态范围大且输入依赖**：
   - 权重是静态的，分布在训练后固定，可以离线精确校准 scale
   - Attention score（QK^T）在不同输入、不同位置差异巨大（有些位置 score=30，有些=-5）
   - 无法用一个固定 scale 覆盖所有输入

2. **Softmax 的放大效应**：
   - Softmax 对输入的微小扰动**非线性放大**
   - 量化误差 δ 经过 softmax：`softmax(x+δ) ≠ softmax(x) + 小量`
   - 尤其当某个 score 远大于其他值时（winner-take-all），微小扰动可能改变 attention 分配

3. **异常值问题**：
   - 某些 token（如序列开头、标点符号）的 attention score 远大于其他值
   - 这些 outlier 把量化范围撑大，正常值被压缩到很少的 bin 中

4. **误差累积**：
   - Attention 在每层都做一次，量化误差逐层累积
   - 后续层的 Q/K 基于前面层（可能已有误差）的输出生成，误差放大

**业界实践**：
- 权重和 KV Cache 优先量化（静态、分布稳定）
- Attention score 计算保持 FP16/FP32
- 如需量化 Attention：使用 per-token/per-head 的动态 scale

### Q: 如何判断一个算子是 latency-bound、memory-bound 还是 compute-bound？

**使用 Roofline 模型**做系统化判断：

**Step 1：计算算术强度（Arithmetic Intensity, AI）**
```
AI = FLOPs / Bytes_accessed（访存量，包括读+写）
单位：FLOP/Byte
```

**Step 2：对比硬件拐点**
```
拐点 = Peak_Compute / Peak_Bandwidth
A100: 312 TFLOPS / 2.0 TB/s = 156 FLOP/Byte
H100: 989 TFLOPS / 3.35 TB/s = 295 FLOP/Byte
```

**Step 3：判断**

| 条件 | 瓶颈类型 | 含义 | 优化方向 |
|---|---|---|---|
| AI < 拐点 | **Memory-bound** | 数据搬运跟不上计算 | 减少访存、提高带宽利用率 |
| AI > 拐点 | **Compute-bound** | 计算单元是瓶颈 | Tensor Core、减少冗余计算 |
| Kernel 极短（<10μs） | **Latency-bound** | Launch 和调度开销主导 | 算子融合 |

**常见算子的类型**：
- GEMM (大矩阵)：compute-bound（AI ≈ M·N·K/(M·K+K·N+M·N) ≈ min(M,N)/2）
- GEMV (decode)：memory-bound（AI ≈ 2，远小于拐点）
- Softmax：memory-bound（AI ≈ 5-10 FLOPs/元素 vs 2×8 bytes 读写）
- LayerNorm：memory-bound（AI ≈ 5）
- Elementwise (ReLU/Add)：memory-bound（AI = 1-2）

**NCU 快速判断**：SOL（Speed of Light）面板直接显示 Memory Throughput % 和 Compute Throughput %，哪个高就是哪个 bound。
