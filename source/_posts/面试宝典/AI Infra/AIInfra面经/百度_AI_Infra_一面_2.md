---
title: '百度 AI Infra 一面 (2)'
date: 2026-04-16 12:15:51
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: CUDA GEMM优化的关键技术？

CUDA 上的高性能 GEMM（通用矩阵乘法）是所有深度学习加速的基础，优化层次从宏观到微观：

1. **分块（Tiling）**：将大矩阵分为 tile（如 128x128），每个 Thread Block 负责一个输出 tile。Tile 大小需匹配 Shared Memory 容量（A100: 192KB 可配）。

2. **寄存器级分块**：每个线程负责计算 8x8 或 16x8 的小块结果，用寄存器存储累加器。这提高了数据复用率（每次从 Shared Memory 读入的数据被多次使用）。

3. **数据预取（Double/Triple Buffering）**：在计算当前 K-tile 的同时，异步加载下一个 K-tile 到 Shared Memory 的另一个 buffer。实现计算和访存的流水线重叠，隐藏 Global Memory 延迟。

4. **向量化加载**：使用 `float4`（128bit）或 `LDG.128` 一次加载 4 个 float，减少内存事务数量。要求源地址 16B 对齐。

5. **避免 Bank Conflict**：Shared Memory 有 32 个 bank。通过 padding（如 `float smem[128][129]`）打破规律性冲突模式，保证 warp 内线程无冲突访问。

6. **利用 Tensor Core**：通过 WMMA（Warp Matrix Multiply-Accumulate）或 MMA PTX 指令调用 Tensor Core。A100 的 FP16 Tensor Core 吞吐是 CUDA Core 的 16 倍。需要特定的数据布局（如 HMMA 16x16x16 格式）。

**典型性能**：优化后的 GEMM kernel 可达硬件峰值 TFLOPS 的 80-95%。CUTLASS 是 NVIDIA 开源的 GEMM 模板库，提供了这些优化的系统化实现。

### Q: 手撕：CUDA向量加法？

（编程题）

### Q: LLM推理部署优化技术有哪些？

系统化的推理优化技术栈（从模型层到系统层）：

**模型压缩**：
1. **量化**：W4A16（4-bit 权重 + FP16 计算，GPTQ/AWQ）或 W8A8（INT8 权重和激活）。模型体积缩 2-8x，推理提速 2-4x。
2. **剪枝/蒸馏**：结构化剪枝去除冗余 head/层，知识蒸馏到小模型。

**注意力优化**：
3. **KV Cache**：缓存历史 Key/Value 避免 O(N) 重复计算。
4. **FlashAttention + Flash Decoding**：Prefill 用 FlashAttention（IO优化），Decode 用 Flash Decoding（KV 维度并行）。
5. **PagedAttention**：分页管理 KV Cache 消除碎片（vLLM）。

**服务优化**：
6. **Continuous Batching**：不同请求动态组 batch，短请求完成后立即替换新请求，不等长请求。
7. **投机解码**：小模型快速草拟 + 大模型并行验证，在不改变输出分布的前提下加速 2-3x。
8. **模型并行**：TP（节点内切分权重）+ PP（多节点流水线）降低单卡显存需求。

**系统优化**：
9. **算子融合 + CUDA Graph**：减少 kernel launch 开销和中间内存读写。
10. **Prefix Caching**：共享相同 system prompt 的 KV Cache，避免重复 prefill。

### Q: 模型量化中Float32和INT8怎么相互转换？

**量化（FP32 -> INT8）**：

对称量化（以 0 为中心）：
```
scale = max(|fp32_values|) / 127
int8_val = round(fp32_val / scale)  # clamp到[-128, 127]
```

非对称量化（适应偏移分布）：
```
scale = (max - min) / 255
zero_point = round(-min / scale)  # 将实际0映射到的INT8值
int8_val = round(fp32_val / scale) + zero_point  # clamp到[0, 255]
```

**反量化（INT8 -> FP32）**：
```
对称：fp32_val = int8_val * scale
非对称：fp32_val = (int8_val - zero_point) * scale
```

**关键考量**：
- 对称量化实现简单（无 zero_point），但如果数据分布不对称会浪费量化范围
- 非对称量化更精确但计算稍复杂（多一次减法）
- 权重通常对称（分布接近 0 中心），激活可能需要非对称（如 ReLU 后全正）
- scale 的确定是核心问题：MinMax（简单但受 outlier 影响）、Percentile（99.9%分位数）、MSE 最小化、KL 散度校准（TensorRT 方法）

### Q: 手撕：最长公共子序列？

（编程题）
