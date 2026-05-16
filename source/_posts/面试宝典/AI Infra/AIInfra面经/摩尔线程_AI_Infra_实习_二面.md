---
title: '摩尔线程 AI Infra 实习 二面'
date: 2026-04-16 12:10:06
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: INT8 和 W4A16 的量化流程和应用场景？

**INT8（W8A8）量化——全面量化方案**：

```
量化流程:
  1. 准备校准数据（128-512 条代表性输入）
  2. 前向推理收集统计信息:
     - 权重: per-channel min/max（静态，只需算一次）
     - 激活: per-tensor 或 per-token min/max（动态，每次推理计算）
  3. 计算量化参数:
     - 对称量化: scale = max(|x|) / 127
     - 非对称量化: scale = (max - min) / 255, zp = round(-min / scale)
  4. 量化: q = clamp(round(x / scale), -128, 127)
  5. 推理执行: INT8 矩阵乘（Tensor Core 加速）→ FP32 累加 → rescale → 输出
```

**W8A8 的核心挑战——激活 outlier**：
```
问题: 激活中某些 channel 的值远大于其他（如最大值 100，其余 < 1）
     → per-tensor scale 被 outlier 撑大 → 正常值量化精度极低

SmoothQuant 解决方案:
  将激活的难度"转移"给权重:
    Y = (X × diag(s)^{-1}) × (diag(s) × W)  
        ↑ 激活除以 s（缩小 outlier）  ↑ 权重乘以 s（补偿）
  
  s 的选择: s_j = max(|X_j|)^α / max(|W_j|)^{1-α}, α∈[0.5, 0.75]
```

**W8A8 适用场景**：
- 硬件支持 INT8 Tensor Core（A100: 624 TOPS, 是 FP16 的 2x）
- 对精度要求较高（PPL 增加 0.1-0.3）
- Prefill 阶段加速效果显著（大矩阵乘 compute-bound，INT8 TC 吞吐 2x）

---

**W4A16 量化——仅量化权重方案**：

```
量化流程:
  1. 分组: 将权重按 group_size=128 分组
  2. 每组独立计算 scale 和 zero_point:
     scale = (max - min) / 15  (4-bit 范围 [0, 15])
     zp = round(-min / scale)
  3. 量化: q = clamp(round(w / scale + zp), 0, 15)
  4. 打包: 2 个 INT4 值打包到 1 个 INT8（节省存储）
  5. 存储: INT4 权重 + FP16 scale/zp (每 128 个权重一组)

推理时:
  load INT4 权重 (0.5 byte/element)
    → 寄存器内 unpack + dequant 为 FP16
    → 与 FP16 激活做 FP16 GEMM (Tensor Core)
```

**W4A16 适用场景**：
- 显存受限（权重减小 4x：7B 从 14GB → 3.5GB）
- Decode 阶段加速（memory-bound，带宽减少 4x → 加速 ~3x）
- 对精度容忍度中等（PPL 增加 0.3-1.0）

**两者对比**：

| 维度 | W8A8 (INT8) | W4A16 |
|---|---|---|
| 量化对象 | 权重 + 激活 | 仅权重 |
| 计算方式 | INT8 Tensor Core GEMM | FP16 Tensor Core GEMM（权重先 dequant） |
| 权重压缩 | 2x (FP16→INT8) | 4x (FP16→INT4) |
| Prefill 加速 | 1.5-2x（INT8 TC 算力高） | 无加速（仍是 FP16 计算） |
| Decode 加速 | 1.5-2x（权重读取减 2x） | 2-3x（权重读取减 4x） |
| 精度损失 | 小（PPL +0.1-0.3） | 中（PPL +0.3-1.0） |
| 实现复杂度 | 高（需要处理激活 outlier） | 中（只需离线量化权重） |
| 代表方法 | SmoothQuant, TensorRT INT8 | AWQ, GPTQ, Marlin |

**实践选择建议**：
```
A100/H100 + 精度优先 → W8A8 (SmoothQuant + TensorRT)
显存紧张 + Decode 为主 → W4A16 (AWQ + Marlin kernel)
H100 + 训练+推理 → FP8 (Transformer Engine, 最简单)
```

---

### Q: 数据分布极不均匀（最小值 -10000，其余在 [-1,1]）应该用什么量化方式？

这是一个典型的 **outlier 问题**——少数极端值将量化范围撑大，导致绝大部分正常值的有效量化位数极低。

**问题量化分析**：
```
假设使用 INT8 对称量化 (范围 [-128, 127]):
  scale = 10000 / 127 ≈ 78.7
  正常值 [-1, 1] 量化后 → round(±1 / 78.7) = 0
  
  结果: 99.9% 的值被量化为 0！精度完全丧失
```

**解决方案（按推荐优先级排序）**：

**1. 离群值单独处理（最推荐，LLM.int8() 方案）**：
```
策略: 检测 outlier channel → 分离处理
  
  正常 channel (|x| < threshold):  INT8 量化，正常参与 INT8 GEMM
  Outlier channel (|x| ≥ threshold): 保持 FP16，参与 FP16 GEMM
  最终结果 = INT8 部分输出 + FP16 部分输出

优点: 无信息损失，outlier 精确处理
缺点: 混合精度 GEMM 实现复杂，有额外开销
典型 threshold: 6.0（实验确定）
```

**2. Per-group 量化（AWQ/GPTQ 方案）**：
```
策略: 每 128 个连续权重共享一个 scale
  
  如果 outlier 只影响一个 group:
    该 group: scale = 10000/127 ≈ 78.7（这个 group 精度差）
    其他 group: scale = 1/127 ≈ 0.008（精度很好）
  
  损失被局限在单个 group 内，不影响全局

优点: 简单高效，是 GPTQ/AWQ 的标准做法
缺点: 如果 outlier 分散在多个 group 中，每个 group 都受影响
```

**3. Clipping + Calibration（截断方案）**：
```
策略: 计算最优截断点，牺牲 outlier 精度换取正常值精度

  不截断: MSE_total = 大量正常值的量化误差 (因 scale 太大)
  截断到 ±c: MSE_total = 正常值的量化误差 (小) + 截断误差 (outlier 被限制到 ±c)
  
  找最优 c: min_c (E[(x - Q(clip(x, -c, c)))²])
  通常取 99.9% 或 99.99% 百分位数

优点: 实现简单
缺点: outlier 信息完全丢失
```

**4. 非均匀量化（适配非均匀分布）**：
```
NF4 (NormalFloat4): 假设权重近似正态分布
  量化点: 在标准正态分布的等概率分位点处设置
  0 附近密集（正态分布概率密度高），尾部稀疏

对于有 outlier 的分布:
  可用对数量化: q = round(sign(x) × log(1 + |x|) / log(1 + max_val) × 127)
  使得大值区间分配更多 bin

缺点: 无硬件加速（非标准格式），需要自定义 dequant kernel
```

**5. SmoothQuant 思想（转移难度）**：
```
如果 outlier 出现在激活中:
  找到 outlier channel → 将该 channel 的缩放因子 s 设大
  激活该 channel 除以 s（outlier 被缩小）
  权重对应 channel 乘以 s（补偿）
  
  效果: 激活分布变平坦 → 量化容易
        权重接受 "难度转移" → 权重是静态的，可离线精确量化
```

**实际工程推荐**：
- 如果是权重 outlier → per-group 量化（group_size=128）解决大部分问题
- 如果是激活 outlier → SmoothQuant 或 LLM.int8() 分离处理
- 极端情况 → 混合精度：outlier channel 保持 FP16，其余 INT8/INT4

---

### Q: 手撕：手写一个 CUDA 算子？

（编程题）
