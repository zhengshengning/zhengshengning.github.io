---
title: '智谱 AI Infra 实习 一面'
date: 2026-04-16 12:07:28
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 推理优化, 面经]
---


<!-- more -->

---

### Q: MinMax和Percentile量化校准算法有什么不同？

量化校准（Calibration）的目标是确定量化范围[min, max]，使得量化后的精度损失最小。两种基础方法有不同的outlier处理策略：

**MinMax校准**：
- 做法：直接使用校准数据中观察到的全局最大值和最小值作为量化范围。
  ```
  scale = (max_val - min_val) / (2^bits - 1)
  ```
- 优点：绝不截断任何值，量化范围完整覆盖数据分布。实现极简。
- 缺点：对**离群值（outlier）极其敏感**——哪怕只有一个极端值就会拉大整个范围，导致大部分正常值只能用很少的量化级别表示，量化精度严重下降。
- 典型问题：LLM的激活中经常出现少量magnitude很大的outlier（如SmoothQuant论文中发现某些channel的激活值可达其他channel的100倍），MinMax在这种场景下表现很差。

**Percentile校准**：
- 做法：使用第p百分位和第(100-p)百分位的值作为量化范围（如p=99.99%），将超出范围的outlier截断（clamp）。
  ```
  range_min = percentile(data, 0.01)
  range_max = percentile(data, 99.99)
  ```
- 优点：对outlier**鲁棒**——牺牲极少数极端值的精度（它们被clamp到边界），换取绝大多数值获得更精细的量化分辨率。
- 参数选择：p值通常99.9%-99.999%，需要在截断误差和量化精度间取平衡。太小的p会截断过多有效数据。
- 适用场景：激活分布有长尾/outlier时（LLM激活的典型特征）。

**量化误差对比**（假设INT8, 数据范围[0,100]但99.9%数据在[0,10]内，有少数outlier到100）：
- MinMax：scale=100/255≈0.39，[0,10]的数据只能用26个量化级别表示。
- Percentile(99.9%)：scale≈10/255≈0.04，[0,10]的数据能用全部255个级别，精度好10倍。

### Q: 还有哪些其他量化校准算法？

除了MinMax和Percentile，更高级的校准算法通过最小化某种量化损失指标来确定最优范围：

**1. KL散度（Entropy Calibration）—— TensorRT默认**：
- 原理：搜索使量化后分布与原始分布的KL散度（相对熵）最小的阈值。
- 做法：收集校准数据的激活直方图（如2048个bin），遍历不同的截断阈值（从128到2048个bin），对每个阈值计算量化前后分布的KL(P||Q)，选择KL最小的阈值。
- 优点：考虑了整体分布形状，而非仅看极值。
- 缺点：需要遍历搜索，比MinMax慢。

**2. MSE（L2 Error Minimization）**：
- 原理：搜索使量化前后**均方误差**最小的range。
- 公式：`min_range |x - dequantize(quantize(x, range))|^2`
- 特点：直接最小化量化误差本身，通常效果好于KL。
- 实现：可以解析求解（假设均匀分布）或网格搜索。

**3. ACIQ（Analytical Clipping for Integer Quantization）**：
- 假设权重/激活服从特定分布（高斯或拉普拉斯）。
- 解析推导出最优clip值的闭式解。
- 优点：无需搜索，计算快。缺点：分布假设不总成立。

**4. Histogram-based（直方图统计）**：
- 收集大量校准样本的激活值直方图。
- 基于直方图统计信息（如标准差的N倍作为范围、或自定义目标函数优化）选择最优范围。
- TensorRT的entropy方法本质上也是基于直方图。

**选择建议**：
- 快速验证：MinMax/Percentile
- 量产部署（CNN/CV模型）：KL散度（TensorRT默认，效果稳定）
- LLM部署：SmoothQuant/AWQ等专用方法（因为LLM的outlier问题更严重）

### Q: SmoothQuant的量化粒度是什么？AWQ和GPTQ的流程？

**SmoothQuant —— 迁移量化难度**

核心观察：LLM中权重分布均匀容易量化，但激活分布有极大的per-channel方差（某些channel值是其他的100倍），直接量化激活会损失精度。

核心思想：在数学等价的前提下，将激活的量化难度"转移"到权重上：
```
Y = (X × diag(s)^{-1}) × (diag(s) × W) = X_smooth × W_smooth
```
- 对激活X的每个channel除以平滑因子s，使激活分布更均匀（容易量化）。
- 对权重W的对应行乘以s（权重变得稍不均匀但仍可接受）。
- s的选择：`s_j = max(|X_j|)^α / max(|W_j|)^(1-α)`，α=0.5是均衡点。

量化粒度：**权重per-channel，激活per-token**。最终W8A8部署，利用INT8 Tensor Core加速。

**AWQ（Activation-aware Weight Quantization）流程**：
1. 用少量校准数据（如128条样本）前向推理，收集每个channel的激活magnitude。
2. 找出"salient"权重通道——对应激活magnitude大的channel（这些权重对输出影响大）。
3. 对salient通道的权重乘以scale（>1），使其量化后相对误差更小（等效于赋予更高的量化精度）。
4. 对所有权重做per-group量化（group_size=128）。
5. 推理时权重保持INT4存储，激活FP16计算（W4A16）。

优点：仅需校准数据，无需梯度计算，一块GPU几分钟完成。

**GPTQ（Generative Post-Training Quantization）流程**：
1. 收集少量校准数据，计算Hessian矩阵 H = X^T × X（反映每个权重位置的敏感度）。
2. **逐列量化**权重矩阵：
   - 量化当前列 w_col → q_col。
   - 计算量化误差 δ = w_col - dequant(q_col)。
   - 将误差**补偿到剩余未量化的列**：`W_remaining += δ × H_row / H_diag`（OBS原理）。
3. 重复直到所有列量化完成。

优点：误差补偿使得量化精度显著优于naive round-to-nearest。4bit量化时perplexity损失通常<0.5。

### Q: NVFP4的原理是什么？怎么做缩放？在哪个维度缩放？

NVFP4是NVIDIA在Blackwell（B100/B200）架构引入的4位浮点格式，配合Tensor Core实现极致压缩率的推理。

**FP4数据格式（E2M1）**：
- 1位符号 + 2位指数 + 1位尾数
- 可表示的值（含符号）：{0, 0.5, 1, 1.5, 2, 3, 4, 6} 及其负数，共16个离散值。
- 相比INT4（均匀分布16个值），FP4的非均匀分布更适合神经网络权重的近似正态分布。

**微缩放（Micro-scaling）—— 两级缩放机制**：

由于FP4本身动态范围极小（0-6），必须配合缩放因子（scale factor）才能表示实际权重的分布。

```
实际值 = FP4_value × scale_per_block
```

- **Block大小**：每32个连续FP4元素共享1个FP8 scale factor。
- **缩放维度**：沿reduction维度（K维度）分组。对于权重矩阵W(N×K)，沿K方向每32个元素一组。
- **存储格式**：32个FP4值（16字节）+ 1个FP8 scale（1字节）= 17字节/32元素。
- **有效位宽**：17/32 × 8 = 4.25 bit/element（比纯4bit多5.6%的overhead）。

**为什么沿reduction维度分组**：
- GEMM的计算是沿K维度做点积。同一组的32个元素会被一起乘加。
- 反量化操作（乘scale）可以在点积结果上一次完成，而非每个元素都乘scale，减少计算开销。

**Blackwell硬件支持**：
- B200 Tensor Core原生支持FP4格式，吞吐比FP8再翻倍。
- FP4 Tensor Core峰值可达9 PFLOPS（vs H100 FP8 约4 PFLOPS）。

### Q: per-tensor、per-channel、per-group，哪个粒度更细？

粒度从粗到细：**per-tensor < per-channel < per-group**。

| 粒度 | 共享scale的范围 | 参数数量开销 | 精度 | 典型应用 |
|------|---------------|-------------|------|---------|
| per-tensor | 整个tensor 1个scale | 最小（1个值） | 最差 | 激活量化的简化方案 |
| per-channel | 每个output channel 1个scale | 中等（OC个值） | 中等 | 权重量化标准做法（INT8） |
| per-group | 每G个连续元素 1个scale | 最大（元素数/G个值） | 最好 | 4bit权重量化（G=128/32） |

**为什么粒度越细精度越高**：
- 同一个scale覆盖的元素越少，这些元素的值域越接近，量化的分辨率越高。
- 极端情况：per-element（每个元素独立scale）= 无损，但开销等于原始数据。

**实际应用中的选择**：
- **权重INT8**：per-channel是标准做法。每个output channel的权重分布相对独立，per-channel能很好地适应各channel不同的值域。
- **激活INT8**：per-tensor或per-token。因为激活是动态的（每次输入不同），per-channel太贵需要逐通道计算scale。
- **权重INT4（LLM部署）**：per-group（G=128或32）。4bit表示范围小（仅16个值），per-channel对LLM权重的分布差异过大难以覆盖，需要更细的per-group。GPTQ/AWQ默认group_size=128。
- **NVFP4（Blackwell）**：per-group（G=32），使用FP8 scale。

**开销分析（以1B参数INT4模型为例）**：
- per-tensor：1个FP16 scale = 2字节额外。
- per-channel（假设4096个channel）：4096 × 2 = 8KB额外。
- per-group（G=128）：1B/128 × 2 = 15.6MB额外（约为模型的3%）。

### Q: 手撕：实现MinMax和Percentile量化校准？

（编程题）
