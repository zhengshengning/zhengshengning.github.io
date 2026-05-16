---
title: '理想汽车 云 AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 手撕：两数之和（找到和等于目标值的两个数）？

（编程题）

---

### Q: BatchNorm 的作用？梯度消失和梯度爆炸的解决思路？

**BatchNorm 作用**：

对 mini-batch 的每个 feature channel 做归一化：`x̂ = (x - μ_batch) / √(σ²_batch + ε)`，再通过可学习参数 γ、β 做线性变换。

核心效果：
- **加速收敛**：保持中间层输入分布稳定（减少 Internal Covariate Shift），允许使用更大 lr（提升 5-10x 收敛速度）
- **正则化**：batch 内统计的随机性提供轻微正则
- **缓解梯度问题**：输出维持在 [-2, 2] 范围，避免激活值过大/过小导致的梯度异常
- **降低初始化敏感性**：BN 会"修正"不好的初始化

---

**梯度消失解决思路（深层网络梯度逐层指数衰减）**：

| 方法 | 原理 |
|---|---|
| ReLU/GELU 激活 | 正区间梯度恒为 1（vs Sigmoid 的 max 梯度 0.25） |
| BatchNorm/LayerNorm | 保持每层输出分布稳定，梯度不会因值域偏移而消失 |
| 残差连接 | 提供梯度直通路径：`∂L/∂x = ∂L/∂(x+F(x)) × (1 + ∂F/∂x)`，至少有常数 1 |
| Xavier/Kaiming 初始化 | 保持每层方差不变，防止信号逐层衰减 |
| LSTM 门控 | 遗忘门让梯度可以沿 cell state 无衰减传播 |

**梯度爆炸解决思路（梯度逐层指数增长）**：

| 方法 | 原理 |
|---|---|
| 梯度裁剪 (Clip) | `clip_grad_norm_(params, max_norm=1.0)` 限制梯度范数 |
| 权重正则化 | L2 decay 防止权重过大 |
| BatchNorm | 约束输出范围 |
| 学习率 Warmup | 初期小 lr 避免大梯度×大步长 |
| 残差连接 | 限制梯度增长速率 |

---

### Q: 算子层面的底层优化方法？

算子优化的核心是**最大化硬件利用率**——让 GPU 的计算单元和内存带宽都不闲着：

**访存优化（解决 memory-bound）**：
- **合并访问**：warp 32 线程访问连续 128 字节 → 1 次内存事务（vs 分散访问 32 次事务）
- **向量化加载（float4）**：一条指令 128-bit，减少事务数 4x
- **Shared Memory Tiling**：Global→Shared 加载一次，block 内复用多次（GEMM 复用率 ∝ tile 大小）
- 效果：memory-bound 算子可提升 2-8x

**计算优化（解决 compute-bound）**：
- **Tensor Core**：16×16×16 矩阵乘加，FP16 吞吐是 CUDA Core 的 16x
- **循环展开**：消除循环控制指令，增加指令级并行（ILP）
- **指令级并行**：相邻独立指令可同时发射（超标量发射）
- 效果：充分利用 Tensor Core 可提升 10x+

**调度优化（解决 latency-bound）**：
- **算子融合**：减少 kernel launch（每次 ~5μs）+ 中间 tensor HBM 读写
- **减少同步**：warp 级原语（`__shfl_sync`）替代 block 级 `__syncthreads`
- **Persistent Kernel**：减少 block 调度开销

**数据复用优化**：
- **寄存器 Tiling**：每线程在寄存器中维护多个输出元素（寄存器带宽无限大）
- 每从 shared memory 读一行/列 → 计算 TM×TN 个输出 → 复用率 = TM+TN

---

### Q: 目标检测的方法有哪些？

**按设计范式分类**：

| 类别 | 代表 | 特点 | 精度 | 速度 |
|---|---|---|---|---|
| 两阶段 | Faster R-CNN, Cascade R-CNN | RPN提候选 + 分类回归 | 高 | 慢 |
| 单阶段 | YOLOv5-v8, SSD, RetinaNet | 直接预测 box + class | 中高 | 快 |
| Anchor-free | CenterNet, FCOS, CornerNet | 无需预定义 anchor | 中高 | 快 |
| Transformer | DETR, Deformable DETR | 端到端，无 NMS | 高 | 中 |
| 3D 检测 | PointPillars, CenterPoint, BEVFusion | 点云/多模态 | 场景定 | 中 |

**核心演进思路**：
- Anchor-based → Anchor-free（减少超参）
- 两阶段 → 单阶段（追求速度）
- 手工 NMS → End-to-End（DETR 无后处理）
- 2D → 3D/BEV（自动驾驶需求）

---

### Q: LSS 的方法和改进？LSS、BEVFormer 的算力比较？

**LSS（Lift-Splat-Shoot）——图像到 BEV 的经典范式**：

```
多视角图像 → 2D 特征提取(CNN/ViT)
    ↓ Lift：为每个像素预测深度分布 D(d)
    ↓ 得到 3D 点云特征：feature × D(d) → 3D volume
    ↓ Splat：将 3D 点投影到 BEV 网格（Pillar Pooling）
    ↓ BEV 特征图 → 下游任务（检测/分割）
```

**改进**：
- **BEVDet**：改进特征提取骨干 + 时序融合
- **BEVDepth**：显式深度监督（用 LiDAR 点云 GT 监督深度估计网络），大幅提升深度准确性
- **BEVStereo**：利用时序立体匹配辅助深度估计

**BEVFormer——基于注意力的范式**：

```
多视角图像 → 2D 特征提取
    ↓ BEV Queries（可学习的 BEV 网格特征）
    ↓ Spatial Cross-Attention：BEV query 从多视图特征中采样（Deformable Attention）
    ↓ Temporal Self-Attention：融合历史帧 BEV 特征
    ↓ BEV 特征 → 下游任务
```

**算力对比**：

| 维度 | LSS 类 | BEVFormer |
|---|---|---|
| 主要计算 | 2D CNN + 深度预测 + Pillar Pool | 2D CNN + Cross-Attention |
| FLOPs | ~100-200 GFLOPs | ~300-500 GFLOPs |
| 延迟 | ~30-50ms | ~80-150ms |
| 精度 | 中 | 高 |
| 部署友好度 | 高（操作简单） | 低（attention 动态性强） |
| 适用 | 量产端侧部署 | 研究/高精度场景 |

LSS 类在算力受限的车端（如 Orin）更友好，BEVFormer 精度更高但部署成本大。
