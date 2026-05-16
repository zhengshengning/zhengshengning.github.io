---
title: '小米 AI Infra 校招 一面'
date: 2026-04-16 12:16:56
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: PagedAttention的作用是什么？

PagedAttention是vLLM提出的KV Cache内存管理机制，借鉴操作系统虚拟内存分页思想，解决了大模型推理中**KV Cache显存管理效率低下**的核心问题。

**背景问题**：
传统KV Cache管理采用连续内存预分配——为每个请求预留 max_seq_len × num_layers × 2（K和V）× hidden_size 的连续显存。这导致：
- 大量内部碎片：实际生成长度远小于预留长度，平均浪费60-80%。
- 外部碎片：请求结束后释放的不连续空间难以复用。
- 并发受限：同样80GB显存，低效管理可能只能服务2-3个请求。

**PagedAttention核心机制**：

1. **Block化管理**：将KV Cache划分为固定大小的block（通常16个token的KV），每个block是独立的内存单元。
2. **Block Table映射**：维护逻辑位置到物理block地址的映射表（类似OS页表），逻辑上连续的KV在物理上可以分散存储。
3. **按需动态分配**：序列生成时每填满一个block才申请下一个，不预留未来空间。
4. **高效回收**：序列结束后立即释放其所有block回到空闲池。

**关键优势**：

| 特性 | 传统连续分配 | PagedAttention |
|------|------------|----------------|
| 显存利用率 | 20-40% | >96% |
| 同显存并发量 | 低 | 高3-5倍 |
| 内存碎片 | 严重 | 几乎无 |
| 前缀共享 | 不支持 | copy-on-write支持 |

**内存共享（Copy-on-Write）**：多个序列共享相同前缀（如system prompt）的KV Cache block，通过引用计数管理。当某个序列需要修改某block时才复制，节省大量重复计算和存储。

**对Beam Search的支持**：不同beam可以共享公共前缀的block，只在分叉点创建新block，显著减少beam search的显存开销。

### Q: FlashAttention的算法流程？

FlashAttention通过分块计算和online softmax实现IO-efficient的精确注意力，以下是详细的算法流程：

**初始化**：
- 将Q按行分块：每块Br行（典型值128）
- 将K、V按行分块：每块Bc行（典型值128）
- SRAM容量约束：Br × d + 2 × Bc × d ≤ SRAM_size

**算法步骤**：

```
For each Q block i (外层循环):
    初始化: Oi = 0, mi = -∞ (行最大值), li = 0 (行指数和)
    
    For each K,V block j (内层循环):
        1. 从HBM加载 Qi (Br×d), Kj (Bc×d), Vj (Bc×d) 到SRAM
        
        2. 计算局部注意力分数:
           Sij = Qi × Kj^T    (Br × Bc 矩阵，在SRAM中)
        
        3. 计算当前block的行最大值:
           mij = rowmax(Sij)   (Br维向量)
        
        4. 计算局部softmax指数:
           Pij = exp(Sij - mij)  (每行减去该行最大值后取exp)
        
        5. 更新全局统计量 (Online Softmax核心):
           mi_new = max(mi, mij)              // 新的全局最大值
           li_new = li × exp(mi - mi_new)     // 修正旧的累积和
                  + rowsum(Pij) × exp(mij - mi_new)  // 加入新block贡献
        
        6. 更新输出 (关键步骤):
           Oi = Oi × (li × exp(mi - mi_new) / li_new)   // 修正旧输出的权重
              + Pij × Vj × (exp(mij - mi_new) / li_new) // 加入新block对V的加权
        
        7. 更新: mi = mi_new, li = li_new
    
    写回 Oi 到 HBM (最终结果)
```

**关键数学保证**：
- 通过Online Softmax的修正因子 `exp(m_old - m_new)`，保证每次更新后Oi都等于当前已处理所有block的正确softmax加权结果。
- 最终结果与标准Attention**数学等价**（非近似）。

**反向传播**：不存储中间的N×N注意力矩阵P，只存储统计量m和l（O(N)空间）。反向时从HBM重新加载Q、K、V重计算P（用saved的m、l验证），然后计算梯度。

**性能数据**：A100上seq_len=2048时，FlashAttention-2比PyTorch标准实现快约3倍，内存使用降低10-20倍。

### Q: 量化的原理是什么？

量化是将高精度浮点数（FP32/FP16）映射为低位宽表示（INT8/INT4/FP8）的过程，核心目的是**减少模型体积和推理计算量**。

**基本数学公式**：

对称量化（zero_point = 0）：
```
量化: q = round(x / scale)
反量化: x_approx = q × scale
其中: scale = max(|x|) / (2^(bits-1) - 1)
```

非对称量化：
```
量化: q = round(x / scale) + zero_point
反量化: x_approx = (q - zero_point) × scale
其中: scale = (max - min) / (2^bits - 1)
      zero_point = round(-min / scale)
```

**按量化时机分类**：

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| PTQ（训练后量化） | 模型训练完后直接量化 | 快速、无需训练 | 低bit时精度下降明显 |
| QAT（量化感知训练） | 训练时模拟量化误差（STE） | 精度好 | 需要重训/微调 |

**按量化粒度分类**：
- **per-tensor**：整个tensor共享一个scale，最粗糙但overhead最小。
- **per-channel**：每个输出channel一组参数，权重量化的标准做法。
- **per-group**：每G个连续元素一组（如G=128），精度最高，GPTQ/AWQ常用。

**大模型常用量化方案**：

| 方案 | 格式 | 方法 | 适用 |
|------|------|------|------|
| GPTQ | W4A16 | 逐列量化+Hessian误差补偿 | 推理部署 |
| AWQ | W4A16 | 激活感知+salient channel保护 | 推理部署 |
| SmoothQuant | W8A8 | 迁移量化难度到权重 | INT8推理 |
| FP8 | E4M3/E5M2 | 直接低精度训练/推理 | Hopper架构 |

**量化的实际收益**：
- 模型体积：FP16→INT4 = 4倍压缩。
- 推理速度：INT8 Tensor Core比FP16快约2倍（A100），FP8比FP16快约2倍（H100）。
- 内存带宽：Decode阶段是memory-bound，权重量化直接按比例降低带宽需求。
