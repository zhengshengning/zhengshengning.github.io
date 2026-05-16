---
title: '识渊科技 AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 介绍一个CUDA算子优化过程？

CUDA算子优化是一个系统化的迭代过程，核心方法论是**Profile → 定位瓶颈 → 针对性优化 → 验证效果**。

**完整优化流程：**

**Step 1: Baseline实现 + Profiling**
```bash
# 使用Nsight Compute分析kernel
ncu --set full ./my_program
```
关键指标：
- `Memory Throughput` SOL%：实际带宽占理论带宽百分比
- `Compute (SM) Throughput` SOL%：计算利用率
- `Achieved Occupancy`：活跃warp比例
- `Warp State Statistics`：stall原因分析

**Step 2: 根据瓶颈类型优化**

| 瓶颈类型 | 判断依据 | 优化方向 |
|----------|----------|---------|
| Memory-bound | Memory SOL% > Compute SOL% | 优化访存 |
| Compute-bound | Compute SOL% > Memory SOL% | 优化计算 |
| Latency-bound | 两者都低 | 增大并行度 |

**Memory-bound优化实例（如Softmax kernel）：**
1. 合并访问：确保连续线程访问连续地址 → 一次128字节事务服务32个线程
2. 向量化加载：`float4* p = (float4*)input; float4 val = p[tid];` → 4倍带宽利用
3. 减少全局内存访问：online softmax一遍扫描替代两遍
4. 共享内存中间缓存：如果需要两遍扫描则第一遍的结果存shared memory

**Compute-bound优化实例（如GEMM kernel）：**
1. 利用Tensor Core：`wmma::mma_sync()` 调用硬件矩阵乘
2. 提高ILP：`#pragma unroll`展开内层循环
3. 寄存器级Tiling：每个线程计算8x8输出元素

**Step 3: 验证优化效果**
- 对比优化前后的SOL%变化
- 检查是否出现新瓶颈（消除memory瓶颈后可能变为compute瓶颈）
- 确认数值正确性（优化不能改变结果）

**性能目标参考：**
- Memory-bound kernel目标：达到理论带宽的80%+
- Compute-bound kernel目标：达到理论算力的70%+
- A100 HBM带宽2039 GB/s，FP16 Tensor Core算力312 TFLOPS

### Q: KNN算法的流程？

KNN（K-Nearest Neighbors，K近邻）是一种基于实例的懒学习(Lazy Learning)算法，不需要显式训练过程。

**算法流程：**

1. **计算距离**：待预测样本与训练集中所有样本的距离
2. **排序选择**：按距离升序排列，选取最近的K个邻居
3. **决策**：
   - 分类任务：K个邻居中多数投票（majority voting）
   - 回归任务：K个邻居目标值的（加权）平均

**距离度量选择：**

| 距离度量 | 公式特点 | 适用场景 |
|----------|----------|---------|
| 欧氏距离 | sqrt(Σ(xi-yi)^2) | 通用，连续值特征 |
| 曼哈顿距离 | Σ|xi-yi| | 高维空间更稳定 |
| 余弦距离 | 1 - cos(x,y) | 文本/稀疏高维（关注方向不关注幅度） |
| 闵可夫斯基距离 | (Σ|xi-yi|^p)^(1/p) | 欧氏(p=2)和曼哈顿(p=1)的泛化 |

**K值选择的影响：**
- K太小：过拟合，对噪声敏感（一个噪声点就可能改变分类结果）
- K太大：欠拟合，将远处不相关样本纳入投票，决策边界过于平滑
- 经验值：sqrt(n)或通过交叉验证选择最优K

**加速搜索的数据结构：**
- **KD-Tree**：空间划分树，平均O(log n)查询（高维退化严重，d>20时接近线性）
- **Ball-Tree**：球形划分，高维表现优于KD-Tree
- **HNSW**：近似最近邻（ANN），通过图结构加速，牺牲精度换取O(log n)查询
- **Faiss/ScaNN**：GPU加速的向量检索库，支持亿级数据

**KNN的优缺点：**
- 优点：简单直观、无需训练、对非线性边界适应好
- 缺点：预测慢O(nd)（d为维度）、内存占用大、对特征尺度敏感（需标准化）、高维失效（维度灾难）

### Q: 数据集有问题时训练怎么解决？

根据数据集问题的类型采取不同策略：

**1. 类别不平衡：**

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| 过采样(SMOTE) | 对少数类合成新样本（插值） | 少数类样本太少时 |
| 欠采样 | 随机移除多数类样本 | 多数类样本充足时 |
| Focal Loss | γ参数降低易分类样本的loss权重 | 目标检测（正负样本1:1000+） |
| Class-weighted Loss | 按类别频率反比加权 | 通用 |
| 过采样+欠采样结合 | 两者混合使用 | 严重不平衡 |

**2. 噪声标签（Label Noise）：**
- **数据清洗**：用已训练模型预测，找出与标签不一致的样本人工复核
- **置信学习(Confident Learning)**：cleanlab库估计noise matrix识别错误标签
- **噪声鲁棒损失**：Symmetric Cross Entropy、Generalized Cross Entropy
- **Co-teaching**：两个模型互相筛选干净样本训练对方
- **Early Stopping**：模型先拟合干净样本再拟合噪声，提前停止可避免学习噪声

**3. 数据量不足：**
- **数据增强**：几何变换、颜色抖动、Mixup/CutMix、RandAugment
- **迁移学习**：预训练模型微调（大模型+小数据最有效的方案）
- **半监督学习**：利用大量无标注数据（如FixMatch、Pseudo-labeling）
- **合成数据**：用生成模型(Diffusion/GAN)生成训练数据

**4. 分布偏移(Distribution Shift)：**
- **Domain Adaptation**：对齐源域和目标域的特征分布（DANN、MMD）
- **Test-Time Adaptation**：推理时在线适应目标域分布（如Tent）
- **Domain-specific augmentation**：模拟目标域的数据变换

### Q: 图像算子了解哪些？

**传统图像处理算子：**

**1. 边缘检测：**
- Sobel：一阶导数近似，分别计算x/y方向梯度，简单快速
- Canny：多步骤（高斯模糊→梯度计算→非极大值抑制→双阈值→连接），效果最好
- Laplacian：二阶导数，检测零交叉点，对噪声敏感

**2. 模糊/平滑（去噪）：**
- 高斯模糊：加权平均（权重为高斯分布），保持边缘相对清晰
- 均值滤波：等权平均，计算最快但模糊边缘
- 中值滤波：取窗口内中值，对椒盐噪声效果最好
- 双边滤波：同时考虑空间距离和像素值差异，保持边缘

**3. 形态学操作：**
- 膨胀(Dilation)：扩大白色区域
- 腐蚀(Erosion)：缩小白色区域
- 开运算(Opening)：先腐蚀后膨胀，去小噪点
- 闭运算(Closing)：先膨胀后腐蚀，填小孔

**4. 深度学习核心算子：**
- 卷积(Conv2d)：特征提取，学习局部模式
- 池化(Pooling)：降采样，Max/Average Pooling
- BatchNorm：归一化激活值，稳定训练
- ReLU/GELU/SiLU：非线性激活函数

**GPU优化角度：**
传统图像算子（如Sobel、高斯模糊）天然适合GPU并行——每个像素的计算独立，可以分配一个线程。CUDA实现时利用纹理内存（texture memory）的空间局部性和硬件插值加速。

### Q: 如何保证系统的高性能？

高性能系统设计需要在多个层面综合优化：

**1. 算法与计算优化（根基）：**
- 选择渐进复杂度最优的算法（O(n log n) vs O(n^2)）
- SIMD向量化：一条指令处理多个数据（如AVX-512处理16个float）
- GPU并行：大规模数据并行计算
- 近似计算：用精度换速度（如近似最近邻替代精确KNN）

**2. 内存优化（常见瓶颈）：**
- **减少拷贝**：零拷贝技术、move语义、引用传递
- **内存池(Memory Pool)**：避免频繁malloc/free的系统调用和碎片
- **缓存友好的数据布局**：SoA替代AoS、连续内存访问
- **预分配**：`vector::reserve()`避免多次扩容拷贝
- **对象复用**：对象池模式减少创建/销毁开销

**3. IO优化：**
- 异步IO（AIO/io_uring）：不阻塞计算线程
- 批量处理（Batching）：合并小IO为大IO减少系统调用次数
- 零拷贝：sendfile、mmap避免用户态-内核态拷贝
- 预取(Prefetch)：预判下一步数据需求提前加载

**4. 并发优化：**
- 无锁数据结构（Lock-Free Queue、CAS操作）
- 减少锁粒度（分段锁、读写锁分离）
- 线程池复用：避免频繁创建销毁线程
- 减少同步点：异步通知替代轮询等待

**5. 系统架构层面：**
- 流水线化(Pipelining)：多阶段重叠执行
- 计算通信重叠（Overlap）
- 负载均衡：Work-stealing调度
- 性能隔离：NUMA-aware内存分配

**6. 持续性能工程：**
- 定期Profiling定位瓶颈（perf、Nsight、火焰图）
- 建立性能基线和回归检测
- Amdahl定律指导优化优先级（先优化占比最大的部分）
