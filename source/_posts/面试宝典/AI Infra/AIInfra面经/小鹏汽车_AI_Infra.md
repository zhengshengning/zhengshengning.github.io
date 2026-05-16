---
title: '小鹏汽车 AI Infra'
date: 2026-04-16 12:12:10
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 什么是算术强度（Arithmetic Intensity）？矩阵乘法的性能瓶颈是什么？

算术强度（Arithmetic Intensity, AI）是衡量算子计算密集度的核心指标：

**定义**：算术强度 = 计算量（FLOPs）/ 数据搬运量（Bytes），单位为 FLOPs/Byte。

**Roofline Model与算术强度的关系**：

GPU有两个硬件上限：
- 峰值算力：如A100 FP16 = 312 TFLOPS
- 峰值带宽：如A100 HBM = 2 TB/s

**拐点（Ridge Point）** = Peak_FLOPS / Peak_Bandwidth = 312T / 2T = **156 FLOPs/Byte**。

- 算术强度 < 拐点 → **Memory-bound**：执行速度受内存带宽限制，计算单元空闲等数据。
- 算术强度 > 拐点 → **Compute-bound**：执行速度受算力限制，内存带宽充裕。

**矩阵乘法 C = A × B（M×K × K×N）的分析**：

- 计算量：2×M×K×N FLOPs（每个输出元素需要K次乘法+K-1次加法 ≈ 2K FLOPs）
- 数据量：(M×K + K×N + M×N) × bytes_per_element
- 算术强度 ≈ 2MKN / [(MK+KN+MN) × sizeof]

当M=N=K=n时：AI ≈ 2n^3 / (3n^2 × 4) = n/6 FLOPs/Byte（FP32）。
- n=1024: AI ≈ 170 > 156 → Compute-bound
- n=64: AI ≈ 10 < 156 → Memory-bound

**矩阵乘法的性能瓶颈取决于**：
1. **矩阵维度**：大矩阵是compute-bound（AI高），小矩阵是memory-bound。Decode阶段的矩阵-向量乘（M=1）严重memory-bound。
2. **分块策略（Tiling）**：影响数据复用率和缓存命中率。理想tile使得数据在shared memory中被多次使用。
3. **数据Layout**：行主序访问连续内存时合并访问高效；转置访问需要shared memory中转。
4. **硬件利用率**：Tensor Core要求特定的数据格式和对齐（如FP16、维度是8/16的倍数）。

### Q: 手撕CUDA：实现二维矩阵转置并逐步优化？

（编程题）

### Q: map和unordered_map的实现与复杂度对比？

| 特性 | std::map | std::unordered_map |
|------|----------|-------------------|
| 底层数据结构 | **红黑树**（自平衡BST） | **哈希表**（开链法/桶+链表） |
| 查找复杂度 | O(log N) 最坏保证 | 平均O(1)，最坏O(N)（全碰撞） |
| 插入复杂度 | O(log N) | 平均O(1)，rehash时O(N) |
| 删除复杂度 | O(log N) | 平均O(1) |
| 有序性 | key按比较器有序 | 无序 |
| 迭代器稳定性 | 插入不失效，删除仅失效当前元素 | **rehash时全部失效** |
| 内存布局 | 每节点独立分配（3指针+颜色位+数据） | 桶数组 + 链表节点 |
| 内存overhead | 约每元素32-48字节（树节点） | 每元素~40字节 + 桶数组 |
| key要求 | 需要 `operator<`（或自定义比较器） | 需要 `std::hash` + `operator==` |

**unordered_map的关键行为**：
- **负载因子**：`load_factor = size / bucket_count`，默认max_load_factor = 1.0。
- **Rehash触发**：当 load_factor > max_load_factor 时自动rehash——申请约2倍桶数组，所有元素重新hash分配。这是O(N)操作且使所有迭代器失效。
- **reserve(n)**：预分配足够桶位避免rehash（已知数据量时推荐）。

**选择建议**：
- 需要有序遍历、范围查询（lower_bound/upper_bound）→ map
- 纯粹的键值查找，追求速度 → unordered_map
- 元素数量少（<100）→ 两者差异不大，甚至sorted vector可能更快（cache友好）
- 需要稳定迭代器（遍历中插入）→ map
- 自定义类型做key → 需要为unordered_map提供hash函数和==运算符

**性能注意**：unordered_map在元素极多时因为链表指针追踪（pointer chasing）导致cache不友好。对于高性能场景可考虑开放寻址hash（如absl::flat_hash_map）。

### Q: 推理中常见的优化手段有哪些？

大模型推理优化是一个多层次系统工程，从模型架构、算法、系统、硬件四个维度综合优化：

**1. 量化 —— 最直接的加速手段**：
- W8A8（SmoothQuant）：权重和激活都INT8，利用INT8 Tensor Core，加速约2倍。
- W4A16（GPTQ/AWQ）：仅量化权重，Decode阶段带宽减半→速度翻倍。
- FP8（H100原生支持）：E4M3训练/推理，E5M2反向，精度接近FP16。

**2. 算子融合 —— 减少内存往返**：
- QKV投影融合：3个独立GEMM→1个大GEMM，减少2次kernel launch和2次输出写回。
- LayerNorm + 残差Add + 下一层输入：消除中间tensor的HBM写+读。
- FFN中SwiGLU：gate和up投影融合为一个GEMM。
- 效果：对memory-bound的小算子效果显著（可加速2-5倍）。

**3. KV Cache —— 避免重复计算**：
- 原理：Decoder自回归生成时缓存历史token的K/V，每步只增量计算新token。
- 优化：PagedAttention（分页管理）、GQA/MQA（减少KV头）、量化KV、Prefix Caching。

**4. FlashAttention —— IO-aware注意力**：
- 分块计算避免N×N矩阵的HBM读写，内存O(N)，速度提升2-4倍。
- FlashDecoding：Decode阶段优化，沿KV长度维度并行。

**5. 投机解码（Speculative Decoding）**：
- 小模型（draft model）快速生成K个候选token，大模型一次前向批量验证。
- 接受率60-80%时等效加速1.5-2.5倍，不影响输出质量（数学等价）。

**6. 连续批处理（Continuous Batching）**：
- 传统静态batch：最长请求完成前所有请求等待。
- Continuous Batching：请求完成立即移出，新请求立即加入，GPU利用率提升2-3倍。

**7. 张量/流水线并行 —— 多卡推理**：
- TP（节点内）：切分单层权重降低延迟，适合TTFT优化。
- PP（跨节点）：切分层降低单卡显存需求。

**8. 编译优化**：
- TensorRT/torch.compile：图级别优化 + 自动代码生成 + 自动调优。
- 效果：通常比naive PyTorch快3-5倍。

**9. 结构化稀疏（2:4 Sparsity）**：
- A100+ Tensor Core原生支持2:4稀疏模式（每4个值中2个为0），加速约2倍无需改变数据类型。
