---
title: '经纬恒润 AI Infra 二面'
date: 2026-04-16 12:20:45
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 常见的多目标跟踪算法有哪些？

多目标跟踪（MOT）的核心挑战是在检测结果基础上维持目标身份的连续性：

- **SORT**：基于卡尔曼滤波预测目标下一帧位置 + 匈牙利算法做 IoU 关联匹配。速度极快（~260 FPS），但仅靠位置关联，遮挡后容易 ID 切换（ID Switch 多）

- **DeepSORT**：在 SORT 基础上引入**外观特征**（ReID 网络提取 128-d appearance embedding），采用**级联匹配**策略：优先匹配上一帧关联过的目标（motion + appearance），再处理长期未匹配的目标。ID Switch 降低 45% 以上

- **ByteTrack**：核心创新是**利用低分检测框**。传统方法丢弃低置信度检测（如 score < 0.5），ByteTrack 将低分框（byte 级别）用于二次匹配——高分框先匹配，未匹配的 track 再与低分框匹配。这样被遮挡导致检测分低的目标不会丢失。MOTA 大幅提升

- **OC-SORT**：针对**遮挡场景**改进卡尔曼滤波——当目标被遮挡时不做观测更新（避免错误更新），恢复时利用虚拟轨迹做重关联。观测中心性（Observation-Centric）替代运动中心性

- **BoT-SORT**：结合**相机运动补偿**（估计帧间全局仿射变换）+ 更强的 ReID 特征 + IoU 融合外观相似度。在需要处理相机抖动的场景下表现更好

**选择建议**：
- 追求速度：SORT/ByteTrack
- 行人跟踪（遮挡多）：DeepSORT/OC-SORT
- 竞赛/高精度：BoT-SORT

### Q: 卡尔曼滤波的原理？

卡尔曼滤波是一种**递归的线性最优状态估计**算法，在高斯噪声假设下给出均方误差最小的估计。分为两步：

**1. 预测（Predict）**——基于运动模型预测下一时刻状态：
```
x̂_k|k-1 = F · x̂_{k-1} + B · u_k        (状态预测)
P_k|k-1  = F · P_{k-1} · F^T + Q         (协方差预测)
```
- F：状态转移矩阵（如匀速运动模型）
- Q：过程噪声协方差（反映模型不确定性）
- P：估计误差协方差

**2. 更新（Update）**——结合观测修正预测：
```
K_k  = P_k|k-1 · H^T · (H · P_k|k-1 · H^T + R)^{-1}   (卡尔曼增益)
x̂_k = x̂_k|k-1 + K_k · (z_k - H · x̂_k|k-1)           (状态更新)
P_k  = (I - K_k · H) · P_k|k-1                           (协方差更新)
```
- H：观测矩阵（将状态映射到观测空间）
- R：观测噪声协方差
- K：卡尔曼增益——自动在预测和观测之间取最优加权

**核心直觉**：
- 当观测噪声 R 大时，K 小，更信赖预测
- 当过程噪声 Q 大时，K 大，更信赖观测
- 卡尔曼增益 K 是最优的加权系数，使后验估计的均方误差最小

**在 MOT 中的应用**：状态向量通常为 `[x, y, w, h, vx, vy, vw, vh]`（位置 + 速度），使用匀速运动模型做预测。

### Q: 常见的推理框架有哪些？各有什么特点？

| 框架 | 厂商 | 定位 | 核心优势 | 局限 |
|---|---|---|---|---|
| TensorRT | NVIDIA | 通用推理优化 | 图优化+量化+AutoTune，延迟最低 | 仅 NVIDIA GPU，算子覆盖有限 |
| ONNX Runtime | 微软 | 跨平台推理 | 多后端（CPU/GPU/NPU），生态好 | 性能非极致，对 LLM 支持有限 |
| vLLM | UC Berkeley | LLM 高吞吐推理 | PagedAttention+Continuous Batching | 仅面向生成式 LLM |
| TensorRT-LLM | NVIDIA | LLM 推理 | TRT 图优化+LLM 调度（PP/TP） | 仅 NVIDIA，API 不够简洁 |
| llama.cpp | 社区 | 轻量级推理 | 极低比特量化，CPU/GPU 通用 | 吞吐不如专业框架 |
| SGLang | UC Berkeley | LLM 推理 | RadixAttention 前缀复用 | 较新，生态不够成熟 |

**选择建议**：
- 追求极致延迟（CV/小模型）：TensorRT
- LLM 生产部署（高吞吐）：vLLM 或 TensorRT-LLM
- 端侧/低资源：llama.cpp
- 跨平台通用：ONNX Runtime
- 多轮对话/结构化生成：SGLang

### Q: CUDA 锁页内存（Pinned Memory）是什么？

锁页内存是通过 `cudaMallocHost()` 或 `cudaHostAlloc()` 在主机端分配的**物理地址固定**的内存。

**为什么需要**：
- 普通 malloc 分配的 pageable memory 可能被 OS 换出到磁盘（swap）
- GPU 的 DMA 引擎需要物理地址固定才能直接传输数据
- 使用 pageable memory 时，CUDA 驱动必须先将数据拷贝到内部的 pinned buffer，再由 DMA 传输——多了一次 CPU 端拷贝

**Pinned Memory 的优势**：
- **传输速度**：Host↔Device 传输带宽从 ~6 GB/s 提升到 ~12 GB/s（PCIe 3.0 满速），约 **2x 加速**
- **支持异步传输**：`cudaMemcpyAsync` 必须使用 pinned memory 才能真正异步（否则退化为同步）
- **实现计算-传输重叠**：配合 CUDA Stream，可以在传输数据的同时执行 kernel

**使用注意**：
- 占用物理内存且不可被换出，分配过多会压缩系统可用内存
- 分配和释放比 malloc 慢（需要通知 OS 锁定页面）
- 典型用量：不超过系统物理内存的 10-20%
- 分配后对 CPU 访问性能无影响（仍然在 CPU cache hierarchy 中）

**适用场景**：频繁 H2D/D2H 传输、需要 overlap 计算与传输的流水线（如推理引擎的输入预处理）、多 stream 并发传输。

**API 对比**：
```cpp
// 标准 pinned memory
cudaMallocHost(&ptr, size);
cudaFreeHost(ptr);

// 带选项的 pinned memory
cudaHostAlloc(&ptr, size, cudaHostAllocMapped);  // mapped: GPU 可直接访问
cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined); // 写合并，适合 GPU 只读
```
