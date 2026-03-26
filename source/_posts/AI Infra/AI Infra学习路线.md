---
title: AI Infra学习路线
date: 2026-03-26 10:00:00
categories:
  - [AI Infra, 学习路线]
tags: [AI Infra, LLM, 分布式训练, 推理优化, CUDA, 学习路线]
---

AI Infra（人工智能基础设施）是大模型时代壁垒最高、最核心的技术高地。本文从硬件基础到推理部署，系统梳理 AI Infra 的完整学习路线，为每个模块列出需要掌握的知识点、推荐学习资料以及可量化的检验标准，帮助从业者建立体系化的知识树。

<!-- more -->

## 目录

- [全景概览：四层架构](#全景概览四层架构)
- [第一层：硬件与通信网络](#第一层硬件与通信网络)
- [第二层：CUDA编程与算子优化](#第二层cuda编程与算子优化)
- [第三层：分布式训练](#第三层分布式训练)
- [第四层：推理与部署](#第四层推理与部署)
- [新人破局指南](#新人破局指南)

---

## 全景概览：四层架构

AI Infra 的本质是**"用系统工程释放硬件算力"**。我们可以将其自底向上分为四个核心层级：

| 层级 | 名称 | 核心关注点 |
|------|------|-----------|
| 第一层 | 硬件与通信网络 | GPU架构、显存带宽、NVLink、InfiniBand |
| 第二层 | CUDA编程与算子优化 | Kernel编写、FlashAttention、AI编译器 |
| 第三层 | 分布式训练 | 数据并行、3D并行、ZeRO、混合精度 |
| 第四层 | 推理与部署 | KV Cache、PagedAttention、量化、Speculative Decoding |

所有的优化都是在 **"计算、通信、显存"** 这个不可能三角中做取舍：ZeRO 是用通信换显存；重计算（Activation Checkpointing）是用计算换显存；量化是用精度换显存和带宽。学习时始终问自己：**这个技术牺牲了什么，换取了什么？**

---

## 第一层：硬件与通信网络

这是所有算力的物理基石。模型再精妙，最终也要转化为硅片上的电子跃迁。

### 1.1 知识点

**计算节点**

- GPU 核心架构：SM（流多处理器）、Tensor Core、CUDA Core 的区别与协作
- 主流 GPU 规格对比：A100 / H100 / H200 的算力、显存带宽、HBM 容量
- Memory Wall：为什么显存带宽瓶颈往往比算力瓶颈更致命
- 存储层次结构：寄存器 > 共享内存 > L1/L2 Cache > HBM > 主机内存

**通信拓扑**

- 单机内部通信：NVLink / NVSwitch 的带宽与拓扑
- 多机间通信：InfiniBand（IB）网络、RoCE 协议
- 集合通信原语：AllReduce、AllGather、ReduceScatter 的含义与开销
- NCCL：NVIDIA 集合通信库的基本用法与调优

### 1.2 推荐资料

| 类型 | 资料 | 说明 |
|------|------|------|
| 官方文档 | NVIDIA GPU 架构白皮书（Ampere / Hopper） | 理解 SM、Tensor Core、HBM 设计 |
| 官方文档 | NVIDIA NCCL 文档 | 集合通信原语与多卡编程 |
| 教程 | NVIDIA Deep Learning Performance Guide | 硬件性能瓶颈分析方法论 |
| 工具 | Nsight Systems User Guide | CPU-GPU 交互分析，判断 host 是否拖后腿 |
| 工具 | Nsight Compute Profiling Guide | Kernel 级下钻，定位 SM / Memory / Tensor Core 瓶颈 |

### 1.3 检验标准

- 能画出 A100/H100 的存储层次图，标注各级带宽和容量
- 能解释 NVLink 与 PCIe 的带宽差异及其对分布式训练的影响
- 能用 `nvidia-smi` 和 `nvidia-smi topo -m` 读懂 GPU 拓扑信息
- 能用 Nsight Systems 说明瓶颈在 host、PCIe/NVLink、kernel、mem 的哪一层
- 能用 Nsight Compute 说清关键 kernel 是 memory bound 还是 compute bound

---

## 第二层：CUDA编程与算子优化

这一层是连接硬件和软件的桥梁，负责把高层的数学计算翻译成 GPU 能最高效执行的机器指令。

### 2.1 知识点

**CUDA 编程基础**

- 编程模型：Grid / Block / Thread 层级，线程索引计算
- 内存模型：全局内存、共享内存、寄存器、常量内存的特性与用法
- 关键概念：Warp、Bank Conflict、Coalesced Access、Occupancy
- 核心直觉："内存访问模式决定运行速度"

**常见算子实现与优化**

- Reduce：并行归约的多种实现与优化（Warp Shuffle、多级归约）
- GEMM：矩阵乘法的分块、向量化、Shared Memory Tiling、利用 Tensor Core
- Softmax：Online normalizer calculation，高效 Softmax kernel
- 算子融合：将多个小算子合并为一个 kernel，减少全局内存读写

**Attention 算子**

- FlashAttention V1/V2：Memory-aware 的精确 Attention 实现，通过 tiling 减少 HBM 访问
- FlashAttention-3：在 Hopper 架构上进一步拉高利用率
- Flash-Decoding / FlashDecoding++：面向 Decode 阶段的 Attention 加速
- FlashInfer：可定制 Attention 引擎，面向 Serving 的可组合格式与异构 KV 存储
- PagedAttention CUDA Kernel：vLLM 中 PagedAttention 的底层实现

**AI 编译器**

- Triton：OpenAI 开源的 GPU 编程语言，大幅降低高效算子编写门槛
- TVM / XLA：计算图优化与代码生成
- `torch.compile`：PyTorch 2.x 的编译模式，理解 Graph Break 与性能收益

### 2.2 推荐资料

| 类型 | 资料 | 说明 |
|------|------|------|
| 入门教程 | 小小将：CUDA编程入门极简教程 | CUDA 零基础入门 |
| 官方文档 | NVIDIA CUDA Programming Guide | CUDA 编程权威参考 |
| Reduce | PeakCrosser：CUDA Reduce 算子优化 | Reduce 实现与优化的详尽总结 |
| GEMM | 猛猿：从啥也不会到CUDA GEMM优化 | 从基础分块到极致优化的 GEMM 教程 |
| GEMM | MegEngine Bot：CUDA 矩阵乘法终极优化指南 | 系统性的 GEMM 优化参考 |
| Softmax | Online normalizer calculation for softmax | NVIDIA 员工的 Softmax 实现论文 |
| Softmax | OneFlow：如何实现一个高效的Softmax CUDA kernel | Softmax kernel 工程实践 |
| 算子融合 | 成诚：OneFlow是如何做到世界最快深度学习框架的 | 算子融合思路与方法 |
| Attention | FlashAttention V1 Paper | Memory-aware Attention 的里程碑论文 |
| Attention | FlashAttention V2 Paper | 更好的并行与分块策略 |
| Attention | FlashAttention-3 Paper | Hopper 架构上的进一步优化 |
| Attention | Flash-Decoding 技术报告（Stanford CRFM） | Decode 阶段的 Attention 加速 |
| Attention | FlashInfer Paper + Repo | 可组合的 Attention 引擎 |
| 解读 | 猛猿：图解FlashAttention V1/V2 系列 | 适合新手入门的图文解读 |
| 解读 | 方佳瑞：深入浅出理解PagedAttention CUDA实现 | vLLM PagedAttention kernel 图文解读 |
| 编译器 | Triton 官方教程 | GPU 编程新范式 |
| 编译器 | PyTorch profiling torch.compile | 抓 Graph Break、编译收益与损耗 |

### 2.3 检验标准

- 能独立编写一个正确的 CUDA Reduce kernel 并做至少两轮优化
- 能解释 Shared Memory Bank Conflict 并写出避免冲突的访问模式
- 能用 Nsight Compute 分析自己写的 kernel，判断是 memory bound 还是 compute bound
- 能解释 FlashAttention 的核心思想：为什么 tiling 能减少 HBM 访问
- 能解释 Attention 推理时的数据流：Q/K/V、KV Cache、Softmax、写回
- 能用 Triton 写出一个简单算子（如向量加法或 Softmax）并对比 CUDA 实现

---

## 第三层：分布式训练

当模型参数量超越单卡显存极限时，分布式训练就是必经之路。这是 AI Infra 目前最活跃、最核心的区域。

### 3.1 知识点

**模型基础**

- Transformer 架构：Attention Is All You Need，理解 Self-Attention、FFN、LayerNorm
- Attention 变种：MHA、MQA、GQA、MLA 的区别与演进（推荐阅读 DeepSeek V2 技术报告）
- FFN 变种：混合专家模型 MoE（DeepSeekMoE）

**数据并行**

- DP（DataParallel）：最基础的数据并行，单进程多卡
- DDP（DistributedDataParallel）：多进程数据并行，理解 AllReduce 梯度同步
- FSDP（Fully Sharded Data Parallel）：PyTorch 原生的 ZeRO-3 实现

**模型并行（3D 并行）**

- 张量并行（TP）：将矩阵乘法沿特定维度切分到多卡，通信密集，通常限于单机
- 流水线并行（PP）：将模型不同层切分到不同机器，像流水线一样传递数据
- 序列并行（SP）：沿序列维度切分，与 TP 配合减少激活显存

**显存优化**

- ZeRO 系列（DeepSpeed）：
  - ZeRO-1：优化器状态切分
  - ZeRO-2：优化器状态 + 梯度切分
  - ZeRO-3：优化器状态 + 梯度 + 参数切分（用通信换显存）
- 混合精度训练：FP16 / BF16 / FP8 训练，减少显存占用与计算开销
- 梯度累积：在有限显存下模拟更大的有效 Batch Size
- Activation Checkpointing（重计算）：用计算换显存，只保存部分激活值

**训练框架**

- Megatron-LM：张量并行与流水线并行的标杆实现
- DeepSpeed：ZeRO 系列的核心实现，丰富的训练优化工具集
- PyTorch FSDP：原生分布式训练方案

### 3.2 推荐资料

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | Attention Is All You Need | Transformer 基础，必读 |
| 解读 | 琳琅阿木：图文详解LLM inference | LLM 模型架构详解 |
| 论文 | DeepSeek V2 技术报告 | MLA 注意力机制 |
| 论文 | DeepSeekMoE Paper | MoE 架构设计 |
| 教程 | 混合专家模型 (MoE) 详解 | MoE 入门 |
| 论文 | Megatron-LM Paper | TP 与 PP 原理的里程碑论文 |
| 论文 | ZeRO Paper（DeepSpeed） | 显存优化的核心方法 |
| 文档 | DeepSpeed 官方文档 | ZeRO 配置与使用 |
| 文档 | PyTorch DDP / FSDP 教程 | 原生分布式训练入门 |
| 解读 | 苏剑林：从MHA、MQA、GQA到MLA | Attention 变种演进 |

### 3.3 检验标准

- 能用 PyTorch DDP 将单卡训练脚本改造为多卡分布式训练
- 能解释 ZeRO-1/2/3 各切分了什么，通信量如何变化
- 能画出 TP + PP 的 3D 并行拓扑图，标注通信位置与通信量
- 能计算给定模型的参数量、优化器状态、梯度所需的显存占用
- 能解释混合精度训练为什么需要 Loss Scaling，以及 BF16 vs FP16 的差异
- 能配置 DeepSpeed ZeRO Stage 2/3 并跑通一个训练任务

---

## 第四层：推理与部署

训练只是万里长征第一步。如何让模型快速、低成本地服务用户，是工业界最关心的问题。

### 4.1 LLM 推理基础

**知识点**

- LLM 推理的两阶段：Prefill（处理输入）与 Decode（逐 token 生成）
- KV Cache：自回归生成中的"显存刺客"，理解其生命周期和碎片问题
- 关键性能指标：TTFT（首 token 延迟）、TPOT（每 token 延迟）、吞吐量（token/s）、P50/P95 尾延迟

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 解读 | 琳琅阿木：图文详解LLM inference——KV Cache | KV Cache 原理详解 |
| 综述 | Towards Efficient Generative LLM Serving: A Survey | CMU 的 LLM 推理综述（算法 + 系统） |

**检验标准**

- 能解释 Prefill 和 Decode 的计算特性差异（compute bound vs memory bound）
- 能给出 KV Cache 的显存占用公式，并据此估算给定 batch size 下的显存需求
- 能把推理链路拆成：tokenize → prefill → decode → sampling → postprocess，并标出每段耗时特征

### 4.2 推理引擎

**知识点**

- PagedAttention：vLLM 提出的虚拟内存分页思想管理 KV Cache，解决碎片化问题
- Continuous Batching：动态组批，请求随到随处理，与传统 static batching 的差异
- Prefix Cache / RadixAttention：复用已计算的 KV Cache，优化重复前缀场景
- Chunked Prefill：将长 prompt 分块处理，减少 prefill 对 decode 的干扰

**主流推理框架**

| 框架 | 核心特性 | 适用场景 |
|------|---------|---------|
| vLLM | PagedAttention、Continuous Batching、Prefix Cache | 通用推理服务，社区活跃 |
| SGLang | RadixAttention、cFSM 结构化输出加速 | 复杂 Agent、多轮生成、结构化输出 |
| TensorRT-LLM | Paged KV、Inflight Batching、深度硬件优化 | 追求极限性能、NVIDIA 生态 |

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | vLLM Paper（PagedAttention） | 推理引擎里程碑论文 |
| 源码 | vLLM GitHub Repo | 实战特性全集 |
| 文档 | vLLM 官方文档 | 部署与配置 |
| 解读 | 猛猿：vLLM 源码解析系列（架构/调度器/BlockManager） | 深入理解 vLLM 内部机制 |
| 论文 | SGLang Paper | RadixAttention + 前端语言 |
| 源码 | SGLang GitHub Repo | 工程实现 |
| 文档 | TensorRT-LLM 官方文档 | 工程落地导向 |
| 论文 | Orca Paper | Continuous Batching 的原始论文 |
| 解读 | 吃果冻不吐果冻皮：Continuous Batching | Continuous Batching 中文解读 |
| 解读 | DefTruth：vLLM Prefix Cache 原理图解 | Prefix Cache 万字详解 |

**检验标准**

- 能用同一模型在 vLLM 和 SGLang 上跑出可对比的 TTFT / TPOT
- 能解释 Continuous Batching 与传统 Static Batching 的差异与收益
- 能说清 KV Cache 的生命周期、碎片问题，以及 PagedAttention / RadixAttention 如何缓解
- 能给出选型建议：低延迟、长上下文、多并发、结构化输出各用哪个框架更合理

### 4.3 量化

**知识点**

- W8A8（SmoothQuant）：将 activation 的 outlier 难题转移到 weights，工程友好
- Weight-only INT4（GPTQ / AWQ）：只量化权重到 3/4-bit，减少显存和带宽占用
- KV Cache 量化（KIVI / Kitty）：对 KV Cache 进行 2-bit 量化，长上下文场景效果显著
- FP8 量化：Hopper 架构原生支持，精度与性能的平衡点
- 量化选择决策树：

```
目标：省显存？省带宽？提吞吐？
├─ 通用、工程友好 → W8A8 (SmoothQuant)
├─ 更省显存/带宽 → INT4 weight-only (AWQ/GPTQ)
└─ 长上下文/大并发 → KV Cache 量化 (KIVI)
```

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | SmoothQuant Paper | W8A8 量化的经典方案 |
| 论文 | GPTQ Paper | Weight-only PTQ 3/4-bit |
| 论文 | AWQ Paper | 基于 activation 分布的 4-bit 量化 |
| 论文 | KIVI Paper | 2-bit KV Cache 量化 |
| 论文 | Marlin Paper（FP16xINT4） | INT4 高性能 matmul kernel |
| 代码 | TensorRT-LLM 量化工具链 | FP8/INT4/AWQ/SmoothQuant 集成 |
| 代码 | vLLM 量化支持 | GPTQ/AWQ/FP8/INT8 多种接入 |

**检验标准**

- 能区分并选择 W8A8 vs Weight-only INT4 的适用场景、代价与收益
- 能在同模型上跑通 FP16 baseline 与 INT4 量化版本，输出精度对比
- 能解释为什么有时"更低 bit 反而更慢"（kernel 开销、packing、带宽、并行度）
- 能给出量化失败排查清单（数值爆炸、输出退化、吞吐无提升）
- 能写出一个可复用的量化评测脚本（吞吐 + 延迟 + 简单质量指标）

### 4.4 Speculative Decoding

**知识点**

- Speculative Sampling：经典框架——用小模型（Draft）批量"猜测"多个 token，大模型（Target）一次性验证，保证分布无偏
- Medusa：不用外部 Draft 模型，通过多个 Decoding Heads 预测多 token 再并行验证
- EAGLE-2：动态 Draft Tree，靠校准置信度更激进地产生可接受 token
- Block Verification：将 token 级验证升级为 block 级联合验证，进一步提速

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | Speculative Sampling Paper | 经典 Speculative Decoding |
| 论文 | Medusa Paper + Repo | 多头解码，免 Draft 模型 |
| 论文 | EAGLE-2 Paper | 动态 Draft Tree |
| 论文 | Block Verification Paper | Block 级联合验证 |
| 代码 | vLLM / TensorRT-LLM 的 Speculative 支持 | 工程落地参考 |
| 教程 | SGLang 结构化输出加速（cFSM） | 结构化生成场景加速 |

**检验标准**

- 能解释 Speculative Decoding 的正确性保证（目标分布不变）
- 能在同模型上跑出 baseline vs speculative 的 TPOT 改善，并分析 TTFT 变化
- 能给出"什么时候 speculative 不赚"的判断依据（任务类型、温度、Draft 质量、batch 大小）
- 能输出 acceptance length / ratio 的统计表
- 能提出至少 1 个与量化 / 批处理 / 长上下文耦合的风险点及规避策略

### 4.5 系统架构：Prefill/Decode 解耦

**知识点**

- 核心问题：Prefill 与 Decode 混合 batching 造成资源耦合与互扰，导致尾延迟爆炸
- DistServe（OSDI'24）：系统化论证并实现 Prefill/Decode 解耦，围绕 goodput 调度
- Splitwise（ISCA）：将 Prefill 和 Decode 分配到不同 GPU 池，优化吞吐与成本
- TaiChi（2025）：将聚合与解耦统一，面向不同 SLO 组合做最优 goodput
- Goodput：满足 SLO 的有效吞吐，区别于裸 QPS

```
传统聚合：请求 → [同一组GPU] prefill + decode 交织 → 互扰、尾延迟爆炸

解耦架构：请求 → Prefill GPU池 → KV 状态迁移 → Decode GPU池
                                    ↑
                          调度器按 SLO 做 goodput 优化
```

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | DistServe（OSDI'24） | Prefill/Decode 解耦的系统化论证 |
| 论文 | Splitwise（ISCA） | Phase Splitting 设计 |
| 论文 | TaiChi（2025） | 聚合与解耦统一框架 |
| 教程 | Disaggregated Inference: 18 Months Later | 解耦为何成为默认配置 |
| 教程 | MLC microserving | 跨引擎编排的可编程 API |

**检验标准**

- 能清楚定义并计算 TTFT SLO、TPOT SLO、goodput
- 能解释 Prefill/Decode 互扰的来源（batch 资源耦合、调度冲突）
- 能给出 Prefill-heavy vs Decode-heavy 的资源配比方案并解释依据
- 能列出至少 3 个系统风险点（KV 迁移延迟、网络带宽、尾延迟、队列震荡）及缓解手段

### 4.6 性能分析与 Benchmark

**知识点**

- 核心指标体系：QPS、TTFT、TPOT、token/s、P50/P95 尾延迟
- 性能分析工具链：
  - `torch.profiler`：PyTorch 官方 profiler，定位算子与 shape
  - Nsight Systems：CPU-GPU 交互全链路分析
  - Nsight Compute：Kernel 级性能下钻
- 压测工具：GenAI-Perf（LLM 指标一站式输出）、Triton Perf Analyzer
- 权威基准：MLPerf Inference（Datacenter），统一口径与规则
- 回归门禁：每次改动输出同一张指标对比表，退化则阻止合并

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 工具 | torch.profiler 文档 | PyTorch 性能分析起点 |
| 工具 | GenAI-Perf | TTFT/TPOT/token throughput 一站式压测 |
| 工具 | Triton Inference Server Quickstart | 在线推理容器化基线 |
| 基准 | MLPerf Inference（Datacenter） | 权威 benchmark 入口 |
| 解读 | MLPerf Inference v5.0 LLM 任务解读 | 理解低延迟 LLM benchmark 趋势 |

**检验标准**

- 能同时报告至少 5 个指标：QPS、TTFT、TPOT、token/s、P50/P95
- 有一套固定的 benchmark 配置（模型、batch、context、并发、硬件）
- 每次改动都能输出同一张指标对比表
- 能把一次性能回退定位到具体 commit / 配置 / 驱动变化
- 能写出"上线门禁"规则：什么指标退化会阻止合并
- 能写出一份"一页纸"性能报告模板

---

## 新人破局指南

### 推荐学习路径

面对这么庞大的技术栈，建议采取**需求驱动**而非自底向上的学习路径：

**基础阶段（0-3个月）**

1. 主攻 Python / C++、Linux 系统基础
2. 精读 Transformer 论文，用 PyTorch 跑通一个小模型训练
3. 学习 CUDA 编程基础，能写出简单的 Reduce / GEMM kernel
4. 尝试用 PyTorch DDP 将训练分布到两张卡上，观察显存和通信变化

**专项深入（3-6个月）**

1. 精读四篇里程碑论文并对照代码：
   - Megatron-LM（TP 与 PP 原理）
   - ZeRO（DeepSpeed 核心）
   - FlashAttention（Memory-aware 算法）
   - vLLM（PagedAttention、KV Cache）
2. 参与开源项目（vLLM、DeepSpeed、SGLang），贡献算子优化或功能模块
3. 掌握量化、Speculative Decoding 等推理优化技术

**工程实践（6个月以上）**

1. 在 GPU 集群上部署百亿 / 千亿参数模型，优化端到端性能
2. 建立完整的性能分析与回归体系
3. 研究 Prefill/Decode 解耦等前沿系统架构
4. 跟踪最新技术迭代（FP8、RDMA 网络优化、新架构适配等）

### 选型决策树

当你遇到推理性能问题时，按以下决策树定位方向：

```
你最痛的是哪一个？
├─ TTFT 很大（首 token 慢）
│  ├─ prompt 很长 → Prefill 优化 / Chunked Prefill / 更快 GEMM
│  ├─ CPU/调度慢 → nsys 找 host bottleneck，换更成熟 runtime
│  └─ 频繁重复前缀 → Prefix/KV 复用
│
├─ TPOT 很大（续杯慢）
│  ├─ decode memory-bound → FlashAttention/FlashInfer + KV Cache 管理
│  ├─ token-by-token 串行 → Speculative / Medusa / EAGLE-2
│  └─ batch 太小 → Continuous Batching + 合理并发
│
├─ 显存爆了
│  ├─ KV Cache 占用大 → PagedAttention + KV 量化(KIVI)
│  └─ 权重占用大 → INT4 weight-only(AWQ/GPTQ) 或 W8A8
│
└─ 尾延迟 P95 爆炸
   ├─ prefill/decode 互扰 → Prefill/Decode 解耦(DistServe/Splitwise)
   └─ SLO 要求苛刻 → TaiChi 类统一调度方案
```

### 核心思维模型

学习 AI Infra 的过程中，始终牢记一个权衡思维：

| 优化技术 | 牺牲了什么 | 换取了什么 |
|---------|-----------|-----------|
| ZeRO | 通信带宽 | 显存空间 |
| Activation Checkpointing | 计算时间 | 显存空间 |
| 量化 | 精度 | 显存 + 带宽 + 吞吐 |
| Speculative Decoding | Prefill 开销 | Decode 速度 |
| Prefill/Decode 解耦 | 系统复杂度 + KV 迁移开销 | 尾延迟 + goodput |
| FlashAttention | 实现复杂度 | 显存 + 速度 |

---
