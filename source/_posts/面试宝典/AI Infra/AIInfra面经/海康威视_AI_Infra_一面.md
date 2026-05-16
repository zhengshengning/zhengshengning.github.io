---
title: '海康威视 AI Infra 一面'
date: 2026-04-16 12:19:12
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 训练优化, 面经]
---


<!-- more -->

---

### Q: 国产AI平台和NVIDIA平台的主要差异？

国产 AI 芯片（华为昇腾/寒武纪/壁仞/海光 DCU 等）与 NVIDIA GPU 平台的差异体现在多个层面：

**1. 软件生态差距（最关键瓶颈）**：
- **NVIDIA**：CUDA 生态经过 15+ 年积累，cuBLAS/cuDNN/TensorRT/NCCL/Triton 等库高度成熟，开发者社区庞大
- **国产平台**：各家自建编程模型（华为 CANN/AscendCL，寒武纪 BANG C，壁仞 BIRCC），算子库覆盖度有限，社区小，遇到问题难以搜索解决方案
- **影响**：模型迁移需要重写或适配算子，新模型/新算子的支持滞后于 NVIDIA 数周到数月

**2. 通信互联差距**：
- **NVIDIA**：NVLink（900 GB/s per GPU on H100）+ NVSwitch（全互联）+ InfiniBand（400 Gbps）形成高速通信网络
- **国产平台**：芯片间互联带宽通常是 NVLink 的 1/5-1/3，缺乏成熟的全互联方案
- **影响**：TP（张量并行）效率受限，通信-计算重叠的设计空间更小，大规模并行训练效率低

**3. 编程模型与开发效率**：
- **NVIDIA**：统一的 CUDA 编程模型，丰富的调试/性能分析工具（Nsight Compute/Systems、cuda-gdb、compute-sanitizer）
- **国产平台**：各厂商接口不统一，调试工具相对简陋，Profile 粒度和准确性有限
- **趋势**：华为的 MindSpore/CANN 在逐步完善，但与 CUDA 生态差距仍然显著

**4. 实际性能（MFU）差距**：
- 单卡理论算力可能接近（如昇腾 910B 标称 FP16 320 TFLOPS vs A100 312 TFLOPS）
- 但实际 MFU（Model FLOPs Utilization）差距大：NVIDIA 大模型训练 MFU 可达 50-60%，国产平台通常 30-40%
- 原因：编译器优化不足、kernel 实现效率低、通信库不够成熟

**5. 兼容性与迁移成本**：
- PyTorch/TensorFlow 对 CUDA 原生支持，国产平台需要通过适配层（如华为的 torch_npu）接入
- 模型迁移通常需要：算子适配（有些算子国产平台不支持需要拆分/替换）、精度对齐（不同硬件的浮点行为差异）、性能调优
- 迁移一个大模型到国产平台典型需要 2-6 个月工程投入

**发展趋势**：国产平台在追赶中，华为昇腾生态最为完善（已支持 Llama/ChatGLM 等主流模型训练推理）。信创需求推动适配加速，但短期内 CUDA 仍是 AI Infra 的主流选择。

---

### Q: 分布式训练的瓶颈？

大规模分布式训练的瓶颈随着规模增长在不同阶段呈现不同特征：

**1. 通信瓶颈（最常见）**：
- **AllReduce 开销**：DDP 中每步梯度同步的通信量 = 2 * model_size * (N-1)/N。7B 模型 64 卡 Ring AllReduce 通信量 ~28GB/step
- **规模效应**：随卡数增加，通信占比从 10% 可能上升到 30-50%
- **缓解方法**：通信计算重叠（overlap backward 和 allreduce）、梯度压缩（FP16/INT8 通信）、增大 batch size 摊薄通信比例
- **TP 通信**：每个 transformer 层需要 2 次 AllReduce（attention + FFN），延迟累积显著

**2. Pipeline 气泡（PP 并行）**：
- 1F1B 调度下，气泡率 = (p-1) / (p-1+m)，p=stage 数，m=micro-batch 数
- 4 stage 16 micro-batch：气泡率 = 3/19 ≈ 16%
- **缓解**：增加 micro-batch 数、interleaved schedule（Megatron-LM 的虚拟 pipeline stage）、zero bubble PP

**3. 显存限制**：
- 模型参数 + 梯度 + 优化器状态 + 激活值：7B FP32 训练约需 7*20=140GB
- 激活值显存随 seq_len 和 batch size 增长快速增加
- **缓解**：ZeRO 切分、激活重计算（checkpoint）、offload 到 CPU/NVMe

**4. 数据 IO 瓶颈**：
- 大规模预训练的数据吞吐需求：1000 GPU * 每 GPU 处理 100K tokens/s = 100M tokens/s
- 数据读取、tokenize、shuffle 的速度需跟上 GPU 消耗速度
- **缓解**：预处理数据存为 MMap 格式、多 worker 并行加载、数据在线 streaming

**5. 容错性与稳定性**：
- 千卡规模训练持续数周，期间硬件故障（GPU 宕机、网络断连、NVLink 错误）概率高
- 1000 卡训练中，平均每 2-4 天可能遇到一次硬件故障
- **缓解**：频繁 checkpoint（每 100-500 步）、弹性训练（故障节点自动踢出）、冗余计算、heartbeat 监控

**6. 负载不均衡**：
- PP 中各 stage 计算量不等（embedding 层、LM head 层特殊）
- MoE 训练中不同专家负载不均
- 异构硬件（不同代 GPU 混用）的速度差异
- **缓解**：精细的 stage 划分（加权而非均匀）、负载均衡的专家路由、同步屏障等待最慢节点

---

### Q: 量化过程中出现掉点如何排查？

量化导致精度下降（掉点）是常见问题，需要系统化排查定位根因并针对性解决：

**Step 1: 逐层敏感度分析（定位问题层）**：
- 方法：每次只量化一层（其他层保持 FP16），观察端到端 PPL/accuracy 变化
- 工具：量化框架的 sensitivity analysis 功能或自定义 hook
- 典型发现：第一层（embedding 后的 linear）、最后一层（LM head）、attention 的 QK 投影层通常最敏感
- 量化标准：单层量化导致 PPL 上升 >0.1 视为"敏感层"

**Step 2: 检查分布异常（Outlier 问题）**：
- 统计每层权重和激活的分布（min/max/std/percentile）
- LLM 中普遍存在 "massive activation" 现象：某些通道的激活值比其他通道大 10-100x
- 可视化直方图：如果分布有长尾（少量极大值），min-max 量化范围被拉大，正常值精度下降
- 解决：SmoothQuant（将激活的 outlier 迁移到权重侧）、AWQ（保护重要通道不量化）

**Step 3: 提升量化粒度**：
- Per-tensor -> Per-channel -> Per-group（128/64/32 元素一组）
- 粒度越细，每组的 scale 越能匹配局部分布，量化误差越小
- Per-group 128 通常是精度和开销的平衡点（额外存储 scale/zp 的开销 ~3%）

**Step 4: 检查校准数据**：
- 校准数据是否代表实际推理数据的分布？
- 用少于 50 条校准数据可能统计不稳定
- 尝试增加校准数据量或更换更有代表性的数据集
- 不同校准数据可能导致 0.1-0.5 PPL 差异

**Step 5: 混合精度策略**：
- 基于 Step 1 的敏感度分析结果，对敏感层保持 FP16/BF16
- 常见保留高精度的层：第一个 Linear、LM Head、attention 的 Q/K 投影
- 通常保留 10-20% 的层为高精度，整体压缩比损失有限（如 W4 -> 有效 W4.5）

**Step 6: 升级量化算法**：
- MinMax PTQ -> GPTQ（逐列优化最小化逐层输出误差）
- GPTQ -> AWQ（识别并保护重要权重通道）
- PTQ -> QAT（有训练资源时，端到端微调量化参数）

**实践经验**：INT8 量化通常不需要特殊处理（PPL 上升 <0.1）；INT4 量化需要使用 GPTQ/AWQ + per-group 才能保证质量（PPL 上升 <0.5）；更激进的 INT2/INT3 通常需要 QAT。

---

### Q: 手撕：合并两个有序数组？

（编程题）
