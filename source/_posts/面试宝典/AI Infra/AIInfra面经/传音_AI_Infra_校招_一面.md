---
title: '传音 AI Infra 校招 一面'
date: 2026-04-16 12:20:53
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 量化的校准数据集有什么要求？

校准数据集（Calibration Dataset）是 PTQ（训练后量化）中用于统计权重/激活分布的关键输入，直接决定量化参数（scale/zero_point）的质量。

**核心要求**：

1. **分布代表性**：校准数据必须覆盖实际推理数据的分布特征。对 LLM 来说，需要涵盖不同长度、不同领域（代码/数学/对话/知识问答）的输入，确保激活值的统计量（min/max/percentile）能反映真实推理场景。如果校准数据全是短文本，而推理时遇到长文本，激活分布可能完全不同。

2. **数量要求**：通常 100-1000 条即可。TensorRT 官方建议 500 条左右。过少（<50）统计不稳定，过多不显著提升精度但增加校准时间。对 GPTQ/AWQ 等高级方法，128 条通常就足够（因为这些方法做逐层优化而非简单统计）。

3. **多样性 > 数量**：10 条覆盖 10 种模式的数据优于 100 条同质数据。需要覆盖各种输入模式（长/短序列、不同值域范围、不同语言/任务类型）。

4. **避免极端分布**：校准数据不应包含过多 outlier，否则 min-max 范围被拉大导致量化精度下降。解决方法是使用 percentile（如 99.99%）而非 min-max 来确定范围。

5. **与训练数据无关**：校准数据不需要标签，只需输入即可（前向推理统计激活分布）。可以从验证集或实际业务数据中随机采样。

**实践建议**：使用 WikiText-2 的随机 128 条样本作为通用校准集；领域模型使用领域数据；多模态模型需要覆盖不同模态组合。

### Q: 量化的原理？

量化（Quantization）的本质是将连续的高精度浮点数映射为离散的低精度整数值，通过牺牲少量数值精度换取存储压缩和计算加速。

**线性量化的数学公式**：

- **量化（Quantize）**：q = clamp(round(x / scale + zero_point), q_min, q_max)
- **反量化（Dequantize）**：x_approx = (q - zero_point) * scale
- **Scale 计算**：scale = (x_max - x_min) / (q_max - q_min)

**对称 vs 非对称量化**：
- **对称量化**：zero_point = 0，scale = max(|x|) / (2^(bits-1) - 1)。优点是计算简单（无 zero_point 偏移），适合权重（通常近似零均值）
- **非对称量化**：zero_point != 0，完整利用量化范围。适合激活值（如 ReLU 后全为正值）

**量化粒度**（从粗到细）：
- **Per-tensor**：整个张量一组 scale/zp，最简单但精度最差
- **Per-channel**：每个输出通道独立 scale/zp，权重量化标配
- **Per-group**：每 128/64 个元素一组，GPTQ/AWQ 使用的粒度，精度接近 FP16

**量化误差分析**：误差 = x - x_approx = x - (round(x/s) * s)，最大舍入误差为 scale/2。因此 scale 越小（量化范围越紧）精度越高，但超出范围的值会被 clamp 产生大误差（截断误差）。好的校准就是在舍入误差和截断误差之间找平衡。

**为什么量化能加速**：INT8 Tensor Core 的吞吐是 FP16 的 2x（A100: 624 vs 312 TOPS），且 INT8 数据传输带宽需求减半，对 memory-bound 算子（如 decode 阶段 GEMV）加速更明显。

### Q: 量化的精度评估方法？

量化后的精度评估需要从局部到全局、多维度进行：

**1. 逐层输出对比（定位问题层）**：
- 对每层分别量化，比较量化前后输出的 MSE（均方误差）和余弦相似度
- MSE > 0.01 或余弦相似度 < 0.99 的层视为"敏感层"
- 工具：PyTorch 的 hook 机制逐层比较，或量化框架的 layer-wise analysis 功能

**2. 端到端任务指标**：
- LLM：Perplexity（WikiText-2/C4 上 PPL 上升 <0.5 可接受）
- 分类：Accuracy drop < 1%
- 生成：BLEU/ROUGE 变化
- 实践标准：INT8 通常 <0.5% 掉点，INT4 通常 <1-2% 掉点

**3. 敏感度分析（Sensitivity Analysis）**：
- 方法：每次只量化一层（其他层保持 FP16），逐层观察 PPL 变化
- 发现规律：第一层 embedding、最后一层 LM head、attention 的 QK 投影层通常最敏感
- 用途：指导混合精度量化策略

**4. 混合精度搜索**：
- 基于敏感度分析结果，敏感层保持 FP16/BF16，非敏感层用 INT8/INT4
- 约束条件：总显存预算、目标加速比
- 搜索算法：贪心（按敏感度排序逐层降精度）或启发式搜索

**5. 分布分析**：
- 可视化权重/激活的直方图，检查是否存在 outlier（如 LLM 中著名的"massive activation"问题）
- Outlier 会导致量化范围过大，使大部分正常值的有效位数减少
- 解决：SmoothQuant（平衡权重和激活的量化难度）、AWQ（保护重要通道）

### Q: 量化计算scale的方法？

Scale 的选择直接决定量化精度，核心挑战是确定最优的量化范围 [x_min, x_max]：

**1. Min-Max（最简单）**：
- scale = (max(x) - min(x)) / (2^bits - 1)
- 优点：实现简单、无损（所有值都在范围内）
- 缺点：对 outlier 极度敏感。一个 outlier 将范围拉大，使 99.9% 正常值的量化精度大幅下降
- 适用：权重量化（分布稳定、outlier 少）

**2. Percentile（分位数截断）**：
- 取 99.9% 或 99.99% 分位数作为范围边界，超出范围的值被 clamp
- 直觉：允许少量极端值被截断，换取大部分值的更高精度
- 参数选择：通常 99.99% 用于权重，99.9% 用于激活（激活 outlier 更多）
- 缺点：需要人工选择分位数，最优值因模型和数据而异

**3. KL 散度校准（TensorRT 方法）**：
- 搜索最优阈值 T，使量化后分布与原始分布的 KL 散度最小
- 具体做法：对激活分布的直方图（2048 bins），尝试不同截断位置，计算 KL(P || Q_quantized)
- 优点：理论上信息损失最小，是 TensorRT INT8 的默认方法
- 缺点：需要遍历搜索，校准时间较长

**4. MSE 最小化（最直观）**：
- 直接搜索使 E[(x - dequant(quant(x)))^2] 最小的 scale
- 可以对 scale 做一维搜索（黄金分割/网格搜索），或用解析近似
- GPTQ/AWQ 在此基础上进一步做逐列/逐组的 MSE 优化
- 优点：目标函数直接对应量化误差

**5. 学习 Scale（LSQ - Learned Step Size Quantization）**：
- 将 scale 作为可训练参数，通过 QAT 的反向传播学习最优值
- 使用 STE（Straight-Through Estimator）传递量化操作的梯度
- 优点：端到端优化，精度最好；缺点：需要训练

**实践选择**：PTQ 快速上线用 Percentile 或 KL 散度；追求极致精度用 MSE 最小化或 GPTQ；有训练资源则用 QAT+LSQ。

### Q: 吞吐量如何计算？

吞吐量（Throughput）衡量系统单位时间内的处理能力，在不同场景下定义和计算方式不同：

**LLM 推理吞吐量**：
- **Token 吞吐量**：tokens/second（生成的 token 数 / 总时间）
- **请求吞吐量**：requests/second（完成的请求数 / 总时间）
- 注意区分：Prefill 吞吐（input tokens/s）和 Decode 吞吐（output tokens/s）通常分开衡量

**计算公式**：
- 单请求延迟 = TTFT（首 token 延迟）+ (output_len - 1) * TPOT（每 token 延迟）
- 系统吞吐 = 并发请求数 * 平均每请求 output tokens / 平均请求延迟

**影响因素分析**：
- **Batch Size**：增大 batch 提升吞吐但增加延迟（throughput-latency tradeoff）
- **模型大小**：参数量决定每步计算和访存量
- **硬件算力**：GPU TFLOPS 决定 compute-bound 场景的上限
- **内存带宽**：HBM 带宽决定 memory-bound 场景（decode 阶段）的上限
- **KV Cache 显存**：限制最大并发 batch size

**理论峰值估算**（decode 阶段，memory-bound）：
- 每生成一个 token 需读取全部模型参数：7B * 2 bytes = 14GB
- A100 HBM 带宽 2TB/s，理论上限：2000/14 = 143 tokens/s（单请求）
- 实际因 KV Cache 读取等开销，约达理论值 60-80%

**优化吞吐的方法**：continuous batching（动态批处理）、量化减少访存、speculative decoding（投机解码）、PagedAttention（减少内存碎片提高可用 batch size）。

### Q: 推理框架有哪些？

主流推理框架按场景和特点分类：

**通用 GPU 推理**：
- **TensorRT**（NVIDIA）：图优化（层融合/常量折叠）+ kernel auto-tuning + INT8/FP8 量化。延迟最优，但构建时间长、灵活性低。适合固定模型的生产部署。
- **ONNX Runtime**（Microsoft）：跨平台（CPU/GPU/NPU），支持多种 EP（Execution Provider）。兼容性最好，但峰值性能不如专用框架。

**LLM 专用推理**：
- **vLLM**：PagedAttention 实现高效 KV Cache 管理 + continuous batching。吞吐量业界领先，适合大规模在线服务。
- **TGI**（HuggingFace）：开箱即用，支持多种模型，集成量化/水印/流式输出。适合快速上线。
- **SGLang**：RadixAttention（前缀树优化 KV Cache 复用）+ 编程式 LLM 调用。适合复杂推理流程。
- **TensorRT-LLM**（NVIDIA）：基于 TensorRT 的 LLM 专用优化，支持 TP/PP + inflight batching + FP8。单卡延迟最优。

**轻量/端侧推理**：
- **llama.cpp**：纯 C/C++ 实现，支持 CPU/GPU/Metal，量化格式丰富（Q2-Q8）。适合端侧和个人部署。
- **MLC-LLM**：基于 TVM 编译优化，支持跨设备（手机/浏览器/GPU）。适合异构部署。
- **ncnn/MNN/TNN**（移动端）：ARM NEON/GPU 优化，轻量级，适合手机端 CV 模型。

**选择建议**：追求吞吐选 vLLM/SGLang；追求延迟选 TensorRT-LLM；端侧选 llama.cpp/MLC-LLM；需要跨平台兼容选 ONNX Runtime。

### Q: 计算图的构建过程？

计算图（Computation Graph）是深度学习编译和推理优化的核心数据结构，将模型表示为有向无环图（DAG）：

**1. 前端解析（Frontend Parsing）**：
- 从训练框架导出模型：PyTorch -> TorchScript/ONNX，TF -> SavedModel/GraphDef
- 将动态图（eager mode）转为静态图（IR 表示）
- 处理动态控制流：将 if/for 转为静态子图或保留动态节点
- 关键挑战：动态 shape 处理（序列长度/batch size 变化）

**2. 图构建（Graph Construction）**：
- 算子（Op）表示为节点，张量（Tensor）流向表示为边
- 每个节点记录：算子类型、输入输出 shape/dtype、属性（如 conv 的 stride/padding）
- 类型推断：根据输入推导所有中间张量的 shape 和 dtype
- 常见 IR：ONNX（跨框架标准）、TorchScript、TVM Relay、StableHLO

**3. 图优化（Graph Optimization）**：
- **常量折叠（Constant Folding）**：编译期计算不变量（如 BatchNorm 的 running_mean/var 合并到 weight）
- **算子融合（Operator Fusion）**：将多个算子合并为一个 kernel（如 Conv+BN+ReLU -> 单 kernel），减少 kernel launch 和中间 tensor 的全局内存读写
- **死代码消除（DCE）**：删除没有消费者的节点
- **布局转换（Layout Transform）**：选择硬件最优的内存布局（NCHW vs NHWC，A100 偏好 NHWC）
- **代数简化**：x*1->x，x+0->x，transpose(transpose(x))->x

**4. 内存规划（Memory Planning）**：
- 分析张量生命周期（liveness analysis）：确定每个张量的创建和最后使用时间
- 内存复用：生命周期不重叠的张量共享内存地址，减少峰值显存
- 实践效果：好的内存规划可减少 30-50% 的中间张量显存占用

**5. 代码生成（Code Generation）**：
- 将优化后的图转为可执行 kernel 序列
- 选择最优 kernel 实现：cuBLAS/cuDNN 调用或自动生成的 kernel
- Auto-tuning：对不同 kernel 配置（tile size/unroll factor）做性能测试选最优

### Q: 卷积算子的实现方式？

卷积是 CNN 的核心算子，不同实现方式适用于不同的 kernel size 和输入规模：

**1. Im2col + GEMM（最通用）**：
- **原理**：将卷积重新组织为矩阵乘法。把输入中每个滑窗位置展开为一列，形成 [C_in*K*K, H_out*W_out] 矩阵，与权重矩阵 [C_out, C_in*K*K] 做 GEMM
- **优点**：可直接调用高度优化的 GEMM 库（cuBLAS），实现简单且性能稳定
- **缺点**：Im2col 导致内存膨胀 K^2 倍（3x3 卷积膨胀 9 倍），额外的数据重排开销
- **适用**：通用场景，是大多数框架的默认实现
- **优化**：Implicit GEMM（不显式展开，在 GEMM kernel 内部隐式计算索引映射）消除内存膨胀

**2. Direct Convolution（直接计算）**：
- **原理**：按卷积定义逐元素计算，7 层嵌套循环
- **优点**：无额外内存开销，实现直观
- **缺点**：数据复用率低，难以充分利用硬件并行性和缓存
- **适用**：1x1 卷积（退化为 GEMM）、depthwise 卷积（每通道独立，内存膨胀不划算）

**3. Winograd 变换（小 kernel 加速）**：
- **原理**：基于 Winograd 最小滤波算法，将卷积转换为元素乘法，减少乘法次数。F(2,3) 将 3x3 卷积的 9 次乘法减为 4 次（2.25x 加速）
- **优点**：显著减少计算量（FLOPs）
- **缺点**：变换本身有开销、数值精度下降（变换矩阵含大系数）、只适用于小 kernel（3x3/5x5）、不适合大 stride
- **适用**：3x3 卷积（CNN 中占比 70%+），ResNet/VGG 等经典模型

**4. FFT 卷积（大 kernel 场景）**：
- **原理**：利用卷积定理——时域卷积等于频域点乘。对输入和 kernel 做 FFT，频域逐元素相乘，再 IFFT 回时域
- **优点**：大 kernel 时计算复杂度从 O(N^2*K^2) 降为 O(N^2*logN)
- **缺点**：FFT 本身开销大，小 kernel 时不如直接计算；需要 padding 到 2 的幂次
- **适用**：kernel size > 7 的场景（如音频处理、某些科学计算）

**实践选择**：3x3 卷积用 Winograd 或 Im2col+GEMM；1x1 卷积直接用 GEMM；depthwise 用 Direct；大 kernel 用 FFT。cuDNN 会根据输入 shape 自动选择最优算法（cudnnFindConvolutionForwardAlgorithm）。

### Q: 矩阵乘分块（Tiling）的原理？

Tiling（分块）是高性能 GEMM 的核心优化技术，通过将大矩阵切分为适配缓存层级的小块，最大化数据复用率：

**基本原理**：
- 计算 C[M,N] = A[M,K] * B[K,N] 时，如果 M/N/K 很大，A/B 无法完整装入 cache
- Tiling 将 C 分为 TM*TN 的小块，每块计算只需 A 的 TM*TK 和 B 的 TK*TN 子矩阵
- 这些子矩阵装入 shared memory/L1 cache 后，被复用 TN/TM 次

**多级 Tiling（与硬件缓存层级对应）**：
- **Block-level tile**：将 C 的 128x128 分配给一个 thread block，对应数据从 Global Memory 加载到 Shared Memory（~100KB）
- **Warp-level tile**：block 内每个 warp 负责 64x32 的子块，从 Shared Memory 加载到寄存器
- **Thread-level tile**：每个线程计算 8x8 的输出元素，对应 8+8 个寄存器值的乘加

**数据复用率分析**：
- 无 tiling 时，每个乘法需从 DRAM 读一次 A 和 B 元素，总访存 = 2MNK
- 有 tiling 时，A 的一个 tile 被 B 的 TN/TK 个 tile 复用，总访存 = MNK/TN + MNK/TM
- 复用率提升：TM * TN 倍（若 TM=TN=128，则 128x 减少全局访存）

**Tile 大小选择原则**：
- Shared Memory tile 大小受限于 SM 上的共享内存容量（A100: 164KB configurable）
- 寄存器 tile 大小受限于每线程可用寄存器数（255 个）
- Tile 过大导致 occupancy 下降（资源不足以启动更多 block），过小导致复用率不够
- 最优点需要 auto-tuning（如 CUTLASS 的 tile size 模板参数搜索）

**双缓冲（Double Buffering）**：在计算当前 tile 时预取下一个 tile，将访存延迟隐藏在计算中。这需要额外的 shared memory/寄存器空间，但能显著提升吞吐。

### Q: 大模型分词器（Tokenizer）？

Tokenizer 是 LLM 的输入预处理模块，将原始文本切分为模型可处理的 token 序列：

**主流算法**：

**1. BPE（Byte-Pair Encoding）**：
- 训练过程：从字符级别开始，迭代合并语料中出现频率最高的相邻 token 对，直到达到目标词表大小
- 优点：无 OOV（未登录词）问题（最坏降级为字节级）、压缩效率好
- 使用：GPT 系列（GPT-2/3/4）、LLaMA 系列
- 词表大小：GPT-2 用 50K，LLaMA-2 用 32K，LLaMA-3 扩大到 128K

**2. WordPiece**：
- 类似 BPE 但合并标准不同：选择使语言模型似然度最大增长的 pair 合并
- 使用 `##` 前缀标记非首子词（如 "playing" -> "play" + "##ing"）
- 使用：BERT、DistilBERT

**3. SentencePiece**：
- 直接在原始文本（包括空格）上训练，不依赖预分词，适合无空格语言（中文/日文）
- 支持 BPE 和 Unigram 两种模式
- Unigram：从大词表出发，迭代删除对似然度影响最小的 token，直到达到目标大小
- 使用：T5、LLaMA

**关键设计决策**：
- **词表大小**：32K（LLaMA-2）vs 128K（LLaMA-3）。更大词表 -> 更短序列（推理步数少）但 embedding 层参数增加
- **Byte-level fallback**：确保任何输入都可编码，避免 UNK token
- **特殊 token**：BOS/EOS/PAD/MASK 等控制 token

**对推理的影响**：平均 token 数直接决定自回归生成步数。同样的文本，128K 词表比 32K 词表少约 15-20% 的 token，意味着推理延迟等比减少。中文场景下大词表优势尤为明显（常见汉字组合可被单 token 表示）。

### Q: ARM NEON是什么？

ARM NEON 是 ARM 处理器的 SIMD（Single Instruction, Multiple Data）扩展指令集，用于移动端和嵌入式设备上的并行计算加速：

**硬件规格**：
- 128 位宽的向量寄存器（32 个 Q 寄存器，或视为 64 个 D 寄存器）
- 单条指令可同时处理：4 个 float32 / 8 个 float16 / 8 个 int16 / 16 个 int8
- ARMv8（64 位）支持完整的 NEON 指令集，包括 FP16 运算

**典型应用场景**：
- **矩阵乘加速**：4x4 tile 的向量化乘加，配合循环展开
- **量化推理**：INT8 GEMM 利用 16 个 int8 并行乘加
- **激活函数**：向量化 ReLU/Sigmoid 的查表或多项式近似
- **数据预处理**：图像 resize、色彩空间转换、归一化

**编程方式**：
- **Intrinsics 函数**：`vld1q_f32`（加载）、`vfmaq_f32`（融合乘加）、`vmaxq_f32`（ReLU）等，C 语言级别的向量操作
- **内联汇编**：极致优化时直接写 ARM 汇编
- **自动向量化**：编译器 `-O3 -ftree-vectorize` 自动 SIMD 化（效果有限）
- **框架封装**：NCNN/MNN 等移动端框架已内置 NEON 优化的算子

**与其他 SIMD 的对比**：
| 特性 | ARM NEON | x86 AVX2 | x86 AVX-512 |
|------|---------|----------|-------------|
| 向量宽度 | 128-bit | 256-bit | 512-bit |
| float32 并行 | 4 | 8 | 16 |
| 典型平台 | 手机/嵌入式 | PC/服务器 | 数据中心 |

**注意事项**：NEON 的 FP16 支持需要 ARMv8.2+ 的 FP16 扩展（如 A76 及以后的核心）；SVE/SVE2 是 ARM 的可变长度向量扩展（Neoverse V1 支持 256-bit）。

### Q: KV Cache是什么？

KV Cache 是 LLM 自回归推理中最核心的优化技术，通过缓存已计算的 Key 和 Value 矩阵避免重复计算：

**原理**：
- 自回归生成第 t 个 token 时，Attention 需要计算 Q_t 与所有历史 token 的 K/V 的点积
- 如果不缓存，每生成一个新 token 就需要对前面所有 token 重新计算 KV（O(t^2) 总计算量）
- KV Cache：将每步计算的 K_t, V_t 追加到缓存中，下一步直接复用，总计算量降为 O(t)

**显存占用计算**：
- 公式：2（K 和 V）* n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
- 示例（LLaMA-2 7B，FP16）：2 * 32 * 32 * 128 * 2048 * 1 * 2 = 1GB（单请求 2048 token）
- 长上下文（128K token）：单请求 KV Cache 可达 64GB，比模型权重还大

**优化方向**：
- **GQA/MQA**：减少 KV heads 数（LLaMA-2 70B 用 GQA 8 组，KV Cache 缩小 4x）
- **KV Cache 量化**：INT8/INT4 量化 KV Cache（FP16 -> INT8 减半显存）
- **PagedAttention（vLLM）**：按页分配 KV Cache，消除内存碎片，提高 GPU 显存利用率从 20-40% 到 >90%
- **Sliding Window Attention**：只保留最近 N 个 token 的 KV Cache（Mistral 用 4096 窗口）
- **Token 剪枝/压缩**：H2O/StreamingLLM 等动态淘汰不重要的 KV 对

**Prefill vs Decode 的不同行为**：
- Prefill 阶段：一次性计算完整输入的 KV 并写入缓存（compute-bound）
- Decode 阶段：每步追加一个 KV 对，读取全部缓存做 attention（memory-bound）

### Q: C++中指针与引用的区别？

指针和引用是 C++ 中两种间接访问对象的机制，表面相似但语义和使用场景有本质差异：

**核心区别**：

| 特性 | 指针（Pointer） | 引用（Reference） |
|------|----------------|-------------------|
| 空值 | 可以为 nullptr | 必须绑定合法对象 |
| 重新绑定 | 可以指向不同对象 | 一旦绑定不可更改 |
| 访问方式 | 需要解引用 `*ptr` | 直接使用，如同原对象 |
| 内存占用 | 有自己的地址（存储目标地址） | 语义上无独立存储（编译器可能用指针实现） |
| 多级间接 | 支持 `int**`（指针的指针） | 不存在引用的引用 |
| 算术运算 | 支持 ptr++/ptr+n | 不支持 |
| sizeof | 返回指针本身大小（8 bytes on 64-bit） | 返回绑定对象的大小 |

**底层实现**：引用在大多数编译器实现中本质上是不可变的指针（const pointer），但语言层面保证了更强的安全性（非空、不可重绑定）。

**使用场景选择**：
- **优先使用引用**：函数参数传递（避免拷贝）、操作符重载、确定不为空的对象
- **必须使用指针**：需要 nullptr 表示"无对象"、需要重新指向、动态分配内存、数据结构（链表/树）、与 C API 交互
- **智能指针优先于裸指针**：需要动态分配时用 unique_ptr/shared_ptr

**常见陷阱**：
- 悬空引用（dangling reference）：返回局部变量的引用（UB）
- 野指针（dangling pointer）：指向已释放内存的指针
- 引用并不保证线程安全，多线程修改被引用对象仍需同步

### Q: 堆和栈的区别？

堆（Heap）和栈（Stack）是程序内存布局中两个核心区域，管理方式和性能特征完全不同：

**对比分析**：

| 特性 | 栈（Stack） | 堆（Heap） |
|------|-------------|------------|
| 管理方式 | 编译器自动管理（RAII） | 程序员手动管理（new/delete）或 GC |
| 分配速度 | 极快（移动栈指针，~1ns） | 较慢（空闲链表查找，~100ns+） |
| 空间大小 | 较小（Linux 默认 8MB） | 很大（受物理内存+虚拟内存限制） |
| 增长方向 | 向低地址增长 | 向高地址增长 |
| 碎片化 | 无碎片（LIFO 连续释放） | 容易碎片化（随机分配释放） |
| 缓存友好 | 极好（连续内存，空间局部性强） | 较差（分散分配） |
| 线程独立性 | 每个线程有独立栈 | 所有线程共享堆 |

**栈的典型用途**：局部变量、函数参数、返回地址、寄存器保存。特点是生命周期严格遵循 LIFO（后进先出），函数返回时自动释放。

**堆的典型用途**：动态大小数据（vector/string 的底层存储）、生命周期超出函数作用域的对象、大型数据结构（MB 级别放堆上避免栈溢出）。

**栈溢出（Stack Overflow）**：递归过深或局部大数组（如 `int arr[1000000]`）会超出栈空间限制。Linux 可通过 `ulimit -s` 调整栈大小。

**堆分配器**：glibc 的 ptmalloc2（基于 arena）、jemalloc（Facebook，减少碎片）、tcmalloc（Google，线程缓存加速）。性能差异显著：tcmalloc 多线程场景下比 ptmalloc2 快 2-5x。

**C++ 最佳实践**：优先栈分配（性能好、安全）；需要堆分配时使用智能指针（unique_ptr/shared_ptr）自动管理；避免裸 new/delete。

### Q: 野指针和智能指针？

**野指针（Dangling Pointer）**：

指向已释放内存或未初始化的指针，解引用会导致未定义行为（UB）——可能崩溃、静默数据损坏、或看似正常运行（最危险）。

**野指针的产生场景**：
1. `delete ptr` 后未置空，再次使用 ptr（use-after-free）
2. 函数返回局部变量的地址
3. 指针未初始化就使用
4. 指向的对象生命周期结束（如 vector 扩容后迭代器失效）

**防范措施**：
- 释放后立即 `ptr = nullptr`（解引用 nullptr 会确定性 crash，比 UB 好调试）
- 使用智能指针自动管理生命周期
- 工具检测：AddressSanitizer（ASan）、Valgrind

**智能指针（Smart Pointer）**：

C++11 引入的 RAII 封装，利用析构函数自动释放堆内存，解决手动 new/delete 的安全问题：

| 类型 | 语义 | 引用计数 | 拷贝 | 开销 | 适用场景 |
|------|------|---------|------|------|---------|
| unique_ptr | 独占所有权 | 无 | 禁止（可 move） | 零开销（与裸指针相同） | 单一所有者（95% 场景） |
| shared_ptr | 共享所有权 | 有（原子操作） | 允许 | 有开销（计数器+原子操作） | 多所有者共享资源 |
| weak_ptr | 弱引用 | 不增加计数 | — | 需升级为 shared_ptr | 打破循环引用、观察者模式 |

**shared_ptr 的代价**：
- 额外的控制块（strong_count + weak_count + deleter + allocator），通常 32-48 bytes
- 每次拷贝/销毁涉及原子操作（多线程下可能有缓存行 ping-pong）
- 循环引用导致内存泄漏（A 持有 B 的 shared_ptr，B 持有 A 的 shared_ptr -> 永远不释放）

**最佳实践**：
- 默认用 `unique_ptr`，需要共享时再升级为 `shared_ptr`
- 用 `make_unique` / `make_shared` 创建（异常安全 + shared_ptr 减少一次内存分配）
- 函数参数传递：不转移所有权时传 `const T&` 或 `T*`，不要传 `shared_ptr`（避免无谓的引用计数操作）
