---
title: '卓驭 AI Infra 实习'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 算子和Tensor是什么关系？

Tensor和算子（Operator）构成计算图的两大基本元素，类似于数据流图中的"边"和"节点"：

**Tensor（数据载体）**：

Tensor是多维数组的抽象，包含两部分信息：
- **数据（Storage）**：实际的数值存储（一段连续内存/显存）。
- **元信息（Metadata）**：shape、dtype、stride、device、layout、requires_grad等。

```python
x = torch.randn(2, 3, 4, device='cuda', dtype=torch.float16)
# shape=(2,3,4), stride=(12,4,1), dtype=float16, device=cuda:0
```

关键设计：Tensor的stride机制使得transpose、slice等操作可以不复制数据（只修改元信息），这是"view"操作的基础。但算子实现时必须考虑non-contiguous tensor的情况。

**算子（Operator，计算变换）**：

算子对一个或多个输入Tensor施加计算变换，产生输出Tensor。每个算子定义了：

| 职责 | 说明 | 示例 |
|------|------|------|
| 前向计算逻辑 | 数值计算的具体实现 | matmul: C = A × B |
| 反向梯度公式 | autograd需要的导数 | matmul反向: dA = dC×B^T, dB = A^T×dC |
| Shape推导 | 输出tensor的shape/dtype | matmul: [M,K]×[K,N]→[M,N] |
| 内存需求 | 需要的workspace/临时buffer | conv可能需要im2col buffer |
| Dispatch规则 | CPU/CUDA/NPU各有不同实现 | matmul → cuBLAS(CUDA) / MKL(CPU) |

**在计算图中的关系**：

```
Tensor A ──→ [MatMul Op] ──→ Tensor C ──→ [ReLU Op] ──→ Tensor D
Tensor B ──↗                                        
```

- 边 = Tensor（带有shape/dtype/device信息的数据流）。
- 节点 = 算子（有具体实现的计算单元）。
- 一个算子可以有多个输入tensor和多个输出tensor。
- tensor的生命周期由计算图决定（最后一个消费者算子执行后可释放）。

**对推理框架的意义**：
- tensor的shape信息用于算子的kernel选择（如小矩阵用不同tile策略）。
- tensor的生命周期分析用于内存规划（buffer复用）。
- tensor的device属性决定算子dispatch到哪个后端实现。

### Q: 计算图是用什么构建的？

计算图是AI框架的核心抽象，不同框架和阶段使用不同的构建方式：

**1. PyTorch动态图（Eager Mode + Autograd）**：

```python
x = torch.randn(3, 4, requires_grad=True)
y = torch.relu(x @ w + b)  # 每步执行时动态构建autograd图
loss = y.sum()
loss.backward()  # 反向遍历图计算梯度后，图被释放
```

实现机制：每个requires_grad的tensor操作会创建一个`Function`节点（如`MmBackward`、`ReluBackward`），记录输入tensor引用和反向计算函数。形成以loss为根的DAG。

优势：调试直观（可print中间值）、支持数据依赖的控制流。
劣势：无法全局优化（每op独立dispatch+launch）。

**2. TorchScript / torch.jit（静态追踪/脚本）**：

```python
# Trace方式：用示例输入执行一遍，记录操作序列
traced = torch.jit.trace(model, example_input)

# Script方式：解析Python AST转为TorchScript IR
@torch.jit.script
def foo(x):
    if x.sum() > 0:  # 支持控制流
        return x * 2
    return x
```

生成的IR可以序列化、跨语言部署、做图优化。但Script方式对Python子集的支持有限。

**3. FX Graph（PyTorch 2.0核心）**：

```python
from torch.fx import symbolic_trace
graph_module = symbolic_trace(model)
# 生成FX Graph: 节点 = call_function/call_method/call_module
print(graph_module.graph)
```

FX通过symbolic tracing在Python层面捕获计算图（不执行实际计算，用proxy对象记录操作）。是torch.compile/TorchDynamo的基础表示。

**4. ONNX（开放交换格式）**：

```python
torch.onnx.export(model, example_input, "model.onnx", opset_version=17)
```

Protocol Buffers序列化的静态计算图，包含：Graph（节点列表+拓扑）→ Node（op_type + inputs/outputs + attributes）→ Tensor（常量权重内嵌）。

用途：框架间模型迁移（PyTorch→TensorRT/OpenVINO/ONNX Runtime）。

**5. MLIR / TVM IR（编译器级表示）**：

```mlir
func.func @matmul(%arg0: tensor<128x512xf16>, %arg1: tensor<512x256xf16>) 
    -> tensor<128x256xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1) outs(%init) -> tensor<128x256xf16>
  return %0
}
```

多级IR设计（High-level → Mid-level → Low-level/Hardware），每级有不同的优化pass。适合做跨硬件的深度编译优化。

**各层次对比**：

| 表示 | 构建时机 | 优化能力 | 灵活性 | 典型用途 |
|------|---------|---------|--------|---------|
| Autograd | 运行时 | 无 | 最高 | 训练/研发 |
| FX Graph | Trace时 | 中（Python变换） | 高 | torch.compile |
| ONNX | 导出时 | 需外部优化器 | 中 | 跨框架部署 |
| TensorRT IR | 编译时 | 最强（全图优化） | 低 | 推理部署 |

### Q: 权重如何导入到自搭建的模型中？

权重迁移是模型移植和框架适配的核心工作，需要解决名称映射、格式转换、精度验证三大问题：

**1. State Dict映射（最常用）**：

```python
# 加载预训练权重
pretrained = torch.load("model.pt")['state_dict']
# 自定义模型
my_model = MyModel()

# 情况1：key完全匹配
my_model.load_state_dict(pretrained, strict=True)

# 情况2：需要key重命名
rename_map = {
    "transformer.h.0.attn.c_attn.weight": "layers.0.attention.qkv_proj.weight",
    "transformer.h.0.mlp.c_fc.weight": "layers.0.ffn.gate_proj.weight",
}
new_state = {rename_map.get(k, k): v for k, v in pretrained.items()}
my_model.load_state_dict(new_state, strict=False)
```

**2. 处理Shape不匹配**：

```python
# 例如：原模型QKV分开存储，新模型合并为一个tensor
q_weight = pretrained["attn.q_proj.weight"]  # [hidden, hidden]
k_weight = pretrained["attn.k_proj.weight"]  # [hidden, kv_dim]
v_weight = pretrained["attn.v_proj.weight"]  # [hidden, kv_dim]
# 合并
qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
new_state["attn.qkv_proj.weight"] = qkv_weight
```

**3. 跨框架转换**：

| 源格式 | 目标格式 | 方法 |
|--------|---------|------|
| TF Checkpoint | PyTorch state_dict | 逐tensor读取+名称映射脚本 |
| HuggingFace safetensors | 自定义模型 | safetensors.load_file()→字典→映射 |
| GGUF (llama.cpp) | PyTorch | 解析GGUF二进制格式+反量化 |
| Megatron分片 | 单文件 | 合并各TP/PP分片的张量 |

**4. 部分加载和验证**：

```python
# strict=False允许不完全匹配
missing, unexpected = my_model.load_state_dict(new_state, strict=False)
print(f"Missing keys: {missing}")      # 新模型有但权重中没有的参数
print(f"Unexpected keys: {unexpected}") # 权重中有但新模型没有的参数

# missing keys通常需要随机初始化（如新增的层）
# unexpected keys通常可忽略（如旧模型的BN统计量在新模型不需要）
```

**5. 大模型分片加载（避免OOM）**：

```python
# safetensors支持lazy loading（按需读取单个tensor）
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    for name in f.keys():
        tensor = f.get_tensor(name)
        # 逐tensor加载到模型对应位置
        set_parameter(my_model, mapped_name(name), tensor)
```

**6. 精度验证**：

加载后必须验证输出正确性：
```python
# 用相同输入对比原模型和加载后模型的输出
with torch.no_grad():
    ref_output = original_model(test_input)
    my_output = my_model(test_input)
    assert torch.allclose(ref_output, my_output, atol=1e-5)
```

常见陷阱：转置遗漏（TF conv权重格式HWIO vs PyTorch OIHW）、缩放因子遗漏、归一化层参数对应错误。

### Q: 大模型推理分为哪些阶段？

大模型（LLM）的自回归推理分为两个性能特征截然不同的阶段：

**1. Prefill阶段（首token生成/提示处理）**：

```
输入: "请解释量子计算的基本原理" (N个token)
处理: 一次性计算所有N个token的注意力，生成完整KV Cache
输出: 第一个生成token + KV Cache[N]
```

性能特征：
- **Compute-bound**：attention计算量 = O(N^2 × d) + FFN计算量 = O(N × 4d^2)。
- 大矩阵乘法，Tensor Core高效利用，GPU利用率可达60-80%。
- 延迟指标：TTFT（Time To First Token），用户等待的首响应时间。
- 典型耗时：4K tokens输入在A100上约100-500ms（取决于模型大小）。

**2. Decode阶段（逐token自回归生成）**：

```
每步: 
  - 输入: 上一步生成的1个token
  - 计算: 1个token与所有历史KV的attention + FFN
  - 输出: 下一个token
  - 更新: KV Cache追加新token的K/V
```

性能特征：
- **Memory-bound**：每步只做batch_size×1个token的计算，但需读取全部模型权重（70B模型=140GB）。
- 算术强度极低：FLOPs/Byte ≈ 2（远低于GPU拐点~100-200）。
- GPU计算利用率<5%（A100: 312T FP16峰值，实际只用几T）。
- 延迟指标：TPOT（Time Per Output Token），影响用户感知的生成流畅度。
- 典型耗时：30-100ms/token（batch_size=1时）。

**3. Post-processing（后处理）**：

- **Sampling**：temperature缩放 → top-k过滤 → top-p(nucleus)采样 → 随机采样。
- **Detokenization**：token ID → text（需要处理BPE的subword合并）。
- **停止判断**：检测EOS token或达到max_length。
- 通常CPU执行，耗时可忽略（<1ms）。

**两阶段对比**：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算模式 | 大batch矩阵乘 | 向量×矩阵 |
| 瓶颈 | Compute-bound | Memory-bound |
| GPU利用率 | 60-80% | <5% |
| 优化策略 | FlashAttention、TP | 量化、投机解码 |
| 主要延迟 | TTFT | TPOT × 生成长度 |
| batch效果 | 接近线性加速 | 加速有限(受带宽限制) |

**PD分离趋势**：由于两阶段性能特征完全不同，生产系统趋向将Prefill和Decode部署在不同GPU集群上：Prefill节点追求高计算密度，Decode节点追求高内存带宽和大容量（如HBM3E 8TB/s）。

### Q: Prefill阶段有没有办法加快KV Cache的生成？

Prefill阶段的KV Cache生成速度直接决定TTFT（首token延迟），以下技术从不同角度加速：

**1. FlashAttention（减少I/O瓶颈）**：
- 标准attention需要将N×N的score矩阵写入HBM再读回做softmax。FlashAttention通过tiling+online softmax在SRAM中完成，I/O减少O(N^2/M)倍（M为SRAM大小）。
- A100实测：seq=4K时Prefill加速2-3x，seq=16K时加速4-6x（长序列收益更大）。
- 计算量不变（仍是O(N^2×d)），但I/O大幅减少使得实际离compute-bound更近。

**2. Chunked Prefill（降低首token延迟的调度策略）**：
- 问题：一个很长的prefill请求（如32K tokens）会独占GPU数百ms，期间所有decode请求被阻塞。
- 解决：将长prefill切分为小块（如每块512 tokens），每块之间穿插一些decode步骤。
- 效果：decode请求的延迟不被长prefill打断，P99 TPOT改善显著。
- 代价：总Prefill时间略增（多次kernel launch开销），但用户体验更好。
- vLLM实现：`--enable-chunked-prefill --max-num-batched-tokens 512`。

**3. 张量并行（计算加速）**：
- TP=N时，每卡只需计算1/N的attention heads和1/N的FFN宽度。
- 单次Prefill延迟接近线性降低：A100 8卡TP=8时，70B模型4K输入Prefill约50ms。
- 通信：每层2次AllReduce（约2×seq_len×hidden_dim×2bytes/NVLink_BW），NVLink下<1ms/层。

**4. Prefix Caching（避免重复计算）**：
- 场景：大量请求共享相同system prompt（如2000 token的指令模板）。
- 实现：首次计算system prompt的KV Cache后缓存，后续请求直接引用。
- 效果：如果system prompt占prefill总长的50%，则TTFT减少约50%。
- 缓存策略：LRU淘汰，基于prefix content hash匹配。

**5. FP8计算（Hopper+硬件加速）**：
- H100支持FP8 Tensor Core，Prefill的GEMM用FP8可达~2x吞吐（vs FP16）。
- FP8 prefill + FP16 decode是当前高端部署的常见配置。
- 精度：E4M3格式在attention中需要配合per-tensor或per-block scale，通常<0.5%质量损失。

**6. 投机预填充（Speculative Prefill）**：
- 对常见的输入pattern（如固定格式的query模板），提前预计算部分KV Cache。
- 类似prefix caching但可以覆盖动态内容的部分共同前缀。
- 适合流量模式可预测的场景（如API的固定前缀）。

### Q: Prompt中token之间有依赖吗？可以并行计算吗？

这个问题涉及Transformer注意力计算的核心机制：

**Prefill阶段——可以完全并行**：

虽然因果注意力（causal attention）要求每个token只能关注其前面的token（下三角mask），但在Prefill阶段，**所有输入token已知**：

```
位置 0 1 2 3 4
   0 [✓ ✗ ✗ ✗ ✗]    token 0只看自己
   1 [✓ ✓ ✗ ✗ ✗]    token 1看0,1
   2 [✓ ✓ ✓ ✗ ✗]    token 2看0,1,2
   3 [✓ ✓ ✓ ✓ ✗]    token 3看0,1,2,3
   4 [✓ ✓ ✓ ✓ ✓]    token 4看所有
```

虽然有mask，但计算时所有K和V都已经存在（因为所有token的输入embedding已知），mask只是将未来位置的注意力权重设为0。这在数学上等价于一次矩阵运算（下三角矩阵乘法），**完全可以并行**。

实现上就是一次大的矩阵乘法：`Score = Q × K^T`（所有token的Q同时与所有K相乘），然后对Score施加causal mask再softmax。GPU的并行性被充分利用。

**Decode阶段——token间有串行依赖**：

```
Step 1: 生成 token_6 (依赖token 0-5的KV Cache)
Step 2: 生成 token_7 (依赖token 0-6的KV Cache，包括step 1刚生成的token_6)
Step 3: 生成 token_8 (依赖token 0-7的KV Cache)
...每步依赖上一步的输出
```

这是**本质的串行依赖**——下一个token的输入（embedding）取决于上一步的采样结果。无法在不知道token_6是什么的情况下计算token_7的输出。

**打破串行的方法**：

| 方法 | 原理 | 加速比 | 代价 |
|------|------|--------|------|
| 投机解码 | 小模型猜K个token，大模型一次验证 | 2-3x | 额外小模型 |
| Medusa | 多头并行预测后续token | 2-3x | 额外参数 |
| Parallel Decoding | 同时生成多个独立位置（非自回归） | N/A | 质量下降 |
| Lookahead Decoding | 利用Jacobi iteration并行生成 | 1.5-2x | 接受率有限 |

投机解码的核心洞察：大模型验证K个token的时间≈生成1个token（读取模型权重的I/O不变），但实际生成了~0.7K个有效token。

### Q: 如何使用Nsight辅助优化？重点关注哪些指标？

NVIDIA Nsight工具套件提供从系统级到指令级的完整性能分析能力：

**Nsight Systems（系统级全景分析）**：

```bash
nsys profile -t cuda,nvtx,osrt --stats=true -o report python inference.py
```

关注的核心问题和指标：

| 问题 | 看什么 | 正常范围 | 异常表现 |
|------|--------|---------|---------|
| GPU是否在忙 | GPU活动率 | >90% | 大量idle gap |
| CPU是否成为瓶颈 | CPU-GPU overlap | 高度重叠 | GPU等CPU |
| 数据传输 | H2D/D2H带宽利用 | 被计算覆盖 | 长时间独占时间线 |
| Kernel粒度 | 单kernel耗时 | >10us | 大量<5us的小kernel |
| 多Stream并行 | Stream overlap | 多stream并行 | 单stream串行 |

典型发现和解决：
- 大量小kernel（<5us）密集排列：→ 算子融合。
- GPU idle期间CPU在执行Python代码：→ 用torch.compile减少Python开销。
- H2D传输阻塞：→ 用pin_memory + 异步传输。

**Nsight Compute（kernel级深度分析）**：

```bash
ncu --set full --target-processes all -o detailed_report python inference.py
```

**核心指标及其意义**：

**1. Occupancy（SM占用率）**：
- 定义：实际活跃Warp数 / SM最大可驻留Warp数。
- 影响因素：寄存器用量、共享内存用量、block大小。
- 经验值：>50%通常可接受，但不是越高越好（有时低occupancy+高ILP更快）。
- 查看：`Occupancy`部分的Achieved vs Theoretical。

**2. Memory Throughput**：
- `DRAM Throughput`：实际HBM带宽利用（A100峰值2TB/s）。
- `L2 Hit Rate`：L2缓存命中率，高命中率意味着有效利用了局部性。
- `Shared Memory Efficiency`：bank conflict导致的效率损失。
- Memory-bound kernel应追求DRAM throughput接近峰值（>70%）。

**3. Compute Throughput**：
- `SM Active`：SM处于活跃状态的时间占比。
- `Tensor Core Utilization`：Tensor Core的使用率（GEMM类kernel应>50%）。
- `FP16/FP32 Pipeline Utilization`：各精度管线的利用率。

**4. Warp Stall原因分析**：
- `Stall Long Scoreboard`：等待全局内存数据返回——memory-bound信号。
- `Stall Barrier`：等待__syncthreads()——同步开销过高。
- `Stall Not Selected`：就绪但未被调度——正常的warp切换。
- `Stall Math Pipe Throttle`：计算管线满载——compute-bound。

**5. Roofline分析**：
- 横轴：算术强度（FLOPs/Byte），纵轴：性能（FLOPS）。
- 位于roofline线以下表示有优化空间。
- 位于memory-bound区域：优化数据复用和访存效率。
- 位于compute-bound区域：优化计算效率（Tensor Core、ILP）。

**优化闭环**：

```
Nsight Systems定位热点kernel
    ↓
Nsight Compute分析该kernel瓶颈类型
    ↓
根据bound类型选择优化策略
    ↓
实施优化（修改kernel代码）
    ↓
重新Profile验证改善
```

### Q: 吞吐量（Throughput）在推理中的含义？

推理场景的吞吐量有多个层次的定义，理解其含义对系统优化至关重要：

**吞吐量的多层定义**：

| 层次 | 定义 | 单位 | 典型值(A100, 70B) |
|------|------|------|-----------------|
| 系统级 | 每秒完成的请求数 | requests/s | 5-50 req/s |
| Token级 | 所有请求每秒生成的token总数 | tokens/s | 1000-5000 tokens/s |
| 用户级 | 单个请求每秒生成token数 | tokens/s/req | 20-50 tokens/s/req |
| 硬件级 | 计算单元的利用效率 | TFLOPS / GB/s | 50% MFU |

**吞吐量 vs 延迟的trade-off**：

```
                吞吐量
           ▲
           │      ╱─── 最大吞吐
           │    ╱
           │  ╱         ← 增大batch size
           │╱
           └──────────────── 延迟 ►
                        P99 latency要求
```

- 增大batch size → GPU更忙 → 吞吐提高 → 但每请求排队等待更久。
- SLA约束（如P99 latency < 2s）限制了batch size的上限。
- 最优操作点：在满足延迟SLA的前提下最大化吞吐。

**影响吞吐的关键因素**：

| 因素 | 影响方式 | 优化方向 |
|------|---------|---------|
| Batch大小 | 大batch提高Compute利用率 | Continuous Batching动态调整 |
| 序列长度分布 | 长序列占GPU时间长 | 短序列优先、Chunked Prefill |
| KV Cache效率 | 碎片限制并发数 | PagedAttention |
| 模型精度 | 低精度减少带宽需求 | W4A16/W8A8量化 |
| 调度策略 | 空闲时间浪费算力 | 迭代级调度 |
| 硬件带宽 | Decode受HBM带宽限制 | HBM3E/多卡TP |

**生产系统的吞吐优化优先级**：
1. Continuous Batching（最基础，2-3x提升）。
2. PagedAttention（提高并发数，2-4x提升）。
3. 量化（W8A8可再提升50-100%）。
4. FlashAttention（Prefill加速2-4x）。
5. PD分离（资源利用率提升30-50%）。

### Q: Memory-bound和Compute-bound的区别？

这是GPU优化中最核心的概念——正确判断bound类型决定了优化方向：

**本质区别**：

```
Memory-bound: GPU计算单元大部分时间在等数据（"巧妇难为无米之炊"）
Compute-bound: GPU内存系统已提供足够数据，但计算能力是瓶颈（"食材充足但厨师忙不过来"）
```

**判断方法——算术强度（Arithmetic Intensity）**：

```
AI = FLOPs / Bytes_Accessed
硬件拐点 = Peak_FLOPS / Peak_Bandwidth

若 AI < 硬件拐点 → Memory-bound
若 AI > 硬件拐点 → Compute-bound
```

A100硬件拐点计算：
- FP16 Tensor Core: 312 TFLOPS / 2 TB/s = **156 FLOPs/Byte**
- FP32 CUDA Core: 19.5 TFLOPS / 2 TB/s = **9.75 FLOPs/Byte**

**典型算子的分类**：

| 算子 | 算术强度 | Bound类型 | 原因 |
|------|---------|----------|------|
| GEMM [M,K]×[K,N] | ~M×N×K/(M×K+K×N+M×N) | Compute (大矩阵) | 大量乘加 |
| Vector Add | 1 | Memory | 每元素仅1次加法 |
| ReLU | 0.5 | Memory | 读+比较+写 |
| Softmax | ~10 | Memory | reduce占主导 |
| LayerNorm | ~10 | Memory | reduce+逐元素 |
| Decode Attention | ~1-2 | Memory | batch=1，读大量KV |
| Prefill Attention | ~N/4 | Compute (长序列) | O(N^2)计算 |

**优化策略对照**：

| 方向 | Memory-bound优化 | Compute-bound优化 |
|------|-----------------|------------------|
| 减少读写 | 算子融合、避免中间buffer | - |
| 提高带宽利用 | 合并访问、向量化(float4) | - |
| 复用数据 | 共享内存缓存、tiling | - |
| 加速计算 | - | Tensor Core利用 |
| 增加ILP | - | 循环展开、每线程多元素 |
| 减少计算量 | - | 低精度(FP16/INT8) |

**Decode阶段为什么是Memory-bound的直觉**：
- 70B模型权重140GB（FP16），每次decode需要读取全部。
- 但每次只做batch_size个向量×矩阵乘法（计算量极小）。
- AI ≈ 2×batch_size（batch=1时AI≈2，远低于A100拐点156）。
- 只有当batch_size > 78时才可能变成compute-bound（但受显存限制无法达到）。

### Q: 模型和算子已固定时，影响推理速度的因素有哪些？

当模型架构和算子实现已确定后，推理性能仍受多个运行时因素影响：

**1. Batch Size（最直接的影响）**：

| Batch Size | Decode AI | GPU利用率 | 吞吐 | 延迟 |
|-----------|-----------|----------|------|------|
| 1 | ~2 | <5% | 低 | 最低 |
| 8 | ~16 | ~10% | 中 | 中 |
| 64 | ~128 | ~50% | 高 | 较高 |
| 128+ | >156 | >80% | 最高 | 高 |

Batch越大，单次weight读取服务更多请求（摊薄带宽成本），但每请求排队延迟增加。

**2. 输入/输出序列长度**：
- 长输入：Prefill计算量O(N^2)增长，TTFT显著增加。
- 长生成：KV Cache线性增长，后期每步decode的attention计算量增加。
- 长上下文：可能触发显存限制，导致请求被throttle或换页。

**3. KV Cache管理效率**：
- 碎片化：传统实现浪费50-80%显存→能服务的并发数减半。
- 换入换出延迟：offloading到CPU的请求重新激活需数ms。
- 分配速度：频繁的cudaMalloc/cudaFree有~100us开销。

**4. 调度策略**：
- 静态batch vs Continuous Batching：后者吞吐提升2-3x。
- 优先级调度：短请求优先可降低平均延迟。
- 抢占策略：长请求被短请求抢占的时机。

**5. 硬件带宽利用**：
- HBM带宽：decode瓶颈。A100:2TB/s, H100:3.35TB/s, B200:8TB/s。
- NVLink带宽：TP通信（AllReduce/AllGather）的开销。
- PCIe带宽：与CPU的数据交换。

**6. 量化精度与硬件单元匹配**：
- FP16用FP16 Tensor Core，INT8用INT8 Tensor Core。
- 错误匹配（如用CUDA Core做FP16计算）性能差10x+。
- FP8在H100上吞吐再翻倍。

**7. 通信开销（多卡推理）**：
- TP每层2次AllReduce：延迟约2-5us（NVLink），数据量=batch×seq×hidden×2bytes。
- 总通信开销 = 层数×2×单次通信延迟，70B模型80层约320-800us/token。

**8. 软件overhead**：
- Python调度延迟：eager mode每op几十us。
- CUDA kernel launch：~5us/kernel。
- 内存分配器：不同allocator（native/CachingAllocator）效率差异。
- torch.compile可减少2-5x软件overhead。

### Q: Prompt长度和上下文长度是否会影响推理速度？

**显著影响**，且影响机制在Prefill和Decode阶段不同：

**Prefill阶段的影响**：

```
Attention计算量 = N² × d × num_heads × num_layers
FFN计算量 = N × 4d² × num_layers × 2（gate+up/down）
```

| Prompt长度 | 计算量(A100, 70B) | TTFT | 关系 |
|-----------|-------------------|------|------|
| 512 | ~0.5T FLOPs | ~30ms | - |
| 2048 | ~8T FLOPs | ~100ms | 平方增长(attention主导) |
| 8192 | ~125T FLOPs | ~800ms | - |
| 32768 | ~2000T FLOPs | ~10s | 长上下文TTFT很高 |

注：FlashAttention将I/O从O(N^2)降为O(N)但计算量不变，仍是O(N^2)。

**Decode阶段的影响**：

每步decode的计算量随已生成上下文长度线性增长：

```
Attention计算（每步）: Q(1×d) × K^T(context_len×d) + Score × V(context_len×d)
访存量 = KV Cache大小 = 2 × layers × kv_heads × head_dim × context_len × 2bytes
```

| 已生成长度 | 每步KV读取(70B GQA) | TPOT影响 | 说明 |
|-----------|-------------------|---------|------|
| 100 | ~65MB | 基本不影响 | 权重读取主导(140GB) |
| 4K | ~2.6GB | +5-10% | 开始影响 |
| 32K | ~21GB | +15-25% | 显著影响 |
| 128K | ~84GB | +50-100% | KV读取与权重读取接近 |

**其他影响**：

1. **显存压力**：长上下文KV Cache占用大量显存，可能导致：
   - 可服务的并发请求数减少。
   - 触发KV Cache offloading（增加延迟）。
   - 极端情况OOM。

2. **数值稳定性**：超长序列的softmax中exp值域变大，FP16可能溢出→FlashAttention通过online safe softmax解决。

3. **位置编码外推**：超出训练长度的位置编码可能导致质量下降（RoPE需要配合NTK-aware scaling或YaRN）。

### Q: 进一步提升大模型推理性能，有哪些可用技术？

在已应用基础优化（KV Cache + 量化 + FlashAttention + Continuous Batching）后，更前沿的技术：

**1. 投机解码（Speculative Decoding）**：

```
Draft Model (1B): 快速生成 [t1, t2, t3, t4, t5]  (5个候选，约5ms)
Target Model (70B): 一次性验证这5个token         (约30ms = 单步时间)
接受: [t1, t2, t3, ✗, ✗] → 实际生成3个有效token
等效速度: 3个token / 30ms = 100 tokens/s (vs 单步33 tokens/s)
```

变体对比：

| 方案 | Draft来源 | 优势 | 劣势 |
|------|----------|------|------|
| 外部小模型 | 独立1-7B模型 | 灵活选择 | 额外显存 |
| Medusa | 主模型+多MLP头 | 无需额外模型 | 需要fine-tune |
| EAGLE | 主模型特征+自回归头 | 接受率高 | 稍复杂 |
| Self-Speculative | 跳层/early-exit | 零额外参数 | 接受率较低 |

**2. MoE（Mixture of Experts，稀疏激活）**：
- 模型总参数量大（如671B），但每次只激活部分expert（如8选2）。
- 每token实际计算量≈等效dense模型的1/4。
- 挑战：expert权重需要全部驻留显存（或动态加载），总显存需求大。
- 代表：DeepSeek-V3（671B总参数，37B活跃参数），Mixtral-8x7B。

**3. 稀疏注意力（减少KV计算量）**：

| 方法 | 机制 | KV Cache节省 | 精度影响 |
|------|------|-------------|---------|
| StreamingLLM | 保留前4个sink token + 最近窗口 | ~90% | 长距离略差 |
| H2O | 保留累积注意力权重top-K | 50-80% | 可调 |
| Quest | 查询感知的KV选择 | 60-80% | 较小 |
| InfLLM | 分块+查询相关block检索 | 大幅 | 可接受 |

**4. 多级缓存/分层存储**：
- GPU HBM：热请求的KV Cache + 模型权重。
- CPU DRAM（CXL扩展）：温请求的KV Cache（重新激活延迟~1ms）。
- NVMe SSD：冷数据的KV Cache（重新激活延迟~10ms）。
- 适合大量长session的场景（如持续对话）。

**5. 分布式KV Cache**：
- 跨GPU/跨节点共享KV Cache：请求迁移时不重新计算。
- 适合多实例推理集群的负载均衡。
- 挑战：远程KV访问的网络延迟。

**6. 新硬件利用**：
- **HBM3E**（B200）：8TB/s带宽（vs A100 2TB/s），decode直接加速4x。
- **CXL内存扩展**：扩大有效内存容量到TB级。
- **FP4支持**（Blackwell）：模型压缩率再翻倍。

**7. MLA（Multi-head Latent Attention）**：
- DeepSeek-V2/V3采用的注意力变体。
- 将KV投影到低维潜在空间（如512维 vs 原始4096维），KV Cache减少8倍。
- 需要在训练时使用MLA架构。

### Q: 稀疏化KV Cache需要改变模型训练吗？

这取决于稀疏化的类型是"推理时的工程优化"还是"模型结构的改变"：

**不需要改训练（纯推理时优化）**：

| 方法 | 机制 | 为何不需要改训练 | 精度风险 |
|------|------|---------------|---------|
| H2O | 按累积attention score丢弃低分token | 模型本身就不依赖低注意力token | 长距离依赖可能受损 |
| Scissorhands | 类似H2O的token剪枝 | 纯推理时启发式 | 同上 |
| Dynamic KV Eviction | 基于LRU或频率淘汰 | 不改变模型计算逻辑 | 信息可能丢失 |
| KV Cache量化 | INT8/FP8存储KV | 精度损失在容差内 | 极端情况累积误差 |

这类方法的本质：模型仍按全注意力计算，只是KV Cache的存储/读取策略变了。风险在于丢弃的token如果后续被重要查询需要，会导致质量下降。

**需要改训练（结构性改变）**：

| 方法 | 机制 | 为何需要训练 | 改动程度 |
|------|------|------------|---------|
| 滑动窗口 | 只看最近W个token | 训练时就要使用窗口attention，否则推理质量严重下降 | 从头训练/大规模续训 |
| GQA/MQA | 减少KV头数 | KV头数是结构超参，影响所有层的表达 | 从头训练 |
| MLA | KV投影到低维 | 需要学习投影矩阵 | 从头训练 |
| StreamingLLM | 保留sink token | 需要让模型学会将全局信息集中到前几个token | 可少量fine-tune |
| Longformer | 局部+全局attention | 混合attention pattern需要训练适配 | 从头训练 |

**判断原则**：

```
如果稀疏化改变了模型"能看到什么信息" → 需要训练（模型要学会在约束下工作）
如果稀疏化只改变了"信息存在哪里" → 不需要训练（纯工程/存储优化）
```

**实践建议**：
- 首先尝试不需要训练的方法（H2O/量化），快速验证质量是否可接受。
- 如果质量要求严格（如<0.5% perplexity增加），使用改训练的方案。
- 新模型设计时直接采用GQA+适中窗口（如LLaMA-3: GQA + 128K），训练时就规划好。
