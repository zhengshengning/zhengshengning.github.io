---
title: '旷视科技 AI Infra 校招'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, AI 创业公司面经]
tags: [AIInfra, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: LLaMA 的模型文件 .bin 做了什么处理？有多个 bin 还是一个？

**模型序列化方式**：
- 通过 `torch.save(state_dict, path)` 将模型参数序列化为 `.bin` 文件
- 底层格式：Python pickle（序列化 dict 结构）+ 原始 tensor 数据（连续二进制）
- pickle 部分记录 key→tensor 的映射关系和 dtype/shape 等元数据

**大模型分片存储（Sharding）**：
- 大模型通常**分成多个 bin 文件**（每个 shard 通常 ~5-10 GB）
- 如 LLaMA-65B：`pytorch_model-00001-of-00013.bin` ... `pytorch_model-00013-of-00013.bin`
- 索引文件 `pytorch_model.bin.index.json` 记录每个参数 key 对应哪个 shard 文件
- 分片原因：单文件过大时加载慢、不方便分布式加载、磁盘 I/O 受限

**加载流程**：
1. 读取 index.json 获取参数→shard 的映射
2. 按需打开对应 shard 文件，mmap 或 read tensor 数据
3. 对多卡场景：每卡只加载自己负责的参数分片

**新格式——Safetensors**：
- HuggingFace 推出的更安全高效的格式
- 优势：无 pickle 执行风险（pickle 可执行任意代码）、支持零拷贝 mmap、header 固定为 JSON
- 加载速度比 .bin 快 2-5x（直接 mmap 无需 unpickle）
- 已成为社区推荐的标准格式

### Q: 怎么初始化模型所有层？

**标准初始化流程**（以 HuggingFace Transformers 为例）：

**1. 构建模型结构**（meta device）：
```python
with torch.device('meta'):  # 不分配实际内存
    model = LlamaForCausalLM(config)
# 此时模型占 0 内存，只有结构定义
```

**2. 加载参数**：
```python
# 方式1：直接加载（需要 2x 峰值显存）
model.load_state_dict(torch.load("model.bin"))

# 方式2：分片加载（内存友好）
from accelerate import load_checkpoint_and_dispatch
model = load_checkpoint_and_dispatch(model, "model_dir/", device_map="auto")
```

**3. 多卡场景（TP/PP）**：
- 初始化时根据切分策略（column/row parallel）只加载本卡的参数分片
- 如 TP=4 的 Q_proj [4096, 4096]：每卡加载 [4096, 1024]
- 实现：在 state_dict 加载时按 rank 做 slice/chunk

**4. 参数初始化（从零训练时）**：
```python
# Xavier 初始化（适合 sigmoid/tanh）
nn.init.xavier_uniform_(linear.weight)
# Kaiming 初始化（适合 ReLU/GELU）
nn.init.kaiming_normal_(linear.weight, mode='fan_in')
# 大模型常用：small normal init
nn.init.normal_(linear.weight, std=0.02)
```

### Q: CPU 数据怎么拷贝到 GPU 上？

**五种 Host→Device 数据传输方式**：

| 方式 | API | 特点 | 适用场景 |
|---|---|---|---|
| 同步拷贝 | `cudaMemcpy(dst, src, size, H2D)` | 阻塞直到完成 | 简单场景 |
| 异步拷贝 | `cudaMemcpyAsync(...)` | 非阻塞，需 pinned memory | 需要重叠的场景 |
| 零拷贝 | `cudaHostAlloc(Mapped)` | GPU 直接访问 host 内存 | 少量/低频访问 |
| Unified Memory | `cudaMallocManaged` | 驱动自动迁移 | 原型开发 |
| PyTorch | `tensor.to('cuda')` | 底层调 cudaMemcpy | 框架级使用 |

**关键细节**：

**异步拷贝的前提条件**：
- 源内存必须是 **Pinned Memory**（`cudaMallocHost` 分配）
- 必须指定非默认 stream
- 如果用 pageable memory 调 Async 版本，实际退化为同步

**零拷贝（Mapped Memory）**：
- GPU 通过 PCIe 直接访问 host 内存，无需显式拷贝
- 延迟很高（PCIe 延迟 ~1μs vs HBM ~100ns），但适合偶尔访问的元数据
- 实际带宽受 PCIe 限制（~32 GB/s，远低于 HBM ~2-3 TB/s）

**Unified Memory**：
- `cudaMallocManaged` 分配的内存 CPU/GPU 都能用同一指针访问
- 驱动按需做 page migration（page fault → 迁移到访问方）
- 方便但性能不最优（page fault 开销大）

**最佳实践**：
- 推理框架中：预分配 pinned memory buffer → asyncMemcpy → 计算重叠
- 大数据量：分 chunk 流水化传输（三级流水）

### Q: 推理框架中怎么调用 forward？

**推理框架的 forward 执行流程**：

```
1. 输入预处理 (CPU)
   ↓
2. Tokenize + H2D 传输 (输入tensor到GPU)
   ↓
3. 执行计算图 (GPU)
   ├─ Embedding Layer kernel
   ├─ for each layer:
   │   ├─ LayerNorm kernel
   │   ├─ QKV Projection kernel (或融合kernel)
   │   ├─ Attention kernel (FlashAttention / PagedAttention)
   │   ├─ O Projection kernel
   │   ├─ LayerNorm kernel
   │   ├─ Gate+Up Projection kernel
   │   ├─ SwiGLU Activation kernel
   │   └─ Down Projection kernel
   ├─ Final LayerNorm kernel
   └─ LM Head kernel (logits)
   ↓
4. D2H 传输 (logits 回 CPU) + Sampling
   ↓
5. Decode token / 判断是否结束
```

**推理引擎的优化**：
- **静态图优化**（TensorRT）：编译期将多层融合为大 kernel，消除中间 tensor
- **动态调度**（vLLM/TRT-LLM）：每个 iteration 根据当前 batch 组成决定 kernel 配置
- **Kernel Dispatch**：每层 forward 本质是通过 CUDA launcher 将对应 kernel 提交到 GPU stream
- **预分配**：所有中间 buffer 在初始化时预分配好，运行时零 malloc 开销

### Q: 算子优化的思路和过程？

**系统化的算子优化流程**：

**Step 1：Profiling 定位热点**
- 使用 Nsight Systems 获取全局时间线
- 找到耗时 Top-5 的 kernel（通常 80% 时间集中在 20% 的 kernel 上）
- 确认这些 kernel 是否可以优化（vs 已经接近峰值）

**Step 2：分析瓶颈类型（Roofline 模型）**
- 计算算术强度 AI = FLOPs / Bytes
- 对比硬件拐点：Peak_Compute / Peak_Bandwidth
- AI < 拐点 → Memory-bound；AI > 拐点 → Compute-bound

**Step 3：选择优化策略**

| 瓶颈类型 | 优化方向 | 具体手段 |
|---|---|---|
| Memory-bound | 减少数据搬运 | 合并访存、向量化(float4)、算子融合、tiling |
| Compute-bound | 提高计算效率 | Tensor Core、减少冗余计算、指令级并行 |
| Latency-bound | 减少 kernel 数 | 算子融合、persistent kernel |

**Step 4：实施优化**
- 编写优化 kernel（Triton/CUDA）
- 关键技术：tiling + shared memory + 双缓冲 + Tensor Core + 向量化

**Step 5：验证**
- 正确性：与 PyTorch reference 对比（cosine similarity > 0.999）
- 性能：NCU profiling 确认 SOL 提升
- 回归测试：不同 shape 下的正确性和性能

### Q: 怎么设计内存管理（显存池）？显存池的数据结构？

**设计目标**：避免频繁 `cudaMalloc`/`cudaFree`（耗时 ~1ms），实现 O(1) 的分配/释放。

**架构设计**：

```
预分配大块显存（如 4GB chunk）
    ↓
┌─────────────────────────────────────┐
│  Slab Allocator（分级管理）          │
│  ├─ Small Block Pool (1MB blocks)   │
│  ├─ Medium Block Pool (16MB blocks) │
│  └─ Large Block Pool (256MB blocks) │
└─────────────────────────────────────┘
    ↓
请求来时：找到最合适大小的 pool → 从 free list 取 block
释放时：归还到对应 pool 的 free list
```

**数据结构选择**：

| 数据结构 | 用途 | 优势 |
|---|---|---|
| Block Table (Hash Map) | 逻辑block→物理block映射 | O(1)查找，PagedAttention核心 |
| Free Block Queue (FIFO) | 管理空闲block | 分配/释放 O(1) |
| Buddy System | 2的幂次大小管理 | 支持分裂/合并，减少碎片 |
| Slab Allocator | 固定几种大小分级 | 无外部碎片，分配极快 |

**PyTorch Caching Allocator 的设计**：
- 维护多个 Block Pool（按设备、按大小分级）
- 首次分配：cudaMalloc 大块（512MB），切分为多个 block
- 后续分配：从 cache 中找最佳匹配（best-fit）
- 释放：标记为 free 但不归还 OS
- OOM 时：执行 empty_cache 尝试回收后重试

**PagedAttention 的显存管理**：
- Block 大小固定（16 tokens × num_kv_heads × head_dim × 2 × dtype）
- Free block 用简单队列管理
- 分配：从队列头取；释放：归还队列尾
- 利用率接近 100%（唯一浪费是最后一个 block 的未填满部分）

### Q: 手撕：三数之和？

（编程题）
