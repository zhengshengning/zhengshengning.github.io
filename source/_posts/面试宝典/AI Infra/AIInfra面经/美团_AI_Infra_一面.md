---
title: '美团 AI Infra 一面'
date: 2026-04-16 12:18:36
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: Decoder-only 和 Encoder-only 模型的优缺点？

这两种架构代表了 Transformer 的两种主要使用范式，本质区别在于**注意力的方向性**和**训练目标**：

**Decoder-only（GPT 系列、LLaMA、Qwen）**：

| 维度 | 特点 |
|---|---|
| 注意力方向 | 单向（因果 mask，每个 token 只能看到之前的 token） |
| 训练目标 | Next Token Prediction：P(x_t \| x_1, ..., x_{t-1}) |
| 核心优势 | 自回归生成能力强、few-shot/in-context learning 能力好 |
| Scaling 特性 | Scaling Law 明确，增大规模持续涨点 |
| 推理特点 | 逐 token 生成（decode 是 memory-bound），但可用 KV Cache 加速 |

**Encoder-only（BERT 系列、RoBERTa）**：

| 维度 | 特点 |
|---|---|
| 注意力方向 | 双向（每个 token 可看到所有其他 token） |
| 训练目标 | Masked Language Model：根据上下文预测被 mask 的 token |
| 核心优势 | 双向上下文理解更充分，适合 NLU 任务 |
| 局限 | 不适合生成（需要任务特定 head）、较难做 ICL |
| 推理特点 | 一次前向得到所有 token 表示（compute-bound） |

**为什么 Decoder-only 成为主流**：

1. **统一范式**：所有任务（分类、生成、推理）都可以转化为文本生成
2. **Scaling 效果**：实验证明 Decoder-only 在大规模下 loss 持续下降，而 Encoder-only 收益递减
3. **Zero/Few-shot**：Decoder-only 通过 prompt 即可完成新任务，无需微调
4. **工程统一**：一个模型服务所有任务，部署和维护成本低

**Encoder-Decoder（T5、BART）的位置**：
- 兼具编码和生成能力
- 适合 seq2seq 任务（翻译、摘要）
- 但规模扩展不如 Decoder-only 经济（编码器的计算不能被 KV Cache 复用）
- 逐渐被 Decoder-only 取代（GPT-4 做翻译/摘要同样出色）

---

### Q: 介绍一下 LLaMA 模型？

LLaMA（Large Language Model Meta AI）是 Meta 开源的 Decoder-only Transformer 系列，是当前开源 LLM 生态的基石模型。

**架构设计（以 LLaMA-2/3 为代表）**：

```
每层结构:
  x → RMSNorm → Multi-Head Attention (with RoPE, GQA) → + x（残差）
    → RMSNorm → SwiGLU FFN → + x（残差）
```

**关键架构选择及原因**：

| 组件 | 选择 | 为什么 |
|---|---|---|
| Normalization | Pre-RMSNorm | 去掉均值中心化，计算减少 ~15%，训练稳定 |
| 位置编码 | RoPE | 相对位置编码，天然支持长度外推，无需训练位置嵌入 |
| 激活函数 | SwiGLU | FFN 中 `SiLU(xW_g) ⊙ xW_u`，带门控机制，效果优于 ReLU/GELU |
| 注意力 | GQA (LLaMA-2 70B+) | 减少 KV head 数，KV Cache 降低 4-8x，推理效率高 |
| Bias | 全部去掉 | 简化模型 + 对量化更友好（无 zero_point） |
| 词表 | SentencePiece BPE | LLaMA-3 升级到 tiktoken（128K 词表），提升多语言能力 |
| FFN dim | 8/3 × hidden | SwiGLU 有 gate+up 两个矩阵，为保持参数量对齐调整维度 |

**各版本演进**：

| 版本 | 规模 | 关键改进 |
|---|---|---|
| LLaMA-1 | 7B/13B/33B/65B | 开源基座，证明小模型+大数据可媲美大模型 |
| LLaMA-2 | 7B/13B/70B | 训练数据 2T tokens，70B 用 GQA，RLHF 对齐 |
| LLaMA-3 | 8B/70B | 15T tokens 训练，128K 词表，8K→128K 上下文 |
| LLaMA-3.1 | 8B/70B/405B | 首个开源 400B+ 模型，支持 128K 上下文 |

**关键数值**（LLaMA-3 8B）：
- 32 层，hidden_dim=4096，32 个 attention heads，8 个 KV heads（GQA 4:1）
- FFN hidden dim = 14336（≈ 8/3 × 4096，取 128 整数倍）
- 词表 128,256，BPE tokenizer
- 训练数据 15T tokens
- 总参数量 ~8B（其中 Embedding 0.5B，每层 ~0.23B）

---

### Q: CUDA 编程模型和内存模型？

**CUDA 编程模型——层次化的并行组织**：

```
Grid (网格)
  ├── Block 0 (线程块)
  │     ├── Warp 0 (线程束, 32 线程)
  │     │     ├── Thread 0
  │     │     ├── Thread 1
  │     │     └── ... Thread 31
  │     ├── Warp 1
  │     └── ... (最多 1024 线程/block)
  ├── Block 1
  └── ... (最多 2^31 blocks)
```

**核心概念**：
- **Thread**：最小执行单元，有唯一 ID（threadIdx + blockIdx × blockDim）
- **Warp（32 线程）**：硬件调度的基本单位，同一 warp SIMT 执行同一指令
- **Block（最多 1024 线程）**：共享 Shared Memory，可通过 `__syncthreads()` 同步
- **Grid**：所有 block 的集合，block 间无直接通信（需通过 Global Memory）
- **SM（Streaming Multiprocessor）**：硬件执行单元，同时驻留多个 block

**Kernel 启动语法**：
```cuda
kernel<<<gridDim, blockDim, sharedMemSize, stream>>>(args);
// gridDim: grid 中 block 的数量（1D/2D/3D）
// blockDim: block 中 thread 的数量（1D/2D/3D）
// sharedMemSize: 动态分配的 shared memory 大小
// stream: CUDA stream（默认 stream 0）
```

---

**CUDA 内存模型——从快到慢的层次结构**：

| 内存类型 | 作用域 | 容量（A100） | 延迟 | 带宽 | 特点 |
|---|---|---|---|---|---|
| 寄存器 | 每线程私有 | 255 个/线程 | ~1 cycle | ~数十 TB/s | 最快，数量有限 |
| Shared Memory | Block 内共享 | 最多 164 KB/SM | ~5-30 cycles | ~19 TB/s | 可编程 L1，需手动管理 |
| L1 Cache | SM 内 | 与 Shared Memory 共享 | ~30 cycles | ~数 TB/s | 硬件管理 |
| L2 Cache | 全局共享 | 40 MB | ~200 cycles | ~5 TB/s | 所有 SM 共享 |
| Global Memory (HBM) | 所有线程 | 80 GB | ~400-800 cycles | 2.0 TB/s | 最大最慢 |
| Constant Memory | 全局只读 | 64 KB | ~1-400 cycles | 有 cache 时极快 | 适合广播场景 |
| Texture Memory | 全局只读 | - | - | - | 空间局部性优化 |

**访存优化的核心思路**：
```
Global Memory（慢） → Shared Memory（中） → 寄存器（快）
         ↑                    ↑                 ↑
    减少次数           增加复用            最大化使用
    (合并访存,           (tiling)         (寄存器 tiling)
     向量化 float4)
```

---

### Q: 使用共享内存时需要注意什么？怎么避免 Bank Conflict？

**Shared Memory 使用注意事项**：

1. **同步要求**：
   - 写后读必须 `__syncthreads()` 同步，否则线程可能读到旧值
   - 常见模式：load to shared → sync → compute → sync → store back
   ```cuda
   __shared__ float s[256];
   s[threadIdx.x] = global_data[idx];  // 写入
   __syncthreads();                      // 同步（确保所有线程写完）
   float val = s[(threadIdx.x + 1) % 256];  // 读取其他线程写入的值
   ```

2. **容量限制与 Occupancy 权衡**：
   - A100：每个 SM 最多 164 KB shared memory
   - 单个 block 使用的 shared memory 越多 → SM 上能同时驻留的 block 越少 → occupancy 下降
   - 经验：单 block 使用 < 48 KB 通常不影响 occupancy

3. **生命周期**：仅在 block 执行期间有效，block 结束后自动释放

4. **动态分配 vs 静态分配**：
   ```cuda
   // 静态分配（编译时确定大小）
   __shared__ float s[BLOCK_SIZE];
   
   // 动态分配（运行时确定大小，通过 kernel launch 第三个参数）
   extern __shared__ float s[];
   kernel<<<grid, block, dynamicSize>>>();
   ```

---

**Bank Conflict 详解**：

Shared Memory 被组织为 **32 个 bank**，每个 bank 宽 4 字节，连续地址循环映射到不同 bank：
```
地址:    0    4    8   12   ...  124  128  132  ...
Bank:    0    1    2    3   ...   31    0    1   ...
```

**冲突产生条件**：同一 warp 的多个线程访问**同一 bank 的不同地址**时串行化。

**三种情况**：
- **无冲突**：32 个线程访问 32 个不同 bank → 1 个 cycle
- **N-way conflict**：N 个线程访问同一 bank → 串行化为 N 个 cycle
- **广播**：多个线程访问同一 bank 的**同一地址** → 广播，无冲突

**避免方法**：

**1. Padding（最简单有效）**：
```cuda
// 二维数组按行存储，每行 32 个 float → 行内无冲突
// 但按列访问时：thread 0 访问 [0][col], thread 1 访问 [1][col]
// 它们的 bank = col（相同）→ 32-way conflict!

// 解决：每行多分配 1 个 float（padding）
__shared__ float s[32][33];  // 注意是 33 不是 32
// 现在 s[i][col] 的实际偏移 = i*33 + col
// thread 0: bank = (0*33 + col) % 32
// thread 1: bank = (1*33 + col) % 32 = (33 + col) % 32 = (1 + col) % 32
// 各不相同 → 无冲突
```

**2. Swizzle（高级技巧）**：
```cuda
// 通过位运算重映射地址
int swizzled_col = col ^ (row % 32);  // XOR 使同列不同行落在不同 bank
```

**3. 128-bit 向量化访问**：
```cuda
// 使用 float4 访问：每线程一次读 16 字节（4 个连续 bank）
// 32 个线程 × 4 bank = 128 bank 访问，循环一轮刚好覆盖所有 bank
float4* shared_vec = reinterpret_cast<float4*>(shared_mem);
float4 val = shared_vec[threadIdx.x];  // 天然无冲突
```

---

### Q: 使用寄存器时需要注意什么？怎么避免 Register Spilling？

**寄存器的核心约束**：

| GPU | 寄存器/SM | 寄存器/线程(最大) | 超限后果 |
|---|---|---|---|
| A100 | 65536 × 32-bit | 255 | Spill to local memory (HBM 速度) |
| H100 | 65536 × 32-bit | 255 | 同上 |

**Occupancy 与寄存器的关系**：
```
假设 SM 有 65536 寄存器，每个 block 256 线程:
  每线程用 32 寄存器 → 每 block 用 8192 → SM 可驻留 8 blocks
  每线程用 64 寄存器 → 每 block 用 16384 → SM 可驻留 4 blocks
  每线程用 128 寄存器 → 每 block 用 32768 → SM 可驻留 2 blocks
  每线程用 255 寄存器 → 每 block 用 65280 → SM 只能驻留 1 block
```
寄存器用得越多 → occupancy 越低 → 延迟隐藏能力越弱。但寄存器 tiling 提高了数据复用。需要找到**最优平衡点**。

**Register Spilling 的危害**：
- 寄存器不够时，编译器将变量溢出到 **local memory**（实际在 HBM 上）
- 延迟从 ~1 cycle 暴增到 ~400 cycles
- 性能可能骤降 2-10x

**避免 Register Spilling 的方法**：

1. **`__launch_bounds__` 编译提示**：
   ```cuda
   __global__ void __launch_bounds__(256, 2)  // 每 block 256 线程, 最少 2 blocks/SM
   kernel() { ... }
   // 告诉编译器：每线程最多用 65536/(256×2) = 128 寄存器
   ```

2. **`-maxrregcount` 编译选项**：
   ```bash
   nvcc -maxrregcount=64 kernel.cu  # 限制每线程最多 64 寄存器
   ```

3. **减少局部变量/中间结果**：
   - 复用变量：`float tmp = a*b; result += tmp; tmp = c*d; result += tmp;`
   - 避免同时存活太多变量

4. **控制循环展开程度**：
   ```cuda
   #pragma unroll 4  // 而非 #pragma unroll（完全展开可能爆寄存器）
   for (int i = 0; i < N; i++) { ... }
   ```

5. **使用 Shared Memory 替代**：
   - 对于大数组（如 tiling 的缓冲区），放 shared memory 而非寄存器
   - 权衡：shared memory 延迟 ~5-30 cycles vs 寄存器 ~1 cycle

6. **调整 Thread Tile 大小**：
   - GEMM 中每线程计算 TM×TN 个输出元素，需要 TM×TN 个寄存器存结果
   - TM=TN=8 → 64 个结果寄存器 + 加载缓冲 → 可能接近上限
   - 如果 spilling 严重，减小到 TM=TN=4

---

### Q: GPU 多线程和 CPU 多线程有什么区别？

两者的设计哲学完全不同——CPU 追求**单线程低延迟**，GPU 追求**大规模高吞吐**：

| 维度 | GPU | CPU |
|---|---|---|
| **线程数量** | 数万-数百万同时存在 | 通常 < 100 个 |
| **单线程能力** | 弱（低频率 ~1.5 GHz，简单 in-order） | 强（高频率 ~5 GHz，乱序执行、分支预测） |
| **上下文切换** | **零开销**（寄存器文件足够大，所有线程常驻） | 昂贵（~1-10μs，需保存/恢复寄存器、TLB flush） |
| **延迟隐藏策略** | 大量线程切换（warp 等内存时切换到另一个 warp） | Cache + 乱序执行 + 预取 |
| **内存模型** | 所有线程共享全局内存（但有 shared/local 层次） | 每线程有独立栈，共享堆 |
| **同步机制** | `__syncthreads()`（block）、`__syncwarp()`（warp）| 互斥锁、信号量、条件变量 |
| **调度单位** | Warp（32 线程 SIMT） | 单线程（OS 调度器） |
| **适合场景** | 高并行度、规则数据并行（SIMD 式） | 复杂控制流、分支密集、串行逻辑 |

**GPU 零开销切换的原理**：
```
CPU: 线程 A 执行 → 中断 → 保存 A 的寄存器到内存 → 加载 B 的寄存器 → 线程 B 执行
     ↑ 开销 ~1-10μs

GPU: Warp A 执行 → 遇到长延迟操作（如内存读取）→ 调度器选择 Warp B → 即时执行 B
     ↑ 开销 0 cycle（因为所有 warp 的寄存器都已在寄存器文件中）
```

**为什么 GPU 能存下所有线程的状态**：
- A100 每个 SM 有 65536 个 32-bit 寄存器 = 256 KB 寄存器文件
- 如果每线程用 32 寄存器，可常驻 2048 个线程（64 个 warp）
- 所有 warp 的上下文都在寄存器文件中，切换只需改变调度器的指针

**延迟隐藏的量化分析**：
```
HBM 访问延迟: ~400 cycles
每 cycle 可执行: 1 warp 的 1 条指令
要完全隐藏内存延迟: 需要 ~400/执行延迟 ≈ 12-15 个 ready warp

如果 occupancy = 50%（如 32 warp/SM）:
  有足够的 warp 隐藏延迟 → 带宽利用率高
如果 occupancy = 12.5%（如 4 warp/SM）:
  可能无法隐藏延迟 → 性能下降
```

---

### Q: FlashAttention 的原理？

FlashAttention 是一种 **IO-aware** 的精确 Attention 实现，通过 Tiling 和 Online Softmax 算法避免在 HBM 中存储完整的 N×N attention 矩阵。

**标准 Attention 的内存问题**：
```
标准实现:
  S = Q × K^T          [N×N] → 写入 HBM
  P = softmax(S)       [N×N] → 读/写 HBM
  O = P × V            [N×d] → 写入 HBM
  
  HBM 读写量: O(N² + Nd) → 当 N >> d 时，N² 主导
  显存占用: O(N²) → seq=4096, fp16 时需 32 MB/head
```

**FlashAttention 的核心思想**：
```
Tiling 实现:
  将 Q 分为 [Q_1, Q_2, ..., Q_Tr] 块 (每块 Br 行)
  将 K, V 分为 [K_1, K_2, ..., K_Tc] 块 (每块 Bc 行)
  
  外循环: 遍历 K/V 块
    内循环: 遍历 Q 块
      在 SRAM 中计算 S_ij = Q_i × K_j^T  (Br×Bc 的小矩阵)
      在 SRAM 中增量更新 softmax (Online Softmax)
      在 SRAM 中累积 O_i = O_i + P_ij × V_j
      
  HBM 读写量: O(N²d²/M) → 其中 M 是 SRAM 大小
  显存占用: O(N) → 只需存 O + softmax 统计量
```

**Online Softmax 算法**（增量计算 softmax，不需要完整 S 矩阵）：
```
维护两个统计量: m (running max) 和 l (running sum of exp)

处理新块 S_ij 时:
  m_new = max(m_old, rowmax(S_ij))
  l_new = l_old × exp(m_old - m_new) + rowsum(exp(S_ij - m_new))
  O_new = O_old × (l_old × exp(m_old - m_new) / l_new) + exp(S_ij - m_new) × V_j / l_new
```

**性能收益**：

| 维度 | 标准 Attention | FlashAttention |
|---|---|---|
| HBM 读写 | O(N² + Nd) | O(N²d²/M) ≈ 减少 ~10x |
| 显存占用 | O(N²) | O(N) |
| 实际加速 | 基准 | 2-4x faster（因减少 HBM 读写） |
| Seq=2K FP16 | ~32 MB/head | ~16 KB/head |

**为什么减少 HBM 读写能加速**：Attention 的算术强度很低（~O(N) FLOPS / O(N²) bytes），是典型的 memory-bound 算子。减少 HBM 读写直接提升性能。

---

### Q: PagedAttention 的原理？

PagedAttention 是 vLLM 的核心创新，借鉴操作系统虚拟内存的**分页机制**解决 KV Cache 的显存碎片问题。

**传统 KV Cache 管理的问题**：
```
传统方案: 为每个请求预分配最大序列长度的连续显存
  
  请求 A (实际 500 tokens, 预分配 2048 tokens): [===使用===|-------浪费-------]
  请求 B (实际 1200 tokens, 预分配 2048 tokens): [=======使用=======|---浪费---]
  请求 C (实际 300 tokens, 预分配 2048 tokens): [==使用==|--------浪费--------]
  
  问题:
  - 内部碎片: 实际利用率 = (500+1200+300)/(2048×3) = 32.6%
  - 外部碎片: 连续空间不够时无法接收新请求（即使总空闲显存足够）
```

**PagedAttention 的设计**：
```
分页管理:
  物理显存被划分为固定大小的 block（如每 block 存 16 tokens 的 KV）
  
  Block Table（逻辑→物理映射）:
    请求 A: logical [0,1,2,...,31] → physical [5, 12, 7, ..., 23]（按需分配）
    请求 B: logical [0,1,2,...,74] → physical [3, 18, 0, ..., 41]
  
  显存布局（非连续）:
    Physical Block 0: [请求 B 的 tokens 32-47]
    Physical Block 1: [空闲]
    Physical Block 2: [空闲]
    Physical Block 3: [请求 B 的 tokens 0-15]
    Physical Block 4: [空闲]
    Physical Block 5: [请求 A 的 tokens 0-15]
    ...
```

**关键机制**：

1. **按需分配**：生成新 token 时才分配新 block，不预分配
2. **无碎片**：物理 block 不需要连续，通过 block table 间接寻址
3. **Copy-on-Write (COW)**：Beam Search 等场景，多个 beam 共享前缀的 block，分叉时才拷贝
4. **利用率接近 100%**：唯一浪费是最后一个 block 的内部碎片（平均浪费 block_size/2 tokens）

**Attention 计算的适配**：
```cuda
// PagedAttention kernel 伪代码
for each query token:
    for each logical block in block_table:
        physical_block = block_table[logical_block_id]
        K_block = kv_cache[physical_block]  // 通过间接寻址读取
        V_block = kv_cache[physical_block]
        score += Q × K_block^T
    output = softmax(score) × V_blocks
```

**性能影响**：
- 间接寻址引入少量 overhead（~2-5% latency increase）
- 但显存利用率从 20-40% 提升到 >95%
- 同等显存下可服务 2-4x 更多并发请求 → 吞吐大幅提升
