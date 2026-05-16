---
title: '字节跳动 AML AI Infra 一二面'
date: 2026-04-16 12:15:22
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: LLM推理有什么主要瓶颈？

LLM推理的瓶颈本质源于自回归生成的计算模式和巨大的模型参数量，具体表现为以下五个层面：

**1. 内存带宽瓶颈（Decode阶段核心瓶颈）**：

Decode阶段每步生成1个token，却需要读取全部模型权重：

```
计算量:  2 × Params × 1(token) ≈ 140 GFLOP (70B模型)
读取量:  Params × sizeof(dtype) = 140 GB (FP16)

Arithmetic Intensity = 140G / 140G = 1 FLOP/Byte
A100峰值: 312 TFLOPS / 2 TB/s = 156 FLOP/Byte

实际利用率: 1/156 = 0.64%！
→ GPU 99.4%的算力被浪费，纯粹在等数据从HBM搬到计算单元
```

这就是为什么Decode阶段是**memory-bound**——算力远远富余，带宽成为瓶颈。单步延迟 ≈ 模型大小 / HBM带宽：
- 7B FP16: 14GB / 2TB/s = 7ms
- 70B FP16: 140GB / 2TB/s = 70ms（单卡不可接受，需要TP）

**2. KV Cache显存限制**：

| 模型 | 每token KV大小 | 4K序列单请求 | 32并发4K | 占比(80GB) |
|------|--------------|------------|---------|-----------|
| LLaMA-7B | 0.5MB | 2GB | 64GB | 80% |
| LLaMA-70B | 1.25MB | 5GB | 160GB | 200%！ |
| Qwen-72B(GQA) | 0.33MB | 1.3GB | 42GB | 53% |

KV Cache显存随seq_len×batch_size线性增长，成为限制最大并发数和最大序列长度的硬约束。GQA通过减少KV heads缓解（8→1），但长序列场景下仍然是瓶颈。

**3. 自回归串行限制**：
- 每个token的生成依赖前一个token的输出（因为需要前一个token的hidden state来计算attention）。
- 无法像Prefill阶段那样并行处理所有token。
- 生成N个token至少需要N步串行推理，延迟 = N × 单步延迟。
- 即使硬件无限强（单步0延迟），也需要N步逻辑串行。
- 这是自回归模型的结构性限制，只能通过投机解码/非自回归方法缓解。

**4. Prefill计算延迟**：
```
Prefill计算量 = 2 × Params × seq_len（GEMM部分）
            + 2 × num_layers × seq_len^2 × d_model（Attention部分）

70B模型, 4K输入:
  GEMM: 2 × 70B × 4096 ≈ 574 TFLOP
  Attention: 2 × 80 × 4096^2 × 8192 ≈ 22 TFLOP
  总计 ≈ 596 TFLOP

A100 FP16峰值312 TFLOPS → 最快1.9秒
实际（考虑效率60%）→ ~3.2秒 TTFT
```

长输入的首token延迟（TTFT）高，直接影响用户体验。Attention的O(n^2)复杂度使得超长序列（>32K）的Prefill变得极其昂贵。

**5. 调度效率低**：
- 不同请求长度差异大（从10 tokens到32K tokens）。
- 静态batching中短请求等待长请求完成→padding浪费。
- Prefill和Decode对计算资源的需求不同（compute-bound vs memory-bound），混合调度导致两者都不最优。
- 请求到达时间不均匀，burst到达时排队延迟增加。

**各瓶颈的量化影响排序（70B模型推理）**：

| 瓶颈 | 影响程度 | 核心指标 | 优化后收益 |
|------|---------|---------|-----------|
| 带宽瓶颈 | 最高 | 单步70ms(FP16单卡) | 量化+TP→10ms |
| KV Cache | 高 | 限制并发32→满载 | PagedAttention→3x并发 |
| 串行生成 | 高 | 延迟=N×单步 | 投机解码→延迟/2-3 |
| Prefill延迟 | 中 | TTFT 3+秒 | PD分离+FlashAttn→1秒 |
| 调度浪费 | 中 | GPU利用率30% | Continuous Batch→75% |

---

### Q: LLM推理主要的优化技术？

每种优化技术针对特定瓶颈，下面按原理和收益详细展开：

**1. KV Cache + PagedAttention**：
- KV Cache消除重复计算（从O(n^2)到O(n) per step）。
- PagedAttention解决显存碎片问题（利用率从~40%提升到>95%）。
- 两者组合效果：同样显存支持3-4x更多并发请求。

**2. FlashAttention减少HBM访问**：
```
标准Attention: Q×K^T(写HBM) → Softmax(读写HBM) → ×V(写HBM)
             HBM访问量: O(N^2) 用于存储NxN attention矩阵

FlashAttention: 分块在SRAM中完成所有计算，使用online softmax
             HBM访问量: O(N^2/SRAM_size) ≈ O(N) 对于大多数实际序列长度
```
- Prefill阶段加速2-4x（IO密集时更显著）。
- 额外收益：显存省N^2（不需要存完整attention矩阵）。
- FlashDecoding：Decode阶段在KV维度并行，利用更多SM资源。

**3. Continuous Batching动态调度**：
```
静态Batching:
  Batch开始: [Req1(100tok), Req2(500tok), Req3(200tok)]
  Req1完成后: [padding, Req2继续, padding]  ← GPU大量空转
  
Continuous Batching:
  Req1完成: 移出 → 立即插入Req4
  每步都保持batch尽可能满 → GPU利用率最大化
```
- 吞吐提升2-3x，平均延迟降低（短请求不再被长请求阻塞）。
- 实现需要细粒度的内存管理（每个请求独立的KV Cache管理）。

**4. 量化减少带宽需求**：

| 量化方案 | 带宽节省 | 精度损失 | 额外收益 |
|---------|---------|---------|---------|
| W8A8 (SmoothQuant) | 2x | <1% | INT8 TC吞吐2x |
| W4A16 (GPTQ/AWQ) | 4x(权重) | 1-3% | 单卡放更大模型 |
| FP8 (H100原生) | 2x | <0.5% | FP8 TC吞吐2x |
| W4A4 | 4x | 3-5% | 极限压缩 |

Decode阶段是memory-bound，量化直接将瓶颈带宽需求降低2-4x→接近等比例加速。

**5. 投机解码减少生成步数**：
```
传统: N tokens需要N步大模型推理
投机解码: 
  小模型draft K=5个token → 大模型1步验证 → 接受j个(j≤K)
  N tokens只需约N/(j+1)步大模型推理
  
  平均接受率α=0.8, K=5时:
  有效加速 ≈ 1/(1-0.8^5) × (5/6) ≈ 2.5x
```
- 关键条件：小模型的延迟远小于大模型；接受率足够高。
- Medusa/EAGLE变体：不用独立小模型，用附加head预测多个token。

**6. 张量并行多卡降低延迟**：
- TP=N时，每卡只需存储和读取1/N的权重→单步延迟降为约1/N。
- 通信开销：每层2次AllReduce，NVLink（900GB/s）下通常<5%总延迟。
- 实际加速：TP=8时约5-6x（通信+同步开销），但延迟从70ms降至~12ms。
- 注意：TP增加batch维度不变→吞吐不变，纯粹降延迟。

**7. PD分离独立优化各阶段**：

| 阶段 | 特征 | 最优配置 |
|------|------|---------|
| Prefill | Compute-bound, 大矩阵乘 | 高算力GPU, 大batch并行 |
| Decode | Memory-bound, 低算术强度 | 高带宽, 大batch以提高AI |

分离后：
- Prefill集群：可以充分利用Tensor Core算力。
- Decode集群：优化batch size以提高带宽利用率，不被Prefill抢占。
- 消除调度干扰：Prefill的burst计算不再阻塞Decode的延迟敏感步骤。

**8. 算子融合减少kernel开销**：
```
未融合: LayerNorm(读HBM→写HBM) + Residual(读HBM→写HBM) + Linear(读HBM→写HBM)
融合后: Fused_LN_Res_Linear(读HBM一次→SRAM中完成LN+Res→Linear→写HBM一次)

节省: 2次中间tensor的HBM读写（每次~数MB），减少2个kernel launch(~10us)
70B模型有240+个小kernel → 融合后减至~80个 → 总kernel launch时间从1.2ms→0.4ms
```

**优化技术组合的实际应用（以vLLM为例）**：
```
vLLM = PagedAttention + Continuous Batching + FlashAttention + 
       Prefix Cache + 量化支持 + 张量并行
       
典型配置（LLaMA-70B, 2×A100）:
  TP=2, W4量化, PagedAttention, Continuous Batching
  → 吞吐: ~2000 tokens/s, TTFT <1s, 延迟 ~30ms/token
```

---

### Q: PagedAttention的原理？

**核心问题——传统KV Cache的显存碎片化**：

```
传统方案——预分配连续显存:
请求到来时分配max_seq_len的连续空间:

显存布局:
[Req1: 4096 slots(实际用了500)][Req2: 4096 slots(实际用了2000)][空闲碎片][Req3...]
        ↑ 浪费88%                      ↑ 浪费51%              ↑ 不可用

问题:
1. 内部碎片: 预分配比实际使用多 → 平均浪费60-80%
2. 外部碎片: 释放后的空间不连续 → 新请求可能放不下
3. 不可预知: 不知道请求最终多长 → 只能按最大长度分配
```

**PagedAttention的设计——借鉴OS虚拟内存**：

```
核心思想: 将KV Cache切分为固定大小的block(如16 tokens)
         通过block table维护逻辑→物理映射
         物理block可以在显存中不连续分布

数据结构:
  Block size = 16 tokens
  每个block存储: 16 × num_kv_heads × head_dim × 2(K+V) × sizeof(dtype)
  
  Block Table (per request):
    Logical Block 0 → Physical Block 7  (存token 0-15的KV)
    Logical Block 1 → Physical Block 2  (存token 16-31的KV)
    Logical Block 2 → Physical Block 15 (存token 32-47的KV)
    ...
    
  Free Block Pool: [Block 0, 3, 4, 6, 8, 9, 10, ...]
```

**分配流程**：
```
1. 新请求到达:
   - 不预分配任何block
   - 开始Prefill时按需从free pool取block

2. 生成过程中:
   - 当前block满(16 tokens)→从free pool取新block，更新block table
   - 每次只分配1个block → 显存浪费最多 = 1个block(最后一个未填满)

3. 请求完成:
   - 归还所有物理block到free pool
   - 即时可用于其他请求（无碎片问题）
```

**Copy-on-Write共享前缀**：
```
Prompt: "You are a helpful assistant. User: "
Request 1和Request 2共享相同前缀:

Block Table (Request 1): [Block 7, Block 2, Block 15, Block 20]
Block Table (Request 2): [Block 7, Block 2, Block 15, Block 23]
                          ↑ 共享前缀blocks        ↑ 各自独立

ref_count[Block 7] = 2, ref_count[Block 2] = 2, ref_count[Block 15] = 2
ref_count[Block 20] = 1, ref_count[Block 23] = 1

如果需要修改共享block(beam search分叉):
  → 先copy再modify（Copy-on-Write语义）
```

**Attention Kernel的适配**：
```cuda
// PagedAttention kernel伪代码
// 输入: query[batch, heads, d], block_tables[batch, max_blocks], 
//       kv_cache[num_physical_blocks, block_size, heads, d]
for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int physical_block = block_tables[request_id][block_idx];
    // 从物理block地址读取KV
    float* k_ptr = &kv_cache[physical_block * block_stride];
    // 计算attention scores for this block
    for (int pos = 0; pos < block_size; pos++) {
        score += query[head] * k_ptr[pos * head_stride];
    }
}
// online softmax累积 + value加权
```

**性能数据对比**：

| 指标 | 传统连续分配 | PagedAttention | 提升 |
|------|------------|---------------|------|
| 显存利用率 | 20-40% | >95% | 2-4x |
| 最大并发(13B,A100) | ~8请求 | ~24请求 | 3x |
| 吞吐量 | baseline | 2-4x | 显著 |
| 内存碎片 | 严重 | 近零（仅最后block） | - |
| Beam Search显存 | K×full copy | 共享+CoW | K倍节省 |

---

### Q: Orca迭代级请求调度是什么？

**Orca（Continuous Batching / Iteration-level Scheduling）**是Yu等人2022年提出的LLM推理调度策略，核心创新在于将调度粒度从"请求级别"细化到"单步迭代级别"：

**传统静态Batching的问题**：
```
时间线:
Step 0: Batch = [Req1(要生成200tok), Req2(要生成50tok), Req3(要生成100tok)]
Step 50: Req2完成 → [Req1继续, padding, Req3继续]  ← GPU 33%空转
Step 100: Req3完成 → [Req1继续, padding, padding]  ← GPU 67%空转
Step 200: Req1完成

总有效步数: 200+50+100 = 350
总消耗步数: 200×3 = 600
GPU有效利用率: 350/600 = 58%
```

**Orca的迭代级调度**：
```
时间线:
Step 0:  Batch = [Req1, Req2, Req3]
Step 50: Req2完成 → 移出 → 插入Req4 → Batch = [Req1, Req4, Req3]
Step 100: Req3完成 → 移出 → 插入Req5 → Batch = [Req1, Req4, Req5]
...
每步都保持batch size = 3 → GPU利用率接近100%
```

**技术实现细节**：

1. **逐步检查完成状态**：每次decode迭代后检查每个请求是否生成了EOS token或达到max_length。完成的请求立即移出当前batch。

2. **动态插入新请求**：空出的slot立即从等待队列中取出优先级最高的新请求。新请求先做Prefill再加入Decode batch（或使用chunked prefill混合调度）。

3. **变长序列处理**：Batch内不同请求长度不同，需要特殊的Attention Kernel支持（如Paged Attention中通过block table处理不同长度）。

4. **调度策略**：
   - FCFS（先来先服务）：公平但可能P99延迟高。
   - Shortest-Job-First：优先短请求，降低平均延迟。
   - 优先级调度：VIP请求抢占普通请求。

**与PagedAttention的协同**：
```
Orca需要: 高效的KV Cache插入/释放
PagedAttention提供: 按block精确分配/回收，无碎片

组合效果:
  请求完成 → 归还所有block → 新请求进入 → 按需取新block
  整个过程无内存碎片、无预分配浪费
```

**Chunked Prefill（进一步优化）**：
```
问题: 长prompt的Prefill计算量大，会阻塞整个batch的Decode

解决: 将Prefill分chunk与Decode混合执行
  Step N: [Decode Req1-7] + [Prefill Req8 chunk1(256 tokens)]
  Step N+1: [Decode Req1-7] + [Prefill Req8 chunk2(256 tokens)]
  ...
  Req8 Prefill完成后加入Decode batch
  
收益: Decode请求的延迟不被Prefill阻塞
```

**性能收益量化**：

| 指标 | 静态Batching | Orca/Continuous Batching | 提升 |
|------|------------|------------------------|------|
| GPU利用率 | 30-50% | 70-90% | 2x+ |
| 吞吐量(tok/s) | baseline | 2-3x | 显著 |
| 平均延迟 | 高（等待长请求） | 低（短请求快速完成） | 30-50%↓ |
| P99延迟 | 极高 | 可控 | 大幅改善 |
| 显存效率 | 低（padding浪费） | 高（无padding） | 2x+ |

**生产系统实现**：
- vLLM：基于Orca + PagedAttention，最主流的开源推理引擎。
- TGI（Hugging Face）：类似实现。
- TensorRT-LLM：NVIDIA官方，in-flight batching。
- SGLang：进一步优化RadixAttention实现更智能的调度。

---

### Q: C++数组下标越界会报什么错？

**C++数组越界是最经典的未定义行为（UB）之一**——语言标准不规定任何特定行为，编译器不保证报错或崩溃：

**栈上数组越界（最常见的隐蔽bug）**：
```cpp
void foo() {
    int a[10];        // 栈上分配40字节
    int x = 42;       // x紧挨a的栈帧位置
    a[10] = 99;       // 越界！写到a后面的内存
    // 可能覆盖了x → x现在是99（静默错误）
    // 可能覆盖了rbp/返回地址 → 函数返回时跳到错误地址 → segfault
    // 可能什么都没发生（padding区域）→ 最难发现的bug
}
```

**不同越界方式的典型表现**：

| 越界类型 | 可能的现象 | 根因 |
|---------|-----------|------|
| 栈上数组写越界 | 覆盖其他局部变量/返回地址 | 栈是连续内存，无边界保护 |
| 栈上数组读越界 | 读到垃圾值或其他变量 | 同上 |
| 堆数组写越界(小) | 覆盖堆元数据→后续free崩溃 | malloc header紧挨数据 |
| 堆数组写越界(大) | SIGSEGV（跨越页边界） | 触碰到未映射页 |
| 负索引越界 | 访问数组前方的内存 | 可能是其他变量或guard page |

**为什么编译器不检查**：
- C/C++的设计哲学："不为不需要的东西付代价"。
- 每次数组访问都做范围检查→循环中的数组访问会慢2-5x（一次比较+一次分支）。
- 编译器可以在编译期检查常量索引的越界（但仅警告，不报错）。

**检测方法（从快到慢）**：

| 工具 | 检测能力 | 运行时开销 | 使用方式 |
|------|---------|-----------|---------|
| -Wall -Wextra | 编译期常量越界警告 | 0 | 编译选项 |
| AddressSanitizer | 栈/堆/全局越界 | 2x时间, 2-3x内存 | `-fsanitize=address` |
| Valgrind(Memcheck) | 堆越界、UAF、泄漏 | 10-50x时间 | `valgrind ./app` |
| Bounds Checking | 精确范围检查 | 2-5x | `-fsanitize=bounds` |
| std::vector::at() | 运行时异常 | 极小 | 替代[]运算符 |

**AddressSanitizer工作原理**：
```
为每个内存区域维护"shadow memory"记录是否可访问:
  [有效区域] [red zone(投毒)] [有效区域] [red zone]
  
数组: [a[0]..a[9]][red zone 32 bytes][其他变量][red zone]
访问a[10] → 命中red zone → ASan立即报告并打印调用栈

输出示例:
  ERROR: AddressSanitizer: stack-buffer-overflow on address 0x7ffd...
  WRITE of size 4 at 0x7ffd... thread T0
  #0 foo() at main.cpp:5
```

**安全的替代方案**：
```cpp
// 1. std::array + at()
std::array<int, 10> arr;
arr.at(10);  // 抛出 std::out_of_range 异常

// 2. std::vector + at()
std::vector<int> vec(10);
vec.at(10);  // 抛出异常

// 3. std::span (C++20) 提供bounds-checked view
std::span<int> s(arr);
s[10];  // 仍然UB，但at()可检查

// 4. GSL的gsl::span  
gsl::span<int> s(arr);  // 可配置debug模式检查

// 生产代码中的平衡:
// - Debug模式: 开启bounds checking和ASan
// - Release模式: 关闭检查以获得性能
```

**AI框架中的相关问题**：
- CUDA kernel中的数组越界：不会立即崩溃，可能写坏其他数据→结果静默错误。需要`compute-sanitizer`检测。
- Tensor shape不匹配：PyTorch/TensorFlow会做运行时shape检查，越界会抛异常。
- 内存访问越界是GPU kernel最常见的bug之一（尤其在手写kernel中）。

---

### Q: Linux下如何Debug和定位错误？

Linux环境下的调试是一个系统化的过程，需要根据问题类型选择合适的工具组合：

**1. GDB（最基础、最强大的交互式调试器）**：

```bash
# 编译时加-g保留调试信息
g++ -g -O0 main.cpp -o app

# 基本使用流程
gdb ./app
(gdb) b main.cpp:42          # 设断点
(gdb) r                       # 运行
(gdb) bt                      # 崩溃时查看调用栈(最常用！)
(gdb) p variable              # 打印变量
(gdb) info locals             # 查看所有局部变量
(gdb) watch *ptr              # 数据断点：ptr指向的值变化时中断
(gdb) thread apply all bt     # 多线程：打印所有线程调用栈

# 条件断点（只在特定条件下停下）
(gdb) b forward if batch_size == 0

# 远程调试（调试服务器上的程序）
gdbserver :1234 ./app
gdb -ex "target remote server:1234" ./app
```

适用场景：逻辑错误、崩溃定位、多线程死锁分析。

**2. Core Dump分析（事后调试/生产环境）**：

```bash
# 启用core dump
ulimit -c unlimited
echo "/tmp/core.%e.%p" > /proc/sys/kernel/core_pattern

# 程序崩溃后生成core文件
./crash_app   # 崩溃 → 生成 /tmp/core.crash_app.12345

# 用GDB加载分析
gdb ./crash_app /tmp/core.crash_app.12345
(gdb) bt                      # 查看崩溃时的完整调用栈
(gdb) frame 3                 # 切到可疑帧
(gdb) info locals             # 查看该帧的局部变量
(gdb) p *this                 # 查看对象状态
```

关键优势：不需要重现问题——生产环境崩溃一次就能分析。

**3. AddressSanitizer（内存错误首选检测工具）**：

```bash
# 编译时启用
g++ -fsanitize=address -g main.cpp -o app_asan
# 或 clang++ -fsanitize=address -fno-omit-frame-pointer -g

# 可检测的错误类型:
# - Stack/Heap buffer overflow（数组越界）
# - Use-after-free（释放后使用）
# - Use-after-return（返回后使用栈变量指针）
# - Double free
# - Memory leak（程序退出时）

# 输出示例:
# ==12345==ERROR: AddressSanitizer: heap-buffer-overflow
# READ of size 4 at 0x602000000014 thread T0
#   #0 0x4006a7 in main /tmp/main.cpp:8
# 0x602000000014 is located 0 bytes to the right of 4-byte region
```

开销：约2x运行时间、3x内存。建议CI中始终开启。

**4. strace（系统调用级别追踪）**：

```bash
# 追踪程序的所有系统调用
strace -f -o trace.log ./app    # -f追踪子进程/线程

# 常见用途:
strace -e trace=open,read,write ./app  # 只看文件IO
strace -e trace=network ./app          # 只看网络调用
strace -c ./app                        # 统计各系统调用次数和耗时

# 典型问题定位:
# - 程序卡住: strace看到阻塞在哪个syscall（如read/futex）
# - 文件找不到: 看open()返回ENOENT的路径
# - 权限问题: 看EACCES的调用
# - 动态库加载失败: 看openat()尝试的路径
```

适用场景：IO问题、权限问题、程序启动失败、动态库加载问题。

**5. dmesg/journalctl（内核级别信息）**：

```bash
# 查看内核消息（OOM、segfault等）
dmesg | tail -20
dmesg | grep -i "oom\|killed\|segfault"

# 典型输出:
# [12345.678] Out of memory: Killed process 5678 (python3), 
#             total-vm:200000000kB, anon-rss:81920000kB
# → 被OOM Killer杀掉，因为占用80GB RSS

# [12345.678] app[5678]: segfault at 0000000000000010 ip 00007f... 
#             sp 00007ffd... error 4 in libfoo.so[...]
# → 空指针偏移16字节处读取（error 4 = user read access）

# systemd管理的服务日志
journalctl -u my-service --since "5 minutes ago"
```

**6. Valgrind（最全面的内存检查，但最慢）**：

```bash
# 内存错误检测
valgrind --leak-check=full --show-leak-kinds=all ./app

# 检测能力:
# - 未初始化内存读取（ASan检测不到！）
# - 内存泄漏（精确到分配位置）
# - 非法读写（但不如ASan准确）

# 缺点: 10-50x运行时间，不适合大型程序的日常使用
```

**7. perf（性能分析和热点定位）**：

```bash
# CPU热点分析
perf record -g ./app           # 采样记录
perf report                    # 查看热点函数

# 统计cache miss等硬件事件
perf stat -e cache-misses,cache-references,instructions ./app

# 火焰图生成
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

**调试决策树——根据症状选工具**：

| 症状 | 首选工具 | 备选 |
|------|---------|------|
| Segfault崩溃 | Core dump + GDB bt | ASan重现 |
| 结果错误（静默bug） | ASan + GDB | Valgrind |
| 内存泄漏 | ASan(LeakSan) | Valgrind |
| 性能问题 | perf + 火焰图 | strace(IO) |
| 程序卡死 | GDB attach + bt | strace |
| 多线程死锁 | GDB thread apply all bt | ThreadSanitizer |
| OOM被杀 | dmesg + memory profiler | /proc/pid/status |
| 文件/网络问题 | strace | ltrace |
| CUDA错误 | compute-sanitizer | cuda-gdb |

**AI Infra特有的调试场景**：
- CUDA illegal memory access：用`compute-sanitizer --tool memcheck ./app`定位kernel内的越界。
- NCCL通信挂起：`NCCL_DEBUG=INFO`环境变量 + GDB attach查看各rank状态。
- PyTorch CUDA OOM：`torch.cuda.memory_summary()`查看显存分配详情。
- 分布式训练某个rank挂住：`py-spy`采样Python调用栈，不需要GDB。

---

### Q: 手撕：反转链表？

（编程题）

---

### Q: 手撕：LRU Cache？

（编程题）
