---
title: '腾讯 AI Infra 校招 一二面1'
date: 2026-04-16 12:17:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: TensorRT-LLM、SGLang/vLLM的源代码架构区别？

这三个框架分别代表了LLM推理领域的三种设计哲学，从底层到上层分别追求**性能极致、通用易扩展、调度创新**。

**TensorRT-LLM架构：**

```
用户模型 (HuggingFace等)
    ↓ 模型定义层 (Python, 类似PyTorch API定义网络结构)
    ↓ Builder (编译为TensorRT Engine)
    ↓ C++ Runtime (高性能执行引擎)
        ├── Batch Manager: inflight batching调度
        ├── GPT Session: 管理KV-Cache和生成循环
        └── TensorRT Engine: 高度融合的CUDA kernel
```

核心特点：
- **编译时优化**：模型在部署前编译为TensorRT Engine，kernel高度融合（如整层Transformer融合为几个大kernel）
- **C++为主**：运行时几乎全C++，Python只用于模型定义和编译阶段
- **性能极致但灵活性低**：Engine绑定特定shape/精度，动态shape支持有限
- **闭源kernel**：核心GEMM/Attention kernel不开源，难以自定义

**vLLM架构：**

```
FastAPI/HTTP Server
    ↓ AsyncLLMEngine (异步请求管理)
    ↓ LLMEngine
        ├── Scheduler (Python): 管理waiting/running/swapped队列
        │     └── BlockManager: PagedAttention的block分配/回收
        ├── ModelRunner: 准备输入tensor，调用模型前向
        │     └── Model (PyTorch): 标准Transformer实现
        └── Worker: 每个GPU一个进程
              └── CacheEngine: KV-Cache的物理管理
```

核心特点：
- **Python为主**：调度器、模型定义均为Python，易于阅读和修改
- **PagedAttention**：核心创新，KV-Cache按block管理（16 token/block），通过block_table间接索引
- **模块化**：模型、调度、执行分层清晰，扩展新模型只需实现forward()
- **生态丰富**：支持100+模型，社区活跃

**SGLang架构：**

```
SGLang Frontend (结构化生成语言)
    ↓ Router/Load Balancer
    ↓ SGLang Runtime
        ├── Scheduler: 更智能的调度策略
        │     ├── RadixAttention: 前缀树管理KV-Cache
        │     └── Chunked Prefill + Decode混合调度
        ├── Model Runner: 与vLLM类似但优化更激进
        └── FlashInfer Backend: 高性能Attention kernel
```

核心特点：
- **RadixAttention**：用Radix Tree（基数树）管理所有请求的KV-Cache前缀，自动发现和复用共享前缀
- **前端语言**：提供结构化生成DSL（如JSON schema约束、多次调用复用）
- **调度优化**：更激进的chunked prefill策略，更好的batch组装
- **FlashInfer集成**：使用FlashInfer库提供高性能的变长attention kernel

**三者对比：**

| 维度 | TensorRT-LLM | vLLM | SGLang |
|------|-------------|------|--------|
| 语言 | C++为主 | Python为主 | Python为主 |
| 性能 | 单次推理最快 | serving高吞吐 | serving最高吞吐 |
| 灵活性 | 低（需重编译） | 高（PyTorch模型） | 高 |
| KV-Cache管理 | 连续分配+池化 | PagedAttention | RadixAttention |
| 前缀复用 | 支持但较基础 | Prefix Caching | Radix Tree自动复用 |
| 适用场景 | 生产部署追求极致性能 | 通用serving/研究 | 复杂应用(多轮/结构化) |
| 自定义难度 | 高 | 中 | 中 |

---

### Q: 为什么要有Continuous Batching？

**传统Static Batching的致命问题：**

```
Static Batching (batch=4):
时间 →
请求A: [=========]                    完成(20 tokens)
请求B: [==================]           完成(40 tokens)
请求C: [====]                         完成(8 tokens) ← 已完成但必须等B
请求D: [======]                       完成(12 tokens) ← 已完成但必须等B
         ^^^^^^^^^^^^^^^^^^ 
         整个batch必须等最长的B完成才能处理新请求
         C和D完成后GPU空转，利用率低
```

问题量化：假设batch中最长请求生成100 tokens，最短5 tokens，则GPU有效利用率可低至 `avg_len/max_len ≈ 30-50%`。

**Continuous Batching的工作方式：**

```
Continuous Batching (iteration-level):
Step 1: [A, B, C, D] → 4个请求同时decode
Step 8: [A, B, -, D] → C完成,从batch移出,E加入
Step 12: [A, B, E, -] → D完成,F加入
Step 20: [-, B, E, F] → A完成,G加入
...每个step动态调整batch组成
```

**核心机制：**

| 特性 | 实现方式 | 收益 |
|------|---------|------|
| iteration级调度 | 每个decode step检查完成/加入 | GPU始终满载 |
| 请求独立完成 | 达到EOS即释放资源 | 消除短请求被长请求阻塞 |
| 动态batch size | 根据显存预算动态调整 | 最大化并行度 |
| Prefill/Decode混合 | 同一step内可混合 | 减少新请求等待时间 |

**性能提升：**
- 吞吐提升：3-5x（请求长度差异越大收益越大）
- GPU利用率：从30-50%提升到80-95%
- 首token延迟(TTFT)：新请求无需等待当前batch完成

**实现挑战：**
- Prefill和Decode的计算模式不同（compute-bound vs memory-bound），混合调度需要平衡
- KV-Cache管理复杂：需要动态分配/释放，PagedAttention解决碎片问题
- Padding浪费：不同请求长度不同导致tensor需要padding（或使用变长kernel）

---

### Q: Python计算密集型任务使用多进程还是多线程？

**答案：必须使用多进程。** 原因在于Python的GIL（Global Interpreter Lock）机制。

**GIL的本质：**

```python
# CPython内部简化逻辑
while True:
    acquire_GIL()              # 获取全局锁
    execute_bytecode(100)       # 执行约100条字节码
    release_GIL()              # 释放锁，给其他线程机会
    # 只有一个线程能持有GIL → 多线程无法真正并行Python代码
```

**为什么有GIL？**
- CPython的内存管理（引用计数）不是线程安全的
- 加入GIL比给每个对象加锁简单得多
- 历史遗留：大量C扩展依赖GIL的线程安全保证

**多线程 vs 多进程对比：**

| 维度 | 多线程(threading) | 多进程(multiprocessing) |
|------|-----------------|----------------------|
| 计算密集型并行 | 不行（GIL限制） | 可以（每个进程独立GIL） |
| IO密集型 | 有效（IO时释放GIL） | 过重（进程开销大） |
| 内存共享 | 直接共享 | 需IPC(pipe/shm) |
| 创建开销 | 小（~10us） | 大（~100us，fork/spawn） |
| 数据传输 | 零成本 | 序列化开销(pickle) |

**绕过GIL的方法：**

```python
# 方法1: multiprocessing（最常用）
from multiprocessing import Pool
with Pool(8) as p:
    results = p.map(compute_heavy_func, data_chunks)

# 方法2: C扩展释放GIL
# NumPy/SciPy的C代码内部释放GIL后并行计算
import numpy as np
np.dot(A, B)  # 底层BLAS释放GIL，多线程有效

# 方法3: Cython显式释放GIL
# cython代码中:
# with nogil:
#     heavy_computation()

# 方法4: concurrent.futures统一接口
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(task, arg) for arg in args]
```

**深度学习框架的选择：**
- **PyTorch DataLoader**：多进程（`num_workers>0`时spawn子进程预处理数据）
- **PyTorch DDP训练**：多进程（每GPU一个进程，NCCL通信）
- **推理serving（vLLM）**：多进程 + 异步IO（每个GPU worker一个进程）

---

### Q: C++继承是怎么实现的？

**单继承的内存布局：**

```cpp
class Base {
    int a;          // offset 0
    virtual void f();
};
class Derived : public Base {
    int b;          // offset after Base
    void f() override;
    virtual void g();
};

// 内存布局 (64位系统):
// Derived对象:
// [vptr (8 bytes)] → 指向Derived的vtable
// [Base::a (4 bytes)]
// [padding (4 bytes)]
// [Derived::b (4 bytes)]
// [padding (4 bytes)]
// 总计: 24 bytes

// Derived的vtable:
// slot 0: &Derived::f  (覆盖了Base::f)
// slot 1: &Derived::g  (新增的虚函数)
```

**多继承的内存布局：**

```cpp
class A { int a; virtual void fa(); };
class B { int b; virtual void fb(); };
class C : public A, public B { int c; void fa() override; };

// C对象内存布局:
// [A的vptr] → C关于A的vtable
// [A::a]
// [B的vptr] → C关于B的vtable (需要this指针调整)
// [B::b]
// [C::c]

// 关键: 通过B指针调用C的方法时，需要调整this指针（thunk）
B* bp = new C();  // bp指向C对象中B子对象的起始位置
bp->fb();         // 如果C重写了fb，调用前需把this从B子对象调整到C起始
```

**虚继承解决菱形问题：**

```cpp
class A { int a; };
class B : virtual public A { int b; };
class C : virtual public A { int c; };
class D : public B, public C { int d; };

// 没有virtual: D中有两份A (B::A和C::A)，二义性
// 有virtual: D中只有一份A，通过虚基类表(vbtable)定位

// D对象布局:
// [B部分: vbptr + b]  vbptr中记录到虚基类A的偏移
// [C部分: vbptr + c]  vbptr中记录到虚基类A的偏移  
// [D::d]
// [A部分: a]          虚基类放在对象末尾（位置动态确定）
```

**虚函数调用机制的性能影响：**

| 调用方式 | 开销 | 原因 |
|---------|------|------|
| 普通函数 | 直接call指令 | 编译时确定地址 |
| 虚函数 | 间接call(vptr→vtable→func) | 运行时查表，可能cache miss |
| final/devirtualization | 直接call | 编译器证明只有一种可能时优化 |

虚函数调用本身开销约2-3ns（cache命中时），但会阻止内联优化，对热循环影响较大。

---

### Q: 手撕：最大子数组之和？

（编程题）

---

### Q: 求一个整数比特位中1的个数？

（编程题）

---

### Q: C++编译时计算（constexpr）？

**constexpr的演进和能力：**

| 版本 | constexpr能力 | 示例 |
|------|-------------|------|
| C++11 | 单return语句函数 | `constexpr int sq(int x) { return x*x; }` |
| C++14 | 允许循环、局部变量、多return | 完整的编译期算法 |
| C++17 | if constexpr（编译期条件分支） | 替代SFINAE的模板选择 |
| C++20 | consteval（强制编译期）、constexpr容器 | `consteval int must_compile_time()` |

**编译时计算的价值：**

```cpp
// 1. 编译期查找表生成（零运行时开销）
constexpr auto make_sin_table() {
    std::array<float, 360> table{};
    for (int i = 0; i < 360; i++)
        table[i] = /* 编译期计算sin */;
    return table;
}
constexpr auto sin_table = make_sin_table();  // 编译时完成

// 2. if constexpr: 编译期条件分支（替代模板特化/SFINAE）
template<typename T>
auto process(T val) {
    if constexpr (std::is_integral_v<T>) {
        return val * 2;           // 只有整数类型编译这个分支
    } else if constexpr (std::is_floating_point_v<T>) {
        return val * 1.5;         // 只有浮点类型编译这个分支
    }
    // 未选择的分支完全不参与编译（不需要合法）
}

// 3. constexpr容器（C++20）
constexpr std::vector<int> get_primes(int n) {
    std::vector<int> primes;
    // ... 编译期筛法 ...
    return primes;
}

// 4. consteval: 强制编译期求值
consteval int compile_time_only(int x) { return x * x; }
int runtime_val = 5;
// compile_time_only(runtime_val);  // 编译错误！必须编译期常量
```

**constexpr vs 模板元编程：**

| 维度 | constexpr | 模板元编程 |
|------|-----------|-----------|
| 语法 | 普通C++语法 | 递归模板特化 |
| 可读性 | 好 | 差 |
| 调试 | 编译器可给出清晰错误 | 错误信息难读 |
| 能力 | C++20后几乎等价 | 图灵完备 |
| 编译速度 | 通常更快 | 深度递归时慢 |

**在CUDA/HPC中的应用：**
- 编译期确定shared memory大小、block配置
- 根据GPU架构选择不同的kernel实现（`if constexpr`配合架构宏）
- 生成查找表避免运行时计算

---

### Q: vLLM中PagedAttention的原理？

**核心动机——传统KV-Cache管理的浪费：**

```
传统连续分配方式:
请求A (实际长度100, 预分配max=2048):
[████████░░░░░░░░░░░░░░░░░░░░░░░░] 实际使用不到5%
                                      浪费95%显存！

多请求时:
[AAAAAAA...............][BBBB...............][CC..................]
                 ↑ 内部碎片                        ↑ 外部碎片
                 
实测: 传统方式的显存有效利用率仅20-40%
```

**PagedAttention设计——借鉴OS虚拟内存：**

```
操作系统:                        PagedAttention:
虚拟页 → 物理页帧               逻辑KV位置 → 物理KV Block
页表映射                         Block Table映射
按需分配物理页                   按需分配物理Block
Copy-on-Write                    CoW for beam search
```

**Block Table结构：**

```
Block大小 = 16 tokens (每block存16个token的KV tensor)

请求A (已生成45 tokens):
逻辑Block:  [Block0] [Block1] [Block2]  (Block2只用了13/16)
Block Table: [7,  3,  15]  → 物理Block编号
                                  
请求B (已生成30 tokens):
逻辑Block:  [Block0] [Block1]  (Block1只用了14/16)
Block Table: [2,  9]

物理显存:
Block 0: [空闲]
Block 1: [空闲]
Block 2: [B-Block0的KV数据]
Block 3: [A-Block1的KV数据]
...
Block 7: [A-Block0的KV数据]
Block 9: [B-Block1的KV数据]
Block 15: [A-Block2的KV数据(部分)]
```

**关键机制：**

| 机制 | 原理 | 收益 |
|------|------|------|
| 非连续存储 | Block散落在显存任意位置 | 消除外部碎片 |
| 按需分配 | 每生成16 tokens才分配新Block | 消除内部碎片（最多浪费1个Block） |
| Block Table间接索引 | 类似页表的逻辑→物理映射 | O(1)查找，灵活管理 |
| Copy-on-Write | beam search分叉时共享Block直到写入时才复制 | 显存节省(beam-1)/beam |
| Prefix Sharing | 相同前缀的请求共享Block（引用计数） | 减少重复prefill |
| 抢占(Preemption) | 显存不足时swap Block到CPU | 避免OOM，保证服务可用 |

**Attention Kernel的修改：**

```cuda
// 传统Attention: KV连续存储，直接偏移访问
K_ptr = kv_cache + seq_offset * head_dim;

// PagedAttention Kernel: 通过block_table间接访问
for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int physical_block = block_table[seq_id][block_idx];
    float* K_block = kv_cache + physical_block * block_size * head_dim;
    // 对该block内的tokens计算attention score
    for (int t = 0; t < tokens_in_block; t++) {
        score += Q * K_block[t * head_dim : (t+1) * head_dim];
    }
}
```

**显存利用率对比：**
- 传统方式：20-40%（大量预分配浪费）
- PagedAttention：>96%（仅最后一个Block有少量内部碎片）
- 实际效果：相同显存下可服务2-4倍的并发请求

---

### Q: CUDA内存模型介绍？

**CUDA内存层次完整架构：**

```
┌─────────────────────────────────────────────────┐
│                  GPU Device                       │
│  ┌──────────SM 0──────────┐  ┌────SM 1────┐     │
│  │ Thread: Registers      │  │            │     │
│  │   (私有, ~1 cycle)      │  │            │     │
│  │                        │  │            │     │
│  │ Block: Shared Memory   │  │            │     │
│  │   (192KB/SM, ~5ns)     │  │            │     │
│  │                        │  │            │     │
│  │ L1 Cache (与Shared共享) │  │            │     │
│  └────────────────────────┘  └────────────┘     │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │        L2 Cache (全SM共享, 40MB, ~50ns)    │    │
│  └──────────────────────────────────────────┘    │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │    HBM (全局内存, 80GB, 2TB/s, ~400ns)     │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

**各内存空间详细参数（以A100为例）：**

| 内存类型 | 容量 | 带宽 | 延迟 | 作用域 | 管理方式 |
|---------|------|------|------|--------|---------|
| 寄存器 | 65536×32bit/SM | ~19TB/s | 1 cycle | 线程私有 | 编译器分配 |
| 共享内存 | 可配置最大164KB/SM | ~19TB/s | ~28 cycles | Block内共享 | 程序员显式 |
| L1 Cache | 与Shared共享192KB/SM | ~19TB/s | ~28 cycles | Block内 | 硬件自动 |
| L2 Cache | 40MB | ~5TB/s | ~200 cycles | 全GPU | 硬件自动 |
| 全局内存(HBM) | 80GB | 2039 GB/s | ~400ns | 全GPU | cudaMalloc |
| 常量内存 | 64KB | ~很高(命中时) | ~4 cycles(命中) | 全GPU只读 | __constant__ |
| 纹理内存 | 与全局共享 | 有专用cache | 与L1类似 | 全GPU只读 | texture引用 |
| Local Memory | 溢出到全局内存 | 与全局相同 | ~400ns(未命中) | 线程私有 | 编译器溢出 |

**内存一致性模型：**

```cuda
// CUDA使用弱内存模型（Relaxed Memory Model）
// 不同线程看到的写入顺序可能不同，除非使用fence

// 1. Block内同步: __syncthreads()
__shared__ float s[256];
s[tid] = compute();
__syncthreads();  // 确保所有线程写入对block内可见
result = s[tid ^ 1];

// 2. 全局内存fence: __threadfence()
data[idx] = value;
__threadfence();  // 确保写入对所有SM可见
flag[idx] = 1;    // 其他block看到flag=1时一定能看到data的新值

// 3. 原子操作: atomicAdd/atomicCAS等
atomicAdd(&global_counter, 1);  // 保证原子性但不保证ordering

// 4. L2缓存管理（Ampere+）
// 可设置L2 cache持久化策略：
cudaAccessPolicyWindow window;
window.base_ptr = persistent_data;
window.num_bytes = size;
window.hitProp = cudaAccessPropertyPersisting;  // 保持在L2中
```

**编程中的最佳实践：**

| 场景 | 推荐做法 | 原因 |
|------|---------|------|
| 多次复用的数据 | 加载到Shared Memory | ~19TB/s vs 2TB/s |
| 线程间通信(warp内) | Warp Shuffle | 无需shared memory |
| 线程间通信(block内) | Shared Memory + syncthreads | 低延迟 |
| 全局同步 | Cooperative Groups / 原子操作 | 跨block |
| 只读数据(广播模式) | 常量内存或L2 persist | 专用cache高效 |
| 寄存器溢出 | 减少每线程变量或降低occupancy | 避免local memory |

---

### Q: 使用Triton实现PagedAttention的思路？

**Triton实现PagedAttention的核心挑战是：在非连续内存布局下高效计算Attention。**

**整体架构：**

```python
@triton.jit
def paged_attention_kernel(
    Q,              # [num_seqs, num_heads, head_dim]  当前step的Q
    K_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]  物理KV存储
    V_cache,        # [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables,   # [num_seqs, max_num_blocks]  逻辑→物理映射
    seq_lens,       # [num_seqs]  每个序列的实际长度
    output,         # [num_seqs, num_heads, head_dim]
    ...
):
```

**实现步骤：**

**Step 1: 加载Q和确定Block范围**

```python
# 每个program instance处理一个(seq, head)对
seq_idx = tl.program_id(0)
head_idx = tl.program_id(1)

# 加载当前序列的Q向量 [head_dim]
q = tl.load(Q + seq_idx * stride_qs + head_idx * stride_qh + 
            tl.arange(0, HEAD_DIM))

# 确定需要处理的KV block数量
seq_len = tl.load(seq_lens + seq_idx)
num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
```

**Step 2: 通过Block Table间接加载KV**

```python
# Online Softmax变量
m_i = float('-inf')   # running max
l_i = 0.0            # running sum of exp
acc = tl.zeros([HEAD_DIM], dtype=tl.float32)  # running output

for block_idx in range(num_blocks):
    # 关键: 通过block_table获取物理block编号
    physical_block_num = tl.load(block_tables + seq_idx * stride_bt + block_idx)
    
    # 计算物理地址并加载K
    k_ptr = K_cache + physical_block_num * stride_kb  # 间接索引!
    k_block = tl.load(k_ptr + offsets...)  # [BLOCK_SIZE, HEAD_DIM]
    
    # 计算attention score
    scores = tl.dot(q[None, :], tl.trans(k_block))  # [1, BLOCK_SIZE]
    
    # 处理padding: 超过seq_len的位置mask为-inf
    valid_mask = (block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < seq_len
    scores = tl.where(valid_mask, scores, float('-inf'))
```

**Step 3: Online Softmax跨Block累积**

```python
    # Online Softmax更新
    m_new = tl.maximum(m_i, tl.max(scores))
    
    # 修正之前的累积值
    alpha = tl.exp(m_i - m_new)
    l_i = l_i * alpha
    acc = acc * alpha
    
    # 当前block的贡献
    p = tl.exp(scores - m_new)
    l_i += tl.sum(p)
    
    # 加载V并累加
    v_block = tl.load(V_cache + physical_block_num * stride_vb + offsets...)
    acc += tl.dot(p, v_block)
    
    m_i = m_new

# 最终归一化
output_val = acc / l_i
tl.store(output + ..., output_val)
```

**性能优化要点：**

| 优化 | 原理 | 实现方式 |
|------|------|---------|
| Block大小选择 | 平衡间接访问开销和并行度 | 通常16-64 tokens/block |
| 预取block_table | 减少间接寻址的延迟 | 循环开始前批量load block numbers |
| GQA支持 | 多个Q head共享一个KV head | kv_head_idx = head_idx // num_q_per_kv |
| 分块并行 | 不同seq和head完全独立 | program_id映射到(seq, head) |
| 向量化加载 | 尽管非连续，block内仍连续 | 每个physical block内用连续load |
| Flash-Decoding | Decode时KV序列长，可按KV分块并行 | 多个program处理同一seq的不同KV块 |

**与vLLM原生CUDA kernel的对比：**

| 维度 | vLLM CUDA kernel | Triton实现 |
|------|-----------------|-----------|
| 开发效率 | 低（需手写CUDA） | 高（Python-like语法） |
| 性能 | 极致优化 | 接近（~90-95%） |
| 可维护性 | 差（数千行CUDA） | 好（数百行Triton） |
| 灵活性 | 修改困难 | 易于实验新策略 |
| GQA/MQA支持 | 需单独kernel | 参数化即可 |
