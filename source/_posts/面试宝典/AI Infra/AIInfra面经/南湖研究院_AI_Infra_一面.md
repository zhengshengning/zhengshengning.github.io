---
title: '南湖研究院 AI Infra 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: Python 的垃圾回收算法？会循环引用吗？

Python 使用**三层垃圾回收机制**协同工作：

**第一层：引用计数（Reference Counting）——主要机制**：
```python
# 每个 PyObject 头部有一个 ob_refcnt 字段
import sys

a = [1, 2, 3]        # refcnt([1,2,3]) = 1
b = a                 # refcnt([1,2,3]) = 2
sys.getrefcount(a)    # 返回 3（函数参数也临时 +1）

del b                 # refcnt -= 1 → 1
del a                 # refcnt -= 1 → 0 → 立即释放!
```

优点：实时释放（引用归零即回收）、确定性析构。
缺点：无法处理循环引用、频繁修改 refcnt 有开销。

**第二层：标记清除（Mark-Sweep）——解决循环引用**：
```python
# 循环引用示例
class Node:
    def __init__(self):
        self.ref = None

a = Node()
b = Node()
a.ref = b    # a → b
b.ref = a    # b → a（循环！）

del a, b     # refcnt(a) = 1 (被 b.ref 引用), refcnt(b) = 1 (被 a.ref 引用)
             # 引用计数永远不为 0！但这两个对象已不可达

# 标记清除算法:
# 1. 从 root 集合（全局变量、栈上变量）出发
# 2. 标记所有可达对象
# 3. 未被标记的对象 = 不可达 = 可回收（即使 refcnt > 0）
```

**只对容器类型做标记清除**（list, dict, tuple, class instance 等），因为只有容器能引用其他对象形成循环。int, str, float 等不可能循环引用。

**第三层：分代回收（Generational GC）**：
```
Generation 0: 新创建的对象（GC 最频繁）
  ↓ 存活过一次 GC
Generation 1: 存活较久的对象（GC 频率中等）
  ↓ 存活过一次 Gen1 GC
Generation 2: 长期存活的对象（GC 频率最低）

触发条件: 当某代新分配对象数超过阈值 (默认 700/10/10)
```

**分代的直觉**：大部分对象生命周期短（临时变量），只需要频繁检查 Gen0。长期存活的对象（如模块级变量）很少成为垃圾，不需要频繁检查。

**避免循环引用的方法**：
```python
import weakref

class Cache:
    def __init__(self, model):
        self.model = weakref.ref(model)  # 弱引用，不增加 refcnt
        
# 或使用 weakref.WeakValueDictionary
cache = weakref.WeakValueDictionary()
```

---

### Q: 内联（inline）什么时候会影响性能？

内联将函数体直接嵌入调用点，消除函数调用开销（压栈/跳转/返回）。但**过度内联会导致性能下降**：

**内联的正面效果**：
- 消除 call/ret 指令（~3-5 cycles）
- 允许编译器跨函数边界优化（常量折叠、死代码消除）
- 消除函数调用的寄存器保存/恢复

**内联的负面效果**：

| 场景 | 问题 | 机制 |
|---|---|---|
| **大函数体** | 代码膨胀 (code bloat) | 每个调用点展开一份副本 → binary 体积暴增 |
| **热路径上的大函数** | I-Cache 压力 | 展开后指令总量超过 L1 I-Cache（32-64 KB）→ cache miss |
| **递归函数** | 无法完全内联 | 编译器只能展开有限层（通常 1-2 层） |
| **虚函数** | 通常无法内联 | 运行时才确定目标（除非编译器做去虚拟化） |
| **循环内大函数** | 循环体膨胀 | 展开后循环体不适合 loop unrolling/vectorization |

**代码膨胀的连锁反应**：
```
函数 A (100 条指令) 被 50 个调用点内联:
  不内联: 100 指令 + 50×5 指令(call/ret) = 350 条指令
  内联:   50×100 = 5000 条指令

L1 I-Cache 32 KB, 每条指令 ~4 bytes → 容纳 ~8000 条指令
  不内联: 350/8000 = 4.4%，绰绰有余
  内联:   5000/8000 = 62.5%，可能导致其他代码被挤出缓存

结果: 虽然每次调用少了 5 cycles，但 I-Cache miss 增加了 50-100 cycles/miss
```

**在 CUDA/GPU 编程中的特殊考量**：
- GPU kernel 中的函数调用开销更大（没有硬件 call stack 的情况下需要保存恢复寄存器）
- 但 GPU 寄存器压力大，内联可能增加寄存器使用 → occupancy 下降
- 编译器（nvcc）通常会自动内联小函数（`__device__` 函数默认内联）
- 可用 `__forceinline__` 或 `__noinline__` 显式控制

---

### Q: inline 对作用域的影响？除了 inline 还有什么会内联？

**inline 对作用域/链接的影响**：

C++ 的 One Definition Rule（ODR）要求每个函数在整个程序中只能有一个定义。但 `inline` 函数获得 ODR 豁免：

```cpp
// header.h
inline int compute(int x) { return x * 2; }  // OK: inline 允许在多个 TU 中定义

// 不加 inline:
int compute(int x) { return x * 2; }  // 错误: 多个 .cpp include 会导致多重定义
```

`inline` 的语义：
- 告诉链接器：如果多个编译单元（.o 文件）中有同名 inline 函数，它们是同一个函数，合并为一份
- **不保证内联优化**：编译器可能忽略 inline 提示（函数太大时）
- 反过来，**不加 inline 也可能被内联**（编译器自动判断）

---

**其他会导致内联的情况**：

| 情况 | 说明 | 是否需要显式 inline |
|---|---|---|
| **类内定义的成员函数** | 隐式 inline | 不需要 |
| **模板函数** | 通常在头文件定义，编译器倾向内联 | 不需要（template 有 ODR 豁免） |
| **constexpr 函数** | 编译期求值时必然"内联" | 不需要（C++17 隐式 inline） |
| **编译器自动内联（-O2/-O3）** | 小函数自动内联 | 不需要 |
| **LTO（Link Time Optimization）** | 跨编译单元内联 | 不需要 |
| **__attribute__((always_inline))** | GCC/Clang 强制内联 | 显式标记 |
| **__forceinline__** | MSVC / CUDA 强制内联 | 显式标记 |

**编译器自动内联的判断标准**：
- 函数体指令数少（通常 < 10-30 条 IR 指令）
- 调用频次高（热路径上的函数更倾向内联）
- 调用开销占比大（函数体短时 call/ret 开销占比高）
- `-O2` 比 `-O1` 更激进内联，`-O3` 和 `-Ofast` 最激进
- `-Os`（优化大小）会抑制内联

---

### Q: 完美转发（Perfect Forwarding）？

完美转发解决的问题：如何将函数参数**按原始类型（左值/右值）**透传给另一个函数，不丢失值类别信息。

**问题场景**：
```cpp
// 想实现一个通用的工厂函数
template<typename T, typename Arg>
T* create(Arg arg) {
    return new T(arg);  // 问题: arg 总是左值! 即使调用者传入右值
}

// 调用者传入右值:
create<string>(string("hello"));  // "hello" 是右值，但 arg 是左值
// → 内部调用 string(左值)，触发拷贝构造而非移动构造!
```

**解决方案——万能引用 + std::forward**：
```cpp
template<typename T, typename... Args>
T* create(Args&&... args) {                     // 万能引用
    return new T(std::forward<Args>(args)...);  // 完美转发
}
```

**引用折叠规则（Reference Collapsing）**：
```
T& &   → T&   (左值引用的引用 → 左值引用)
T& &&  → T&   (左值引用的右值引用 → 左值引用)
T&& &  → T&   (右值引用的引用 → 左值引用)
T&& && → T&&  (右值引用的右值引用 → 右值引用)

规则: 只要有一个 & 出现，结果就是 &（左值引用"传染"）
```

**模板参数推导 + 引用折叠的协作**：
```cpp
template<typename T>
void wrapper(T&& arg) { ... }

// 传入左值:
int x = 42;
wrapper(x);      // T 推导为 int&，参数类型 = int& && = int& (左值引用)

// 传入右值:
wrapper(42);     // T 推导为 int，参数类型 = int&& (右值引用)
```

**std::forward 的实现**：
```cpp
template<typename T>
T&& forward(std::remove_reference_t<T>& arg) {
    return static_cast<T&&>(arg);
}

// 当 T = int&:  返回 int& &&  = int& (保持左值)
// 当 T = int:   返回 int&&       (保持右值)
```

**实际应用场景**：
```cpp
// 1. emplace_back（避免额外拷贝/移动）
template<typename... Args>
void emplace_back(Args&&... args) {
    new (end_ptr) T(std::forward<Args>(args)...);  // 原地构造
}

// 2. make_unique / make_shared
template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// 3. 通用 wrapper（如 timing 包装器）
template<typename F, typename... Args>
auto timed_call(F&& f, Args&&... args) {
    auto start = clock::now();
    auto result = std::forward<F>(f)(std::forward<Args>(args)...);
    auto elapsed = clock::now() - start;
    return result;
}
```

---

### Q: 右值引用的意义？

右值引用（`T&&`）是 C++11 引入的核心特性，解决了**不必要的深拷贝**问题。

**核心价值——移动语义**：
```cpp
// 场景: vector 返回大数据
std::vector<float> compute_output(int n) {
    std::vector<float> result(n);
    // ... 计算 ...
    return result;  // C++11 之前: 拷贝构造 (O(n))
                    // C++11 之后: 移动构造 (O(1))
}

// 移动构造 vs 拷贝构造
class Tensor {
    float* data_;
    size_t size_;
public:
    // 拷贝构造: O(n) — 分配新内存 + 复制所有元素
    Tensor(const Tensor& other) : size_(other.size_) {
        data_ = new float[size_];
        memcpy(data_, other.data_, size_ * sizeof(float));
    }
    
    // 移动构造: O(1) — 直接"偷走"资源
    Tensor(Tensor&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;  // 源对象置空（转移所有权）
        other.size_ = 0;
    }
};
```

**三个核心应用场景**：

1. **容器操作加速**：
   ```cpp
   std::vector<Tensor> tensors;
   tensors.push_back(Tensor(1000000));  // 临时对象是右值 → 移动（O(1)）
   
   // vector 扩容时移动所有元素（如果 move ctor 是 noexcept）
   // 1000 个 Tensor 扩容: 拷贝 O(n×size) vs 移动 O(n)
   ```

2. **资源转移（unique_ptr）**：
   ```cpp
   std::unique_ptr<Model> model = load_model("path");
   std::unique_ptr<Model> server_model = std::move(model);
   // 所有权从 model 转移到 server_model，model 变为空
   // 不需要拷贝整个 Model 对象
   ```

3. **完美转发**（配合万能引用 T&&）：
   ```cpp
   template<typename... Args>
   auto make_kernel(Args&&... args) {
       return Kernel(std::forward<Args>(args)...);
   }
   ```

**std::move 的本质**：
```cpp
// std::move 不移动任何东西！它只是将左值转换为右值引用（类型转换）
template<typename T>
std::remove_reference_t<T>&& move(T&& arg) noexcept {
    return static_cast<std::remove_reference_t<T>&&>(arg);
}

// 真正的移动发生在移动构造函数/移动赋值运算符中
Tensor a(1000);
Tensor b = std::move(a);  // std::move 将 a 转为右值 → 调用移动构造
// 此后 a 处于 valid-but-unspecified 状态（通常为空）
```

**noexcept 的重要性**：
- STL 容器扩容时，如果移动构造是 noexcept，则使用移动
- 如果移动构造可能抛异常，STL 退回使用拷贝（保证异常安全）
- 因此移动构造函数务必加 `noexcept`

---

### Q: FasterTransformer（FT）框架的特点？

FasterTransformer 是 NVIDIA 推出的高性能 Transformer 推理库（现已演进为 TensorRT-LLM），代表了 CUDA 原生 LLM 推理的极致优化水平。

**核心特点**：

| 维度 | 特点 | 详情 |
|---|---|---|
| **语言** | 纯 C++/CUDA | 无 Python 开销，适合生产部署 |
| **算子优化** | 极致手写 CUDA kernel | 融合 MHA、融合 LayerNorm+Add、融合 GEMM+Bias+Act |
| **并行支持** | TP + PP | 多卡推理，NVLink/NCCL 通信 |
| **量化** | INT8/FP8 | 借助 Tensor Core 加速 |
| **模型支持** | GPT/BERT/T5/LLaMA/Bloom 等 | 主流架构预实现 |
| **内存管理** | 手动 memory pool | 预分配+复用，减少运行时分配 |

**FT 的优化手段（后续被 TensorRT-LLM 继承）**：

```
1. 算子融合:
   - Attention: QKV GEMM 融合 + Softmax + V×Attn 融合
   - FFN: Linear+Bias+Activation 融合
   - Post-process: LayerNorm+Residual+Bias 融合为单 kernel

2. 内存优化:
   - KV Cache 预分配 + 复用
   - 激活值 scratch buffer 复用
   - Weight pre-packing（适合 Tensor Core 的 layout）

3. 并行策略:
   - TP: column/row parallel linear（与 Megatron 方式一致）
   - PP: 多 stage 流水线
   - 通信与计算重叠
```

**FT → TensorRT-LLM 的演进**：

| 维度 | FasterTransformer | TensorRT-LLM |
|---|---|---|
| 接口 | C++ API | Python + C++ plugin |
| 图优化 | 手动 | 基于 TensorRT 编译器 |
| 动态 batch | 基础支持 | In-flight Batching (Continuous Batching) |
| 量化 | INT8 | INT8/INT4/FP8 + AWQ/GPTQ/SmoothQuant |
| KV Cache | 连续分配 | 类 PagedAttention 分页管理 |
| 投机解码 | 无 | 内置支持 |
| 多模态 | 有限 | 支持 VLM |
| 维护 | 已停止 | 积极维护 |

**为什么 FT 被替代**：
- C++ 纯手动实现新模型开发周期长（每支持一个新模型需要数周）
- 缺乏动态 batch/PagedAttention 等现代推理框架特性
- TensorRT-LLM 继承了 FT 的所有优化 + 添加了编译器自动化 + Python 易用性

---

### Q: 线程同步的方式？

**CPU 线程同步机制**：

| 方式 | 原理 | 适用场景 | 开销 |
|---|---|---|---|
| **Mutex（互斥锁）** | 同时只有一个线程进入临界区 | 通用互斥 | 无竞争 ~25ns，竞争 ~μs |
| **Spinlock（自旋锁）** | 忙等待（循环 CAS） | 临界区极短（< 1μs） | CPU 占用高 |
| **RWLock（读写锁）** | 多读单写 | 读多写少 | 略高于 mutex |
| **Condition Variable** | 等待条件 + 通知唤醒 | 生产者-消费者 | park/unpark ~μs |
| **Semaphore（信号量）** | 计数型，允许 N 个线程同时访问 | 限流/资源池 | 类似 mutex |
| **Barrier（屏障）** | 所有线程到达后才继续 | 分阶段并行计算 | 取决于最慢线程 |
| **Atomic（原子操作）** | 无锁的 CAS/fetch_add | 计数器、标志位 | ~1-10ns |

**选择决策树**：
```
需要互斥?
  ├─ 是 + 临界区 < 100ns → Spinlock
  ├─ 是 + 临界区较长 → Mutex
  └─ 否 → Atomic（如果操作足够简单）

需要等待条件?
  └─ Condition Variable + Mutex 组合

读多写少?
  └─ RWLock (shared_mutex)

多线程到齐后一起前进?
  └─ Barrier (std::barrier C++20)
```

---

**CUDA 中的同步机制**：

| 级别 | 方式 | 作用域 | 使用场景 |
|---|---|---|---|
| **Warp 内** | 隐式同步（SIMT lockstep） | 32 线程 | 同一 warp 天然同步 |
| **Warp 内** | `__syncwarp(mask)` | 指定线程 | divergent path 后重新同步 |
| **Block 内** | `__syncthreads()` | 全 block | shared memory 读写同步 |
| **Grid 内** | `cooperative_groups` | 全 grid | persistent kernel 通信 |
| **Device 级** | `cudaDeviceSynchronize()` | host 等待 device | 确保所有 kernel 完成 |
| **Stream 级** | `cudaStreamSynchronize()` | 指定 stream | 等待特定 stream 完成 |
| **Event** | `cudaEventRecord/Wait` | 跨 stream | stream 间依赖 |

**`__syncthreads()` 使用规则**：
```cuda
// 正确: 所有线程都到达同一个 syncthreads
__shared__ float s[256];
s[tid] = global_data[idx];
__syncthreads();  // 确保所有线程写完
float val = s[255 - tid];  // 安全读取其他线程写入的数据

// 错误: 条件分支中 sync（可能死锁）
if (tid < 128) {
    __syncthreads();  // 只有一半线程到达 → 死锁!
}
```

---

### Q: CUDA Stream 有什么要求？如何实现设备级重叠？

**CUDA Stream 基本概念**：

Stream 是 GPU 上的有序命令队列。同一 stream 内的操作按顺序执行，不同 stream 间的操作可以并发。

**核心要求**：

1. **异步操作需指定 stream**：
   ```cuda
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   
   // 指定 stream 的异步操作
   cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
   kernel<<<grid, block, 0, stream2>>>(d_b);
   ```

2. **异步传输需要 Pinned Memory**：
   ```cuda
   // 只有 pinned memory 才能真正异步传输
   float* h_data;
   cudaHostAlloc(&h_data, size, cudaHostAllocDefault);  // pinned
   cudaMemcpyAsync(d_data, h_data, size, H2D, stream);  // 真正异步
   
   // 普通 malloc 的内存: cudaMemcpyAsync 实际会退化为同步
   // 因为 DMA 引擎需要物理地址（pinned memory 锁页不换出）
   ```

3. **Default Stream (stream 0) 的隐式同步**：
   - 如果不指定 stream，操作在 default stream 执行
   - Default stream 会隐式与所有其他 stream 同步（blocking）
   - 使用 `--default-stream per-thread` 编译可避免

---

**设备级重叠（计算与传输 Overlap）**：

GPU 有独立的硬件引擎：
```
硬件引擎:
  - Copy Engine (H2D): 负责 Host→Device 传输
  - Copy Engine (D2H): 负责 Device→Host 传输
  - Compute Engine (SM): 负责 kernel 计算
  
三者可以同时工作！
```

**经典三级流水线**：
```cuda
// 将数据分为 N 份，用 3 个 stream 流水线处理
for (int i = 0; i < N; i++) {
    // Stream i%3 中的三个阶段可以与其他 stream 重叠
    cudaMemcpyAsync(d_in + i*chunk, h_in + i*chunk, chunk_size, H2D, stream[0]);
    kernel<<<grid, block, 0, stream[1]>>>(d_in + (i-1)*chunk, d_out + (i-1)*chunk);
    cudaMemcpyAsync(h_out + (i-2)*chunk, d_out + (i-2)*chunk, chunk_size, D2H, stream[2]);
}

// 时间线 (理想情况):
// Stream 0: [H2D_0] [H2D_1] [H2D_2] ...
// Stream 1:    [Compute_0] [Compute_1] [Compute_2] ...
// Stream 2:       [D2H_0] [D2H_1] [D2H_2] ...
//           ↑ 三者在时间上重叠，总时间 ≈ max(H2D, Compute, D2H) × N
```

**依赖管理——CUDA Event**：
```cuda
cudaEvent_t event;
cudaEventCreate(&event);

// Stream 1 在 stream 0 的 H2D 完成后才开始计算
cudaMemcpyAsync(..., stream0);
cudaEventRecord(event, stream0);       // 在 stream0 中记录事件
cudaStreamWaitEvent(stream1, event);   // stream1 等待该事件
kernel<<<..., stream1>>>(...);         // 确保 H2D 完成后才计算
```

**实际收益**（以 PCIe 4.0 + A100 为例）：
```
不重叠:
  H2D (1 GB): 32 ms + Compute: 50 ms + D2H (1 GB): 32 ms = 114 ms

重叠:
  总时间 ≈ max(32, 50, 32) + startup = ~55 ms (加速 ~2x)
```

---

### Q: Git 协作流程？

**标准 Feature Branch 工作流**：

```
main ─────●────────●──────────●──────────── (始终可部署)
           \      ↑ PR merge    \
            \    /                \
  feature/x  ●─●─●──── Code Review ──→ squash merge
                                    \
  feature/y                          ●─●─● ...
```

**完整流程**：
```bash
# 1. 创建 feature 分支
git checkout main && git pull
git checkout -b feature/flash-attention-v3

# 2. 开发 + 提交（原子化 commit）
git add src/attention.cu
git commit -m "feat: implement FlashAttention v3 forward pass"
git add tests/test_attention.py
git commit -m "test: add FlashAttention v3 correctness tests"

# 3. 推送到远程
git push -u origin feature/flash-attention-v3

# 4. 创建 Pull Request → Code Review
#    - CI 自动运行测试
#    - Reviewer 提出修改建议
#    - 修改后 push → CI 重跑

# 5. 合并（三种方式）
#    - Squash Merge: 压缩为一个 commit（保持主线干净）
#    - Rebase Merge: 线性历史（无 merge commit）
#    - Merge Commit: 保留分支历史（有 merge commit）

# 6. 清理
git checkout main && git pull
git branch -d feature/flash-attention-v3
```

**冲突解决**：
```bash
# 方式 1: Rebase（推荐，线性历史）
git fetch origin
git rebase origin/main
# 解决冲突 → git add → git rebase --continue

# 方式 2: Merge
git merge origin/main
# 解决冲突 → git add → git commit
```

**Commit Message 规范（Conventional Commits）**：
```
feat: 新功能
fix: 修复 bug
perf: 性能优化
refactor: 重构（不改变功能）
test: 添加/修改测试
docs: 文档
chore: 构建/工具链
```

---

### Q: 手撕：智能指针（shared_ptr）？

（编程题）

---

### Q: 手撕：前 K 大的元素？

（编程题）
