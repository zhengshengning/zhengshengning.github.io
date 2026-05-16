---
title: '沐曦 AI Infra 实习 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: CUDA 编程的并行性和并发性有什么区别？

这是理解 GPU 执行模型的重要概念区分：

**并行性（Parallelism）——空间维度的"同时"**：

多个执行单元在**同一时刻**执行相同（或类似）操作。在 CUDA 中表现为：
- **Warp 内 SIMT**：同一 warp 的 32 个线程在同一 cycle 执行同一条指令（对不同数据）
- **SM 内多 warp**：一个 SM 可以同时执行多个 warp 的指令（如 A100 每 SM 有 4 个 warp scheduler）
- **跨 SM 并行**：多个 SM 同时执行不同 block（A100 有 108 个 SM）

```
时间 →
SM 0:  [Warp 0 执行 ADD] [Warp 0 执行 MUL] ...
       [Warp 1 执行 ADD] [Warp 1 执行 MUL] ...  ← 空间并行
       [Warp 2 执行 ADD] ...
SM 1:  [Block 2 的 Warp 在同时执行]              ← 跨 SM 并行
```

**并发性（Concurrency）——时间维度的"重叠"**：

多个任务在时间上重叠执行，但不一定在同一时刻。在 CUDA 中表现为：

1. **多 Stream 并发**：
   ```
   Stream A: [Kernel 1 计算]     [Kernel 3 计算]
   Stream B:    [H2D 传输]  [Kernel 2 计算]
   Stream C:                   [D2H 传输]
              ↑ 不同硬件单元同时工作
   ```
   - Copy Engine（DMA）和 Compute Engine（SM）是独立硬件
   - 可以传输和计算同时进行（需要不同 stream + pinned memory）

2. **SM 上多 Block 并发**：
   - 单个 SM 上可能驻留 2-16 个 block（取决于资源）
   - 同一时刻只有部分 warp 在执行，其余在等待（memory access、sync 等）
   - 调度器在 ready warp 间切换 → 并发执行

3. **Graph-level 并发**：
   - CUDA Graph 中无依赖的 kernel 可以并发（如果 SM 资源足够）

**总结对比**：

| 维度 | 并行性 | 并发性 |
|---|---|---|
| 本质 | 多个执行单元同时做 | 多个任务时间上重叠 |
| CUDA 体现 | warp 内 32 线程同步执行 | 多 stream 的 kernel/传输重叠 |
| 硬件需求 | 足够的计算单元 | 独立的硬件功能单元 |
| 目的 | 加速计算（数据并行） | 隐藏延迟、提高利用率 |
| 类比 | 32 人同时搬砖 | 一边搬砖一边等水泥搅拌 |

---

### Q: C++ 的三大特性？

**封装（Encapsulation）**：

将数据（成员变量）和操作数据的方法（成员函数）绑定在类中，通过访问控制隐藏内部实现：

```cpp
class KVCache {
private:
    float* data_;          // 内部存储细节对外隐藏
    size_t capacity_;
    size_t size_;
    
public:
    void append(const float* kv, int num_tokens);  // 只暴露接口
    float* get_block(int block_id);
    size_t available() const { return capacity_ - size_; }
};
```

核心价值：
- 降低耦合（修改内部实现不影响外部使用者）
- 保护数据完整性（外部无法直接修改 private 数据）
- 明确的接口契约（public 方法定义了"能做什么"）

---

**继承（Inheritance）**：

子类继承父类的属性和方法，实现代码复用和层次化设计：

```cpp
class BaseAttention {
public:
    virtual Tensor forward(Tensor Q, Tensor K, Tensor V) = 0;  // 纯虚函数
protected:
    int num_heads_;
    int head_dim_;
};

class FlashAttention : public BaseAttention {
public:
    Tensor forward(Tensor Q, Tensor K, Tensor V) override;  // 具体实现
private:
    int block_size_;  // FlashAttention 特有参数
};

class PagedAttention : public BaseAttention {
public:
    Tensor forward(Tensor Q, Tensor K, Tensor V) override;
private:
    BlockTable block_table_;  // PagedAttention 特有
};
```

继承方式：public（最常用）、protected、private。

---

**多态（Polymorphism）**：

同一接口、不同实现。分为编译时多态和运行时多态：

```cpp
// 运行时多态（虚函数 + 动态绑定）
BaseAttention* attn = create_attention(config);  // 可能返回 Flash 或 Paged
Tensor output = attn->forward(Q, K, V);          // 运行时决定调用哪个实现

// 编译时多态（模板）
template<typename T>
T reduce_sum(T* data, int n);  // T=float, T=half, T=int8_t 各有特化版本

// 编译时多态（函数重载）
void launch_kernel(float* data, int n);    // FP32 版本
void launch_kernel(half* data, int n);     // FP16 版本
void launch_kernel(int8_t* data, int n);   // INT8 版本
```

**虚函数表（vtable）机制**：
- 每个有虚函数的类维护一个虚函数表（函数指针数组）
- 每个对象有一个 vptr 指向所属类的 vtable
- 调用虚函数时：`obj->vptr->vtable[func_index](args)`
- 开销：一次额外的间接寻址（~几 ns），在 GPU kernel 中应避免

---

### Q: map 和 unordered_map 的底层实现区别？

| 维度 | std::map | std::unordered_map |
|---|---|---|
| **底层结构** | 红黑树（自平衡 BST） | 哈希表（开链法：bucket 数组 + 链表/红黑树） |
| **有序性** | key 有序（按 `operator<` 排列） | 无序（按 hash 值分桶） |
| **查找** | O(log n) 保证 | 平均 O(1)，最坏 O(n)（hash 冲突严重时） |
| **插入** | O(log n) + 可能旋转 | 平均 O(1)，最坏 O(n)（rehash 时 O(n)） |
| **删除** | O(log n) | 平均 O(1) |
| **内存布局** | 节点分散分配（指针连接） | 桶数组 + 节点链表 |
| **迭代器稳定性** | 插入/删除不影响其他迭代器 | rehash 会使所有迭代器失效 |
| **空间开销** | 每节点 3 指针 + color | 桶数组 + 每节点 1-2 指针 |
| **Cache 友好度** | 差（节点分散在堆上） | 中（桶数组连续，链表仍分散） |

**选择指南**：
```
需要有序遍历（如 range query）        → std::map
纯查找性能优先（如缓存/hash table）    → std::unordered_map
key 类型没有好的 hash 函数             → std::map（只需 operator<）
关注最坏情况性能                       → std::map（保证 O(log n)）
大数据量 + 频繁查找                    → std::unordered_map（常数时间）
```

**unordered_map 的 rehash**：
- 当 load_factor = size / bucket_count > max_load_factor（默认 1.0）时触发
- Rehash：分配更大的桶数组（通常 2x），所有元素重新 hash 分配
- 时间复杂度 O(n)，但均摊后仍是 O(1)

---

### Q: new、malloc 和智能指针的区别？底层原理？

**三者对比**：

| 维度 | malloc | new | 智能指针 |
|---|---|---|---|
| 语言 | C | C++ | C++ (C++11+) |
| 分配 | 原始字节 | 对象（分配+构造） | 对象 + 自动管理 |
| 释放 | free() | delete | 自动（析构时） |
| 失败行为 | 返回 NULL | 抛 std::bad_alloc | 同 new |
| 类型安全 | void*（需强转） | 返回类型化指针 | 类型化包装 |

**malloc 底层原理**：
```
用户调用 malloc(size):
  
  小块 (< 128KB, glibc):
    → ptmalloc 从 arena 的 free bin 中查找合适块
    → 找不到: brk() 系统调用扩展堆（增量分配）
    → Bins 分类: fast bins (< 80B), small bins, large bins, unsorted bin
  
  大块 (≥ 128KB):
    → mmap() 系统调用直接从 OS 映射虚拟内存
    → 释放时 munmap() 归还 OS
    
内存组织:
  [chunk header (8-16 bytes)] [user data ...] [padding]
  header 包含: size + flags (prev_inuse, is_mmapped, non_main_arena)
```

**new 的执行流程**：
```cpp
MyClass* p = new MyClass(args);
// 等价于:
void* mem = operator new(sizeof(MyClass));  // 1. 分配内存（底层调 malloc）
MyClass* p = static_cast<MyClass*>(mem);
p->MyClass(args);                            // 2. 在分配的内存上调用构造函数
```

**智能指针详解**：

```cpp
// unique_ptr: 独占所有权，零开销
std::unique_ptr<float[]> buffer(new float[1024]);
// 底层: 仅包含一个裸指针，sizeof(unique_ptr) == sizeof(void*)
// 析构时自动 delete，不可拷贝，可移动

// shared_ptr: 共享所有权，引用计数
std::shared_ptr<Model> model = std::make_shared<Model>();
// 底层:
//   对象指针 → [Model 数据]
//   控制块指针 → [strong_count | weak_count | deleter | allocator]
// sizeof(shared_ptr) == 2 × sizeof(void*)

// weak_ptr: 弱引用，不增加 strong_count
std::weak_ptr<Model> cache = model;
// 用于打破循环引用 + 观察者模式
// lock() → shared_ptr 或 nullptr (对象已销毁时)
```

**make_shared 的优势**：
```cpp
// 方式 1: 两次分配
auto p = std::shared_ptr<T>(new T());  // 分配 T + 分配控制块（两次 new）

// 方式 2: 一次分配（推荐）
auto p = std::make_shared<T>();  // T 和控制块在连续内存中（一次 new）
// 优势: 减少分配次数 + 更好的 cache locality
```

---

### Q: 介绍一下 C++ Lambda 表达式？

Lambda 是 C++11 引入的匿名函数对象，本质是编译器生成的匿名类的实例。

**完整语法**：
```cpp
[capture](parameters) mutable -> return_type { body }
```

**捕获方式详解**：
```cpp
int x = 10;
float y = 3.14;
std::vector<int> data = {1, 2, 3};

auto f1 = [x]() { return x; };          // 值捕获 x（拷贝一份）
auto f2 = [&x]() { x++; };              // 引用捕获 x（可修改原始 x）
auto f3 = [=]() { return x + y; };      // 值捕获所有用到的变量
auto f4 = [&]() { x++; y += 1.0; };     // 引用捕获所有
auto f5 = [&x, y]() { x += y; };        // x 引用，y 值（混合）
auto f6 = [data = std::move(data)]() {}; // C++14: 移动捕获（初始化捕获）
```

**Lambda 的编译器实现**：
```cpp
// 源码:
auto lambda = [x, &y](int a) -> int { return x + y + a; };

// 编译器生成的等价代码:
class __anonymous_lambda_42 {
private:
    int x_;         // 值捕获 → 成员变量
    float& y_;      // 引用捕获 → 引用成员
public:
    __anonymous_lambda_42(int x, float& y) : x_(x), y_(y) {}
    int operator()(int a) const { return x_ + y_ + a; }
};
auto lambda = __anonymous_lambda_42(x, y);
```

**mutable 关键字**：
```cpp
int count = 0;
auto counter = [count]() mutable { return ++count; };
// 默认 operator() 是 const，不能修改值捕获的成员
// mutable 去掉 const 限定，允许修改副本（不影响原始变量）
```

**常见用途**：
```cpp
// 1. STL 算法
std::sort(vec.begin(), vec.end(), [](int a, int b) { return a > b; });

// 2. CUDA kernel 的 host-side 配置
auto launch_config = [block_size, grid_size](auto kernel) {
    kernel<<<grid_size, block_size>>>();
};

// 3. 异步回调
auto future = std::async([&model]() { return model.forward(input); });

// 4. 条件过滤
auto outliers = std::count_if(values.begin(), values.end(),
    [threshold](float v) { return std::abs(v) > threshold; });
```

**注意事项**：
- **悬垂引用**：引用捕获时，如果 lambda 的生命周期超过被捕获变量 → undefined behavior
- **大对象捕获**：值捕获大对象（如 vector）会拷贝 → 用 `[v = std::move(v)]` 或引用
- **this 捕获**：`[this]` 捕获 this 指针，注意对象生命周期

---

### Q: 手撕 Transformer 结构（PyTorch）？

（编程题）

---

### Q: 手撕：两数之和？

（编程题）
