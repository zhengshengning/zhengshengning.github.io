---
title: '网易 AI Infra 校招'
date: 2026-04-16 12:08:36
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 设计一个高吞吐、低延迟的模型推理服务时，重点考虑哪些架构层面和工程层面的问题？

**系统架构全景：**

```
客户端请求
    ↓ (HTTP/gRPC, 流式SSE)
┌──────────────────────────────────────┐
│ API Gateway / Load Balancer          │
│  - 限流、认证、路由                    │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Scheduler (调度器)                    │
│  ├── 请求队列(优先级+FCFS)           │
│  ├── Batch组装(continuous batching)  │
│  ├── 显存预算管理                     │
│  └── 抢占/降级策略                    │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ Inference Engine (执行引擎)           │
│  ├── Model: 量化模型(FP8/INT8)       │
│  ├── Attention: FlashAttention       │
│  ├── KV-Cache: PagedAttention        │
│  └── CUDA Graph (Decode加速)         │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│ GPU Cluster (硬件层)                  │
│  ├── TP: 节点内NVLink多卡            │
│  ├── PP: 跨节点流水线                 │
│  └── EP: MoE专家并行                 │
└──────────────────────────────────────┘
```

**架构层面的关键决策：**

| 决策点 | 选项 | 考虑因素 |
|--------|------|---------|
| 模型优化 | FP8/INT8/INT4量化 | Decode阶段INT4最快(memory-bound), Prefill阶段FP8平衡 |
| KV-Cache管理 | PagedAttention vs 连续分配 | 长序列必须分页，短固定长度可连续 |
| 调度模式 | Continuous Batching | 吞吐提升3-5x，几乎无理由不用 |
| PD分离 | Prefill和Decode分开 | Prefill(compute)和Decode(memory)资源需求不同 |
| 并行策略 | TP2/TP4/TP8 | 延迟敏感选大TP，吞吐优先选小TP+多实例 |
| 冗余/容错 | 多副本+健康检查 | 单点故障时自动切换 |

**工程层面的关键问题：**

**1. 调度器设计：**

```python
# 调度决策的核心逻辑(每iteration执行一次):
def schedule_step():
    # 1. 检查running请求是否需要新KV block
    for req in running:
        if req.needs_new_block() and not has_free_blocks():
            preempt(req)  # 抢占: swap to CPU or recompute
    
    # 2. 尝试加入新请求
    while waiting and has_resources_for_prefill():
        req = waiting.pop()
        allocate_kv_blocks(req)
        running.add(req)
    
    # 3. 尝试恢复被抢占的请求
    while swapped and has_free_blocks():
        req = swapped.pop()
        swap_in(req)
        running.add(req)
```

**2. 延迟优化：**

| 延迟类型 | 目标 | 优化手段 |
|---------|------|---------|
| TTFT(首token延迟) | <500ms | Chunked Prefill避免长序列阻塞 |
| TPOT(每token延迟) | <30ms | CUDA Graph消除launch开销 |
| P99延迟 | <2x P50 | 抢占长请求保护短请求 |
| 端到端延迟 | <1s(短回复) | 异步流式返回 |

**3. 吞吐优化：**

```
吞吐 = batch_size × tokens_per_second / avg_output_length

提高batch_size:
  - PagedAttention: 显存利用率接近100%
  - 量化: INT4权重减少4x，INT8 KV-Cache减少2x
  - 长序列offload: 超长KV swap到CPU

提高tokens_per_second:
  - FlashAttention: 减少HBM IO
  - Tensor Core: FP8/INT8矩阵乘
  - CUDA Graph: 消除CPU开销

减少avg_output_length:
  - 投机解码: 一次验证多个token
```

**4. 监控和SLA保障：**

| 监控指标 | 正常范围 | 告警阈值 | 含义 |
|---------|---------|---------|------|
| GPU利用率 | >80% | <60% | 调度不满或通信瓶颈 |
| Queue Depth | <100 | >500 | 请求堆积需要扩容 |
| P99延迟 | <200ms/token | >500ms/token | 性能退化 |
| KV-Cache利用率 | 70-90% | >95% | 即将触发抢占 |
| Throughput(tokens/s) | 稳定 | 下降>20% | 可能硬件故障 |

### Q: C++程序内存中栈、堆和静态/全局存储区的特点与主要区别？

**进程内存布局（从低地址到高地址）：**

```
┌──────────────────────────────┐ 高地址
│         内核空间              │
├──────────────────────────────┤ 0x7FFF...
│    栈 (Stack) ↓              │ 向低地址增长, 默认8MB
│    ↓↓↓↓↓↓↓↓↓↓↓              │
│                              │
│          (空闲)              │
│                              │
│    ↑↑↑↑↑↑↑↑↑↑↑              │
│    堆 (Heap) ↑               │ 向高地址增长, brk/mmap
├──────────────────────────────┤
│    BSS段 (未初始化全局变量)    │ 程序加载时清零
├──────────────────────────────┤
│    Data段 (已初始化全局变量)   │ 从可执行文件加载
├──────────────────────────────┤
│    Text段 (代码)             │ 只读+可执行
└──────────────────────────────┘ 低地址 0x400000
```

**三个区域的详细对比：**

| 维度 | 栈(Stack) | 堆(Heap) | 静态/全局区 |
|------|-----------|----------|------------|
| 管理方式 | 编译器自动(RAII) | 程序员手动(new/delete) | 系统管理 |
| 分配速度 | 极快(~1ns, 移动rsp) | 慢(~100ns, 空闲链表搜索) | 程序加载时一次分配 |
| 大小限制 | 小(默认8MB, ulimit -s) | 大(受虚拟地址空间限制) | 编译时确定 |
| 生命周期 | 随函数调用/返回 | 手动控制(直到delete) | 程序全程 |
| 碎片 | 无(LIFO连续分配) | 有(频繁alloc/free) | 无 |
| 线程安全 | 每线程独立栈 | 需要锁(或线程局部分配器) | 需要同步 |
| Cache友好 | 极好(时空局部性) | 较差(分散) | 取决于访问模式 |

**实际编程中的考虑：**

```cpp
void example() {
    // 栈分配: 局部变量, 小对象
    int arr[1024];           // 4KB在栈上, OK
    // int big[1000000];     // 4MB在栈上, 可能栈溢出!
    
    // 堆分配: 大对象, 生命周期不确定的对象
    auto* big = new int[1000000];  // 堆上, 安全
    // 问题: 忘记delete → 内存泄漏
    
    // 更好的做法: 智能指针
    auto big_ptr = std::make_unique<int[]>(1000000);
    // 自动释放, 无泄漏
    
    // 静态/全局区
    static int call_count = 0;  // 只初始化一次, 函数间保持
    call_count++;
}

// 全局区: 全局变量
int g_config = 42;         // .data段(已初始化)
int g_buffer[4096];        // .bss段(未初始化, 清零)
```

**性能影响量化：**

| 操作 | 耗时 | 原因 |
|------|------|------|
| 栈分配(局部变量) | ~0 ns | 编译器只调整rsp |
| 堆分配(小对象<256B) | ~50-100 ns | tcmalloc/jemalloc线程局部缓存 |
| 堆分配(大对象>128KB) | ~1-10 us | mmap系统调用 |
| 系统调用(brk/mmap) | ~500 ns-2 us | 用户态→内核态切换 |

### Q: C++中new/delete与malloc/free的主要区别？

| 维度 | new/delete | malloc/free |
|------|-----------|-------------|
| 类型安全 | 返回具体类型指针 | 返回void*需强转 |
| 构造/析构 | 调用构造函数/析构函数 | 只分配/释放原始内存 |
| 错误处理 | 抛std::bad_alloc异常 | 返回NULL |
| 大小计算 | 自动计算(sizeof) | 手动指定字节数 |
| 可重载 | operator new可重载 | 不可重载 |
| 数组版本 | new[]/delete[] | 需手动计算size*count |
| 对齐 | C++17 aligned new | 需要memalign/posix_memalign |

**关键细节：**

```cpp
// new的底层实现(简化):
T* operator new(size_t size) {
    void* p = malloc(size);      // 1. 分配内存
    if (!p) throw std::bad_alloc();
    return static_cast<T*>(p);
}
// 然后在分配的内存上调用构造函数: new(p) T(args...)

// delete的底层实现:
void operator delete(void* p) {
    p->~T();    // 1. 调用析构函数
    free(p);    // 2. 释放内存
}

// 为什么new[]和delete[]必须配对?
auto* arr = new int[100];
// 编译器在数组前额外存储元素个数: [100][int0][int1]...[int99]
delete[] arr;  // 根据存储的100知道要调用100次析构函数
// delete arr;  // UB! 只析构第一个元素, 且free地址错误
```

**现代C++的推荐：**
- 避免裸new/delete → 使用`std::make_unique`/`std::make_shared`
- 需要原始内存 → `std::aligned_alloc`(C++17)
- 容器管理 → `std::vector`（RAII自动管理堆内存）
- 自定义分配器 → `std::pmr::memory_resource`(C++17)

### Q: 深拷贝和浅拷贝的概念？什么情况下必须使用深拷贝？

**核心区别：**

```cpp
class Buffer {
    int* data;
    size_t size;
public:
    Buffer(size_t n) : size(n), data(new int[n]) {}
    
    // 浅拷贝(默认): 只复制指针值
    // Buffer b2 = b1;  → b2.data == b1.data (指向同一块内存!)
    
    // 深拷贝: 复制指针指向的内容
    Buffer(const Buffer& other) : size(other.size), data(new int[other.size]) {
        std::memcpy(data, other.data, size * sizeof(int));
    }
};
```

```
浅拷贝后:
b1.data ──→ [1, 2, 3, 4, 5]  ← b2.data (共享!)
            修改b1影响b2
            析构b1后b2的指针悬空(dangling pointer)
            两次delete → double free → UB

深拷贝后:  
b1.data ──→ [1, 2, 3, 4, 5]  (独立副本)
b2.data ──→ [1, 2, 3, 4, 5]  (独立副本)
            完全独立, 无风险
```

**必须深拷贝的场景：**

| 场景 | 原因 | 如果不深拷贝的后果 |
|------|------|-----------------|
| 类含裸指针成员 | 默认拷贝只复制指针值 | double free, 悬空引用 |
| 函数返回局部对象(含指针) | 局部内存将被释放 | 返回的指针指向已释放内存 |
| STL容器存指针(clone语义) | 容器拷贝只复制指针 | 多个容器共享底层对象 |
| 多线程中传递对象 | 避免共享状态的数据竞争 | Race condition |

**Rule of Five (C++11):**

如果需要自定义以下任一个，通常需要自定义全部五个：
1. 析构函数
2. 拷贝构造函数（深拷贝）
3. 拷贝赋值运算符（深拷贝）
4. 移动构造函数（转移所有权）
5. 移动赋值运算符（转移所有权）

**Rule of Zero（推荐）:**
使用智能指针等RAII类型管理资源，完全不需要自定义上述五个函数。

### Q: std::unique_ptr、std::shared_ptr和std::weak_ptr的设计意图和区别？

**三种智能指针的内部实现：**

```cpp
// unique_ptr: 零开销抽象
template<typename T>
class unique_ptr {
    T* ptr;  // 只有一个裸指针，sizeof == sizeof(T*)
public:
    unique_ptr(const unique_ptr&) = delete;  // 禁止拷贝!
    unique_ptr(unique_ptr&& other) : ptr(other.ptr) {
        other.ptr = nullptr;  // 移动: 转移所有权
    }
    ~unique_ptr() { delete ptr; }
};

// shared_ptr: 引用计数
template<typename T>
class shared_ptr {
    T* ptr;
    ControlBlock* ctrl;  // 额外堆分配的控制块
    // sizeof == 2 * sizeof(void*)
};
struct ControlBlock {
    std::atomic<int> strong_count;  // shared_ptr引用数
    std::atomic<int> weak_count;    // weak_ptr引用数
    // + deleter + allocator
};

// weak_ptr: 不增加strong_count
// 只增加weak_count，不阻止对象销毁
// lock() → 检查strong_count>0? 返回shared_ptr : 返回空
```

**性能对比：**

| 操作 | unique_ptr | shared_ptr | 裸指针 |
|------|-----------|-----------|--------|
| 创建 | 与new相同 | new + ControlBlock分配 | 与new相同 |
| 拷贝 | 不允许 | 原子递增(~5-20ns) | 零成本 |
| 销毁 | delete | 原子递减+检查(~5-20ns) | 手动delete |
| 内存开销 | 0 extra | 控制块(~32-48 bytes) | 0 |
| 解引用 | 零开销 | 零开销 | 零开销 |

**使用场景和最佳实践：**

```cpp
// 1. unique_ptr: 明确单一所有者(最常用)
auto model = std::make_unique<NeuralNetwork>(config);
// 所有权传递:
auto engine = std::make_unique<InferenceEngine>(std::move(model));

// 2. shared_ptr: 多处共享生命周期
auto kv_block = std::make_shared<KVBlock>(block_size);
// beam search中多个候选共享同一个KV block:
auto beam1_ref = kv_block;  // strong_count=2
auto beam2_ref = kv_block;  // strong_count=3

// 3. weak_ptr: 观察但不拥有
class Cache {
    std::unordered_map<Key, std::weak_ptr<Value>> entries;
    std::shared_ptr<Value> get(Key k) {
        auto it = entries.find(k);
        if (it != entries.end()) {
            if (auto sp = it->second.lock()) {  // 检查是否还活着
                return sp;  // 还活着，返回shared_ptr
            }
            entries.erase(it);  // 已被销毁，清理
        }
        return nullptr;
    }
};

// 4. 打破循环引用:
struct Node {
    std::shared_ptr<Node> next;   // 强引用
    std::weak_ptr<Node> prev;     // 弱引用(避免循环)
};
```

**make_shared为什么优于分步构造？**

```cpp
// 分步构造: 两次堆分配
auto p = std::shared_ptr<T>(new T(args));
// 1. new T → 堆上分配对象
// 2. shared_ptr构造 → 堆上分配ControlBlock

// make_shared: 一次堆分配(对象和控制块连续)
auto p = std::make_shared<T>(args);
// 一次分配 [ControlBlock | T对象] → cache友好, 减少分配开销
// 缺点: 弱引用存在时整块内存不能释放(即使对象已析构)
```

### Q: 虚函数表如何实现运行时多态？虚函数与纯虚函数的区别？

**vtable机制的完整细节：**

```cpp
class Animal {
public:
    virtual void speak() { cout << "..."; }      // 虚函数(有默认实现)
    virtual void move() = 0;                      // 纯虚函数(无实现)
    virtual ~Animal() {}                          // 虚析构
    void eat() { cout << "eating"; }             // 非虚函数
};
class Dog : public Animal {
public:
    void speak() override { cout << "Woof!"; }   // 重写
    void move() override { cout << "Run"; }      // 必须实现
};
class Cat : public Animal {
public:
    void speak() override { cout << "Meow!"; }
    void move() override { cout << "Sneak"; }
};
```

```
内存布局:
Dog对象: [vptr | Animal成员 | Dog成员]
          ↓
Dog的vtable:
  [0] → &Dog::speak     (覆盖了Animal::speak)
  [1] → &Dog::move      (实现了纯虚函数)
  [2] → &Dog::~Dog      (虚析构)
  [3] → &Animal::eat    (非虚函数不在vtable中!)
  [4] → typeinfo(Dog)   (RTTI信息)

Cat的vtable:
  [0] → &Cat::speak
  [1] → &Cat::move
  [2] → &Cat::~Cat
```

**调用过程的汇编级理解：**

```cpp
Animal* a = new Dog();
a->speak();

// 编译器生成的代码(伪汇编):
// mov rax, [a]           ; 读取对象起始处的vptr
// mov rbx, [rax + 0]     ; 读取vtable第0槽(speak的地址)
// call rbx               ; 间接调用 → Dog::speak()
```

**虚函数 vs 纯虚函数：**

| 维度 | 虚函数(virtual) | 纯虚函数(= 0) |
|------|---------------|--------------|
| 基类实现 | 有默认实现 | 无实现(或提供可选实现) |
| 派生类 | 可重写可不重写 | 必须重写(否则仍为抽象类) |
| 实例化 | 基类可实例化 | 基类不可实例化(抽象类) |
| 用途 | 提供可覆盖的默认行为 | 定义接口/强制实现 |

**虚析构函数的重要性：**

```cpp
Animal* a = new Dog();
delete a;
// 如果~Animal()不是virtual:
//   只调用~Animal() → Dog的资源泄漏!
// 如果~Animal()是virtual:
//   通过vtable找到~Dog() → 正确析构
```

### Q: 什么是内存对齐？规则是什么？如何计算结构体sizeof？

**为什么需要内存对齐？**

```
CPU通过总线一次读取4/8字节(自然对齐的地址):
  对齐的int (地址0x08): 一次读取 [0x08-0x0B] → 1次总线事务
  未对齐int (地址0x03): 需要两次读取 [0x00-0x03] + [0x04-0x07] → 2次事务 + 移位拼接

某些架构(如ARM早期): 未对齐访问直接产生硬件异常
x86: 允许未对齐但性能下降(跨cache line时约2-3x慢)
```

**对齐规则（结构体布局算法）：**

```
规则:
1. 每个成员的起始偏移 = min(成员大小, 对齐系数) 的整数倍
2. 结构体总大小 = 最大对齐值的整数倍(尾部填充)
3. 默认对齐系数 = 成员自然对齐(char=1, short=2, int=4, double=8, 指针=8)

计算示例:
struct A {           // 假设64位, 默认对齐
    char a;          // offset 0, size 1
                     // padding 3 bytes (下一个int需要4对齐)
    int b;           // offset 4, size 4
    char c;          // offset 8, size 1
                     // padding 3 bytes (总大小需要4对齐: 12是4的倍数)
};
// sizeof(A) = 12

struct B {           // 调整成员顺序
    int b;           // offset 0, size 4
    char a;          // offset 4, size 1
    char c;          // offset 5, size 1
                     // padding 2 bytes
};
// sizeof(B) = 8  ← 比A小33%! 仅靠调整成员顺序

struct C {
    char a;          // offset 0, size 1
                     // padding 7 bytes (double需要8对齐)
    double b;        // offset 8, size 8
    char c;          // offset 16, size 1
                     // padding 7 bytes (总大小需要8对齐: 24)
};
// sizeof(C) = 24

// 优化: 大成员在前, 小成员集中
struct C_opt {
    double b;        // offset 0, size 8
    char a;          // offset 8, size 1
    char c;          // offset 9, size 1
                     // padding 6 bytes
};
// sizeof(C_opt) = 16  ← 节省33%
```

**控制对齐：**

```cpp
#pragma pack(1)    // 关闭对齐(紧凑布局，用于网络协议/文件格式)
struct Packet {
    char type;     // offset 0
    int length;    // offset 1 (不填充!)
    char data[0];  // 柔性数组
};
#pragma pack()     // 恢复默认

// C++11 alignas
struct alignas(64) CacheLine {  // 强制64字节对齐(cache line大小)
    int data[16];
};

// CUDA中:
// 向量化加载要求128-bit对齐: float4需要16字节对齐
// 共享内存数组对齐影响bank conflict
```

### Q: C++11中右值引用和移动语义的概念？

**左值 vs 右值：**

```cpp
int x = 42;          // x是左值(有名字, 有地址, 持久存在)
int& ref = x;        // OK: 左值引用绑定左值

int&& rref = 42;     // OK: 右值引用绑定右值(字面量/临时对象)
// int&& bad = x;    // 错误: 右值引用不能绑定左值
int&& moved = std::move(x);  // OK: move将左值"转为"右值引用
```

**移动语义解决的问题：**

```cpp
class HugeBuffer {
    float* data;
    size_t size;
public:
    // 拷贝构造: O(N)，分配新内存+复制数据
    HugeBuffer(const HugeBuffer& other) : size(other.size) {
        data = new float[size];
        memcpy(data, other.data, size * sizeof(float));  // 昂贵!
    }
    
    // 移动构造: O(1)，窃取资源
    HugeBuffer(HugeBuffer&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;  // 源对象放弃所有权
        other.size = 0;
    }
};

// 使用场景:
vector<HugeBuffer> v;
HugeBuffer buf(1000000);

v.push_back(buf);              // 拷贝: 100万float复制
v.push_back(std::move(buf));   // 移动: 只转移指针, O(1)
// 此后buf处于"有效但未指定状态"(moved-from)

// vector扩容时自动移动(如果移动构造是noexcept):
// 旧容量不够 → 分配新内存 → 移动所有元素(而非拷贝!) → 释放旧内存
```

**完美转发（Perfect Forwarding）：**

```cpp
// 问题: 包装函数如何保持参数的值类别?
template<typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
    // Args&&是万能引用(universal reference)
    // std::forward保持参数原始的左值/右值属性
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// 如果传入左值: forward保持为左值引用 → 触发拷贝构造
// 如果传入右值: forward保持为右值引用 → 触发移动构造
```

**在高性能计算中的应用：**
- tensor容器的移动避免大数据拷贝
- RVO/NRVO(返回值优化)配合移动语义：函数返回大对象零开销
- emplace_back直接在容器内构造对象，避免临时对象

### Q: CUDA的SIMT编程模型，thread/block/grid的层次关系？

**SIMT模型的本质：**

```
SIMT = SIMD + Multi-Threading
  - SIMD: 一条指令操作多个数据(Intel AVX: 256/512bit)
  - SIMT: 一条指令在32个线程上执行(但每个线程有独立PC和寄存器)

关键区别于SIMD:
  - SIMD: 程序员显式使用向量指令(intrinsic)
  - SIMT: 程序员写标量代码, 硬件自动32线程并行执行
  - SIMT允许分支发散(不同线程走不同路径, 但会串行化)
```

**层次结构映射到硬件：**

```
Grid (整个kernel的线程组织)
├── Block 0 ──→ SM 0       (被调度到某个SM执行)
│   ├── Warp 0 (thread 0-31)  ──→ 32个CUDA Core并行
│   ├── Warp 1 (thread 32-63) ──→ 32个CUDA Core并行
│   └── ...
├── Block 1 ──→ SM 3       (可能在不同SM)
│   ├── Warp 0
│   └── ...
└── Block N ──→ SM k

硬件约束(A100):
  - SM数量: 108个
  - 每SM最大block数: 32
  - 每SM最大线程数: 2048 (= 64 warps)
  - 每block最大线程数: 1024
  - 每SM共享内存: 192KB (可配置L1/Shared比例)
  - 每SM寄存器: 65536个32-bit
```

**线程ID计算：**

```cuda
// 1D grid, 1D block:
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D grid, 2D block:
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int tid = y * gridDim.x * blockDim.x + x;

// 3D情况类推...

// block/grid维度配置:
dim3 block(256);                    // 1D: 256 threads/block
dim3 grid((N + 255) / 256);         // 足够cover N个元素

dim3 block(16, 16);                 // 2D: 16×16 = 256 threads/block
dim3 grid((W+15)/16, (H+15)/16);   // 2D图像
```

**Block内协作与Warp执行：**

| 层级 | 同步方式 | 共享资源 | 通信方式 |
|------|---------|---------|---------|
| Warp内(32线程) | 隐式同步(lock-step) | 寄存器(shuffle) | __shfl_sync |
| Block内 | __syncthreads() | 共享内存 | shared memory |
| 跨Block | 无直接同步 | 全局内存 | 原子操作+fence |
| 跨Block(全局) | cooperative_groups | 全局内存 | grid.sync() |

### Q: CUDA内核中线程局部变量存储在何处？与寄存器分配的关系？

**线程局部变量的存储决策：**

```cuda
__global__ void kernel() {
    int a = 5;           // 寄存器(最快, 1 cycle)
    float b[4];          // 小数组 → 可能在寄存器中
    float c[1024];       // 大数组 → 溢出到local memory(慢!)
    
    // 编译器决定:
    // 能放寄存器 → 寄存器 (首选)
    // 寄存器不够 → local memory (物理上在全局内存, 经L1/L2缓存)
}
```

**寄存器分配与Occupancy的关系：**

```
A100每SM: 65536个32-bit寄存器, 最多2048个线程

假设kernel每线程用128个寄存器:
  65536 / 128 = 512线程可同时驻留 = 16 warps
  Occupancy = 16/64 = 25% (理论最大64 warps/SM)

假设kernel每线程用32个寄存器:
  65536 / 32 = 2048线程 = 64 warps
  Occupancy = 64/64 = 100%

Trade-off:
  更多寄存器/线程 → 更少active warps → 更差的延迟隐藏
  更少寄存器/线程 → 更多active warps → 更好的延迟隐藏
  但: 寄存器太少 → 溢出到local memory → 性能急剧下降
```

**控制寄存器使用：**

```cuda
// 方法1: 编译器选项
// nvcc --maxrregcount=128  限制每线程最大寄存器数

// 方法2: __launch_bounds__提示
__global__ void __launch_bounds__(256, 2)  // 每block最多256线程, 至少2个block/SM
kernel() {
    // 编译器据此计算每线程可用寄存器数:
    // 65536 / (256 * 2) = 128 寄存器/线程
}

// 方法3: 查看实际使用情况
// nvcc -Xptxas -v  输出:
// ptxas info: Used 76 registers, 4096 bytes smem, ...
```

**Register Spilling的性能影响：**

| 存储位置 | 延迟 | 带宽 | 何时发生 |
|---------|------|------|---------|
| 寄存器 | 1 cycle | ~19 TB/s | 默认(变量少) |
| L1 Cache(local memory命中) | ~28 cycles | 数TB/s | 溢出但被缓存 |
| L2 Cache(local memory) | ~200 cycles | ~5 TB/s | L1未命中 |
| HBM(local memory) | ~400 cycles | 2 TB/s | L2也未命中(最坏) |

### Q: 如何使用共享内存减少全局内存的重复访问？以矩阵乘法为例？

**GEMM分块使用共享内存的核心思想：**

```
C = A × B, A: [M,K], B: [K,N], C: [M,N]

不使用共享内存:
  每个线程计算C[i][j] = Σ A[i][k] * B[k][j], k=0..K-1
  每个A[i][k]被同行的N个线程重复读取
  每个B[k][j]被同列的M个线程重复读取
  总全局内存读取: M×N×K + M×N×K = 2MNK次

使用共享内存(Tiled GEMM):
  将C分成TILE×TILE的小块, 每个block计算一个小块
  每次加载A和B的一个tile到共享内存, 复用TILE次
  总全局内存读取: 2MNK/TILE次 → 减少TILE倍!
```

**代码实现：**

```cuda
#define TILE 32

__global__ void gemm_shared(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float As[TILE][TILE];  // A的tile缓存
    __shared__ float Bs[TILE][TILE];  // B的tile缓存
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    
    // 沿K维度滑动tile窗口
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Step 1: 协作加载 —— 每个线程加载一个元素到共享内存
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? 
                                        A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? 
                                        B[b_row * N + col] : 0.0f;
        
        // Step 2: 同步 —— 确保所有线程加载完毕
        __syncthreads();
        
        // Step 3: 计算 —— 从共享内存读取,每个元素被复用TILE次
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            // As[threadIdx.y][k] 被同列的TILE个线程共享
            // Bs[k][threadIdx.x] 被同行的TILE个线程共享
        }
        
        __syncthreads();  // 确保本轮计算完毕再加载下一轮
    }
    
    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**性能分析：**

| 方案 | 全局内存读取 | 带宽需求 | 相对加速 |
|------|------------|---------|---------|
| 朴素(无共享内存) | 2MNK float | 极大 | 1x |
| Tiled(TILE=16) | 2MNK/16 float | 减少16x | ~10-15x |
| Tiled(TILE=32) | 2MNK/32 float | 减少32x | ~20-25x |
| 多级Tiling+寄存器 | 更少 | 接近峰值 | ~50-80x |

### Q: 什么是Warp Shuffle指令？在规约操作中有什么优势？

**Warp Shuffle允许warp内32个线程直接交换寄存器数据：**

```cuda
// 核心API:
int __shfl_sync(mask, val, src_lane);      // 获取src_lane的val
int __shfl_down_sync(mask, val, delta);    // 获取lane+delta的val
int __shfl_up_sync(mask, val, delta);      // 获取lane-delta的val  
int __shfl_xor_sync(mask, val, lane_mask); // 获取lane^mask的val

// mask = 0xFFFFFFFF 表示所有32个lane参与
```

**用Shuffle实现Warp内Reduce：**

```cuda
__device__ float warp_reduce_sum(float val) {
    // 5次shuffle完成32个线程的求和
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);  // lane i += lane i+16
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);   // lane i += lane i+8
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);   // lane i += lane i+4
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);   // lane i += lane i+2
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);   // lane i += lane i+1
    return val;  // lane 0 holds the sum
}

// 可视化(8个lane简化示例):
// 初始:   [a0, a1, a2, a3, a4, a5, a6, a7]
// delta=4: [a0+a4, a1+a5, a2+a6, a3+a7, ...]
// delta=2: [a0+a4+a2+a6, a1+a5+a3+a7, ...]
// delta=1: [sum_all, ...]
```

**Shuffle vs 共享内存 Reduce对比：**

| 维度 | Warp Shuffle | Shared Memory |
|------|-------------|---------------|
| 适用范围 | 仅32线程(warp内) | 任意block大小 |
| 额外内存 | 无(直接寄存器交换) | 需要声明shared数组 |
| 同步 | 隐式(warp硬件同步) | 需要__syncthreads() |
| 延迟 | 1-2 cycles | ~28 cycles |
| Bank Conflict | 不存在 | 可能(需padding) |
| 代码量 | 更简洁 | 更多 |

**Block级Reduce的组合策略：**

```cuda
// 256线程的block reduce: shuffle + shared memory
__shared__ float shared[8];  // 只需要warp数量个元素

float val = input[tid];  // 每个线程的值

// Phase 1: Warp内reduce(shuffle, 无需shared memory)
val = warp_reduce_sum(val);  // 每个warp得到部分和

// Phase 2: Warp间reduce(shared memory)
int lane = threadIdx.x % 32;
int warp_id = threadIdx.x / 32;
if (lane == 0) shared[warp_id] = val;  // 8个warp的结果
__syncthreads();

// Phase 3: 第一个warp做最终reduce
if (warp_id == 0) {
    val = (lane < 8) ? shared[lane] : 0;
    val = warp_reduce_sum(val);  // 最终结果在thread 0
}
```

### Q: 向量化加载（float4/int4）进行合并访存的原理和性能收益？

**合并访存(Coalesced Access)的硬件机制：**

```
Warp内32个线程同时执行一条load指令:
  thread 0: load addr[0]
  thread 1: load addr[1]
  thread 2: load addr[2]
  ...
  thread 31: load addr[31]

硬件合并判断:
  如果所有地址落在同一个128字节的cache line内
  → 合并为1次内存事务(128 bytes)
  
  如果地址分散到32个不同的cache line
  → 32次内存事务(每次128 bytes, 只用了4 bytes → 3% 效率!)
```

**float4向量化的效果：**

```cuda
// 未向量化: 每个线程加载1个float(4 bytes)
float val = input[tid];
// 32线程连续访问 → 合并为1次128B事务
// 但每次事务只传4×32=128 bytes有效数据 → 效率100% (已经很好)

// 问题出在: 如果需要加载更多数据
// 加载128个float需要4次事务(每次32个线程各加载1个)

// 向量化: 每个线程加载1个float4(16 bytes)  
float4 val4 = reinterpret_cast<float4*>(input)[tid];
// 32线程 × 16字节 = 512字节 → 需要4次128B事务
// 但只需1条load指令(编译器生成LDG.128指令)
// 效果: 相同数据量, 指令数减少4倍, 指令调度开销减少4倍
```

**性能收益来源：**

| 收益来源 | 解释 | 量化 |
|---------|------|------|
| 减少指令数 | float4一条指令读4个float | 指令调度压力减少4x |
| 更宽的内存事务 | LDG.128 vs LDG.32 | 带宽利用率不变但指令效率高 |
| 隐藏延迟 | 更少的outstanding loads | 寄存器压力降低 |
| ILP提升 | 计算指令和少量load交替 | 流水线更满 |

**使用约束和注意事项：**

```cuda
// 约束1: 地址必须对齐(float4需要16字节对齐)
float4* ptr = reinterpret_cast<float4*>(base);  // base必须16字节对齐
// cudaMalloc保证256字节对齐, 通常满足

// 约束2: 元素数量必须是4的倍数
// 处理不对齐的尾部:
int vec_n = n / 4;
int remainder = n % 4;
// 主体用float4, 尾部逐个处理

// 约束3: 数据类型组合
// float4, int4, half2(FP16), char4等
// 原则: 尽量一次load 128 bit

// 实际加速(memory-bound kernel):
// 向量化前: 带宽利用率60-70% (指令开销限制)
// 向量化后: 带宽利用率85-95% (接近硬件峰值)
// 加速比: 通常1.5-2.5x
```

### Q: 什么是共享内存的Bank Conflict？如何产生？如何解决？

**Bank的物理结构：**

```
共享内存分32个Bank, 每个Bank宽4字节:
地址 → Bank映射: bank = (addr / 4) % 32

Bank 0:  byte[0-3],   byte[128-131], byte[256-259], ...
Bank 1:  byte[4-7],   byte[132-135], byte[260-263], ...
Bank 2:  byte[8-11],  byte[136-139], byte[264-267], ...
...
Bank 31: byte[124-127], byte[252-255], ...

每个Bank每个时钟周期可以服务一个地址的读/写
32个Bank并行 → 一个时钟周期可服务32个不同Bank的请求
```

**Bank Conflict示例：**

```cuda
__shared__ float s[32][32];  // 32×32 float数组

// 无冲突: 连续线程访问连续地址(不同Bank)
float val = s[threadIdx.x][0];  
// thread 0→bank 0, thread 1→bank 1, ... → 0冲突

// 2-way冲突: stride=2
float val = s[0][threadIdx.x * 2];
// thread 0→bank 0, thread 1→bank 2, thread 2→bank 4...
// thread 16→bank 0 (冲突!), thread 17→bank 2 (冲突!)
// 每个bank被2个线程访问 → 2-way conflict → 延迟2x

// 32-way冲突(最坏): 所有线程访问同一bank
float val = s[threadIdx.x][0];  // 如果按列访问32×32矩阵
// s[0][0]=bank 0, s[1][0]=bank 0 (因为一行32个float=128B=32个bank循环)
// 所有线程都访问bank 0 → 32-way conflict → 延迟32x
```

**解决方案：**

```cuda
// 方案1: Padding (最常用)
__shared__ float s[32][32 + 1];  // 每行多1个float(4字节)

// 原来: s[0][0]=bank0, s[1][0]=bank0 (间距128B=32 banks 回到同一bank)
// Padding: s[0][0]=bank0, s[1][0]=bank1 (间距132B=33 banks 错开1个!)
// 代价: 浪费 32×4=128 bytes共享内存

// 方案2: 数据Swizzle (更高级)
// 在加载/存储时对地址做异或变换:
int swizzled_col = col ^ row;  // 异或使得同列不再映射到同bank
float val = s[row][swizzled_col];

// 方案3: 调整访问模式
// 如果stride导致冲突, 改用其他等效的访问模式
// 例如: 转置时用padding或使用32×33的tile
```

**特殊情况——Broadcast(不是冲突)：**

```cuda
// 多个线程读同一bank的同一地址: 硬件广播, 无冲突!
float val = s[0][0];  // 所有线程读s[0][0] → broadcast, 1个周期
// 只有同bank不同地址才冲突!
```

### Q: 如何通过Padding避免Bank Conflict？

**Padding的数学原理：**

```
32×32的float矩阵在共享内存中:
  s[i][j] 的地址 = base + (i * 32 + j) * 4 bytes
  Bank号 = ((i * 32 + j) * 4 / 4) % 32 = (i * 32 + j) % 32

同列元素(j固定):
  s[0][j]: bank = (0*32+j)%32 = j
  s[1][j]: bank = (1*32+j)%32 = j  ← 同一bank!
  s[2][j]: bank = (2*32+j)%32 = j  ← 同一bank!
  ...
  所有s[*][j]都在bank j → 列访问32-way冲突!

Padding后: s[32][33]
  s[i][j] 地址 = base + (i * 33 + j) * 4 bytes
  Bank号 = (i * 33 + j) % 32

同列元素(j固定):
  s[0][j]: bank = (0*33+j)%32 = j
  s[1][j]: bank = (1*33+j)%32 = (j+1)%32  ← 错开1个bank!
  s[2][j]: bank = (2*33+j)%32 = (j+2)%32  ← 错开2个bank!
  ...
  所有s[*][j]都在不同bank → 0冲突!
```

**Padding的代价和适用场景：**

| 维度 | 代价 | 可接受性 |
|------|------|---------|
| 额外共享内存 | 每行多1个元素(~3%) | 通常可接受 |
| 索引计算 | 无额外计算(编译器处理) | 零开销 |
| 总内存浪费 | TILE行 × 4字节 | 很小 |

实际应用：矩阵转置、GEMM的tile加载、卷积的feature map缓存等场景中广泛使用padding。

### Q: CPU缓存的工作原理？时间局部性、空间局部性和缓存替换策略？

**CPU缓存层次完整架构：**

```
┌─────────────────────────────────────────┐
│ CPU Core                                 │
│  ┌──────────┐  ┌──────────┐            │
│  │ L1-D     │  │ L1-I     │  每核独占  │
│  │ 32-48KB  │  │ 32-48KB  │  ~1-3ns    │
│  │ 4-5 cycles│  │ 4-5 cycles│           │
│  └─────┬────┘  └─────┬────┘            │
│        └──────┬───────┘                  │
│         ┌─────┴──────┐                   │
│         │ L2 Cache    │  每核独占        │
│         │ 256KB-1MB   │  ~5-12ns        │
│         │ 12-15 cycles│                  │
│         └─────┬──────┘                   │
└───────────────┼──────────────────────────┘
          ┌─────┴──────┐
          │ L3 Cache    │  多核共享
          │ 4-60 MB     │  ~20-40ns
          │ 30-50 cycles│
          └─────┬──────┘
          ┌─────┴──────┐
          │ Main Memory │  DRAM
          │ GB-TB       │  ~60-100ns
          │ 200+ cycles │
          └─────────────┘
```

**局部性原理：**

| 类型 | 定义 | 利用方式 | 示例 |
|------|------|---------|------|
| 时间局部性 | 最近访问的数据很可能再次被访问 | 缓存保留最近访问的数据 | 循环变量、热点函数 |
| 空间局部性 | 某地址被访问后，附近地址也可能被访问 | cache line(64B)为单位加载 | 数组顺序遍历 |

**Cache Line和性能影响：**

```cpp
// 空间局部性的量化影响:
int arr[1024][1024];

// 行优先遍历(空间局部性好): ~0.5ms
for (int i = 0; i < 1024; i++)
    for (int j = 0; j < 1024; j++)
        sum += arr[i][j];  // 连续地址, 每64B(16个int)只miss一次

// 列优先遍历(空间局部性差): ~5ms (10x慢!)
for (int j = 0; j < 1024; j++)
    for (int i = 0; i < 1024; i++)
        sum += arr[i][j];  // 每次跳4KB, 每个访问都miss
```

**缓存替换策略：**

| 策略 | 原理 | 优缺点 | 应用 |
|------|------|--------|------|
| LRU | 替换最久未使用的行 | 实现复杂(完全排序), 精确 | 小容量cache |
| 伪LRU(Tree-PLRU) | 用二叉树近似LRU | 实现简单(每组只需N-1 bit) | 现代L1/L2 |
| Random | 随机选择替换 | 最简单, 对某些pattern反而好 | ARM某些实现 |
| RRIP | 预测将来不再使用的行 | 自适应, 抗pollution | Intel L3 |

**写策略：**

| 策略 | 写命中 | 写不命中 | 适用层级 |
|------|--------|---------|---------|
| Write-Back | 写到cache, 标dirty | Write-Allocate(先读入再写) | L1/L2(减少写带宽) |
| Write-Through | 同时写cache和下层 | No-Write-Allocate | 某些L1 |

**与GPU缓存的关键区别：**
- CPU缓存：大而复杂(多路组相联, coherence协议)，硬件全自动管理
- GPU L1/L2：较小但带宽极高，部分可由程序员管理(共享内存)
- GPU设计哲学：用大量线程隐藏延迟，而非用大缓存减少miss

### Q: 手撕：实时找出数据流中出现频率最高的前K个元素？

（编程题）
