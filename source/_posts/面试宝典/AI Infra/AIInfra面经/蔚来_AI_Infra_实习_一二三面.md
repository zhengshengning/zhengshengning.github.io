---
title: '蔚来 AI Infra 实习 一二三面'
date: 2026-04-16 12:18:52
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: TensorRT的理解？TensorRT和OpenVINO的区别？

**TensorRT核心优化流程：**

```
ONNX/TF/PyTorch模型
    ↓ Parser (导入模型)
Network Definition (TRT内部表示)
    ↓ Builder (核心优化引擎)
    ├── Layer Fusion: Conv+BN+ReLU → 单个kernel
    ├── Precision Calibration: FP32→FP16/INT8自动校准
    ├── Kernel Auto-Tuning: 对每层尝试所有实现,选最快
    ├── Tensor Memory Planning: 分析生命周期,复用buffer
    └── Multi-Stream: 独立分支并行执行
    ↓
Optimized Engine (.engine/.plan文件)
    ↓ Runtime (执行)
推理结果
```

**TensorRT关键优化技术：**

| 优化 | 原理 | 收益 |
|------|------|------|
| 层融合 | Conv+BN+ReLU合并为一个kernel | 减少kernel launch和中间IO |
| INT8量化 | 校准集统计分布→KL散度找最优threshold | 速度2-4x, 精度损失<1% |
| Kernel Auto-Tuning | 运行时测试所有可用kernel实现 | 选择当前硬件上最快的 |
| Dynamic Tensor Memory | 分析tensor生命周期做in-place复用 | 减少30-50%显存 |
| Multi-Stream Execution | 独立路径分配到不同stream | 提高SM利用率 |

**TensorRT vs OpenVINO对比：**

| 维度 | TensorRT | OpenVINO |
|------|----------|----------|
| 厂商 | NVIDIA | Intel |
| 目标硬件 | NVIDIA GPU(仅) | Intel CPU/GPU/VPU/FPGA |
| 核心优势 | GPU推理性能极致 | 跨Intel硬件统一部署 |
| 量化方案 | INT8(entropy/MinMax校准) | INT8/INT4(NNCF工具) |
| 开源程度 | 核心kernel闭源 | 大部分开源 |
| 灵活性 | 自定义plugin扩展 | Model Optimizer灵活转换 |
| 动态shape | 支持(需指定范围) | 支持 |
| 延迟 | GPU上最低 | CPU上很优(Winograd/VNNI) |
| 典型场景 | 云端/边缘GPU推理 | 边缘设备(无GPU)/CPU服务器 |
| LLM支持 | TensorRT-LLM | OpenVINO GenAI |

**选择建议：**
- 有NVIDIA GPU且追求极致性能 → TensorRT
- Intel硬件部署(含CPU推理) → OpenVINO
- 需要模型快速上线且不限硬件 → ONNX Runtime(支持TRT/OpenVINO后端)

### Q: C++面向对象的三大特性如何体现？

**封装（Encapsulation）：**

```cpp
class KVCache {
private:
    float* data_;          // 隐藏实现细节
    size_t capacity_;
    size_t used_;
    
public:
    // 只暴露必要接口
    void append(const float* kv, size_t len);
    float* get_block(int block_idx) const;
    size_t available() const { return capacity_ - used_; }
};
// 外部无法直接操作data_指针，只能通过接口使用
// 好处: 修改内部实现(如分页存储)不影响使用者
```

**继承（Inheritance）：**

```cpp
class Attention {  // 基类: 通用attention接口
public:
    virtual Tensor forward(Tensor q, Tensor k, Tensor v) = 0;
    virtual ~Attention() = default;
};

class MHA : public Attention {  // 多头注意力
    Tensor forward(Tensor q, Tensor k, Tensor v) override;
};

class GQA : public Attention {  // 分组查询注意力
    int num_kv_groups_;
    Tensor forward(Tensor q, Tensor k, Tensor v) override;
};
// 代码复用 + 统一接口 + 方便扩展新的attention变体
```

**多态（Polymorphism）：**

```cpp
// 运行时多态(虚函数):
void run_inference(Attention* attn, Tensor q, Tensor k, Tensor v) {
    auto output = attn->forward(q, k, v);  // 运行时决定调用MHA还是GQA
}

// 编译时多态(模板):
template<typename AttnImpl>
void run_inference(AttnImpl& attn, Tensor q, Tensor k, Tensor v) {
    auto output = attn.forward(q, k, v);  // 编译时确定类型, 可内联
}
// 优势: 零虚函数开销, 编译器可优化(但失去运行时灵活性)
```

### Q: C++继承的分类和特点？

| 继承类型 | 基类public成员变为 | 基类protected成员变为 | 语义 |
|---------|------------------|---------------------|------|
| public | public | protected | is-a关系(Dog is an Animal) |
| protected | protected | protected | 实现继承(限制对外暴露) |
| private | private | private | has-a替代(实现细节) |

**菱形继承问题：**

```cpp
class Device {};                      // 基类
class GPU : public Device {};         // GPU is a Device
class CPU : public Device {};         // CPU is a Device
class APU : public GPU, public CPU {};// APU继承GPU和CPU
// 问题: APU中有两份Device! → 访问Device成员二义性

// 虚继承解决:
class GPU : virtual public Device {}; 
class CPU : virtual public Device {};
class APU : public GPU, public CPU {};
// APU中只有一份Device（通过vbptr间接定位）
```

**虚继承的代价：**
- 额外的vbptr(虚基类指针)存储
- 访问虚基类成员需要间接寻址(多一次指针解引用)
- 对象布局更复杂(虚基类放在对象末尾)
- 实际C++项目中建议避免深层多继承，改用组合+接口

### Q: C++虚函数的原理和作用？

**vtable完整机制：**

```cpp
class Shape {
public:
    virtual double area() { return 0; }
    virtual void draw() = 0;  // 纯虚函数
    virtual ~Shape() {}
    int x, y;  // 非虚成员
};

class Circle : public Shape {
public:
    double area() override { return 3.14159 * r * r; }
    void draw() override { /* OpenGL绑定... */ }
    double r;
};
```

```
Circle对象内存布局:
  [vptr (8B)] → Circle_vtable
  [Shape::x (4B)]
  [Shape::y (4B)]
  [Circle::r (8B)]
  Total: 24 bytes

Circle_vtable:
  [0]: &Circle::area       (覆盖了Shape::area)
  [1]: &Circle::draw       (实现了纯虚函数)
  [2]: &Circle::~Circle    (虚析构)
  [-1]: &typeinfo(Circle)  (RTTI, dynamic_cast使用)
```

**虚函数调用的代码生成：**

```
Shape* s = get_shape();  // 可能是Circle或Rectangle
s->area();

汇编:
  mov rax, [s]          ; 读vptr(对象首8字节)
  mov rbx, [rax + 0]    ; 读vtable[0] = area的函数地址
  call rbx              ; 间接调用

开销: 2次内存读取(vptr + vtable entry) + 1次间接call
     vs 非虚函数的直接call
     额外 ~2-5ns(cache命中时)
```

**虚函数vs纯虚函数的设计意图：**
- 虚函数：提供合理的默认行为，派生类可以选择性重写
- 纯虚函数：强制派生类必须实现，基类只定义接口契约
- 含纯虚函数的类=抽象类：不可实例化，只能作为接口使用

### Q: vector、map、unordered_map的底层实现？

**三种容器的完整对比：**

| 维度 | vector | map | unordered_map |
|------|--------|-----|---------------|
| 底层结构 | 动态数组(连续内存) | 红黑树 | 哈希表(开链法) |
| 有序性 | 插入顺序 | key有序 | 无序 |
| 随机访问 | O(1) | O(log n) | 平均O(1) |
| 插入(尾部/一般) | O(1)均摊/O(n) | O(log n) | 平均O(1), 最坏O(n) |
| 删除 | O(n)(中间)/O(1)(尾) | O(log n) | 平均O(1) |
| 内存布局 | 连续 | 每个节点堆分配 | 桶数组+链表节点 |
| Cache友好 | 极好 | 差(节点分散) | 中等 |
| 迭代器失效 | 插入扩容时全失效 | 只有被删除的失效 | rehash时全失效 |

**vector扩容机制：**

```cpp
// capacity不够时:
// 1. 分配2倍新空间(MSVC是1.5倍)
// 2. 移动所有元素到新空间(C++11用移动构造)
// 3. 释放旧空间
// 均摊O(1): n次push_back只有log(n)次扩容

// 性能建议:
vec.reserve(expected_size);  // 预分配，避免扩容
vec.shrink_to_fit();         // 释放多余capacity
```

**map(红黑树)的性质：**
- 每个节点红或黑色
- 根节点和叶子(NIL)为黑
- 红节点的子节点必须为黑
- 从根到叶的路径黑节点数相同
- 保证：最长路径不超过最短路径2倍 → O(log n)

**unordered_map内部结构：**

```
桶数组(bucket array):
  [0] → nullptr
  [1] → Node(k1,v1) → Node(k5,v5) → nullptr  (链表)
  [2] → Node(k2,v2) → nullptr
  [3] → nullptr
  ...
  [n-1] → Node(km,vm) → nullptr

当 load_factor = size/bucket_count > max_load_factor(默认1.0):
  → rehash: bucket_count翻倍, 所有元素重新hash分配
  → O(n)开销, 迭代器全失效
```

**选择指南：**
- 频繁随机访问、尾部增删 → `vector`
- 需要有序遍历、范围查询 → `map`
- 只需快速查找/插入/删除 → `unordered_map`
- 小数据量(<100) → `vector`(即使需要查找，cache优势可能胜过算法优势)

### Q: 多进程间通信方式有哪些？

| IPC方式 | 速度 | 方向 | 适用场景 | 关键限制 |
|---------|------|------|---------|---------|
| 管道(pipe) | 快 | 半双工 | 父子进程 | 只能亲缘进程 |
| 命名管道(FIFO) | 快 | 半双工 | 任意进程 | 文件系统可见 |
| 共享内存(shm) | 最快 | 双向 | 大数据量高频通信 | 需自行同步 |
| 消息队列 | 中 | 双向 | 结构化消息 | 有大小限制 |
| 信号量 | - | - | 进程同步 | 只能做同步不能传数据 |
| 信号(signal) | 快 | 单向 | 异步通知 | 只能传少量信息 |
| Socket | 中 | 全双工 | 网络/跨机器 | 开销最大(协议栈) |
| 内存映射文件(mmap) | 快 | 双向 | 文件共享/大数据 | 需文件系统 |

**深度学习中常用的IPC：**

```python
# PyTorch DataLoader: 共享内存传递tensor
# 原理: worker进程将预处理好的tensor放入共享内存
# 主进程直接从共享内存读取，零拷贝

# PyTorch DDP: NCCL (GPU间通信)
# GPU↔GPU: NVLink/PCIe P2P (不经CPU)
# 机器间: InfiniBand RDMA (零拷贝网络)

# vLLM: Unix Domain Socket + 共享内存
# Scheduler(Python)和Worker(Python/C++)通过IPC通信
```

### Q: 多线程如何进行资源共享和同步？

**同步原语的完整对比：**

| 原语 | 适用场景 | 性能 | 特点 |
|------|---------|------|------|
| mutex | 保护临界区 | ~25ns(无竞争) | 同一时刻只一个线程 |
| shared_mutex | 读多写少 | 略高于mutex | 多读单写 |
| spinlock | 极短临界区 | 极快(无竞争) | 忙等(消耗CPU) |
| condition_variable | 等待/通知 | 无忙等 | 配合mutex使用 |
| atomic | 简单变量操作 | 最快(无锁) | 只适合简单操作 |
| semaphore | 控制并发数 | 中等 | 允许N个线程同时进入 |
| barrier | 同步点(所有线程到齐) | 中等 | C++20 std::barrier |

**典型使用模式：**

```cpp
// 1. 生产者-消费者(condition_variable)
std::queue<Task> queue;
std::mutex mtx;
std::condition_variable cv;

// 生产者:
{
    std::lock_guard lock(mtx);
    queue.push(task);
}
cv.notify_one();  // 唤醒一个等待的消费者

// 消费者:
std::unique_lock lock(mtx);
cv.wait(lock, [&]{ return !queue.empty(); });  // 等待直到有数据
auto task = queue.front(); queue.pop();
lock.unlock();
process(task);

// 2. 无锁计数器(atomic)
std::atomic<int> counter{0};
counter.fetch_add(1, std::memory_order_relaxed);  // 原子递增

// 3. RAII锁(避免忘记解锁)
{
    std::lock_guard<std::mutex> lock(mtx);  // 构造时加锁
    // ... 临界区 ...
}  // 析构时自动解锁(即使异常也能正确释放)
```

**避免死锁的策略：**
- `std::lock(m1, m2)`: 同时获取多把锁(内部使用try-and-back-off)
- `std::scoped_lock(m1, m2)`: C++17 RAII版本
- 固定加锁顺序: 所有线程按相同顺序获取锁
- 超时机制: `try_lock_for()`

### Q: TVM的原理和作用？

**TVM的核心设计——计算与调度分离：**

```python
# 1. 描述计算(WHAT to compute):
import tvm
from tvm import te

N = 1024
A = te.placeholder((N, N), name='A')
B = te.placeholder((N, N), name='B')
k = te.reduce_axis((0, N), name='k')
C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
# 只描述了C[i][j] = sum(A[i][k] * B[k][j]) —— 纯数学定义

# 2. 描述调度(HOW to compute):
s = te.create_schedule(C.op)
# 分块:
xo, xi = s[C].split(C.op.axis[0], factor=32)
yo, yi = s[C].split(C.op.axis[1], factor=32)
ko, ki = s[C].split(k, factor=4)
# 重排循环顺序:
s[C].reorder(xo, yo, ko, xi, ki, yi)
# 并行化外层:
s[C].parallel(xo)
# 向量化最内层:
s[C].vectorize(yi)
```

**TVM编译栈全景：**

```
PyTorch/TensorFlow/ONNX模型
    ↓ Frontend (模型导入)
Relay IR (高层图级IR)
    ↓ Graph-level Optimization
    ├── 算子融合(Fusion)
    ├── 常量折叠(Constant Folding)
    ├── 死代码消除
    └── 布局变换(NCHW→NHWC)
    ↓
TIR (Tensor IR, 低层循环级IR)
    ↓ Schedule/AutoTVM/Ansor
    ├── Loop Tiling
    ├── Vectorization
    ├── Unrolling
    ├── Thread binding (GPU)
    └── Memory hierarchy mapping
    ↓ Code Generation
目标代码 (CUDA/LLVM/Metal/OpenCL/C)
```

**自动调度(AutoScheduler/Ansor)：**

| 维度 | AutoTVM | Ansor(AutoScheduler) |
|------|---------|---------------------|
| 搜索空间 | 手写模板定义 | 自动生成(sketch-based) |
| 工程量 | 每个算子写模板 | 几乎零手工(定义计算即可) |
| 搜索效率 | 模板内搜索(空间小) | 更大空间但更智能(cost model引导) |
| 性能 | 好 | 通常更好 |

**TVM vs 其他编译器：**

| 维度 | TVM | TensorRT | Triton |
|------|-----|----------|--------|
| 目标 | 通用编译器(多后端) | NVIDIA GPU推理 | GPU kernel开发 |
| 优势 | 支持NPU/DSA等多硬件 | GPU极致性能 | 开发效率高 |
| 劣势 | 编译时间长 | 只支持NVIDIA | 只能写kernel |
| 适用 | 部署到多种硬件 | 生产GPU推理 | 自定义算子 |

### Q: 图优化和算子调度方法？

**图优化（Graph-level Optimization）：**

**1. 算子融合（最重要的优化）：**

```
融合前:                     融合后:
[Conv] → HBM               [Conv+BN+ReLU] (一个kernel)
  ↓ 写出中间结果               不需要中间结果落地HBM
[BatchNorm] → HBM
  ↓ 
[ReLU] → HBM
3次HBM读写                  1次HBM读写 → 节省2次IO
```

融合分类：

| 类型 | 模式 | 示例 |
|------|------|------|
| Element-wise融合 | 逐元素操作串联 | ReLU+Add+Mul |
| Reduction+Element-wise | reduce后接逐元素 | Softmax内部(max+sub+exp+sum+div) |
| 复杂融合 | 多种pattern混合 | Conv+BN+ReLU |
| 注意力融合 | 整个attention块 | FlashAttention |

**2. 常量折叠：**

```python
# 优化前: 运行时计算
scale = 1.0 / sqrt(head_dim)  # head_dim=128, 每次推理都算
q = q * scale

# 优化后: 编译时计算
q = q * 0.08838834  # 直接用常量
```

**3. 布局变换：**

```
不同硬件偏好不同数据布局:
  CPU: NCHW (通道优先, 适合向量化)
  GPU Tensor Core: NHWC (通道最后, 适合矩阵乘)
  某些NPU: NC/4HW4 (通道分组)

编译器在图中插入transpose节点, 或fusion时内部处理
```

**算子调度（Operator Scheduling）：**

```
计算图:
  A ──→ C ──→ E
  B ──→ D ──↗

调度决策:
1. 拓扑排序确定合法执行顺序: [A,B,C,D,E] 或 [B,A,D,C,E] 等
2. 内存规划: 
   A的输出只被C使用 → C执行完后释放A的buffer
   B的输出只被D使用 → D执行完后释放B的buffer
   → 内存复用: A和D可以共享同一buffer(生命周期不重叠)
3. 并行调度:
   A和B独立 → 可并行(不同stream)
   C依赖A, D依赖B → 各自在依赖完成后执行
4. Stream分配:
   Stream 0: A → C → E
   Stream 1: B → D
   (C和D可能并行执行)
```

### Q: C++多继承的问题和解决？

**菱形继承问题详解：**

```cpp
class Animal { public: int age; void breathe(); };
class Dog : public Animal {};  // Dog有Animal::age
class Pet : public Animal {};  // Pet也有Animal::age
class PetDog : public Dog, public Pet {};
// PetDog有两份age! 
// petdog.age 编译错误: ambiguous
// 需要 petdog.Dog::age 或 petdog.Pet::age

// 内存布局:
// PetDog: [Dog部分: [Animal::age] | Dog成员] [Pet部分: [Animal::age] | Pet成员] [PetDog成员]
//                    ^^^^第一份               ^^^^第二份
```

**虚继承的解决方案：**

```cpp
class Animal { public: int age; void breathe(); };
class Dog : virtual public Animal {};
class Pet : virtual public Animal {};
class PetDog : public Dog, public Pet {};
// PetDog只有一份Animal::age ✓
// petdog.age 无歧义 ✓

// 内存布局(含虚基类偏移):
// PetDog: [Dog部分: vbptr, Dog成员]
//         [Pet部分: vbptr, Pet成员]
//         [PetDog成员]
//         [Animal部分: age]  ← 虚基类放在末尾,通过vbptr间接访问
```

**虚继承的代价：**
- 每个虚继承路径增加一个vbptr(虚基类指针)
- 访问虚基类成员多一次间接寻址
- 构造时最派生类负责初始化虚基类(中间类的初始化被忽略)
- 对象大小增加

**实际建议：**
- 优先使用组合(composition)而非多继承
- 如果需要多个接口，使用纯虚基类(无数据成员的抽象接口)
- Java/Go等语言直接不允许多继承(只允许多接口)

### Q: C++智能指针有哪些？用过吗？

```cpp
// 1. unique_ptr — 工厂模式返回(最常用)
class ModelFactory {
public:
    static std::unique_ptr<Model> create(const Config& cfg) {
        if (cfg.type == "llama") return std::make_unique<LlamaModel>(cfg);
        if (cfg.type == "gpt") return std::make_unique<GPTModel>(cfg);
        return nullptr;
    }
};
auto model = ModelFactory::create(config);
// 所有权唯一, 零额外开销, 自动释放

// 2. shared_ptr — KV-Cache共享(beam search)
auto kv_block = std::make_shared<KVBlock>(block_size);
// beam search中多条路径共享前缀的KV:
std::vector<std::shared_ptr<KVBlock>> beams(beam_width, kv_block);
// 所有beam_path指向同一块KV数据, 引用计数=beam_width
// Copy-on-Write: 需要修改时才复制

// 3. weak_ptr — 缓存/观察者
class PrefixCache {
    std::unordered_map<TokenSeq, std::weak_ptr<KVBlock>> cache_;
public:
    std::shared_ptr<KVBlock> lookup(const TokenSeq& prefix) {
        auto it = cache_.find(prefix);
        if (it != cache_.end()) {
            if (auto sp = it->second.lock()) return sp;  // 还活着
            cache_.erase(it);  // 已被释放,清理
        }
        return nullptr;
    }
};
```

**常见陷阱：**
- `shared_ptr`循环引用 → 内存泄漏(用`weak_ptr`打破)
- `shared_ptr`的原子引用计数在高并发下有竞争开销
- `unique_ptr`不能直接传值(需要`std::move`)
- 从`this`构造`shared_ptr`需要`enable_shared_from_this`

### Q: static和const的区别？

| 维度 | static | const |
|------|--------|-------|
| 核心作用 | 控制生命周期和可见性 | 控制可修改性 |
| 修饰局部变量 | 延长到程序结束，只初始化一次 | 不可修改 |
| 修饰全局变量 | 限制为内部链接(当前文件可见) | 不可修改(C++中默认内部链接) |
| 修饰类成员变量 | 属于类，所有对象共享 | 初始化后不可修改 |
| 修饰成员函数 | 无this指针，只能访问static成员 | 不修改对象状态(const this) |
| 存储位置 | 静态存储区(.data/.bss) | 取决于原始声明 |

```cpp
class Config {
    static int instance_count;       // 所有对象共享的计数器
    static Config& getInstance();    // 单例模式(Meyers Singleton)
    
    const int max_batch_size;        // 构造后不可修改
    void print() const;              // 不修改对象, 可被const对象调用
};

// static局部变量的线程安全(C++11保证):
Config& Config::getInstance() {
    static Config instance;  // 首次调用时构造，线程安全
    return instance;
}
```

### Q: C++异步实现方法？

```cpp
// 1. std::async — 最简单(自动管理线程)
auto future = std::async(std::launch::async, [](int x) {
    return heavy_computation(x);
}, 42);
// ... 做其他事 ...
int result = future.get();  // 阻塞等待结果

// 2. std::thread + promise — 手动控制
std::promise<int> prom;
auto fut = prom.get_future();
std::thread t([&prom]() {
    prom.set_value(compute());  // 设置结果
});
int val = fut.get();  // 获取结果
t.join();

// 3. 线程池模式(生产环境常用):
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    // submit()提交任务, worker线程从队列取任务执行
};

// 4. C++20协程(最现代):
task<int> async_inference(Model& model, Tensor input) {
    auto preprocessed = co_await async_preprocess(input);
    auto result = co_await model.forward(preprocessed);
    co_return postprocess(result);
}
// 无需创建线程, 挂起/恢复开销极低(~纳秒级)
```

**在AI Infra中的异步应用：**
- Prefill和Decode异步调度（不同stream）
- 数据预取和计算overlap（cp.async）
- 多请求异步处理（async serving）
- 权重加载和推理并行

### Q: Python列表切片反转？

```python
# 三种方式:
lst = [1, 2, 3, 4, 5]

# 1. 切片(返回新列表, 不修改原列表):
reversed_lst = lst[::-1]  # [5, 4, 3, 2, 1]
# 原理: [start:stop:step], step=-1从后往前

# 2. reversed()(返回迭代器):
reversed_lst = list(reversed(lst))  # 需要list()转换

# 3. 原地反转:
lst.reverse()  # 修改lst本身, 返回None

# 切片的灵活用法:
lst[1:4]      # [2, 3, 4] — 下标1到3
lst[::2]      # [1, 3, 5] — 每隔一个取
lst[-3:]      # [3, 4, 5] — 最后3个
lst[::-2]     # [5, 3, 1] — 反向每隔一个
```

### Q: Python装饰器的原理？

```python
# 装饰器的本质: 接受函数,返回新函数的高阶函数

# 基础装饰器(计时):
import time
from functools import wraps

def timer(func):
    @wraps(func)  # 保留原函数的__name__, __doc__等
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)  # 调用原函数
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer  # 等价于: inference = timer(inference)
def inference(model, input):
    return model(input)

# 带参数的装饰器(三层嵌套):
def retry(max_attempts=3, delay=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=5, delay=0.5)
def call_api(url):
    ...

# 类装饰器(有状态):
class CacheResult:
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

@CacheResult
def expensive_compute(x, y):
    return x ** y
```

### Q: 手撕：反转链表？

（编程题）

### Q: 手撕：C++实现NMS + IOU？

（编程题）
