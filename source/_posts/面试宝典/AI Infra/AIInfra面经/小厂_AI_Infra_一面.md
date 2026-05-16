---
title: '小厂 AI Infra 一面'
date: 2026-04-16 12:16:51
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 如何设计一个分布式推理框架？

**分布式推理框架的分层架构：**

```
┌────────────── API Layer ──────────────────────┐
│ HTTP/gRPC Server, 流式SSE, 负载均衡            │
└──────────────────┬────────────────────────────┘
                   ↓
┌────────────── Scheduler Layer ────────────────┐
│ 请求管理 │ Batch组装 │ 显存预算 │ 抢占策略      │
└──────────────────┬────────────────────────────┘
                   ↓
┌────────────── Execution Layer ────────────────┐
│ Model Runner │ KV-Cache Manager │ CUDA Graph   │
└──────────────────┬────────────────────────────┘
                   ↓
┌────────────── Communication Layer ────────────┐
│ NCCL │ AllReduce/AllGather │ Send-Recv         │
└──────────────────┬────────────────────────────┘
                   ↓
┌────────────── Hardware Layer ─────────────────┐
│ GPU Cluster │ NVLink/InfiniBand │ Storage      │
└───────────────────────────────────────────────┘
```

**各层关键设计决策：**

**1. 模型切分策略（决定通信模式）：**

| 策略 | 适用场景 | 通信模式 | 延迟影响 |
|------|---------|---------|---------|
| TP(张量并行) | 同节点NVLink多卡 | AllReduce每层2次 | 降低延迟 |
| PP(流水线并行) | 跨节点 | 层间Send/Recv | 增加延迟(有bubble) |
| EP(专家并行) | MoE模型 | All-to-All | 取决于负载均衡 |
| TP+PP组合 | 大模型多节点 | 节点内TP+节点间PP | 平衡延迟和扩展性 |

```
配置示例(70B模型, 8卡推理):
  方案1: TP=8 (单节点8卡, 每层AllReduce)
    延迟最低, 但需要NVLink高带宽
  方案2: TP=4, PP=2 (4卡一组做TP, 2组做PP)
    可跨节点, 但PP有层间延迟
```

**2. 调度器设计：**

```python
class Scheduler:
    def __init__(self, max_batch_tokens, gpu_memory_budget):
        self.waiting = PriorityQueue()    # 等待prefill的请求
        self.running = []                  # 正在decode的请求
        self.swapped = []                  # 被抢占到CPU的请求
        
    def schedule_step(self):
        # 核心约束: GPU显存预算
        budget = self.gpu_memory_budget
        
        # 1. Running请求继续(分配新KV block)
        for req in self.running:
            if req.needs_block():
                if budget.can_allocate():
                    budget.allocate(req)
                else:
                    self.preempt(req)  # 显存不足→抢占
        
        # 2. 加入新请求(考虑prefill token预算)
        total_tokens = sum(r.decode_token_count for r in self.running)
        while self.waiting and total_tokens < self.max_batch_tokens:
            req = self.waiting.get()
            if budget.can_allocate(req.prefill_blocks):
                budget.allocate(req)
                self.running.append(req)
                total_tokens += req.input_length
            else:
                break
```

**3. KV-Cache管理(PagedAttention)：**

| 设计点 | 方案 | 原因 |
|--------|------|------|
| Block大小 | 16 tokens/block | 平衡碎片和管理开销 |
| 分配策略 | 按需分配(lazy) | 最大化利用率 |
| 回收策略 | 引用计数 | 支持CoW和前缀共享 |
| 跨设备 | TP各rank独立管理各自head的KV | KV按head切分 |
| Swap | GPU↔CPU via PCIe | 抢占时保留计算进度 |

**4. 弹性和容错：**
- 健康检查: 定期ping每个worker, 超时认为故障
- 故障恢复: 将故障节点的请求重路由到健康节点(重新prefill)
- 弹性扩缩: 根据QPS动态增减推理实例(K8s HPA)
- 请求重试: 客户端超时自动重试到其他实例

### Q: 算子优化的一般方法？

**算子优化的系统方法论：**

```
Profile(Nsight Compute)
    ↓ 确定瓶颈类型
    ├── Memory-bound (SM SOL低, Memory SOL高)
    │   ├── 优化全局内存访问模式
    │   ├── 利用共享内存缓存
    │   ├── 向量化加载(float4)
    │   └── 算子融合减少中间IO
    │
    ├── Compute-bound (SM SOL高, Memory SOL低)
    │   ├── Tensor Core加速
    │   ├── 循环展开提高ILP
    │   ├── 减少分支发散
    │   └── 使用快速数学函数
    │
    └── Latency-bound (两者SOL都低)
        ├── 增大block size/occupancy
        ├── 异步操作(cp.async)
        ├── 减少同步点
        └── CUDA Graph消除launch开销
```

**Memory-bound优化详解：**

```cuda
// 优化1: 合并访问(Coalesced Access)
// 坏: thread i 访问 data[i * stride]  (stride>1 → 多次事务)
// 好: thread i 访问 data[base + i]     (连续 → 1次事务)

// 优化2: 向量化加载
float4 val = reinterpret_cast<float4*>(data)[tid]; // 128-bit加载

// 优化3: 共享内存缓存(数据复用)
__shared__ float tile[32][33];  // +1 padding避免bank conflict
tile[ty][tx] = global[...];
__syncthreads();
// 后续从tile多次读取(~19TB/s vs HBM 2TB/s)

// 优化4: 算子融合
// 原来: kernel1(input→tmp) + kernel2(tmp→output) = 2次HBM读写
// 融合: kernel(input→output) 中间结果在寄存器/shared中 = 1次读写
```

**Compute-bound优化详解：**

```cuda
// 优化1: Tensor Core (WMMA)
#include <mma.h>
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
wmma::load_matrix_sync(a, A_ptr, lda);
wmma::load_matrix_sync(b, B_ptr, ldb);
wmma::mma_sync(c, a, b, c);  // 16x16x16 矩阵乘一条指令!
// FP16 Tensor Core: 312 TFLOPS(A100) vs FP32 CUDA Core: 19.5 TFLOPS

// 优化2: 循环展开
#pragma unroll 8
for (int k = 0; k < K; k++) {
    acc += a[k] * b[k];  // 展开后编译器可做指令调度
}

// 优化3: Warp Shuffle替代Shared Memory
float sum = __shfl_down_sync(0xFFFFFFFF, val, 16); // 1 cycle!
```

**Occupancy与性能的关系：**

```
Occupancy不是越高越好!

高Occupancy(>75%):
  + 更多warp → 更好的延迟隐藏(Memory-bound有利)
  - 每线程寄存器少 → 可能溢出到local memory

低Occupancy(25-50%):
  + 每线程更多寄存器 → 更多数据留在寄存器
  - 可能无法充分隐藏延迟

最优点: 需要实验确定
  Memory-bound kernel: 通常需要高occupancy
  Compute-bound kernel: 中等occupancy即可(寄存器更重要)
```

### Q: C++反射机制？

**C++缺乏原生反射的问题和解决方案：**

```cpp
// 问题: 无法在运行时根据字符串创建对象
// Java: Object obj = Class.forName("MyClass").newInstance();
// C++: 不支持! 需要手动实现

// 方案1: 手动注册(框架最常用)
class OpRegistry {
    using Creator = std::function<std::unique_ptr<Op>()>;
    static std::unordered_map<std::string, Creator>& registry() {
        static std::unordered_map<std::string, Creator> r;
        return r;
    }
public:
    static void register_op(const std::string& name, Creator creator) {
        registry()[name] = creator;
    }
    static std::unique_ptr<Op> create(const std::string& name) {
        return registry().at(name)();
    }
};

// 注册宏(自动注册):
#define REGISTER_OP(ClassName) \
    static bool _reg_##ClassName = [] { \
        OpRegistry::register_op(#ClassName, \
            []{ return std::make_unique<ClassName>(); }); \
        return true; \
    }();

// 使用:
class MatMulOp : public Op { ... };
REGISTER_OP(MatMulOp);  // 自动注册到全局map

auto op = OpRegistry::create("MatMulOp");  // 运行时按名字创建
```

```cpp
// 方案2: 有限RTTI (typeid/dynamic_cast)
Base* b = get_object();
if (typeid(*b) == typeid(Derived)) { ... }
// 缺陷: 只能判断类型,不能获取成员列表/调用任意方法

// 方案3: C++26反射提案(未来)
// consteval auto members = std::meta::members_of(^MyClass);
// for (auto m : members) { /* 编译期遍历成员 */ }
```

**在AI框架中的应用：**
- PyTorch C++算子注册: `TORCH_LIBRARY(myops, m) { m.def("my_op", ...); }`
- TensorRT plugin注册: 工厂模式 + 注册表
- ONNX Runtime: 自定义算子通过字符串名字注册

### Q: 工厂模式是什么？

**三种工厂模式及其在AI框架中的应用：**

```cpp
// 1. 简单工厂(Static Factory)
class AttentionFactory {
public:
    static std::unique_ptr<Attention> create(const std::string& type) {
        if (type == "mha") return std::make_unique<MHAttention>();
        if (type == "gqa") return std::make_unique<GQAttention>();
        if (type == "mla") return std::make_unique<MLAttention>();
        throw std::runtime_error("Unknown attention: " + type);
    }
};
// 使用: auto attn = AttentionFactory::create(config.attn_type);

// 2. 工厂方法(Factory Method) — 子类决定创建什么
class ModelBuilder {
public:
    virtual std::unique_ptr<Layer> createAttention() = 0;
    virtual std::unique_ptr<Layer> createFFN() = 0;
    
    std::unique_ptr<Model> build() {
        auto model = std::make_unique<Model>();
        for (int i = 0; i < num_layers; i++) {
            model->add(createAttention());  // 子类决定类型
            model->add(createFFN());
        }
        return model;
    }
};
class LlamaBuilder : public ModelBuilder {
    std::unique_ptr<Layer> createAttention() override {
        return std::make_unique<GQAttention>(num_kv_heads);
    }
};

// 3. 注册表工厂(Registry Factory) — 开闭原则最优
class LayerRegistry {
    static std::map<std::string, std::function<Layer*()>> creators;
public:
    template<typename T>
    static void reg(const std::string& name) {
        creators[name] = []{ return new T(); };
    }
    static Layer* create(const std::string& name) {
        return creators.at(name)();
    }
};
// 新增类型无需修改工厂代码(开闭原则):
LayerRegistry::reg<FlashAttention>("flash_attn");
LayerRegistry::reg<PagedAttention>("paged_attn");
```

**工厂模式的价值：**
- 解耦创建和使用（用户不需要知道具体类）
- 方便扩展新类型（不修改已有代码）
- 支持配置驱动（根据config字符串创建对象）
- AI框架中广泛使用：TensorRT plugin、ONNX Runtime、PyTorch算子注册

### Q: C++模板编程的作用和特点？

**模板的三个层次：**

```cpp
// Level 1: 泛型函数/类(最基本)
template<typename T>
T max(T a, T b) { return a > b ? a : b; }
// 编译器为每种T生成一份代码: max<int>, max<float>, ...

template<typename T, int N>
class FixedArray { T data[N]; };  // 编译期确定大小
FixedArray<float, 1024> buf;      // 栈上分配,无堆开销

// Level 2: SFINAE / Concepts(类型约束)
// C++20 Concepts:
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<Numeric T>
T add(T a, T b) { return a + b; }
// add("hello", "world");  // 编译错误! string不满足Numeric

// Level 3: 模板元编程(编译期计算)
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};
template<>
struct Factorial<0> {
    static constexpr int value = 1;
};
// Factorial<10>::value 在编译时计算完毕 = 3628800
```

**CRTP(Curiously Recurring Template Pattern) — 静态多态：**

```cpp
// 替代虚函数的零开销方案:
template<typename Derived>
class KernelBase {
public:
    void launch() {
        static_cast<Derived*>(this)->impl();  // 编译期确定调用
    }
};

class ReduceKernel : public KernelBase<ReduceKernel> {
public:
    void impl() { /* Reduce实现 */ }
};

// vs 虚函数:
// 虚函数: 运行时查vtable → 间接调用 → 无法内联
// CRTP: 编译时确定类型 → 直接调用 → 可内联 → 零开销!
```

**模板在CUDA/HPC中的应用：**
- 编译期确定tile大小/block配置
- 零开销的kernel参数化(不同精度/不同策略同一模板)
- cuBLAS/cutlass中大量使用模板元编程生成特化kernel

### Q: C++右值引用的应用场景？

**四大核心应用场景：**

```cpp
// 1. 移动语义(避免深拷贝)
class Tensor {
    float* data; size_t size;
public:
    Tensor(Tensor&& other) noexcept  // 移动构造
        : data(other.data), size(other.size) {
        other.data = nullptr;  // 窃取资源
    }
};
std::vector<Tensor> tensors;
tensors.push_back(Tensor(1000000));  // 移动! 不拷贝100万float

// 2. 完美转发(通用包装函数)
template<typename... Args>
auto make_op(Args&&... args) {
    return std::make_unique<Op>(std::forward<Args>(args)...);
    // forward保持每个参数的左值/右值属性
}

// 3. 返回值优化(RVO配合移动)
Tensor create_tensor(int n) {
    Tensor t(n);
    // ... 填充数据 ...
    return t;  // 编译器: RVO(零拷贝) 或 移动(如果RVO不适用)
}

// 4. emplace构造(避免临时对象)
std::vector<std::pair<std::string, Tensor>> cache;
cache.emplace_back("layer_0", Tensor(4096));
// 直接在vector内部构造pair, 不创建临时对象再移动
```

### Q: C++ static变量的初始化规则？

**初始化的三个阶段：**

```cpp
// 阶段1: 零初始化(程序加载时, 所有static变量先清零)
static int x;       // x = 0
static float* p;    // p = nullptr
static bool flag;   // flag = false

// 阶段2: 常量初始化(编译时可求值的初始化)
static const int MAX = 1024;         // 编译时确定
static constexpr double PI = 3.14;   // 编译时确定

// 阶段3: 动态初始化(运行时, main之前)
static std::string name = "hello";   // 需要调用构造函数
static auto config = load_config();  // 需要运行时函数调用
```

**Static Initialization Order Fiasco：**

```cpp
// file_a.cpp:
int compute_value();
static int a = compute_value();  // 动态初始化

// file_b.cpp:
extern int a;
static int b = a + 1;  // 依赖a, 但a可能还没初始化!

// 不同编译单元的动态初始化顺序是未定义的!
// b可能是0+1=1(a还没初始化) 或 正确值+1

// 解决: Construct on First Use (Meyers Singleton)
int& get_a() {
    static int a = compute_value();  // 首次调用时初始化
    return a;
}
// file_b.cpp:
static int b = get_a() + 1;  // 保证a已初始化
```

### Q: C++内存泄漏的原因与解决？

**常见泄漏场景和解决：**

| 场景 | 原因 | 解决方案 |
|------|------|---------|
| 忘记delete | new后提前return或throw | RAII(unique_ptr) |
| 异常路径泄漏 | catch后未清理 | RAII(析构函数自动清理) |
| 循环引用 | shared_ptr A→B, B→A | weak_ptr打破环 |
| 基类析构非virtual | delete基类指针不调用派生析构 | virtual ~Base() |
| 容器存裸指针 | 容器清空但指针指向的对象没释放 | 存智能指针 |
| 数组delete错误 | new[]配delete(非delete[]) | 使用vector代替裸数组 |

**检测工具：**

```bash
# Valgrind Memcheck(最权威)
valgrind --leak-check=full ./my_app
# 输出: 泄漏字节数、泄漏位置(哪行new的)

# AddressSanitizer(编译时插桩, 更快)
g++ -fsanitize=address -g my_app.cpp
./a.out
# 输出: 泄漏检测 + 使用已释放内存 + 缓冲区溢出

# LeakSanitizer(ASan的子集, 只检测泄漏)
g++ -fsanitize=leak -g my_app.cpp
```

**RAII原则的实践：**

```cpp
// Rule of Zero: 如果所有成员都是RAII类型,不需要自定义析构
class SafeModel {
    std::unique_ptr<float[]> weights;  // 自动释放
    std::vector<Layer> layers;          // 自动释放
    std::shared_ptr<KVCache> cache;     // 引用计数管理
    // 无需手写析构函数、拷贝/移动构造!
};
```

### Q: 有符号和无符号字符的最值？

```cpp
// signed char: 8-bit补码表示
// 范围: [-128, 127]
// 最小: 10000000b = -128
// 最大: 01111111b = 127

// unsigned char: 8-bit无符号
// 范围: [0, 255]
// 最小: 00000000b = 0
// 最大: 11111111b = 255

// 常见陷阱:
signed char a = 127;
a++;  // 未定义行为(UB)! 有符号溢出

unsigned char b = 0;
b--;  // 定义行为: wrap around → 255

// 混合比较陷阱:
int x = -1;
unsigned int y = 1;
if (x < y) { ... }  // 可能为false!
// x被隐式转换为unsigned: (unsigned)-1 = 4294967295 > 1
// 编译器会警告: -Wsign-compare
```

**在量化中的应用：**
- INT8量化: signed范围[-128, 127], 对称量化用[-127, 127]
- UINT8: 某些框架的激活量化用[0, 255]
- 量化计算: INT8×INT8→INT32避免溢出

### Q: 链接两个库中有同名函数会怎样？

**不同链接类型的行为：**

```
场景: libA.a 和 libB.a 都定义了 void helper()

静态链接:
  g++ main.o -lA -lB
  链接器按顺序搜索: 先找到libA中的helper → 使用它
  libB中的helper被忽略(除非libB中其他符号也被需要时)
  
  如果都是强符号(非static的函数定义):
  g++ main.o libA.o libB.o  → 链接错误: multiple definition!

动态链接:
  运行时按加载顺序搜索符号(默认RTLD_GLOBAL):
  先dlopen的库中的符号优先
  后加载的同名符号被遮盖(shadowed)
```

**解决方案：**

```cpp
// 1. 命名空间隔离(最推荐)
namespace libA { void helper(); }
namespace libB { void helper(); }

// 2. static限制链接性
static void helper() { ... }  // 只在本文件可见

// 3. 版本脚本(Linux .map文件)
// libA.map:
// { global: libA_*; local: *; };
// 只导出libA_前缀的符号,helper变为local

// 4. dlopen隔离
void* handle = dlopen("libB.so", RTLD_LOCAL);  // RTLD_LOCAL!
auto func = (void(*)())dlsym(handle, "helper");
// libB的helper不污染全局符号表

// 5. 弱符号(weak symbol)
__attribute__((weak)) void helper() { /* 默认实现 */ }
// 强符号定义覆盖弱符号, 不会报多重定义错误
```

**C++中的ODR(One Definition Rule)：**
- 同一符号在整个程序中只能有一个定义
- 违反ODR是未定义行为(不一定报错,可能默默使用错误版本)
- inline函数/模板可以在多个编译单元中有定义(但必须完全相同)
