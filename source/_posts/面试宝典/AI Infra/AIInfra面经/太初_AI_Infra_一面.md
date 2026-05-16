---
title: '太初 AI Infra 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: C++如何管理内存？

C++的内存管理分为**自动管理**（栈）和**手动管理**（堆），理解各区域特点是避免内存问题的基础。

**内存区域划分：**

| 区域 | 分配方式 | 生命周期 | 大小 | 特点 |
|------|---------|---------|------|------|
| 栈(Stack) | 编译器自动 | 函数作用域 | 有限(2-8MB) | 极快(移动SP)，LIFO |
| 堆(Heap) | new/malloc手动 | 手动控制 | 几乎无限 | 较慢(搜索空闲块)，易碎片 |
| 全局/静态区 | 编译器 | 程序全程 | 固定 | 全局变量、static变量 |
| 常量区(.rodata) | 编译器 | 程序全程 | 固定 | 字符串字面量，只读 |
| 代码区(.text) | 编译器 | 程序全程 | 固定 | 机器指令 |

**现代C++内存管理最佳实践——RAII + 智能指针：**

```cpp
// 不推荐：手动管理
int* p = new int[100];
// ... 如果中间抛异常，内存泄漏！
delete[] p;

// 推荐：RAII自动管理
auto p = std::make_unique<int[]>(100);  // 析构时自动释放
// 或
std::vector<int> v(100);  // 容器管理内存
```

**常见内存问题及防范：**
- 内存泄漏：使用智能指针、RAII、Valgrind/ASan检测
- 悬垂指针：使用weak_ptr观察、不返回局部变量地址
- 双重释放：unique_ptr保证唯一所有权
- 缓冲区溢出：使用容器替代裸数组、`at()`替代`[]`

### Q: 共享内存了解吗？

"共享内存"在不同上下文中有两个含义：

**1. 操作系统IPC共享内存（进程间通信）：**
- 多个进程将同一块物理内存映射到各自的虚拟地址空间
- **最快的IPC方式**（无需经过内核拷贝，直接读写）
- 需要配合同步机制（信号量/互斥锁）防止竞态条件

```cpp
// Linux POSIX共享内存
int fd = shm_open("/my_shm", O_CREAT | O_RDWR, 0666);
ftruncate(fd, SIZE);
void* ptr = mmap(NULL, SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
// 多个进程mmap同一个shm对象即可共享
```

**2. CUDA GPU共享内存（Shared Memory）——SM内的高速SRAM：**

| 属性 | GPU共享内存 | GPU全局内存(HBM) |
|------|-----------|-----------------|
| 位置 | SM片上SRAM | 片外HBM |
| 大小 | 192KB/SM (A100) | 80GB (A100) |
| 延迟 | ~5ns (~28 cycles) | ~400ns (~200+ cycles) |
| 带宽 | ~19 TB/s (全芯片) | 2 TB/s |
| 作用域 | Block内所有线程共享 | 所有线程可见 |
| 管理方式 | 程序员显式管理(__shared__) | 自动 |

```cuda
__global__ void kernel() {
    __shared__ float smem[256];  // Block内所有线程共享
    smem[threadIdx.x] = global_data[idx];  // 加载到共享内存
    __syncthreads();  // 同步，确保所有线程加载完成
    // 后续从smem读取，速度快80倍
}
```

**CUDA共享内存的核心作用：**
- 作为程序员管理的**L1 cache**
- 数据复用：一次从HBM加载，多次从共享内存读取（如GEMM tiling）
- Block内线程通信：线程间交换数据（如reduce操作）
- 避免重复全局内存访问

### Q: 智能指针的原理？能管理多个连续空间吗？

**三种智能指针的实现原理：**

| 指针 | 所有权 | 内部实现 | 开销 | 线程安全 |
|------|--------|---------|------|---------|
| unique_ptr | 独占 | 裸指针 + deleter | 零开销（与裸指针相同） | N/A（不共享） |
| shared_ptr | 共享 | 裸指针 + 控制块(引用计数) | 控制块+原子操作 | 引用计数原子操作是线程安全的 |
| weak_ptr | 观察 | 裸指针 + 控制块(弱引用计数) | 同shared_ptr | 同上 |

**shared_ptr控制块结构：**
```
┌──────────────────┐
│  strong_count    │  ← shared_ptr个数（归零时释放对象）
│  weak_count      │  ← weak_ptr个数（归零时释放控制块）
│  deleter         │  ← 自定义释放函数
│  allocator       │  ← 自定义分配器
└──────────────────┘
```

**make_shared的优势：** 一次内存分配同时分配对象和控制块（连续内存），而`shared_ptr<T>(new T())`需要两次分配。

**管理连续空间（数组）：**

```cpp
// unique_ptr管理数组——原生支持
auto arr = std::make_unique<int[]>(100);  // C++14
arr[0] = 42;  // operator[]可用
// 析构时自动调用delete[]

// shared_ptr管理数组
// C++17起：
auto arr = std::make_shared<int[]>(100);
// C++17前需要自定义deleter：
std::shared_ptr<int> arr(new int[100], [](int* p){ delete[] p; });
// 或用std::default_delete<int[]>：
std::shared_ptr<int> arr(new int[100], std::default_delete<int[]>());
```

**注意事项：**
- shared_ptr<int[]>（C++17）支持operator[]，shared_ptr<int>（自定义deleter）不支持
- 优先使用`std::vector`管理连续空间（更安全、更方便）
- 只有在需要共享所有权或自定义deleter时才用智能指针管理数组

### Q: C++多态有哪些？

**多态分为编译期多态和运行期多态：**

**1. 编译期多态（静态多态）——编译时确定调用：**

```cpp
// 函数重载
void print(int x);     // 编译器根据参数类型选择
void print(double x);

// 模板
template<typename T>
T max(T a, T b) { return a > b ? a : b; }
// 编译时为每种类型生成特化版本

// CRTP（奇异递归模板模式）——静态多态替代虚函数
template<typename Derived>
class Base {
    void interface() { static_cast<Derived*>(this)->impl(); }
};
class Derived : public Base<Derived> {
    void impl() { /* 具体实现 */ }
};
```

**2. 运行期多态（动态多态）——运行时确定调用：**

```cpp
class Shape {
public:
    virtual double area() = 0;  // 纯虚函数
    virtual ~Shape() {}
};
class Circle : public Shape {
    double area() override { return 3.14 * r * r; }
};

Shape* s = new Circle();
s->area();  // 运行时通过vtable查找Circle::area()
```

**两种多态的对比：**

| 维度 | 编译期多态 | 运行期多态 |
|------|-----------|-----------|
| 性能 | 零开销（内联优化） | 有vtable间接跳转开销 |
| 灵活性 | 类型编译时确定 | 运行时可切换 |
| 容器 | 不同类型不能放同一容器 | 基类指针容器可存不同派生类 |
| 代码膨胀 | 模板实例化导致代码膨胀 | 无 |
| 可扩展性 | 添加新类型需重编译 | 添加新派生类无需修改已有代码 |
| 典型使用 | STL容器/算法、数值计算库 | 插件系统、GUI框架、游戏引擎 |

### Q: 虚函数的实现原理？

**虚函数通过vtable（虚函数表）+ vptr（虚表指针）实现运行时多态。**

**内存布局：**
```
对象内存:
┌──────────┐
│  vptr    │ ──→  vtable (类级别，所有对象共享)
│  成员1   │      ┌────────────────┐
│  成员2   │      │ &TypeInfo      │  RTTI信息
│  ...     │      │ &虚函数1实现    │  slot 0
└──────────┘      │ &虚函数2实现    │  slot 1
                  │ ...            │
                  └────────────────┘
```

**动态分派过程（通过基类指针调用虚函数）：**
```
Base* ptr = new Derived();
ptr->virtualFunc();
// 编译器生成的代码等价于：
// ptr->vptr[slot_of_virtualFunc]()  // 间接函数调用
```

**继承时vtable的变化：**
```cpp
class Base { virtual void f(); virtual void g(); };
class Derived : public Base { void f() override; };  // 重写f

Base的vtable:    [&Base::f,    &Base::g]
Derived的vtable: [&Derived::f, &Base::g]  // f槽位被覆盖，g继承
```

**性能开销：**
- 一次间接跳转（读vptr → 读vtable → 跳转），约2-3个时钟周期
- 阻止内联优化（编译器无法在编译时确定目标函数）
- 在热路径上（如每帧调用百万次），可考虑CRTP静态多态替代

### Q: 构造函数里面调用虚函数会怎样？

**不会触发多态——只会调用当前正在构造的类的版本。**

**原因：** 构造顺序是从基类到派生类，基类构造时派生类部分尚未初始化。

```cpp
class Base {
public:
    Base() { 
        whoAmI();  // 调用Base::whoAmI()，不是Derived版本
    }
    virtual void whoAmI() { cout << "Base" << endl; }
};

class Derived : public Base {
    int* data;
public:
    Derived() : data(new int[100]) {}
    void whoAmI() override { 
        cout << "Derived, data=" << data << endl;  // 如果被调用，data未初始化！
    }
};

Derived d;  // 输出"Base"，而非"Derived"
```

**底层机制：**
1. 构造Derived时先调用Base构造函数
2. Base构造函数开始时，vptr被设置为指向**Base的vtable**
3. Base构造中调用虚函数 → 通过vptr查Base的vtable → 调用Base版本
4. Base构造完成后，进入Derived构造，vptr更新为指向**Derived的vtable**

**这是C++的安全设计：** 如果基类构造时调用了派生类的虚函数，而派生类成员尚未初始化，会访问未初始化的内存——未定义行为。

**析构函数中同理：** Derived析构先执行（成员已销毁），然后Base析构时vptr已回退到Base的vtable，虚函数调用不会分派到Derived。

### Q: GPU的架构？

**NVIDIA GPU架构层次（以A100为例）：**

```
GPU芯片
├── GPC (Graphics Processing Cluster) × 7
│   ├── TPC (Texture Processing Cluster) × 2
│   │   └── SM (Streaming Multiprocessor) × 2  [总共108个SM]
│   │       ├── 4个Processing Block，每个包含:
│   │       │   ├── 16个FP32 CUDA Core [总共64/SM]
│   │       │   ├── 16个INT32 Core
│   │       │   ├── 1个Tensor Core [总共4/SM，第三代]
│   │       │   ├── 1个Warp Scheduler + Dispatch Unit
│   │       │   └── Register File (16384 × 32bit)
│   │       ├── 共享内存/L1 Cache: 192KB (可配置划分)
│   │       ├── L0 Instruction Cache
│   │       └── 4个SFU (特殊功能单元: sin/cos/sqrt)
├── L2 Cache: 40MB (全SM共享)
├── HBM2e Controller: 5个 × 2通道 (80GB, 2039 GB/s)
├── NVLink: 12个 (600 GB/s双向)
└── PCIe Gen4 x16
```

**关键数值（A100 SXM）：**

| 组件 | 数量/规格 | 用途 |
|------|-----------|------|
| SM | 108个 | 基本计算单元 |
| FP32 Core | 6912个 | 单精度浮点计算 |
| Tensor Core (3rd Gen) | 432个 | 矩阵乘加加速 |
| 寄存器 | 65536/SM × 32bit | 线程私有最快存储 |
| 共享内存 | 192KB/SM | Block内共享高速缓存 |
| L2 Cache | 40MB | 全局缓存 |
| HBM | 80GB, 2039GB/s | 显存 |

**编程模型映射：**
- Grid → 整个GPU
- Block → 一个SM上执行（受资源限制可能多个block共享SM）
- Warp(32 threads) → SM的基本调度单位
- Thread → CUDA Core执行

### Q: Reduce优化怎么做？

GPU上的Reduce（规约）操作优化是经典的并行算法问题。以sum reduce为例，按优化递进：

**优化级别递进：**

**Level 0: 朴素实现（严重的warp divergence）**
```cuda
// 问题：交错索引导致活跃线程不连续
for (int s = 1; s < blockDim.x; s *= 2)
    if (tid % (2*s) == 0) sdata[tid] += sdata[tid + s];
// 只有偶数线程工作，warp内一半线程idle
```

**Level 1: 连续线程工作（消除warp divergence）**
```cuda
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
// 前半线程连续工作，同一warp内线程要么都工作要么都idle
```

**Level 2: 第一次加载时做计算（减少idle线程）**
```cuda
// 每个线程加载两个元素并相加
sdata[tid] = input[i] + input[i + blockDim.x];
// block处理的数据量翻倍，idle线程减半
```

**Level 3: Warp-level优化（最后32个元素无需__syncthreads）**
```cuda
// 当活跃线程 <= 32时，在同一个warp内，硬件保证同步
if (tid < 32) {
    // Warp shuffle替代共享内存
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
}
```

**Level 4: 向量化加载 + 完整优化**
```cuda
// float4加载：一次读取128bit
float4 val4 = reinterpret_cast<float4*>(input)[tid];
float local_sum = val4.x + val4.y + val4.z + val4.w;
// 然后做warp shuffle + shared memory reduce
```

**性能目标：** 优化后的reduce应该达到HBM带宽的90%+（因为reduce是纯memory-bound操作）。

### Q: GEMM优化怎么做？

GEMM优化的核心是**多级分块(Hierarchical Tiling) + 数据复用最大化**。

**分块层次（从大到小）：**

```
全局内存层: C[M,N] = A[M,K] × B[K,N]
    ↓ Block Tile (Thread Block负责)
共享内存层: C_tile[BM, BN] += A_tile[BM, BK] × B_tile[BK, BN]
    ↓ Warp Tile (Warp负责)
寄存器层: C_frag[WM, WN] += A_frag[WM, 1] × B_frag[1, WN]  (外积)
    ↓ Thread Tile (单线程计算)
计算: c[TM, TN] += a[TM] * b[TN]  (每线程8×8=64个FMA)
```

**关键优化技术：**

| 技术 | 目的 | 实现方式 |
|------|------|---------|
| 共享内存Tiling | 减少HBM访问 | 加载A/B的小块到共享内存复用 |
| 双缓冲(Double Buffer) | 隐藏加载延迟 | Ping-pong缓冲区，加载与计算重叠 |
| 寄存器Tiling | 最大化数据复用 | 每线程8×8输出，外积方式计算 |
| 避免Bank Conflict | 消除共享内存串行化 | Padding(+1列)或Swizzle |
| Tensor Core | 硬件加速MMA | wmma::mma_sync或PTX mma指令 |
| 向量化加载 | 提高带宽利用 | float4/LDS.128一次加载128bit |
| 循环展开 | 提高ILP | #pragma unroll内层K循环 |

**典型性能层次：**
- 朴素实现：峰值的2-5%
- 共享内存tiling：峰值的30-50%
- +寄存器tiling+向量化：峰值的70-80%
- +Tensor Core+双缓冲+bank conflict消除：峰值的85-95%（Cutlass水平）
