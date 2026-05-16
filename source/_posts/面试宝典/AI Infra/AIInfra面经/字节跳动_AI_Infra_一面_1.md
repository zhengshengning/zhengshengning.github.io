---
title: '字节跳动 AI Infra 一面 (1)'
date: 2026-04-16 12:17:20
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: C++虚函数的原理？

虚函数是C++实现运行时多态的核心机制，通过间接寻址实现"基类指针调用派生类方法"：

**实现机制（底层）**：

```cpp
class Base {
public:
    virtual void foo() { cout << "Base"; }    // vtable[0]
    virtual void bar() { cout << "Base"; }    // vtable[1]
    int data;
};

class Derived : public Base {
public:
    void foo() override { cout << "Derived"; } // 替换vtable[0]
    // bar() 继承，vtable[1]仍指向Base::bar
};
```

编译器为每个含虚函数的类生成一个**虚函数表（vtable）**：

```
Base的vtable:                 Derived的vtable:
┌───────────────────┐        ┌───────────────────┐
│ &Base::foo        │  [0]   │ &Derived::foo     │  [0] ← override
│ &Base::bar        │  [1]   │ &Base::bar        │  [1] ← 继承
└───────────────────┘        └───────────────────┘
```

每个对象的内存布局：
```
Base对象 (16 bytes on 64-bit):     Derived对象 (16 bytes):
┌──────────┐                        ┌──────────┐
│ vptr (8B)│ → 指向Base的vtable     │ vptr (8B)│ → 指向Derived的vtable
├──────────┤                        ├──────────┤
│ data (4B)│                        │ data (4B)│
├──────────┤                        ├──────────┤
│ padding  │                        │ padding  │
└──────────┘                        └──────────┘
```

**虚函数调用过程**（`base_ptr->foo()`）：
```
1. 读取对象的vptr（对象首8字节）
2. 从vtable中按偏移取函数地址：addr = vptr[0]（foo是第0个虚函数）
3. 通过函数指针间接调用：call addr
```

等效汇编（伪代码）：
```asm
mov rax, [rdi]        ; rax = *this（即vptr）
call [rax + 0]        ; 调用vtable[0]指向的函数
```

**性能开销分析**：

| 开销项 | 代价 | 对比非虚函数 |
|--------|------|------------|
| 对象大小 | +8字节(vptr) | 多8B/对象 |
| 调用延迟 | 多1次内存读取(vptr→vtable) | 直接call vs 间接call |
| 内联优化 | 无法内联（编译期不知道调用目标） | 可内联 |
| Cache影响 | vtable可能不在L1 cache | 无额外cache pressure |

实际影响：对于热路径中的小函数（如虚析构），性能差异可达10-20%。对于大函数（如operator forward），调用开销可忽略。

**常见面试追问**：
- **纯虚函数**：`virtual void foo() = 0;`——类变为抽象类，不能实例化。vtable中该位置为0或__cxa_pure_virtual。
- **虚析构函数**：基类必须virtual析构，否则`delete base_ptr`不会调用派生类析构（内存泄漏）。
- **多重继承**：对象有多个vptr（每个基类一个），vtable变为多组。
- **RTTI**：vtable中还包含type_info指针，`dynamic_cast`和`typeid`依赖它。

### Q: C++内存对齐的规则？

**内存对齐规则（x86-64默认，#pragma pack(8)）**：

**规则1：成员对齐**——每个成员的起始偏移必须是 `min(成员自身大小, pack值)` 的整数倍。

**规则2：结构体总大小**——必须是 `min(最大成员大小, pack值)` 的整数倍（尾部padding）。

**规则3：嵌套结构体**——按其最大成员的对齐值参与外层对齐计算。

```cpp
struct Example {
    char a;    // 偏移0, 大小1  → [0]
               // padding 7字节 (下一成员double需要8对齐)
    double b;  // 偏移8, 大小8  → [8-15]
    int c;     // 偏移16, 大小4 → [16-19]
    char d;    // 偏移20, 大小1 → [20]
               // padding 3字节 (总大小需要8的倍数)
};  // sizeof = 24

// 重排优化（大到小排列）
struct Optimized {
    double b;  // 偏移0, 大小8  → [0-7]
    int c;     // 偏移8, 大小4  → [8-11]
    char a;    // 偏移12, 大小1 → [12]
    char d;    // 偏移13, 大小1 → [13]
               // padding 2字节
};  // sizeof = 16 (节省33%!)
```

**为什么需要对齐**：

1. **硬件效率**：CPU通过数据总线（64位/8字节宽度）读取内存。未对齐的8字节double可能跨两次总线事务读取（性能减半或更差）。
2. **原子性**：对齐的自然大小访问在x86上保证原子性。未对齐的变量修改可能在两次写入之间被其他线程观察到不完整状态。
3. **SIMD要求**：SSE/AVX指令要求16/32字节对齐。`__m256`类型必须32字节对齐，否则段错误。
4. **Cache效率**：对齐到cache line（64B）可避免false sharing。

**AI框架中的对齐要求**：
- GPU全局内存：128字节对齐确保coalesced access的最大效率。
- Tensor Core：部分操作要求256字节对齐。
- `cudaMalloc`返回的指针保证256字节对齐。
- `torch::Tensor`的storage保证至少64字节对齐。

**控制对齐**：
```cpp
#pragma pack(push, 1)  // 紧凑排列（序列化/网络协议用）
struct Packed { ... };
#pragma pack(pop)

alignas(64) float cache_line_aligned_array[16];  // 64字节对齐（避免false sharing）
```

### Q: C++动态库和静态库的区别？

**全面对比**：

| 维度 | 静态库(.a/.lib) | 动态库(.so/.dll/.dylib) |
|------|----------------|----------------------|
| 链接时机 | 编译时嵌入 | 运行时加载 |
| 最终可执行文件 | 包含库代码（大） | 不包含（小） |
| 更新方式 | 必须重新编译 | 替换.so即可（ABI兼容时） |
| 内存占用 | 每个进程各一份库代码 | 多进程共享同一份（通过mmap） |
| 启动速度 | 快（无动态链接） | 慢（需要resolve符号） |
| 符号解析 | 编译时完全解析 | 运行时通过PLT/GOT跳转 |
| 部署复杂度 | 简单（单一二进制） | 需确保.so路径正确 |
| 版本管理 | 无（编译时已固定） | 需要ABI兼容/版本控制 |

**动态链接的底层机制**：

```
可执行文件调用 printf():
  1. 首次调用: call PLT[printf]
  2. PLT跳转到GOT[printf]（首次为dynamic linker地址）
  3. Dynamic linker解析printf的实际地址，写入GOT[printf]
  4. 后续调用: PLT→GOT直接跳转到实际地址（无linker开销）
```

PLT（Procedure Linkage Table）+ GOT（Global Offset Table）实现延迟绑定：
- 首次调用有~100ns解析开销。
- 后续调用仅多一次间接跳转（~1ns，通常可被branch predictor覆盖）。

**AI框架中的选择**：

| 场景 | 选择 | 原因 |
|------|------|------|
| CUDA Runtime (libcudart.so) | 动态库 | 多版本共存、驱动兼容 |
| cuDNN/cuBLAS | 动态库 | 频繁更新、多应用共享 |
| PyTorch C++ extension | 动态库(.so) | torch.ops动态加载 |
| 嵌入式/车载部署 | 静态库 | 减少依赖、确定性 |
| 高频交易系统 | 静态库 | 消除PLT跳转延迟 |

**常见问题**：
- `LD_LIBRARY_PATH`：指定动态库搜索路径。
- `ldd binary`：查看二进制的动态库依赖。
- Symbol versioning：同一.so可包含多个版本的符号实现。
- `-fPIC`：编译动态库时必须使用Position-Independent Code（使代码可加载到任意地址）。
