---
title: '小鹏汽车 AI Infra 一面'
date: 2026-04-16 12:11:26
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: C++中const的作用范围有哪些？

const是C++中表达"不可修改"语义的关键字，在不同上下文中有不同含义和编译期行为：

**1. 修饰变量（顶层const）**：
```cpp
const int x = 42;  // x的值不可修改，必须初始化
// x = 43;  // 编译错误
```
编译器可能将const整型放入符号表而非分配内存（类似#define但有类型安全）。

**2. 修饰指针（底层const vs 顶层const）**：
```cpp
const int* p1;       // 指向常量的指针：*p1不可修改，p1可修改（底层const）
int* const p2 = &a;  // 常量指针：p2不可修改，*p2可修改（顶层const）
const int* const p3 = &a;  // 双重const
```
助记法：从右向左读——`const int*` = "pointer to const int"。

**3. 修饰函数参数**：
```cpp
void process(const std::string& s);  // 承诺不修改s，且避免拷贝开销
void update(const int* data, int n); // 不修改data指向的内容
```
const引用传参（`const T&`）是最常用的参数传递方式：无拷贝开销 + 可接受右值 + 承诺不修改。

**4. 修饰成员函数（const限定）**：
```cpp
class Widget {
    int getValue() const;  // 承诺不修改对象状态，可被const对象调用
    // 内部this指针变为 const Widget*
};
const Widget w;
w.getValue();  // OK：const对象只能调用const成员函数
```
这是C++接口设计中表达"查询操作"vs"修改操作"的核心手段。

**5. 修饰返回值**：
```cpp
const std::string& getName() const;  // 返回const引用，防止外部通过返回值修改内部状态
```

**6. constexpr（编译期常量）**：
```cpp
constexpr int square(int x) { return x * x; }
constexpr int val = square(5);  // 编译时求值为25
```
C++11引入constexpr，C++14/17逐步放宽约束，C++20支持constexpr容器和动态分配。

**const与线程安全**：C++标准库将const成员函数视为线程安全的（多个线程可以同时调用const方法），这是C++11的重要约定。

### Q: 动态链接库的依赖顺序问题？

动态链接库的加载顺序是C/C++项目中常见的"隐蔽bug"来源，需要理解链接器和加载器的工作机制：

**链接时（ld linker）的符号解析规则**：
- 从左到右扫描命令行中的库文件。
- 遇到未定义符号时，在后续库中查找定义。
- 被依赖的库必须放在依赖者的**右边**（后面）。
```bash
# 正确：main.o依赖libA，libA依赖libB
gcc main.o -lA -lB
# 错误：libB在前面，找不到libA中的符号
gcc main.o -lB -lA  # undefined reference错误
```
- 循环依赖解决：`-Wl,--start-group -lA -lB --end-group`（强制多次扫描）。

**运行时（ld.so dynamic linker）的搜索顺序**：
1. `RPATH`（编译时嵌入ELF中，`-Wl,-rpath,/path`）
2. `LD_LIBRARY_PATH` 环境变量
3. `RUNPATH`（优先级低于LD_LIBRARY_PATH）
4. `/etc/ld.so.cache`（ldconfig生成的缓存）
5. 默认路径：`/lib`、`/usr/lib`、`/lib64`、`/usr/lib64`

**常见问题与调试工具**：
- `ldd binary`：查看所有动态库依赖及实际解析路径。
- `nm -D libfoo.so`：查看动态符号表。
- `LD_DEBUG=libs ./binary`：打印加载过程详细信息。
- `ldconfig`：更新共享库缓存。
- `chrpath`/`patchelf`：修改已编译binary的RPATH。

**常见陷阱**：
- 符号冲突（symbol interposition）：先加载的库的同名符号会覆盖后加载的。
- ABI兼容性：库升级后如果ABI变了（如类增加了成员变量），需要重新编译使用者。
- dlopen的RTLD_GLOBAL vs RTLD_LOCAL：控制符号是否对后续dlopen可见。

### Q: C++四种Cast的区别与底层含义？

C++提供四种显式类型转换运算符，取代C风格的`(Type)expr`强转，每种有明确的语义和安全级别：

**1. static_cast —— 编译期已知转换**
```cpp
double d = 3.14;
int i = static_cast<int>(d);         // 数值转换（截断）
Base* b = static_cast<Base*>(derived_ptr);  // 向上转型（安全）
Derived* d = static_cast<Derived*>(base_ptr);  // 向下转型（不安全，无检查）
```
- 底层：编译时进行类型检查和可能的代码生成（如浮点到整数的截断指令）。
- 适用：相关类型间的已知安全转换。不安全的向下转换可能导致UB（如base_ptr实际不指向Derived）。

**2. dynamic_cast —— 运行时安全转换**
```cpp
Base* b = getObject();
Derived* d = dynamic_cast<Derived*>(b);  // 成功返回有效指针，失败返回nullptr
Derived& dr = dynamic_cast<Derived&>(*b); // 失败抛std::bad_cast
```
- 底层：依赖RTTI（RunTime Type Information），查询虚函数表中的类型信息。沿继承链查找是否存在目标类型。
- 开销：需要遍历类型信息（最坏O(继承深度)），约比static_cast慢10-100倍。
- 限制：基类必须有虚函数（否则无RTTI信息）。

**3. const_cast —— 添加/移除const**
```cpp
const char* cs = "hello";
char* s = const_cast<char*>(cs);  // 去除const
// 但修改原本是const的对象是UB！
```
- 底层：不生成任何代码，纯编译器层面的类型标记修改。
- 唯一合法场景：当你确定原始数据不是const的，但API返回了const指针（如C接口兼容）。

**4. reinterpret_cast —— 按位重新解释**
```cpp
int* ip = ...;
char* cp = reinterpret_cast<char*>(ip);  // 以字节视角看int
uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);  // 指针转整数
```
- 底层：不生成任何代码，仅改变编译器对该地址中数据的类型解释。
- 最危险：几乎不做任何检查，误用会导致UB、对齐问题、alias violation。
- 合法使用：序列化（按字节读写对象）、底层系统编程、与C库交互。

**安全级别从高到低**：dynamic_cast > static_cast > const_cast > reinterpret_cast。

### Q: 给出CUDA Kernel代码，识别其功能？

分析CUDA Kernel的系统化方法：

**Step 1：看Kernel签名** —— 确定输入输出类型和维度
```cuda
__global__ void mystery_kernel(float* input, float* output, int N, int C, int H, int W)
// → 4D tensor操作，可能是图像处理或CNN相关
```

**Step 2：看线程索引计算** —— 确定并行维度和数据映射
```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;  // 一维线性索引 → 逐元素操作
int row = blockIdx.y * blockDim.y + threadIdx.y;   // 二维索引 → 矩阵操作
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

**Step 3：看内存访问模式** —— 确定数据流向
- 只有全局内存读 → element-wise或reduce
- 使用共享内存 → 需要线程协作（GEMM/转置/规约）
- 有原子操作 → 多线程竞争写（直方图/reduce）

**Step 4：看核心计算逻辑** —— 确定算法功能

常见kernel模式识别：
| 模式 | 特征 |
|------|------|
| 逐元素操作 | 线性tid索引，每线程处理一个元素 |
| 矩阵乘法 | 共享内存分块，双重循环累加 |
| 规约（Reduce） | 树形折半累加，`__syncthreads()`间隔减半 |
| 转置 | 共享内存+padding，写入时行列交换 |
| 卷积 | 多重循环遍历kernel窗口 |
| Softmax | 先求max（reduce），再exp和sum（reduce），最后归一化 |

**实战技巧**：关注 `__syncthreads()` 的位置（划分计算阶段）、shared memory的使用pattern（数据复用方式）、边界条件检查（if tid < N）。

### Q: 手撕：计算平面上两个任意角度矩形的重叠面积？

（编程题）
