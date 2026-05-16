---
title: '字节跳动 AI Infra 一面 (2)'
date: 2026-04-16 12:17:24
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: CUDA编程一般怎么优化？

CUDA优化的核心思路：根据kernel的bound类型选择对应策略，通过Profiling驱动迭代优化。

**1. 访存优化（Memory-bound kernel的首要任务）**：

**合并全局内存访问（Coalesced Access）**：
```cuda
// 好：相邻线程访问相邻地址 → 一次128字节事务
float val = data[blockIdx.x * blockDim.x + threadIdx.x];  // 连续

// 差：stride访问 → 每线程触发独立事务
float val = data[threadIdx.x * stride];  // stride>1时严重浪费
```
Warp内32线程访问128字节对齐的连续地址时，硬件合并为1次内存事务。非合并访问可能产生32次独立事务，带宽利用率降至1/32。

**共享内存缓存**：
```cuda
__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1消除bank conflict
// 协作加载 → 同步 → 计算阶段从shared memory读取
```
Global memory延迟~400 cycles，Shared memory延迟~20 cycles。数据复用N次时，总延迟从N×400降为400+N×20。

**向量化读写**：
```cuda
float4 vec = *reinterpret_cast<float4*>(&input[idx]);  // 128位单次传输
// 比4次float读取少3次指令+事务overhead
```
要求地址16字节对齐。FP16情况下用half2/half4。

**避免Bank Conflict**：
```cuda
// 32个bank，每bank连续4字节
__shared__ float smem[32][33];  // padding一列消除列访问时的conflict
```

**2. 计算优化（Compute-bound kernel的重点）**：

**利用Tensor Core**：
```cuda
// WMMA API (16×16×16 FP16矩阵乘)
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
// 单条指令 = 数百次FMA，吞吐16x于CUDA Core
```

**减少Warp Divergence**：
```cuda
// 差：同warp线程走不同分支
if (tid % 2 == 0) pathA(); else pathB();  // 串行化两条路径

// 好：相邻线程走相同分支
if (tid < N/2) pathA(); else pathB();  // divergence只在warp边界
```

**循环展开**：
```cuda
#pragma unroll 4  // 展开4次，暴露ILP
for (int i = 0; i < 16; i++)
    sum += a[i] * b[i];
```
展开后编译器可以交错不同迭代的指令，隐藏流水线延迟（指令级并行ILP）。

**快速数学函数**：`__expf()`/`__rsqrtf()`等用硬件SFU单元，比标准函数快5-10x（精度损失<2 ULP）。

**3. 并行度优化**：

**提高Occupancy**：
- 减少每线程寄存器用量：`__launch_bounds__(256, 2)` 限制编译器。
- 合理划分shared memory：不是越多越好（过多则block数受限）。
- Block大小选128或256（通常最优）。

**注意**：高Occupancy不是目标——如果增加Occupancy需要spill寄存器到local memory（400 cycle延迟），反而更慢。**真正的目标是最大化性能，Occupancy只是手段之一**。

**4. 延迟隐藏**：

**Double Buffering**：
```cuda
// 阶段1: 加载buffer_0
cp_async(smem_0, global_0);
// 阶段2: 加载buffer_1 + 计算buffer_0
cp_async(smem_1, global_1);
compute(smem_0);  // 计算和加载并行!
// 阶段3: 加载buffer_0 + 计算buffer_1
cp_async(smem_0, global_2);
compute(smem_1);
// ... 流水线持续
```

**5. 算子融合**：
- 合并多个小kernel减少：launch开销(~5us)×N次 + 中间tensor HBM写入/读取。
- 典型收益：3个小op融合后加速2-5x（对memory-bound操作）。

**6. Profiling驱动**：
- 不要靠猜测优化——先Profile确定瓶颈，再针对性优化。
- Nsight Compute的Roofline图一目了然地告诉你"离峰值还有多远、在哪个方向"。

---

### Q: 卷积如何优化？

卷积优化的核心是将其转化为高效的矩阵运算或利用数学恒等变换减少计算量：

**方法对比**：

| 方法 | 原理 | 适用场景 | 加速比(vs naive) | 缺陷 |
|------|------|---------|----------------|------|
| Im2col+GEMM | 展开为矩阵乘 | 通用 | 10-50x | 额外内存(9x for 3×3) |
| Winograd | 减少乘法次数 | 3×3, stride=1 | 比GEMM再快2-4x | 数值精度略差 |
| FFT | 频域卷积 | kernel>7×7 | 大kernel快 | 小kernel开销大 |
| 隐式GEMM | 不显式展开 | 通用(cuDNN默认) | 接近GEMM性能 | 索引计算复杂 |
| Direct | 直接滑窗 | 特殊形状 | 低 | 难以充分优化 |

**Im2col + GEMM详解**：
```
输入: [N, C_in, H, W]  kernel: [C_out, C_in, kH, kW]

展开后:
  输入矩阵: [N×H_out×W_out,  C_in×kH×kW]
  权重矩阵: [C_out,  C_in×kH×kW]
  输出 = 权重 × 展开输入^T → [C_out, N×H_out×W_out]
```
优势：直接调用cuBLAS的高度优化GEMM（利用Tensor Core）。
代价：3×3 conv展开后输入增大9倍——但对于中等特征图这通常可接受。

**Winograd F(m,r) 变换**：
- F(2,3)：2×2输出tile + 3×3 kernel → 在4×4输入tile上计算。
- 乘法次数：从3×3=9次/输出降为4×4/4=4次/输出（减少56%）。
- F(4,3)：进一步减少到4×4/(16-6)=2.25次/输出。
- 为什么cuDNN 3×3 conv默认选Winograd：在有限的精度损失下获得30-50%额外加速。
- 限制：stride>1或dilation>1时不适用；r越大数值稳定性越差。

**数据Layout对性能的影响**：
- NCHW（PyTorch默认）：channel连续，适合CPU SIMD（按channel向量化）。
- NHWC（TensorFlow/cuDNN推荐）：HWC连续，Tensor Core需要16字节对齐的连续channel数据。
- A100上NHWC比NCHW快10-30%（Tensor Core更高效利用）。

**cuDNN自动选择最优算法**：
```cpp
cudnnFindConvolutionForwardAlgorithm(handle, ..., &algo);
// 或 cudnnGetConvolutionForwardAlgorithm_v7() 基于启发式快速选择
```
cuDNN会根据输入shape/kernel/stride自动选择Winograd/ImplicitGEMM/FFT等。实际部署中通常让cuDNN auto-tune（首次慢一点，后续用缓存）。

---

### Q: 共享内存的作用和使用？

共享内存（Shared Memory）是SM上的高速可编程SRAM，是CUDA优化中最关键的资源：

**物理特性（A100）**：

| 特性 | 数值 | 对比 |
|------|------|------|
| 容量/SM | 192KB（可配，与L1 Cache共享） | HBM: 80GB |
| 延迟 | ~20 cycles | HBM: ~400 cycles |
| 带宽 | ~19 TB/s/SM（理论） | HBM: 2 TB/s(全GPU) |
| Bank数 | 32 banks | - |
| Bank宽度 | 4 bytes/bank | - |
| 访问方式 | 程序员显式管理 | HBM通过L1/L2 Cache |

**典型使用模式**：

```cuda
__shared__ float smem[TILE_SIZE][TILE_SIZE + 1];  // +1 padding避免bank conflict

// 阶段1：协作加载（所有线程参与，一次加载一个tile）
smem[ty][tx] = global_data[row * width + col];
__syncthreads();  // 必须！确保所有线程加载完成

// 阶段2：计算（从shared memory读取，无HBM延迟）
for (int k = 0; k < TILE_SIZE; k++)
    sum += smem[ty][k] * other_smem[k][tx];
__syncthreads();  // 必须！确保所有线程读取完成才能进入下一轮加载
```

**Bank Conflict详解**：
```
32个bank，每bank连续4字节:
  地址 0-3   → Bank 0
  地址 4-7   → Bank 1
  ...
  地址 124-127 → Bank 31
  地址 128-131 → Bank 0 (循环)

无conflict: warp内32线程各访问不同bank（1 cycle）
2-way conflict: 2线程访问同一bank不同行 → 串行化为2次（2 cycle）
32-way conflict: 全部访问同一bank → 串行化32次！

// 例: smem[32][32]按列访问
smem[threadIdx.x][col];  // threadIdx.x=0,1,...,31 访问同一列不同行
// 32个线程访问 smem[0][col], smem[1][col], ..., smem[31][col]
// stride=32*4=128 bytes, 正好跨32 banks → 无conflict!

// 但按行stride=4 访问: smem[row][threadIdx.x*stride]时可能冲突
// 解决: smem[32][33]  padding使stride不是32的倍数
```

**动态分配**：
```cuda
extern __shared__ float dynamic_smem[];  // 声明动态共享内存
kernel<<<grid, block, smem_bytes>>>();   // launch时指定大小
```

**Shared Memory vs L1 Cache**：
- A100上两者共享192KB SRAM，可配置比例（如164KB shared + 28KB L1）。
- `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size)` 配置超过默认上限的shared memory。

**使用注意事项**：
- Shared memory用量直接影响每SM可驻留的Block数（进而影响Occupancy）。
- 一定要`__syncthreads()`确保写入可见（否则读到旧数据）。
- 不要在条件分支中放`__syncthreads()`（可能死锁——部分线程不执行sync）。

---

### Q: C的malloc和C++的new有什么区别？

**全面对比**：

| 维度 | malloc/free | new/delete |
|------|------------|-----------|
| 本质 | C标准库函数 | C++运算符（可重载） |
| 头文件 | `<stdlib.h>` | 无需（语言内置） |
| 类型安全 | 返回void*，需手动强转 | 返回正确类型指针 |
| 大小指定 | 需要`sizeof` | 自动计算 |
| 构造/析构 | 不调用 | 自动调用 |
| 失败行为 | 返回NULL | 抛`std::bad_alloc` |
| 重载 | 不可 | 可重载`operator new` |
| 数组 | 手动计算 | `new[]`/`delete[]` |
| 配对 | `malloc`↔`free` | `new`↔`delete` |

**底层实现（Linux）**：

```
new (或 malloc) 申请内存
    ↓
operator new() → malloc() → 
    ├── 小内存 (<128KB): brk() 扩展堆顶
    └── 大内存 (>=128KB): mmap() 分配独立页
```

两者最终都调用OS系统调用（brk/mmap），但new额外做了：
1. 调用`operator new(size)`分配内存（默认实现就是malloc）。
2. 在分配的内存上调用构造函数（placement new语义）。
3. 失败时调用new_handler或抛异常（而非返回NULL）。

**为什么混用会出问题**：
```cpp
// 危险！malloc分配的不会调用构造函数
std::string* s = (std::string*)malloc(sizeof(std::string));
// s指向的内存未初始化！使用s→内部指针是野指针

// 正确
std::string* s = new std::string("hello");

// 也危险：用free释放new分配的对象
free(s);  // 不会调用析构函数！string内部堆内存泄漏
delete s; // 正确：先析构再释放
```

**AI框架中的内存管理**：
- PyTorch的CachingAllocator：基于cudaMalloc但缓存释放的block（避免频繁系统调用）。
- 自定义operator new：可以用来实现内存池、对齐分配、统计追踪。
- `aligned_alloc`/`posix_memalign`：当需要特定对齐时使用。

---

### Q: C++四种强制类型转换？

**四种cast的语义和安全级别**：

| Cast | 安全性 | 运行时检查 | 典型用途 |
|------|--------|-----------|---------|
| static_cast | 中 | 无 | 数值转换、向上转型 |
| dynamic_cast | 高 | 有（RTTI） | 安全向下转型 |
| const_cast | 低 | 无 | 去除const |
| reinterpret_cast | 最低 | 无 | 按位重解释 |

**static_cast**：
```cpp
double d = 3.14;
int i = static_cast<int>(d);          // 数值转换（截断）

Derived* dp = new Derived();
Base* bp = static_cast<Base*>(dp);     // 向上转型（安全，隐式也行）
Derived* dp2 = static_cast<Derived*>(bp); // 向下转型（不安全！编译期不检查）
```
编译期完成，不产生运行时代码。如果向下转型错误（bp实际不是Derived），行为未定义。

**dynamic_cast**：
```cpp
Base* bp = get_object();
Derived* dp = dynamic_cast<Derived*>(bp);  // 运行时检查
if (dp) {
    dp->derived_method();  // 安全使用
} else {
    // bp实际不是Derived类型
}
```
需要RTTI（虚函数表中的type_info），有运行时开销（遍历继承链）。引用形式失败抛`std::bad_cast`。

**const_cast**：
```cpp
const int* cp = &value;
int* p = const_cast<int*>(cp);  // 去除const限定
*p = 42;  // 如果原始对象确实非const，合法；如果原始是const，UB！
```
唯一能去除const/volatile的cast。常见于调用不接受const参数的旧API。

**reinterpret_cast**：
```cpp
float f = 3.14f;
int bits = *reinterpret_cast<int*>(&f);  // 查看float的二进制表示

// CUDA中常见：将指针转为不同类型指针
float4* vec_ptr = reinterpret_cast<float4*>(float_ptr);  // 向量化读取
```
最危险——只是告诉编译器"把这些字节当作另一种类型看待"。不做任何转换或检查。

**AI框架中的使用场景**：
- `reinterpret_cast`：CUDA kernel中的向量化读写（`float4*`）、类型punning查看内存布局。
- `static_cast`：精度转换（FP32→FP16）、enum→int。
- `dynamic_cast`：在算子dispatch中检查具体子类类型（如检查TensorImpl的具体device type）。

---

### Q: 深拷贝和浅拷贝的区别？

**核心区别在于是否复制指针指向的资源**：

```cpp
class MyArray {
    int* data;
    int size;
public:
    // 浅拷贝（默认生成）
    MyArray(const MyArray& other) : data(other.data), size(other.size) {}
    // data指针被复制 → 两个对象共享同一块堆内存！

    // 深拷贝（手动实现）
    MyArray(const MyArray& other) : size(other.size) {
        data = new int[size];                    // 分配新内存
        std::memcpy(data, other.data, size * sizeof(int));  // 复制内容
    }
};
```

**浅拷贝的致命问题**：
```cpp
MyArray a(100);       // a.data → [heap block A]
MyArray b = a;        // b.data → [heap block A] (同一块!)
// b析构: delete[] b.data → heap block A被释放
// a析构: delete[] a.data → double free! 崩溃
```

**何时必须深拷贝（Rule of Three/Five）**：
- 类管理动态资源（new/malloc/文件句柄/GPU显存等）时。
- 如果定义了析构函数（释放资源），通常也需要定义拷贝构造和拷贝赋值。
- C++11后扩展为Rule of Five（加上移动构造和移动赋值）。

**PyTorch中的例子**：
```python
# 浅拷贝（共享底层storage）
b = a          # 赋值：共享data pointer
c = a.view(4, 4)  # view：共享storage，不同stride

# 深拷贝
d = a.clone()  # 分配新storage，复制数据
e = a.detach().clone()  # 深拷贝且脱离计算图
```

Tensor的storage是引用计数管理的——多个Tensor可以共享同一个Storage（不同offset/stride）。只有`clone()`会创建独立的storage。

---

### Q: 智能指针的种类和实现原理？

**三种智能指针各自解决不同的所有权语义问题**：

**unique_ptr（独占所有权）**：
```cpp
// 实现核心：禁止拷贝，只允许移动
template<typename T>
class unique_ptr {
    T* ptr;
public:
    explicit unique_ptr(T* p) : ptr(p) {}
    ~unique_ptr() { delete ptr; }
    unique_ptr(const unique_ptr&) = delete;            // 禁止拷贝！
    unique_ptr& operator=(const unique_ptr&) = delete;
    unique_ptr(unique_ptr&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }  // 移动
    T& operator*() { return *ptr; }
    T* operator->() { return ptr; }
};
```
- 大小：与裸指针相同（8字节），**零开销抽象**。
- 适用：函数返回动态对象、独占资源管理（如文件句柄、CUDA stream）。

**shared_ptr（共享所有权）**：
```cpp
// 实现核心：引用计数（在堆上的控制块中）
template<typename T>
class shared_ptr {
    T* ptr;
    ControlBlock* ctrl;  // 包含: strong_count, weak_count, deleter
public:
    shared_ptr(T* p) : ptr(p), ctrl(new ControlBlock(1, 0)) {}
    shared_ptr(const shared_ptr& other) : ptr(other.ptr), ctrl(other.ctrl) {
        ctrl->strong_count.fetch_add(1);  // 原子+1
    }
    ~shared_ptr() {
        if (ctrl->strong_count.fetch_sub(1) == 1) {  // 原子-1，最后一个
            delete ptr;
            if (ctrl->weak_count == 0) delete ctrl;
        }
    }
};
```
- 大小：16字节（ptr + ctrl指针）。
- 开销：拷贝/析构时原子操作（~10ns in x86）、堆上control block。
- 适用：多处共享同一对象（如PyTorch的Storage在多个Tensor间共享）。

**weak_ptr（观察不拥有）**：
```cpp
// 不增加强引用计数，只增加弱引用计数
shared_ptr<T> sp = make_shared<T>();
weak_ptr<T> wp = sp;  // weak_count++, strong_count不变

// 使用时需要lock()检查对象是否存活
if (auto locked = wp.lock()) {  // 返回shared_ptr（暂时+1强引用）
    locked->do_something();     // 安全使用
}  // locked析构，强引用-1
```
- 解决循环引用问题：A→B且B→A时，用weak_ptr打破环。
- 适用：缓存、观察者模式、打破shared_ptr循环。

**make_shared优于new + shared_ptr**：
```cpp
auto p = make_shared<T>(args);  // 一次分配（对象和控制块在一块内存中）
auto p = shared_ptr<T>(new T(args));  // 两次分配（对象和控制块分开）
```
make_shared更高效（减少内存分配次数、更好的cache局部性）。

---

### Q: 如何防止内存泄漏？

**系统化的防泄漏策略（从设计到检测）**：

**1. RAII设计原则（最根本）**：
```cpp
// 资源的获取即初始化，释放绑定在析构函数
class CudaBuffer {
    void* ptr;
public:
    CudaBuffer(size_t size) { cudaMalloc(&ptr, size); }
    ~CudaBuffer() { cudaFree(ptr); }  // 无论如何都会释放
    // 禁止拷贝，允许移动
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
};
// 作用域结束或异常抛出时，析构函数自动调用 → 不泄漏
```

**2. 智能指针使用规范**：
```cpp
// 独占: unique_ptr（首选，零开销）
auto model = std::make_unique<Model>(config);

// 共享: shared_ptr（仅当确实需要多处共享时）
auto buffer = std::make_shared<CudaBuffer>(size);

// 避免裸new
auto p = new Resource();  // BAD: 如果后续代码抛异常，p泄漏
auto p = make_unique<Resource>();  // GOOD: 异常安全
```

**3. 容器自动管理**：
```cpp
std::vector<Tensor> activations;  // vector析构时自动析构所有元素
// 不需要手动管理每个Tensor的生命周期
```

**4. 检测工具**：

| 工具 | 用途 | 开销 | 使用方式 |
|------|------|------|---------|
| AddressSanitizer | 越界/UAF/泄漏 | 2x运行时 | `-fsanitize=address` |
| LeakSanitizer | 专注泄漏检测 | 极小 | `-fsanitize=leak`（默认含在ASan中） |
| Valgrind | 全面内存检查 | 10-50x | `valgrind --leak-check=full ./app` |
| CUDA-memcheck | GPU内存错误 | 中等 | `compute-sanitizer ./app` |

**5. GPU显存泄漏防范（AI框架特有）**：
```python
# PyTorch中常见泄漏场景
loss_history = []
for batch in loader:
    loss = model(batch)
    loss_history.append(loss)  # BAD! loss关联计算图，整个图不释放

    loss_history.append(loss.item())  # GOOD: .item()取标量，释放计算图

# 检查显存
print(torch.cuda.memory_summary())  # 查看是否有unreleased tensor
```

---

### Q: GDB的基本使用？

GDB是Linux下最强大的C/C++调试工具，AI Infra开发中常用于调试CUDA host代码、内存问题、多线程问题：

**核心命令分类**：

**断点和执行**：
```bash
(gdb) b main.cpp:42       # 在文件第42行设断点
(gdb) b MyClass::forward  # 在函数入口设断点
(gdb) b kernel.cu:100 if n>1000  # 条件断点
(gdb) r arg1 arg2         # 运行程序（带参数）
(gdb) n                   # 下一行（不进入函数）
(gdb) s                   # 步入函数
(gdb) c                   # 继续执行到下一个断点
(gdb) finish              # 执行完当前函数返回
```

**检查状态**：
```bash
(gdb) p variable          # 打印变量值
(gdb) p *array@10         # 打印数组前10个元素
(gdb) p/x ptr             # 十六进制打印指针
(gdb) bt                  # 打印调用栈（backtrace）
(gdb) bt full             # 打印调用栈+每帧的局部变量
(gdb) info locals         # 当前帧所有局部变量
(gdb) frame 3             # 切换到第3帧查看
(gdb) display expr        # 每步自动打印表达式
```

**内存调试**：
```bash
(gdb) x/16xb ptr          # 以hex byte格式查看ptr开始的16字节
(gdb) x/4gx ptr           # 以giant(8字节)格式查看4个值
(gdb) watch variable      # 数据断点：变量被修改时中断
(gdb) watch *(int*)0x7fff1234  # 监视特定内存地址
```

**多线程调试**：
```bash
(gdb) info threads        # 列出所有线程
(gdb) thread 3            # 切换到线程3
(gdb) thread apply all bt # 打印所有线程的调用栈
(gdb) set scheduler-locking on  # 只运行当前线程（其他线程暂停）
```

**实用技巧**：
```bash
# 崩溃后分析core dump
$ ulimit -c unlimited      # 允许生成core文件
$ ./crash_program           # 崩溃生成core
$ gdb ./crash_program core  # 加载core分析
(gdb) bt                   # 看崩溃时的调用栈

# 附加到运行中的进程
$ gdb -p <pid>

# 条件断点+命令序列
(gdb) b forward if batch_size == 0
(gdb) commands
> p input.shape
> bt 5
> c
> end
```

**AI Infra调试场景**：
- Segfault：通常是越界访问或空指针。用`bt`定位，然后`p`检查指针。
- CUDA错误：host端看到`cudaErrorIllegalAddress`——可能kernel越界。用`compute-sanitizer`代替GDB。
- 死锁：`thread apply all bt`看哪些线程在等锁，分析锁顺序。

---

### Q: 手撕：图的最短连通路径长度？

（编程题）
