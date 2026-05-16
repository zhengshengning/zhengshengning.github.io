---
title: 'AI Infra 综合面经题库 (6)'
date: 2026-04-16 12:17:04
categories:
  - [求职面试,综合面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: CUDA Graph的作用和原理？kernel launch流程是什么？

**CUDA Graph**：将一系列CUDA操作（kernel、memcpy）录制为一个图（DAG），之后一次launch整个图。消除了多次kernel launch的CPU开销（每次launch约5-10us），对于大量小kernel场景提升显著。

**Kernel Launch流程**：CPU端准备kernel参数 → 将launch command写入GPU command buffer → GPU从command buffer取指令 → 在SM上调度执行。CUDA Graph将所有command一次性提交，GPU连续执行无需等待CPU。

### Q: 如何确定CUDA kernel的blockSize和gridSize？

- **blockSize**：通常是32的倍数（Warp大小）。考虑因素：寄存器/共享内存使用量（影响occupancy）、具体算法需要的线程协作模式。常用128/256/512。可用`cudaOccupancyMaxPotentialBlockSize`自动选择。
- **gridSize**：`gridSize = ceil(问题规模 / blockSize)`。确保有足够block填满所有SM（至少SM数量的数倍）。过多block也无妨（硬件会排队调度）。

### Q: 什么是Default Stream？它存在什么问题？

Default Stream（stream 0）是CUDA的默认执行队列。问题：它与其他stream有隐式同步——default stream中的操作会等待之前所有stream的操作完成，之后其他stream的操作也会等待default stream完成。这破坏了多stream并发的能力。解决：使用per-thread default stream（编译选项`--default-stream per-thread`）或总是使用显式创建的stream。

### Q: __threadfence的作用？

`__threadfence()`保证当前线程在该点之前对全局内存/共享内存的写操作对其他线程可见（内存屏障）。三个级别：
- `__threadfence_block()`：对Block内所有线程可见。
- `__threadfence()`：对设备上所有线程可见。
- `__threadfence_system()`：对系统中所有线程（包括host）可见。

常用于实现跨block的同步和无锁数据结构。

### Q: 如何debug CUDA kernel？

- **compute-sanitizer**（原cuda-memcheck）：检测越界访问、竞争条件。
- **cuda-gdb**：GPU调试器，可设断点、查看线程状态。
- **printf**：kernel内直接打印（适合小规模调试）。
- **Nsight Compute**：分析kernel性能瓶颈。
- **验证方法**：与CPU参考实现对比结果、逐步缩小问题范围（减少线程/block数量）。

### Q: Unified Memory和Zero-Copy Memory的区别？

- **Unified Memory**：CPU和GPU看到相同的虚拟地址，驱动自动在CPU和GPU间迁移页面（按需）。编程简单但可能有隐式page fault开销。
- **Zero-Copy Memory**：使用`cudaHostAllocMapped`分配的pinned host memory，GPU通过PCIe直接访问，不做数据拷贝。延迟高（PCIe往返），但适合GPU只访问少量数据的场景。

### Q: CUDA sort如何实现？

常用radix sort（基数排序）：
1. 从最低位到最高位逐位排序。
2. 每一轮：计算每个线程的位值 → 前缀和（scan）计算目标位置 → scatter到目标位置。
3. 利用共享内存加速block内排序，block间归并。

CUB/Thrust库提供了高度优化的实现。对于少量数据可用bitonic sort（适合warp/block内排序）。

### Q: sin函数在GPU的哪个硬件单元计算？该单元还能算什么？

sin函数在**SFU（Special Function Unit）**上计算。SFU还能计算：cos、exp、log、rsqrt（快速倒数平方根）、rcp（倒数）等超越函数。SFU吞吐量低于ALU（通常每SM每周期只有少量SFU），大量使用会成为瓶颈。可用`__sinf()`等快速近似版本。

### Q: Volta架构的ITS（Independent Thread Scheduling）是什么？

Volta引入的Independent Thread Scheduling使Warp内的每个线程拥有独立的程序计数器和调用栈（之前整个Warp共享一个PC）。影响：
- 允许Warp内线程真正独立执行不同路径（非简单masking）。
- 使细粒度同步成为可能。
- 但可能导致原来依赖隐式Warp同步的代码失效，需要显式使用`__syncwarp()`。

### Q: 3090上单个block能用的共享内存最大有多少？

3090（Ampere GA102）单SM共享内存最大可配置为100KB（与L1 Cache共享128KB物理存储，可配置100KB shared + 28KB L1）。单个block最多可使用48KB共享内存（默认限制），通过`cudaFuncSetAttribute`设置`cudaFuncAttributeMaxDynamicSharedMemorySize`可提升到每SM的最大值。

### Q: PTX与SASS的区别？

- **PTX（Parallel Thread Execution）**：NVIDIA的虚拟ISA，与具体GPU架构无关。类似汇编的中间表示，可跨架构兼容（JIT编译到具体架构）。
- **SASS（Streaming Assembly）**：具体GPU架构的原生机器码。每代架构SASS不同。用`cuobjdump`查看。

PTX是编译的中间产物，最终由驱动JIT编译为目标GPU的SASS执行。手写PTX可做微架构级优化但需了解硬件细节。

### Q: GPU性能xx TFLOPS是怎么计算的？

`TFLOPS = CUDA Cores数 * 2 * 时钟频率(GHz)`（FMA指令每周期2个FP操作）。

例如A100：6912 CUDA Cores * 2 * 1.41GHz ≈ 19.5 TFLOPS（FP32）。Tensor Core额外提供更高的矩阵运算吞吐（如A100 FP16 Tensor Core: 312 TFLOPS）。

### Q: C++虚函数实现机制？单继承、多继承、虚继承的内存布局？

**虚函数机制**：每个含虚函数的类有一个vtable，对象头部有vptr指向vtable。

**内存布局**：
- 单继承：对象只有一个vptr，派生类vtable继承并覆盖基类条目。
- 多继承：对象有多个vptr（每个基类一个），地址调整thunk处理不同基类子对象偏移。
- 虚继承：增加虚基类指针/偏移表，虚基类子对象被共享（只有一份），通过偏移量间接访问。

### Q: C++四种cast的区别？

- **static_cast**：编译期类型转换，用于相关类型间转换（上下行转换无运行时检查、基本类型转换）。
- **dynamic_cast**：运行时类型检查的多态向下转换，失败返回nullptr（指针）或抛异常（引用）。
- **const_cast**：去除/添加const/volatile修饰。
- **reinterpret_cast**：位级别重新解释，最危险，用于无关类型间的转换。

### Q: C++三种智能指针？

- **unique_ptr**：独占所有权，不可拷贝只能移动，零开销抽象。
- **shared_ptr**：共享所有权，引用计数管理（原子操作，有额外开销），最后一个释放时删除对象。
- **weak_ptr**：不增加引用计数，用于打破shared_ptr的循环引用，使用前需lock()转为shared_ptr。

### Q: C++函数模板声明与定义能否分离？

通常不能分离到不同编译单元（.h和.cpp），因为模板需要在使用处实例化，编译器需要看到完整定义。解决方法：定义放在头文件中；或在.cpp中显式实例化所有需要的类型（`template class Foo<int>;`）。

### Q: CRTP（Curiously Recurring Template Pattern）静态多态是什么？

```cpp
template<typename Derived>
class Base {
    void interface() { static_cast<Derived*>(this)->impl(); }
};
class D : public Base<D> {
    void impl() { /* ... */ }
};
```
派生类将自身作为模板参数传给基类，基类通过static_cast调用派生类方法，实现编译期多态。无虚函数开销（无vptr、无间接调用），适合性能敏感场景。

### Q: vector的resize和reserve的区别？

- **reserve(n)**：预分配容量至少为n，不改变size，不构造元素。
- **resize(n)**：改变size为n，若n>当前size则构造新元素（默认初始化或指定值），若n<当前size则销毁多余元素。

### Q: 手撕：CUDA实现reduction？

（编程题）

### Q: 手撕：CUDA实现softmax？

（编程题）

### Q: 手撕：CUDA实现matrix transpose？

（编程题）

### Q: 手撕：CUDA实现avg pooling？

（编程题）

### Q: 手撕：CUDA计算两组bbox的IoU？

（编程题）

### Q: 手撕：C++实现NMS？

（编程题）

### Q: 手撕：C++实现conv2d？

（编程题）

### Q: 手撕：C++实现双线性插值？

（编程题）

### Q: 手撕：C++实现LayerNorm？

（编程题）

### Q: 手撕：C++实现单例模式？

（编程题）
