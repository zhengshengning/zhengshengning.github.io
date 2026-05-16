---
title: '百度 AI Infra (2)'
date: 2026-04-16 12:18:05
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: OpenMP并行区怎么开？

使用 `#pragma omp parallel` 编译器指令开启并行区域。编译器遇到该指令时 fork 线程组并行执行代码块。常用方式：`#pragma omp parallel for` 自动将循环迭代均分到线程；`num_threads(N)` 指定线程数（默认取 OMP_NUM_THREADS 或 CPU 核数）；`schedule(dynamic)` 动态调度适合负载不均场景。编译需加 `-fopenmp` 标志。

### Q: for循环中sum++在并行区和串行区结果是否相同？如何修正？

不相同——`sum++` 包含读-改-写三步非原子操作，多线程并发导致数据竞争和更新丢失。修正方法（按性能从优到劣）：

1. **reduction 子句**：`#pragma omp parallel for reduction(+:sum)`。编译器为每个线程创建私有副本，结束后自动归约。零同步开销，最优选择。
2. **atomic**：`#pragma omp atomic` 修饰 sum++。硬件原子指令保证读-改-写不可分割。每次迭代有原子开销。
3. **私有副本合并**：线程用 private 变量累加后 critical 区合并。
4. **critical 互斥区**：完全串行化临界区，丧失并行意义，仅适合复杂逻辑。

### Q: GPU内存架构？

分层设计：寄存器(每SM 256KB, ~1cycle) -> 共享内存/L1(每SM 192KB可配, ~20cycles) -> L2 Cache(40MB, ~200cycles) -> 全局内存HBM(80GB, ~400cycles) -> 常量/纹理内存(只读+专用缓存)。

优化核心：最大化片上数据复用，减少 HBM 访问。将算术强度提升到超过硬件计算-带宽比（A100 约 156 FLOPs/Byte），使 kernel 变为 compute-bound。

### Q: Cache映射分类和特点？

**直接映射**：块映射到唯一行，硬件简单查找快但冲突率高（thrashing）。**全相联**：可放任意行，冲突率最低但需CAM并行比较，硬件贵（适合TLB等小容量）。**组相联**（主流）：映射到固定组内任意行，N路兼顾冲突率和复杂度。现代CPU典型 L1 4-8路，L2/L3 8-16路。

### Q: C++智能指针？

**shared_ptr**：引用计数共享所有权（原子操作线程安全，控制块额外16-32B），`make_shared` 一次分配更高效。**unique_ptr**：独占零开销（delete拷贝只允许move），默认首选。**weak_ptr**：弱引用不增计数，`lock()` 提升为 shared_ptr，解决循环引用/缓存探测。选择策略：unique -> shared -> weak。

### Q: C++多态及实现？

**静态多态**：函数重载/模板，编译期决议，零运行时开销。**动态多态**：虚函数+继承。编译器为含虚函数的类生成 vtable（函数地址表），对象开头有 vptr 指向所属类的 vtable。通过基类指针调虚函数：读vptr -> 查vtable -> 间接调用。开销：每对象+8B(vptr)，每次调用多一次间接寻址，虚函数不能内联。

### Q: 动态链接与动态绑定？

**动态绑定**（语言层面）：C++ 多态核心——运行时通过 vtable 确定虚函数调用目标。**动态链接**（OS层面）：程序运行时加载共享库(.so/.dll)并解析符号。节省内存（多进程共享）、便于更新。PLT/GOT 实现延迟绑定（首次调用时解析）。

### Q: map底层实现？

红黑树（自平衡BST），插入/删除/查找 O(log n)，元素按 key 有序。红黑树性质保证最坏树高 <= 2*log(n+1)。适合需要有序遍历、范围查询的场景。纯查找选 unordered_map（O(1)平均）。

### Q: AVL树与红黑树区别？

AVL 严格平衡（高度差<=1）查找更快（树更矮）；红黑树近似平衡但删除旋转最多3次（AVL可能O(logn)次），插删效率更高。STL 选红黑树因实际应用插删频繁，综合性能更优。

### Q: unordered_map底层？

哈希表+拉链法，bucket 数组每项是链表。平均 O(1) 查找/插入，最坏 O(n)。load_factor>1.0 时 rehash 翻倍扩容。自定义类型需提供 hash 和 ==。元素无序，不支持范围查询。cache 局部性比连续容器差。

### Q: 哈希冲突解决？

**链地址法**（拉链，STL采用）：每桶一个链表，load factor 可>1。**开放定址法**：冲突时探测下一空位（线性/二次/双重哈希），cache友好但 load factor 需<0.75。**再哈希法**：备选哈希函数。**公共溢出区**。

### Q: 计算机存储层次？

寄存器(<1ns, KB级) -> L1(~1ns, 32-64KB) -> L2(~5ns, 256KB-1MB) -> L3(~15ns, 8-64MB) -> DRAM(~80ns, 16-256GB) -> SSD(~100us, TB级) -> HDD(~10ms, TB级)。利用时间/空间局部性，小而快的缓存覆盖大部分访问。

### Q: 程序员可见寄存器及PC作用？

x86-64：通用寄存器（RAX-R15, 16个64位）、SIMD（XMM/YMM/ZMM）、段寄存器（FS/GS做TLS）、标志寄存器（RFLAGS）。PC/RIP 存下条指令地址，CPU 据此取指执行。用户态不可直接写PC，只能通过 jmp/call/ret 改变。

### Q: main前运行函数？

1. 全局对象构造函数（CRT 初始化阶段调用，跨翻译单元顺序未定义——SIOF 问题）
2. `__attribute__((constructor))`（GCC/Clang，注册到 .init_array）
3. 静态变量初始化表达式

### Q: 条件断点？

GDB：`break N if condition`。每次触发都评估表达式，高频循环中显著减慢。也可 `condition breakpoint_id expr`。IDE 右键断点设置条件即可。

### Q: 堆栈监视？

GDB：`bt`(调用栈)、`bt full`(含局部变量)、`frame N`(切换栈帧)、`info locals/args`。ASan(`-fsanitize=address`)检测栈溢出/use-after-free ~2x开销；Valgrind 无需重编译但慢10-20x。

### Q: 程序内存分段？

栈(局部变量/调用帧, 向下增长, 默认8MB) -> 空闲/mmap区域 -> 堆(malloc/new, 向上增长) -> BSS(未初始化全局, 加载时清零) -> 数据段(已初始化全局) -> 代码段(只读指令)。ASLR 随机化基址。

### Q: deque实现？

分段连续：中控 map 数组存指针指向固定大小缓冲区。O(1) 头尾插删（操作端缓冲区），扩容只需新增缓冲区。随机访问需计算所在缓冲区+偏移。cache 局部性不如 vector，适合头尾操作频繁的场景（如 BFS 队列）。

### Q: 寄存器vs Cache vs主存？

寄存器最快(0-1 cycle, 直连ALU)。Cache: SRAM(6管/bit, 无需刷新, 快但贵~$100/GB)。主存: DRAM(1管+1电容/bit, 需64ms刷新, 密度高但慢~$3/GB)。都是易失性存储。分层原因：SRAM太贵，DRAM太慢。

### Q: CUDA内存分类？

寄存器(线程私有, 片上, 1cycle) -> 本地内存(线程私有, 实际在HBM, 溢出时使用) -> 共享内存(Block共享, 片上SM, ~20cycles, 手动管理) -> 全局内存(HBM, ~400cycles, 需合并访问) -> 常量内存(64KB只读+缓存) -> 纹理内存(2D空间局部性优化)

### Q: __global__关键字？

修饰 Kernel 函数：Host调用Device执行，必须 void 返回，指定 `<<<grid, block, sharedMem, stream>>>` 启动配置。异步执行（CPU发起后立即返回）。对比：`__device__` GPU调GPU执行；`__host__` CPU调CPU执行。

### Q: 获取GPU最大线程数？

`cudaGetDeviceProperties(&prop, id)` 获取 `maxThreadsPerBlock`(通常1024)、`maxThreadsDim[3]`([1024,1024,64])、`maxGridSize[3]`([2^31-1,65535,65535])。实际可用线程还受寄存器/共享内存用量约束（影响 occupancy）。

### Q: 贪心vs动态规划？

贪心：每步局部最优不回溯，O(n logn)，需证明贪心选择性质（如 Huffman/Dijkstra）。DP：分解重叠子问题，记录中间结果保证全局最优，O(n^2)+O(n)空间（如 LCS/背包）。判断：局部最优→全局最优用贪心；否则用 DP。

### Q: 图搜索算法？

BFS(队列逐层, 无权最短路, O(V+E))、DFS(栈/递归深入回溯, 拓扑排序/环检测, O(V+E))、Dijkstra(正权单源最短路, 优先队列, O((V+E)logV))、Bellman-Ford(负权, O(VE))、Floyd(全对最短路, O(V^3))、A*(启发式f=g+h)。
