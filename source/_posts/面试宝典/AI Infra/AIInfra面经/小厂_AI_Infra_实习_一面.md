---
title: '小厂 AI Infra 实习 一面'
date: 2026-04-16 12:20:49
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 内存分级（CPU/GPU内存层次结构）？

**CPU和GPU内存层次对比：**

```
CPU内存层次:                         GPU内存层次:
┌───────────────────┐               ┌───────────────────┐
│ 寄存器 (~0.3ns)   │               │ 寄存器 (~0.3ns)   │
│ 每核数百个         │               │ 65536×32bit/SM   │
├───────────────────┤               ├───────────────────┤
│ L1 Cache (~1-3ns) │               │ Shared Mem/L1     │
│ 32-48KB/核        │               │ 192KB/SM (~5ns)  │
├───────────────────┤               ├───────────────────┤
│ L2 Cache (~5-12ns)│               │ L2 Cache (~50ns)  │
│ 256KB-1MB/核      │               │ 40MB (全GPU共享)  │
├───────────────────┤               ├───────────────────┤
│ L3 Cache (~20-40ns)│              │                   │
│ 4-60MB (多核共享) │               │                   │
├───────────────────┤               ├───────────────────┤
│ 主存 DRAM (~100ns)│               │ HBM (~400ns)      │
│ 数十-数百GB       │               │ 80GB, 2TB/s      │
│ ~50 GB/s带宽      │               │                   │
└───────────────────┘               └───────────────────┘
```

**关键差异的设计哲学：**

| 维度 | CPU | GPU |
|------|-----|-----|
| 设计目标 | 减少延迟(大cache) | 隐藏延迟(多线程) |
| Cache比例 | 占芯片面积>50% | 占芯片面积<20% |
| 程序员参与 | 硬件透明管理 | 部分需显式管理(Shared Mem) |
| 带宽 | 中等(DDR ~50GB/s) | 极高(HBM ~2TB/s) |
| 延迟隐藏 | 靠Cache + 乱序执行 + 推测执行 | 靠大量warp切换(Latency Hiding) |

**编程优化启示：**
- CPU：优化Cache命中率(数据局部性、预取)
- GPU：优化带宽利用率(合并访问、向量化)和占用率(Occupancy)
- 两者共同：减少主存访问次数(数据复用)

### Q: GPU内存层次结构？

**NVIDIA GPU完整内存层次（以A100为例）：**

| 内存类型 | 容量 | 带宽 | 延迟 | 作用域 | 管理方式 | 典型用途 |
|---------|------|------|------|--------|---------|---------|
| 寄存器 | 256KB/SM | ~19TB/s | 1 cycle | 线程私有 | 编译器 | 局部变量、中间结果 |
| Shared Memory | 0-164KB/SM | ~19TB/s | ~28 cycles | Block共享 | 程序员__shared__ | Tile缓存、Reduce |
| L1 Cache | 与Shared共享192KB | ~19TB/s | ~28 cycles | Block内 | 硬件自动 | 自动缓存全局访问 |
| L2 Cache | 40MB | ~5TB/s | ~200 cycles | 全GPU | 硬件自动 | 全局内存cache |
| 全局内存(HBM) | 80GB | 2039GB/s | ~600 cycles | 全GPU | cudaMalloc | 模型权重、数据 |
| 常量内存 | 64KB | 很高(命中) | ~4 cycles(命中) | 全GPU只读 | __constant__ | 超参数、查找表 |
| Local Memory | 溢出到HBM | 同全局 | 同全局 | 线程私有 | 编译器溢出 | 寄存器溢出数据 |

**关键优化策略：**

```cuda
// 策略1: 数据从HBM → Shared Memory → 寄存器
__shared__ float tile[32][32];
tile[ty][tx] = global_data[...];  // HBM→Shared (一次)
__syncthreads();
float reg = tile[ty][tx];  // Shared→Register (多次复用)

// 策略2: 减少寄存器使用避免溢出
// 每线程寄存器过多 → 溢出到Local Memory(HBM速度!) → 性能骤降
// 用 __launch_bounds__ 提示编译器控制寄存器数

// 策略3: 利用L2 Cache Persistence (Ampere+)
// 小权重矩阵设为L2 persistent → 多次推理不用再从HBM加载
```

### Q: 影响GPU性能的硬件因素？

**GPU性能的三角模型：**

```
              Compute
             (算力)
            /       \
           /  Kernel  \
          /  受限于其中 \
         /   一个因素    \
        ──────────────────
       Bandwidth        Latency
       (带宽)           (延迟/并行度)
```

**各因素的详细参数和影响：**

| 因素 | A100指标 | H100指标 | 如何成为瓶颈 |
|------|---------|---------|------------|
| FP16算力 | 312 TFLOPS | 989 TFLOPS | 大矩阵乘(高算术强度) |
| INT8算力 | 624 TOPS | 1979 TOPS | 量化推理 |
| HBM带宽 | 2039 GB/s | 3350 GB/s | 小batch推理(低算术强度) |
| NVLink带宽 | 600 GB/s | 900 GB/s | 多卡通信(TP AllReduce) |
| SM数量 | 108 | 132 | 决定最大并行block数 |
| L2 Cache | 40 MB | 50 MB | 工作集超过L2时性能下降 |
| 共享内存/SM | 192 KB | 228 KB | 限制每block可用的tile大小 |
| 寄存器/SM | 65536×32bit | 65536×32bit | 限制occupancy |

**实际场景中的瓶颈分析：**

| 场景 | 主要瓶颈 | 原因 | 优化方向 |
|------|---------|------|---------|
| LLM Decode(batch=1) | HBM带宽 | 每token读全部权重 | 量化减少权重大小 |
| LLM Prefill(长序列) | Compute | 大矩阵乘 | FP8/INT8加速 |
| Reduce/LayerNorm | HBM带宽 | 算术强度低 | 算子融合 |
| 多卡训练 | 互联带宽 | AllReduce通信 | 梯度压缩/overlap |
| 小kernel连续调用 | CPU launch延迟 | 每次launch~5us | CUDA Graph |

**Roofline分析实例：**

```
A100 FP16:
  峰值算力: 312 TFLOPS
  峰值带宽: 2039 GB/s
  平衡点: 312T / 2039G ≈ 153 FLOP/Byte (FP16, 1元素=2B)
         = 76.5 FLOP/Element

GEMM [M=4096, K=4096, N=4096]:
  FLOPs = 2×4096³ = 137.4 GFLOPs
  数据量 = (M×K + K×N + M×N) × 2B = ~100MB
  AI = 137.4G / 100M ≈ 1374 FLOP/Byte >> 153 → Compute-bound ✓

逐元素Add [N=1M]:
  FLOPs = 1M
  数据量 = 3M × 4B = 12MB (读2个+写1个)
  AI = 1M / 12M ≈ 0.08 FLOP/Byte << 153 → Memory-bound ✓
```

### Q: 递归的优缺点？

**递归的本质：**

```cpp
// 递归 = 函数调用自身 + 每次调用有栈帧开销
void factorial(int n) {
    if (n <= 1) return 1;       // 基准条件(终止)
    return n * factorial(n-1);  // 递归调用(减小问题规模)
}

// 每次调用的栈帧:
// ┌──────────────┐
// │ 返回地址      │
// │ 参数n         │  ← 每帧~几十字节
// │ 局部变量      │
// │ 保存的寄存器  │
// └──────────────┘
// 深度D → D个栈帧 → D × frame_size 字节栈空间
```

**优缺点对比：**

| 优点 | 缺点 |
|------|------|
| 代码简洁自然(分治/树/回溯) | 函数调用开销(~5-10ns/次) |
| 天然表达数学归纳/递推关系 | 栈溢出风险(默认栈~8MB, 深度>10万可能溢出) |
| 容易证明正确性 | 重复计算(朴素递归Fibonacci: O(2^n)) |
| 回溯/剪枝自然实现 | 调试较困难(深层栈追踪) |

**优化方法：**

```cpp
// 1. 记忆化(Memoization) — 消除重复计算
unordered_map<int, long> memo;
long fib(int n) {
    if (n <= 1) return n;
    if (memo.count(n)) return memo[n];
    return memo[n] = fib(n-1) + fib(n-2);
}
// O(2^n) → O(n)

// 2. 尾递归优化 — 编译器转循环
long factorial_tail(int n, long acc = 1) {
    if (n <= 1) return acc;
    return factorial_tail(n-1, n * acc);  // 尾部位置递归
}
// 编译器可优化为循环(gcc -O2), 无栈增长

// 3. 手动改迭代 — 显式用栈模拟
void dfs_iterative(TreeNode* root) {
    stack<TreeNode*> s;
    s.push(root);
    while (!s.empty()) {
        auto* node = s.top(); s.pop();
        process(node);
        if (node->right) s.push(node->right);
        if (node->left) s.push(node->left);
    }
}
```

### Q: DFS和BFS在解方程组中的区别？

**图算法在稀疏线性代数中的应用：**

稀疏矩阵可以看作图：非零元素A[i][j]表示节点i和j之间有边。

**DFS的应用：**

| 应用 | 算法 | 作用 |
|------|------|------|
| 强连通分量 | Tarjan/Kosaraju | 将大方程组分解为独立子系统 |
| 拓扑排序 | DFS后序反转 | 确定求解顺序(无环时) |
| 填入分析 | Elimination Tree | 预测LU分解的非零模式 |
| 矩阵重排序 | 嵌套剖分(Nested Dissection) | 减少填入元素 |

**BFS的应用：**

| 应用 | 算法 | 作用 |
|------|------|------|
| 带宽最小化 | Cuthill-McKee(RCM) | 减少矩阵带宽→减少填入 |
| 层次划分 | Level Set | 确定并行度(同层可并行) |
| 图分区 | BFS-based partitioning | 多处理器负载均衡 |
| 最短路径 | BFS(无权) | 确定通信距离 |

**在数值计算中的具体区别：**

```
稀疏方程组 Ax = b:

DFS方法(识别结构):
  对A的图做DFS → 发现强连通分量
  如果分量间无环 → 按拓扑序依次求解各分量
  规模从n降为max(n_component) → 小规模独立子问题

BFS方法(发现并行性):
  对A的图做BFS → 得到level sets
  同一level的节点互不依赖 → 可并行计算
  Level 0 → Level 1 → Level 2 → ...
  每level内的变量可同时求解
  
并行度 = max(level内节点数)
```

### Q: C++静态变量的特点？

**三种static作用域的详解：**

```cpp
// 1. 函数内static(局部static) — 单例/计数器
int call_count() {
    static int count = 0;  // 只初始化一次(首次调用时)
    return ++count;         // 生命周期=程序全程
}
// call_count() → 1, 2, 3, ... (跨调用保持)
// C++11保证: 初始化是线程安全的(Magic Static)

// 2. 类static成员 — 所有对象共享
class Connection {
    static int active_count;  // 声明(所有Connection对象共享)
    static Connection* instance;
public:
    static Connection& getInstance() {  // 工厂方法
        static Connection inst;          // Meyers Singleton
        return inst;
    }
};
int Connection::active_count = 0;  // 定义(类外, .cpp文件中)

// 3. 文件作用域static — 内部链接(限制可见性)
// file_a.cpp:
static int helper = 42;        // 只在file_a.cpp可见
static void internal_func() {} // 只在file_a.cpp可见
// file_b.cpp中无法访问helper和internal_func
// 等价于C++的匿名namespace: namespace { int helper = 42; }
```

**static变量的存储和初始化：**

| 特性 | 全局/文件static | 局部static | 类static |
|------|---------------|-----------|---------|
| 存储位置 | .data(有初值) / .bss(无初值) | 同左 | 同左 |
| 初始化时机 | main之前(动态初始化) | 首次执行到(延迟初始化) | 显式定义处 |
| 线程安全 | 保证(C++11) | 保证(C++11 Magic Static) | 需自行保证 |
| 析构时机 | main之后(反序析构) | main之后 | main之后 |

### Q: 堆和栈的区别？

| 维度 | 栈(Stack) | 堆(Heap) |
|------|-----------|----------|
| 管理 | 编译器自动(RAII) | 程序员手动(new/delete) |
| 分配速度 | 极快(~1ns, sub rsp指令) | 慢(~50-200ns, 搜索空闲块) |
| 释放速度 | 极快(add rsp) | 中等(归还到空闲链表) |
| 大小 | 固定(通常2-8MB) | 灵活(受虚拟地址空间限制) |
| 方向 | 向低地址增长 | 向高地址增长 |
| 碎片 | 无(LIFO天然连续) | 有(频繁malloc/free后碎片化) |
| Cache | 极好(活跃数据连续) | 较差(分散分配) |
| 线程 | 每线程独立栈 | 进程内共享(需同步) |
| 溢出 | Stack Overflow(段错误) | 返回NULL/抛bad_alloc |
| 适用 | 小对象、生命周期确定 | 大对象、生命周期不确定 |

```cpp
// 性能对比示例:
void stack_alloc() {
    int arr[1000];  // ~0ns, 编译器只调整rsp
}

void heap_alloc() {
    int* arr = new int[1000];  // ~100ns, 搜索空闲内存
    delete[] arr;               // ~50ns, 归还内存
}

// 为什么大数组不能放栈上:
void bad() {
    int huge[1000000];  // 4MB! 可能超出栈限制→段错误
}
void good() {
    auto huge = std::make_unique<int[]>(1000000);  // 堆上安全分配
}
```

### Q: 图算法了解哪些？

**图算法分类及复杂度：**

| 类别 | 算法 | 时间复杂度 | 适用条件 |
|------|------|-----------|---------|
| **遍历** | BFS | O(V+E) | 无权最短路/层次遍历 |
| | DFS | O(V+E) | 连通性/拓扑/回溯 |
| **最短路径** | Dijkstra | O((V+E)logV) | 非负权单源 |
| | Bellman-Ford | O(VE) | 有负权/检测负环 |
| | Floyd-Warshall | O(V³) | 全源最短路 |
| | A* | 启发式 | 带启发函数的单目标 |
| **最小生成树** | Prim | O(ElogV) | 稠密图 |
| | Kruskal | O(ElogE) | 稀疏图(配合并查集) |
| **拓扑排序** | Kahn(BFS) | O(V+E) | DAG/编译依赖 |
| | DFS后序 | O(V+E) | 同上 |
| **强连通分量** | Tarjan | O(V+E) | 有向图分解 |
| | Kosaraju | O(V+E) | 两次DFS |
| **匹配** | 匈牙利 | O(V³) | 二分图最大匹配 |
| **网络流** | Dinic | O(V²E) | 最大流/最小割 |
| **连通性** | 并查集 | 近O(1)/操作 | 动态连通性 |

**在HPC/AI中的图算法应用：**
- **拓扑排序**：计算图调度(TVM/TensorRT中算子执行顺序)
- **图分区**：分布式训练中的模型切分
- **BFS/层次划分**：稀疏矩阵并行化分析
- **最短路径**：网络拓扑感知的通信路由
- **匹配**：目标跟踪中的检测-轨迹关联(匈牙利算法)

### Q: 手撕（思路题）：CUDA实现100万个浮点数相加？

（编程题）
