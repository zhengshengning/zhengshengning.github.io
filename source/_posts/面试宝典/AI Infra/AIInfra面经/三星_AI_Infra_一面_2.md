---
title: '三星 AI Infra 一面 (2)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: GEMM优化方法有哪些？

GEMM（General Matrix Multiplication，通用矩阵乘法）是深度学习中最核心的算子，优化的目标是最大化硬件利用率。核心优化方法如下：

**1. 分块（Tiling）—— 最核心的优化：**
- 将大矩阵C[M,N] = A[M,K] * B[K,N]分成小块计算
- 多级分块：Block Tile（线程块负责一个C子块）→ Warp Tile → Thread Tile
- **为什么有效**：小块数据能放入共享内存/寄存器，重复使用而不需反复从HBM加载，将全局内存带宽需求降低了tile_size倍

**2. 共享内存缓存 + 双缓冲：**
```
// 双缓冲伪代码：加载下一轮数据与计算当前数据重叠
load A_tile[0], B_tile[0] to shared memory
for k = 0 to K/TILE_K:
    load A_tile[(k+1)%2], B_tile[(k+1)%2]  // 异步预取
    __syncthreads()
    compute C_partial += A_tile[k%2] * B_tile[k%2]  // 当前计算
```

**3. 寄存器级分块（Thread Tile）：**
- 每个线程计算C矩阵的多个元素（如8x8），而非只算一个
- 数据复用：从共享内存读取的一行A和一列B可被复用多次
- 计算访存比从1:1提升到8:1（外积方式）

**4. 其他关键优化：**

| 技术 | 作用 | 效果 |
|------|------|------|
| 向量化加载(float4/LDS.128) | 一次加载128bit数据 | 减少load指令数 |
| 循环展开(#pragma unroll) | 消除循环开销，提高ILP | 更多指令级并行 |
| 避免Bank Conflict | Padding或Swizzle共享内存 | 消除串行化瓶颈 |
| Tensor Core (WMMA/MMA) | 硬件矩阵乘加单元 | 吞吐提升数倍(A100: 312 TFLOPS FP16) |
| 数据预取(Software Prefetch) | 提前加载下一轮数据到缓存 | 隐藏访存延迟 |

**性能参考：** A100上cuBLAS的GEMM峰值可达硬件理论算力的90%+，手写kernel在标准shape上通常能达到85-95%（Cutlass模板库水平）。

### Q: 深度学习框架前端怎么注册算子的？为什么加个宏定义就能注册了？

框架使用**静态注册机制**，利用C++全局/静态对象在程序加载时（main函数之前）自动执行构造函数的语言特性。

**原理详解：**

```cpp
// 1. 全局注册表（单例map）
class OpRegistry {
    static map<string, OpCreator>& registry() {
        static map<string, OpCreator> r;
        return r;
    }
public:
    static void Register(const string& name, OpCreator creator) {
        registry()[name] = creator;
    }
};

// 2. 注册宏展开为静态对象定义
#define REGISTER_OP(name, cls) \
    static auto __reg_##name = []() { \
        OpRegistry::Register(#name, []() { return new cls(); }); \
        return 0; \
    }();
// 或者用一个辅助注册类：
#define REGISTER_OP(name, cls) \
    static OpRegistrar __reg_##name(#name, []() { return new cls(); });

// 3. 用户只需一行
REGISTER_OP(MyCustomOp, MyCustomOpImpl);
```

**为什么宏定义就能注册？**
- 宏展开后定义了一个**全局/静态对象**
- C++保证全局对象的构造函数在main()之前执行（静态初始化阶段）
- 构造函数执行时自动把算子信息注册到全局map中
- 每个编译单元中的注册宏独立工作，链接后所有算子都已注册

**关键注意事项：**
- Static Initialization Order Fiasco：不同编译单元的全局对象初始化顺序不确定，注册表本身需用局部static保证先于注册代码初始化
- 动态库(DSO)中的注册对象在dlopen时构造，dlclose时析构
- PyTorch使用`TORCH_LIBRARY`宏（基于相同原理），TensorFlow使用`REGISTER_OP`宏

### Q: 有向无环图（DAG）用什么数据结构实现？

**常用表示方式：**

| 表示方式 | 适用场景 | 空间复杂度 | 查询邻居 | 判断边存在 |
|----------|----------|-----------|---------|-----------|
| 邻接表 | 稀疏图（边数远小于V^2） | O(V+E) | O(度数) | O(度数) |
| 邻接矩阵 | 稠密图 | O(V^2) | O(V) | O(1) |
| 边列表 | 特殊算法（如Kruskal） | O(E) | O(E) | O(E) |

**邻接表实现：**
```cpp
// 每个节点维护出边列表
vector<vector<int>> adj(n);         // adj[u] = {v1, v2, ...} 表示u->v1, u->v2
// 带权图
vector<vector<pair<int,int>>> adj;  // adj[u] = {(v, weight), ...}
```

**DAG的特殊性质：**
- 无环 → 一定存在拓扑排序（至少有一个入度为0的节点）
- 可进行动态规划（拓扑序上DP）
- 最长/最短路径可用拓扑排序+DP在O(V+E)内解决

**应用场景：** 任务调度（依赖关系）、编译系统（Makefile依赖）、深度学习计算图、版本控制(git commit graph)、Course prerequisite

### Q: 拓扑排序的实现？

拓扑排序是对DAG的节点进行线性排序，使得对每条有向边(u,v)，u在排序中出现在v之前。

**方法一：Kahn算法（BFS，更直观）：**

```cpp
vector<int> topoSort(int n, vector<vector<int>>& adj) {
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; u++)
        for (int v : adj[u]) indegree[v]++;
    
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (indegree[i] == 0) q.push(i);  // 入度0的节点入队
    
    vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int v : adj[u]) {
            if (--indegree[v] == 0) q.push(v);  // 新的入度0节点
        }
    }
    // 如果order.size() < n，说明有环
    return order;
}
```

**方法二：DFS后序逆序：**
```cpp
void dfs(int u, vector<vector<int>>& adj, vector<bool>& visited, stack<int>& st) {
    visited[u] = true;
    for (int v : adj[u])
        if (!visited[v]) dfs(v, adj, visited, st);
    st.push(u);  // 后序：所有后继都处理完后压栈
}
```

**两种方法对比：**

| 特性 | Kahn (BFS) | DFS后序 |
|------|-----------|---------|
| 时间复杂度 | O(V+E) | O(V+E) |
| 检测环 | 输出节点数 < V | 需要额外标记（三色法） |
| 输出 | 直接得到顺序 | 需要反转 |
| 多个有效排序 | 用优先队列可得字典序最小 | 不方便控制 |
| 并行性发现 | 自然发现可并行的节点（同时入度为0） | 不直观 |

### Q: 二叉树中序遍历？

中序遍历(Inorder Traversal)的访问顺序为：**左子树 → 根节点 → 右子树**。

**递归实现（简洁）：**
```cpp
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);  // 访问根
    inorder(root->right, result);
}
```

**迭代实现（用显式栈模拟递归）：**
```cpp
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;
    while (curr || !st.empty()) {
        while (curr) {           // 一路压入左子节点
            st.push(curr);
            curr = curr->left;
        }
        curr = st.top(); st.pop();
        result.push_back(curr->val);  // 访问
        curr = curr->right;           // 转向右子树
    }
    return result;
}
```

**重要性质：**
- 对**BST（二叉搜索树）**做中序遍历可得到**有序序列**——这是验证BST合法性的经典方法
- Morris遍历可以O(1)空间完成中序遍历（利用线索化——将空的右指针指向后继）
- 时间复杂度O(n)，递归空间O(h)，Morris空间O(1)

### Q: 满二叉树的定义和性质？

满二叉树(Full Binary Tree/Perfect Binary Tree)是每一层节点数都达到最大值的二叉树。

**性质：**
- 深度为k的满二叉树有 **2^k - 1** 个节点
- 第i层有 **2^(i-1)** 个节点（i从1开始）
- 叶子节点数 = 内部节点数 + 1
- 所有叶节点在同一层（最后一层）
- 每个非叶节点恰好有两个子节点

**注意术语区分（中英文定义差异）：**

| 中文概念 | 英文对应 | 定义 |
|----------|----------|------|
| 满二叉树 | Perfect Binary Tree | 每层都满 |
| 完全二叉树 | Complete Binary Tree | 除最后一层外都满，最后一层靠左排列 |
| 真二叉树 | Full/Proper Binary Tree | 每个节点要么0个要么2个子节点 |

完全二叉树适合用数组存储（堆的底层结构），满二叉树是完全二叉树的特例。

### Q: 单例模式如何实现？

**C++11推荐写法——Meyers' Singleton（利用局部static线程安全特性）：**

```cpp
class Singleton {
public:
    static Singleton& getInstance() {
        static Singleton instance;  // 线程安全，延迟初始化
        return instance;
    }
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
private:
    Singleton() = default;
    ~Singleton() = default;
};
```

**为什么这是最佳实现：**
- C++11标准保证局部static变量的初始化是线程安全的（编译器插入同步机制）
- 延迟初始化（首次调用时才创建），避免不必要的资源占用
- 代码极简，无需手动加锁
- 自动调用析构函数（程序退出时）

**编译器底层实现原理（GCC为例）：**
```cpp
// 编译器大致生成的代码
static bool __guard = false;
static char __storage[sizeof(Singleton)];
if (!__guard) {  // 实际用原子操作+互斥锁实现
    lock();
    if (!__guard) {
        new (__storage) Singleton();
        __guard = true;
    }
    unlock();
}
```

### Q: C++模板函数是怎么编译的？

模板函数采用**编译时实例化**（instantiation）机制——编译器遇到模板被具体类型使用时，为该类型生成一份特化代码。

**编译过程：**

```
源代码定义模板 → 编译器解析模板语法（不生成代码）
                          ↓
            遇到具体类型调用（如 max<int>(a,b)）
                          ↓
           为int类型生成特化函数代码（隐式实例化）
                          ↓
           每个编译单元独立生成 → 链接器去重（COMDAT）
```

**关键规则：**
- 模板定义通常放在**头文件**中（因为编译器实例化时需要看到完整定义）
- 如果分离声明和定义到.cpp文件，需要显式实例化：`template int max<int>(int, int);`
- Two-Phase Lookup：第一阶段检查不依赖模板参数的名称，第二阶段（实例化时）检查依赖模板参数的名称

**编译时间和代码膨胀问题：**
- 每种类型组合生成一份代码（`vector<int>`和`vector<double>`是完全不同的类）
- 大量模板实例化导致编译时间长、目标文件大
- 缓解方法：extern template（C++11）声明在其他编译单元已实例化、减少不必要的模板参数组合

### Q: C++ STL迭代器怎么安全删除元素？

迭代器失效是C++容器操作中最常见的bug来源之一。安全删除的关键是**正确获取删除后的有效迭代器**。

**序列容器（vector/deque）——erase返回下一个有效迭代器：**
```cpp
vector<int> vec = {1, 2, 3, 4, 5};
for (auto it = vec.begin(); it != vec.end(); ) {
    if (*it % 2 == 0)
        it = vec.erase(it);  // erase返回指向下一个元素的迭代器
    else
        ++it;
}
```

**关联容器（map/set）：**
```cpp
// C++11后：erase也返回下一个迭代器
for (auto it = myMap.begin(); it != myMap.end(); ) {
    if (shouldRemove(it->second))
        it = myMap.erase(it);
    else
        ++it;
}

// C++11前的经典写法（后置++在erase前保存下一个位置）
for (auto it = myMap.begin(); it != myMap.end(); ) {
    if (shouldRemove(it->second))
        myMap.erase(it++);  // 先复制it，再++，再erase旧的
    else
        ++it;
}
```

**各容器迭代器失效规则：**

| 容器 | insert失效范围 | erase失效范围 |
|------|---------------|---------------|
| vector | 插入点及之后所有（可能触发重新分配则全部失效） | 删除点及之后所有 |
| deque | 中间插入则全部失效；首尾插入不失效 | 中间删除全部失效；首尾删除仅该元素 |
| list | 不失效 | 仅被删除的迭代器失效 |
| map/set | 不失效 | 仅被删除的迭代器失效 |
| unordered_map | 可能rehash导致全部失效 | 仅被删除的迭代器失效 |

**C++20推荐写法——std::erase_if：**
```cpp
std::erase_if(vec, [](int x) { return x % 2 == 0; });
```

### Q: C++ STL基本容器有哪些？

**STL容器分类及特性对比：**

**1. 序列容器（有序排列，按插入顺序）：**

| 容器 | 底层 | 随机访问 | 头部操作 | 尾部操作 | 中间操作 |
|------|------|---------|---------|---------|---------|
| vector | 动态数组 | O(1) | O(n) | 均摊O(1) | O(n) |
| deque | 分段连续数组 | O(1) | O(1) | O(1) | O(n) |
| list | 双向链表 | O(n) | O(1) | O(1) | O(1) |
| forward_list | 单向链表 | O(n) | O(1) | O(n) | O(1) |
| array (C++11) | 固定数组 | O(1) | - | - | - |

**2. 关联容器（按key有序排列，红黑树实现）：**
- `set` / `multiset`：有序集合，查找/插入/删除 O(log n)
- `map` / `multimap`：有序键值对，查找/插入/删除 O(log n)

**3. 无序关联容器（哈希表实现，C++11）：**
- `unordered_set` / `unordered_multiset`：平均O(1)查找
- `unordered_map` / `unordered_multimap`：平均O(1)查找

**4. 容器适配器（基于其他容器封装）：**
- `stack`：LIFO，默认基于deque
- `queue`：FIFO，默认基于deque
- `priority_queue`：堆，默认基于vector

**选择指南：**
- 需要随机访问+尾部增删 → vector（最常用）
- 需要频繁中间插入删除 → list
- 需要有序+快速查找 → map/set
- 需要最快查找（无序） → unordered_map/unordered_set
- 需要固定大小 → array
