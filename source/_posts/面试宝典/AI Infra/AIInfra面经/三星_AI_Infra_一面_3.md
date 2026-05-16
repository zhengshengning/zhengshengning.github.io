---
title: '三星 AI Infra 一面 (3)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 编译器IR转换的过程？

编译器IR（Intermediate Representation，中间表示）转换是将源代码逐步降低（lowering）为机器码的过程。每一层IR在不同的抽象级别上进行针对性优化。

**典型编译流程（以LLVM为例）：**

```
源代码 → AST（抽象语法树）
         ↓ 语义分析
    高层IR（保留语义信息：循环结构、类型系统）
         ↓ 高层优化（常量折叠、内联、死代码消除）
    中层IR（LLVM IR：SSA形式，与平台无关）
         ↓ 中层优化（循环优化、向量化、标量优化）
    低层IR（接近机器码：SelectionDAG → MachineIR）
         ↓ 后端优化（寄存器分配、指令调度、peephole优化）
    目标机器码
```

**各层IR的设计目的：**

| IR层级 | 代表 | 保留的信息 | 主要优化 |
|--------|------|-----------|---------|
| 高层IR | Clang AST, TorchScript | 类型系统、控制流语义 | 函数内联、常量传播 |
| 中层IR | LLVM IR, MLIR | SSA形式、内存模型 | 循环变换、向量化、别名分析 |
| 低层IR | MachineIR, SASS | 寄存器、指令编码 | 寄存器分配、指令选择、调度 |

**MLIR的多层Dialect设计：**

MLIR是LLVM的子项目，通过Dialect机制支持任意数量的IR层级：
- `linalg` dialect：高层线性代数操作
- `affine` dialect：仿射循环和内存访问分析
- `scf` dialect：结构化控制流（for/if/while）
- `gpu` dialect：GPU线程层次映射
- `llvm` dialect：对接LLVM IR

每层保留该层需要的语义信息做优化，渐进式lowering避免信息丢失——这是MLIR相比传统编译器（只有少数IR层）的核心优势。

### Q: 算子融合的原理和方式？

算子融合（Op Fusion）将多个独立算子合并为一个kernel执行，是深度学习编译器中最重要的优化之一。

**为什么融合能加速？**
- **减少显存读写**：中间结果留在寄存器/共享内存中，不需要写回HBM再重新读取。对于element-wise算子，IO开销可能比计算本身大10倍以上
- **减少Kernel Launch开销**：每次launch约5-10us，串联多个小kernel的launch开销显著
- **提高缓存利用率**：数据在高速缓存中被连续使用

**融合模式分类：**

| 模式 | 示例 | 数据流特征 |
|------|------|-----------|
| Element-wise融合 | ReLU + Add + Mul | 一对一映射，直接合并计算 |
| Reduce融合 | Element-wise + Reduce | 先逐元素再规约，一个kernel内完成 |
| Broadcast融合 | Bias + Broadcast + Add | 合并广播与计算 |
| 复合算子融合 | Conv + BN + ReLU | 利用BN是element-wise的特性 |
| 注意力融合 | FlashAttention | QK^T + softmax + V全部在SRAM中完成 |
| 通信计算融合 | GEMM + AllReduce | 计算完成的部分立即开始通信 |

**框架如何实现融合？**
- **图级融合（静态）**：TVM/XLA/TensorRT在计算图层面pattern matching识别可融合子图，生成融合kernel
- **JIT融合（动态）**：PyTorch的torch.compile/NVFuser对运行时计算图做即时融合
- **手动融合**：库级别预先实现常见融合组合（如cuDNN的fused Conv+BN+ReLU）

**融合的限制条件：**
- 不能跨Reduce维度融合（数据依赖）
- 融合后的kernel不能超过寄存器/共享内存限制
- 动态shape可能影响融合决策

### Q: Attention的实现原理？

Self-Attention是Transformer的核心机制，允许序列中任意两个位置直接交互。

**计算公式：**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V$$

**详细计算步骤：**

1. **线性投影**：从输入X计算Q、K、V
   - Q = X · W_Q, K = X · W_K, V = X · W_V
   - 维度：X: [seq_len, d_model], W_Q/K/V: [d_model, d_k]

2. **注意力分数计算**：S = QK^T / sqrt(d_k)
   - QK^T: [seq_len, seq_len]，衡量每对token间的相似度
   - **除以sqrt(d_k)的原因**：当d_k较大时，QK^T的方差约为d_k，导致softmax梯度消失（进入饱和区）。除以sqrt(d_k)使方差恢复为1，保持梯度稳定

3. **Softmax归一化**：A = softmax(S)，将分数转为概率分布（每行和为1）

4. **加权求和**：Output = A · V，用注意力权重对V做加权聚合

**多头注意力（Multi-Head Attention）：**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(X·W_Qi, X·W_Ki, X·W_Vi)
```
- 将d_model拆分为h个头，每个头维度d_k = d_model / h
- 不同头学习不同的注意力模式（如语法关系、语义关系等）
- 计算量与单头相同（总参数量不变），但表达能力更强

**计算复杂度分析：**
- QK^T矩阵乘：O(n^2 * d)
- Softmax：O(n^2)
- AV矩阵乘：O(n^2 * d)
- 总计：O(n^2 * d)，n为序列长度
- 显存：需要存储n*n的attention矩阵（FlashAttention优化此问题）

### Q: Softmax的公式和数值稳定性？

**标准Softmax公式：**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**数值稳定性问题：**
- 当x_i很大时（如>100），exp(x_i)溢出为inf → 结果为nan
- 当x_i很小时（如<-700），exp(x_i)下溢为0 → 分母为0

**Safe Softmax（减去最大值）：**
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

数学上等价（分子分母同除以e^max），但保证指数的最大输入为0，避免溢出。

**实现需要两遍扫描：**
1. 第一遍：找到max(x)
2. 第二遍：计算exp(x_i - max)和sum

**Online Softmax（FlashAttention使用，单遍/分块计算）：**

核心思想是增量更新：处理新的数据块时，如果发现新的max值，修正之前的累积结果。

```python
# Online softmax伪代码
m = -inf  # 当前最大值
d = 0     # 当前分母累积
for block in blocks:
    m_new = max(m, max(block))
    d = d * exp(m - m_new) + sum(exp(block - m_new))  # 修正旧的d
    m = m_new
```

这使得softmax可以分块计算而不需要预先知道全局max——FlashAttention利用这一特性在分块处理QK^T时增量维护softmax状态。

### Q: 卷积算子的优化方法？

卷积是CNN的核心算子，有多种实现和优化策略，各适用于不同场景：

**1. Im2col + GEMM（最通用）：**
- 将卷积转为矩阵乘法：将输入按卷积窗口展开为矩阵，卷积核展开为矩阵，做GEMM
- 优点：可直接利用高度优化的BLAS库（cuBLAS）
- 缺点：需要额外的展开空间（显存开销增加数倍）
- cuDNN中的主要实现方式之一

**2. Winograd变换（小核加速）：**
- 通过数学变换减少乘法次数：F(2x2, 3x3)只需16次乘法代替36次
- 适合3x3等小卷积核，加速比可达2-4倍
- 缺点：数值精度稍差（FP16下误差可能累积）、只适合小核

**3. FFT卷积（大核加速）：**
- 时域卷积 = 频域逐元素乘法
- 对于大卷积核(>7x7)效率更高
- 缺点：FFT变换本身有开销，对小核不划算

**4. 直接卷积（Direct Convolution）：**
- 直接按卷积定义计算，通过分块、向量化、数据重排优化
- 适合内存受限场景（无需展开空间）
- Arm CPU上常用此方式配合NEON指令

**各方法适用场景对比：**

| 方法 | 最佳场景 | 计算复杂度 | 额外内存 |
|------|----------|-----------|---------|
| Im2col + GEMM | 通用，GPU主流 | 与标准卷积相同 | 大（展开后矩阵） |
| Winograd | 3x3核，stride=1 | 减少~2.25x乘法 | 中等 |
| FFT | 大核(>7x7) | O(N^2 log N) | 大（频域表示） |
| Direct | 内存受限、特殊shape | 标准 | 无 |

**5. 算子融合优化：**
- Conv + BN + ReLU融合：BN在推理时等价于线性变换，可以吸收到Conv的权重和bias中
- 即：W_fused = W * gamma / sqrt(var + eps), b_fused = (b - mean) * gamma / sqrt(var + eps) + beta

### Q: 拓扑排序的实现？

**BFS方法（Kahn算法）：**

```cpp
vector<int> topoSort(int n, vector<vector<int>>& adj) {
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; u++)
        for (int v : adj[u]) indegree[v]++;
    
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (indegree[i] == 0) q.push(i);
    
    vector<int> result;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        result.push_back(u);
        for (int v : adj[u])
            if (--indegree[v] == 0) q.push(v);
    }
    return result;  // result.size() < n 表示有环
}
```

**核心思想：** 不断移除入度为0的节点，直到图为空或无法继续（有环）。时间复杂度O(V+E)。

**应用场景：**
- 编译系统的依赖构建顺序（Makefile/Bazel）
- 深度学习计算图的执行顺序调度
- 课程先修关系排列
- 检测图中是否有环（若无法输出所有节点则有环）

**并行性发现：** 同一轮中所有入度为0的节点可以并行处理——这在任务调度中非常重要。

### Q: BFS和DFS的区别和应用？

**核心对比：**

| 维度 | BFS（广度优先） | DFS（深度优先） |
|------|----------------|----------------|
| 数据结构 | 队列 | 栈（递归/显式栈） |
| 搜索策略 | 逐层扩展（由近及远） | 沿一条路径深入再回溯 |
| 空间复杂度 | O(最大层宽度) | O(最大深度) |
| 最短路径 | 保证（无权图） | 不保证 |
| 完整遍历 | 先近后远 | 先深后广 |

**BFS最佳应用：**
- 无权图最短路径（如迷宫最少步数、单词接龙最短变换）
- 多源BFS（如0-1矩阵中每个1到最近0的距离）
- 层序遍历（二叉树按层输出、拓扑排序）
- 状态空间中最少操作次数（如华容道）

**DFS最佳应用：**
- 回溯搜索（N皇后、数独、排列组合）
- 拓扑排序（后序逆序）
- 强连通分量（Tarjan/Kosaraju）
- 连通性判断、环检测
- 树上问题（直径、LCA、路径搜索）

**空间使用场景分析：**
- 图很宽（如完全二叉树底层2^h个节点）→ BFS空间爆炸，用DFS
- 图很深（如链式结构）→ DFS可能栈溢出，用BFS或迭代加深DFS(IDDFS)
- 需要最短路径 → 必须BFS（DFS找到的路径不一定最短）
