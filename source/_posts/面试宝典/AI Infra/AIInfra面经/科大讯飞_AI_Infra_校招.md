---
title: '科大讯飞 AI Infra 校招'
date: 2026-04-16 12:09:58
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 高性能计算, 面经]
---


<!-- more -->

---

### Q: Flash Attention 的核心优化点是什么？

FlashAttention 通过**重新组织计算顺序**来减少对 HBM 的访问，其核心优化点包括：

**1. 分块加载（Tiling）**：
- 将 Q/K/V 分块加载到 GPU 的 SRAM（Shared Memory + Register File，总共 ~20 MB per SM）中计算
- 避免在 HBM 中物化完整的 N×N attention 矩阵（seq_len=4096 时约 32MB/head）
- 分块大小取决于 SRAM 容量，通常 block_size = 64-256 tokens

**2. Online Softmax（核心数学创新）**：
- 传统 softmax 需要两遍扫描：第一遍求全局 max/sum，第二遍计算 exp/normalize
- Online Softmax 只需一遍：维护 running max `m` 和 running sum `l`，每处理一个新 block 增量更新：
  ```
  m_new = max(m_old, max(S_block))
  l_new = l_old × exp(m_old - m_new) + sum(exp(S_block - m_new))
  O_new = O_old × (l_old×exp(m_old-m_new)/l_new) + softmax_block × V_block / l_new
  ```
- 数学上与标准 softmax **完全等价**（非近似），精度无损

**3. 显存优化**：
- 不存储 N×N 的 S 矩阵和 P 矩阵（传统实现存这两个 O(N²) 的中间结果）
- 只保存最终输出 O 和 logsumexp（log(l) + m），用于反向传播时重计算
- 显存复杂度：O(N²) → **O(N)**

**4. 减少 HBM 读写**：
- 标准实现 HBM 访问量：O(N²·d)（读写 S 和 P 矩阵）
- FlashAttention HBM 访问量：O(N²·d²/M)，其中 M 是 SRAM 大小
- 当 d（head_dim，通常 64-128）远小于 M（~数 MB）时，大幅减少 HBM 访问
- 实测：对 seq_len=2048 约减少 **5-9x** 的 HBM 读写量

---

### Q: Self-Attention 为什么要除以 √d？

**直观解释**：当 head_dim（d）较大时，Q 和 K 的点积值绝对值会变大，导致 softmax 进入饱和区（梯度几乎为 0），模型无法有效学习。

**数学推导**：
- 假设 Q、K 的每个元素独立且均值为 0、方差为 1
- Q 和 K 的点积 `q·k = Σ_{i=1}^d q_i × k_i`
- 均值：E[q·k] = 0
- **方差：Var[q·k] = d**（d 个独立乘积项方差之和）
- 即点积的标准差 ∝ √d，d=128 时标准差 ≈ 11.3

**不除以 √d 的后果**：
- 点积值分布在 [-30, 30]（d=128 时）
- softmax 输入这么大的值后：softmax([30, -30]) ≈ [1.0, 0.0]
- 梯度 softmax'(x) ≈ 0（饱和区），模型几乎无法学习 attention pattern

**除以 √d 后**：
- 点积值归一化为方差 ≈ 1，分布在 [-3, 3] 范围
- softmax 处于梯度敏感区域，训练正常
- 这是缩放点积注意力（Scaled Dot-Product Attention）名称的由来

---

### Q: 回调函数怎么实现？

回调函数（Callback）是一种将**控制权反转**的设计模式——调用方不关心具体实现，被调用方在适当时机执行传入的函数。

**C++ 中的实现方式**（按演进顺序）：

**1. 函数指针**（最基础，C 风格）：
```cpp
typedef void (*Callback)(int result);
void asyncTask(Callback cb) { cb(42); }
```
局限：无法携带状态（无闭包），不支持成员函数

**2. std::function + Lambda**（现代 C++ 推荐）：
```cpp
void asyncTask(std::function<void(int)> cb) {
    cb(42);
}
// 调用：支持闭包，可捕获外部状态
int x = 10;
asyncTask([&x](int result) { x += result; });
```
优点：类型安全、支持闭包、可存储任意可调用对象。代价：可能有堆分配（小对象优化可避免）

**3. 虚函数/接口**（OOP 风格）：
```cpp
class ICallback {
    virtual void onComplete(int result) = 0;
};
```
适合需要多个回调方法的场景（如事件监听器接口）

**4. 模板参数**（编译期绑定，零开销）：
```cpp
template<typename F>
void asyncTask(F&& cb) { cb(42); }
```
编译期确定类型，可完全内联，性能最优。缺点：不支持运行时切换，代码膨胀

**选择建议**：
- 性能敏感路径：模板参数（零开销）
- 通用接口：`std::function`（灵活性最好）
- C 兼容/系统级：函数指针

---

### Q: 显存越界（Out-of-bounds access）怎么排查？

CUDA 中显存越界是最常见的 bug 之一，表现为结果错误、程序崩溃或 `cudaError: illegal memory access`。

**排查工具和方法**：

**1. compute-sanitizer（首选）**：
```bash
compute-sanitizer --tool memcheck ./program
```
- 能检测：越界读写、未对齐访问、未初始化读取、race condition
- 缺点：运行速度慢 10-100x，适合 debug build
- 输出精确到 kernel 名称、线程 ID、访问地址

**2. CUDA_LAUNCH_BLOCKING=1**：
```bash
CUDA_LAUNCH_BLOCKING=1 ./program
```
- 强制所有 kernel 同步执行（默认异步）
- 这样 `cudaGetLastError()` 能定位到具体哪个 kernel 出错
- 否则错误报告可能延迟到后续 kernel

**3. 代码级边界检查**：
```cuda
__global__ void kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;  // 边界保护！
    data[idx] = ...;
}
```
这是最根本的预防措施——永远检查索引合法性

**4. printf 调试**：
```cuda
if (idx >= N) {
    printf("OOB! thread=(%d,%d,%d), block=(%d,%d,%d), idx=%d, N=%d\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z, idx, N);
}
```

**5. Nsight Compute/Systems**：通过 profiler 观察异常行为（如 global memory access 异常模式）

**常见越界原因**：
- 线程索引计算错误（二维/三维 grid 的索引公式出错）
- 共享内存越界（未考虑 padding 或 bank conflict 处理时数组大小不够）
- 未正确处理边界 block（最后一个 block 线程数 > 实际剩余数据量）
- 多维 tensor 的 stride 计算错误
- 动态 shape 变化但 grid 配置未更新
