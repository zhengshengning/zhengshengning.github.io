---
title: '太初 AI Infra 实习 一面 (2)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 手撕Softmax并在此基础上做并行优化？

**Softmax标准实现（数值稳定版）：**

```cpp
// 串行版本 - 需要两遍扫描
void softmax(float* input, float* output, int n) {
    // Pass 1: 找最大值
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++)
        max_val = fmaxf(max_val, input[i]);
    
    // Pass 2: 计算exp和sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    // Pass 3: 归一化
    for (int i = 0; i < n; i++)
        output[i] /= sum;
}
```

**GPU并行优化思路——多级规约：**

**Level 1: Warp级并行（32线程协作处理一行）**
```cuda
// 利用warp shuffle做高效reduce
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;  // lane 0持有结果
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

**Level 2: Block级并行（多个warp协作）**
```cuda
// 共享内存做跨warp规约
__shared__ float shared_max[32];  // 每个warp的局部max
__shared__ float shared_sum[32];

// 每个warp内先reduce，然后warp间通过shared memory汇总
```

**Level 3: Online Softmax（分块增量计算，FlashAttention核心）：**

```python
# Online Softmax伪代码 - 单遍扫描
m_prev = -inf   # 当前已知最大值
d_prev = 0      # 当前分母累积
o_prev = 0      # 当前输出累积

for block_i in range(num_blocks):
    x_block = load(input[block_i])
    m_new = max(m_prev, max(x_block))
    
    # 修正因子：旧的max需要更新
    correction = exp(m_prev - m_new)
    
    # 更新分母
    d_new = d_prev * correction + sum(exp(x_block - m_new))
    
    # 更新输出（如果在做attention）
    o_new = o_prev * correction + exp(x_block - m_new) @ V_block
    
    m_prev, d_prev, o_prev = m_new, d_new, o_new

output = o_prev / d_prev
```

**为什么Online Softmax重要？**
- 标准Softmax需要两遍扫描（先找max，再计算）→ 数据从HBM读两次
- Online Softmax只需一遍扫描 → 数据只读一次，带宽节省50%
- 关键：不需要预先知道全局max，通过增量修正实现正确结果
- FlashAttention利用此特性在分块处理QK^T时无需存储完整N×N矩阵

**GPU实现选择策略：**
- 行长度 <= 32：一个warp处理一行（warp shuffle即可）
- 行长度 <= 1024：一个block处理一行（shared memory跨warp reduce）
- 行长度 > 1024：多个block处理一行（需要全局同步或两阶段reduce）
