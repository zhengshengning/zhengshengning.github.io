---
title: '元戎启行 AI Infra 校招 一面 (2)'
date: 2026-04-16 12:19:36
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: CUDA实现Concat算子，如果有四个维度，每个维度该如何拼接？

Concat算子沿指定维度拼接多个tensor，CUDA实现的核心挑战是**高效的地址映射**——从输出的全局线性索引计算出对应的输入tensor编号和源地址偏移。

**不同拼接维度的实现策略**：

**沿最外层维度（dim=0）拼接**——最简单：
```
输入A: [2,3,4,5], 输入B: [3,3,4,5]
输出: [5,3,4,5]
```
- 每个输入tensor在内存中是完整的连续块。
- 直接按顺序拷贝：先拷贝A的全部元素（2×3×4×5=120个），再拷贝B的全部元素（3×3×4×5=180个）。
- 可以直接用 `cudaMemcpy` 或高效的 `float4` 向量化拷贝。

**沿最内层维度（dim=3）拼接**——内存连续性最好：
```
输入A: [2,3,4,5], 输入B: [2,3,4,3]
输出: [2,3,4,8]
```
- 每个"行"（最内层维度）内的元素连续，拼接后的行也连续。
- 可以用float4向量化拷贝每一行的数据。

**沿中间维度（dim=1或dim=2）拼接**——最通用但最复杂：
```
输入A: [2,3,4,5], 输入B: [2,2,4,5]  (沿dim=1拼接)
输出: [2,5,4,5]
```
- 需要逐"切片（slice）"拷贝：对于输出的每个(n, c, h, w)位置，确定它来自哪个输入tensor。
- 关键是预计算每个输入tensor在concat维度上的累积偏移。

**通用CUDA实现**：
```cuda
__global__ void concat_kernel(float* output, float** inputs, 
                              int* concat_offsets, int num_inputs,
                              int concat_dim, int* dims, int ndim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;
    
    // 从输出全局索引反算各维度坐标
    int coords[4];
    int remaining = tid;
    for (int d = 0; d < ndim; d++) {
        coords[d] = remaining / output_strides[d];
        remaining %= output_strides[d];
    }
    
    // 确定concat维度上的坐标落在哪个输入tensor中
    int concat_coord = coords[concat_dim];
    int input_idx = 0;
    while (input_idx < num_inputs - 1 && concat_coord >= concat_offsets[input_idx + 1])
        input_idx++;
    
    // 计算源偏移
    coords[concat_dim] -= concat_offsets[input_idx];
    int src_offset = /* 用coords计算在输入tensor中的线性偏移 */;
    
    output[tid] = inputs[input_idx][src_offset];
}
```

**关键优化**：
1. **沿最后一维concat时用向量化拷贝**：连续元素直接float4搬运。
2. **预计算偏移表**：`concat_offsets[]` 放constant memory或传入kernel，避免循环查找。
3. **合并内存访问**：保证warp内线程访问连续的输出地址（自然满足，因为按output tid线性映射）。
4. **大维度拆分**：如果concat维度很大，用二分查找替代线性查找确定input_idx。
5. **特化kernel**：对常见case（如dim=0、dim=-1）编写特化版本，避免通用版本的开销。

---

### Q: 手撕：实现LRU Cache？

（编程题）
