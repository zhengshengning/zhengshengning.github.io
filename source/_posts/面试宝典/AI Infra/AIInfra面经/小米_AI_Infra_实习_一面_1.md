---
title: '小米 AI Infra 实习 一面 (1)'
date: 2026-04-16 12:18:47
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 如何实现膨胀卷积（Dilated Convolution）的CUDA kernel，并进行优化？

膨胀卷积（又称空洞卷积、Atrous Convolution）在标准卷积基础上引入dilation rate，使卷积核元素间有间隔，**在不增加参数和计算量的情况下指数级扩大感受野**。

**膨胀卷积的数学关系**：
- 有效卷积核大小 = kernel_size + (kernel_size - 1) × (dilation - 1)
- 例如：3×3 kernel + dilation=2 → 等效5×5感受野，但仅9个参数和9次乘加。

**CUDA实现要点**：

```cuda
// 核心索引计算
int input_h = output_h * stride + kh * dilation;  // 注意dilation影响采样位置
int input_w = output_w * stride + kw * dilation;
if (input_h >= 0 && input_h < H && input_w >= 0 && input_w < W) {
    sum += input[n][c][input_h][input_w] * weight[oc][c][kh][kw];
}
```

**优化策略分析**：

1. **共享内存缓存输入tile**：
   - 挑战：dilation使得输入采样稀疏。一个输出tile需要的输入区域大小 = output_tile + (kernel_size-1) × dilation，远大于标准卷积。
   - dilation=4、kernel=3×3时，输出8×8 tile需要加载 (8+2×4)=16 × 16 的输入区域（256 elements），但实际只使用其中的稀疏位置。
   - 解决：只加载实际需要的稀疏位置到shared memory，用间接索引表避免无用数据加载。

2. **Im2col转GEMM**：
   - 将膨胀卷积展开为矩阵乘法形式，利用cuBLAS/Tensor Core加速。
   - 隐式GEMM（cuDNN方式）避免显式im2col的额外内存开销。

3. **访存优化**：
   - 向量化加载：当dilation使得连续输出对应的输入地址跨度为1时，可用float4加载。
   - 数据layout：NHWC格式下channel维度连续，便于向量化。

4. **小kernel循环展开**：
   - 3×3 kernel（最常见）可将9次乘加完全展开，消除循环控制开销和分支预测失败。

**性能瓶颈分析**：dilation越大，输入访问越稀疏，L1/L2 cache命中率越低。对于大dilation（如16/32），建议使用FFT方法或直接依赖cuDNN的自动调优选择最佳实现。
