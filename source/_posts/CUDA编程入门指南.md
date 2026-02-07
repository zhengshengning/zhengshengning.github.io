---
title: CUDA编程入门指南
date: 2026-02-07 18:30:00
tags: [CUDA, GPU编程, 并行计算, NVIDIA]
categories: CUDA编程
---

# CUDA编程入门指南

CUDA (Compute Unified Device Architecture) 是 NVIDIA 推出的并行计算平台和编程模型，它允许开发者利用 GPU 的强大并行计算能力来加速各种应用程序。

<!-- more -->

## 什么是 CUDA？

CUDA 是一种通用并行计算架构，它使得 GPU 能够解决复杂的计算问题。通过 CUDA，开发者可以：

- 利用 GPU 的数千个核心进行并行计算
- 大幅提升计算密集型任务的性能
- 使用 C/C++ 等熟悉的编程语言

## 环境配置

### 1. 硬件要求

- NVIDIA GPU（支持 CUDA 的显卡）
- 足够的显存（建议至少 2GB）

### 2. 软件安装

**安装 CUDA Toolkit：**

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# 设置环境变量
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

**验证安装：**

```bash
nvcc --version
nvidia-smi
```

## 第一个 CUDA 程序

### Hello World 示例

创建文件 `hello_cuda.cu`：

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA 核函数
__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");
    
    // 启动 10 个线程的核函数
    helloFromGPU<<<1, 10>>>();
    
    // 等待 GPU 完成
    cudaDeviceSynchronize();
    
    return 0;
}
```

**编译运行：**

```bash
nvcc hello_cuda.cu -o hello_cuda
./hello_cuda
```

## CUDA 编程基础概念

### 1. 核函数（Kernel）

核函数是在 GPU 上执行的函数，使用 `__global__` 关键字声明：

```cpp
__global__ void myKernel(int *data) {
    int idx = threadIdx.x;
    data[idx] = data[idx] * 2;
}
```

### 2. 线程层次结构

CUDA 使用三层线程组织结构：

- **Grid（网格）**：所有线程的集合
- **Block（线程块）**：线程的分组
- **Thread（线程）**：最小执行单元

```cpp
// 启动核函数：2 个 Block，每个 Block 有 256 个线程
myKernel<<<2, 256>>>(data);
```

### 3. 内存管理

CUDA 提供了显式的内存管理函数：

```cpp
// 分配设备内存
int *d_data;
cudaMalloc(&d_data, size * sizeof(int));

// 数据传输：主机 -> 设备
cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

// 数据传输：设备 -> 主机
cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

// 释放设备内存
cudaFree(d_data);
```

## 向量加法示例

完整的向量加法程序：

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000

// 核函数：向量加法
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    size_t bytes = N * sizeof(float);
    
    // 分配主机内存
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 分配设备内存
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // 数据传输：主机 -> 设备
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // 启动核函数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // 数据传输：设备 -> 主机
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.2f\n", i, h_c[i]);
    }
    
    // 释放内存
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

## 性能优化技巧

### 1. 合理选择线程块大小

```cpp
// 推荐：256 或 512
int threadsPerBlock = 256;
```

### 2. 使用共享内存

```cpp
__global__ void useSharedMemory() {
    __shared__ float shared_data[256];
    shared_data[threadIdx.x] = /* 某个值 */;
    __syncthreads();  // 同步所有线程
}
```

### 3. 内存合并访问

确保相邻线程访问相邻内存地址，以提高带宽利用率。

### 4. 避免分支发散

尽量减少线程束（warp）内的条件分支。

## 常用库推荐

- **cuBLAS**：线性代数运算
- **cuFFT**：快速傅里叶变换
- **cuDNN**：深度学习加速
- **Thrust**：C++ 模板库，类似 STL

## 调试工具

- **cuda-gdb**：CUDA 调试器
- **nvidia-smi**：GPU 监控工具
- **nvprof / nsys**：性能分析工具

```bash
# 监控 GPU 使用情况
nvidia-smi -l 1

# 性能分析
nvprof ./your_cuda_app
```

## 学习资源

1. **官方文档**：[NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
2. **在线课程**：Coursera、Udacity 上的 GPU 编程课程
3. **开源项目**：GitHub 上的 CUDA 示例代码
4. **书籍推荐**：《CUDA by Example》、《Professional CUDA C Programming》

## 总结

CUDA 编程为开发者提供了强大的并行计算能力，通过本文的介绍，您应该已经掌握了：

- CUDA 的基本概念和架构
- 如何编写和编译 CUDA 程序
- 内存管理和数据传输
- 线程组织和核函数调用
- 基本的性能优化技巧

随着实践的深入，您将能够利用 GPU 加速更复杂的应用，如科学计算、深度学习、图像处理等领域。

**开始您的 CUDA 之旅吧！** 🚀

---

*本文最后更新时间：2026-02-07*