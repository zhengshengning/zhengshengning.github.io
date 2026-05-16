---
title: '小厂 AI Infra 实习 (4)'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 其他公司面经]
tags: [AIInfra, 推理优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: TensorRT量化是PTQ吗？为什么用TensorRT量化？

**TensorRT主要支持PTQ（Post-Training Quantization），也有限支持QAT。**

**TensorRT PTQ校准流程：**

```
┌──────────────────────────────────────────────────────────┐
│ 1. 准备校准数据集 (通常500-1000张代表性输入)               │
│ 2. FP32模型前向推理收集每层激活值分布                       │
│ 3. 为每层确定最优量化阈值(threshold):                      │
│    ├── EntropyCal: 最小化KL散度(FP32分布 vs 量化后分布)    │
│    ├── MinMax: 直接用min/max作为阈值(简单但可能不最优)      │
│    └── Percentile: 截断极端值(如99.99%分位)               │
│ 4. 生成量化后的TensorRT Engine                            │
└──────────────────────────────────────────────────────────┘
```

**EntropyCal(KL散度校准)的原理：**

```
目标: 找到threshold T使得:
  原始FP32分布 P 与 量化后分布 Q 的KL散度最小
  
  KL(P||Q) = Σ P(x) × log(P(x)/Q(x))

搜索过程:
  对每个候选T (从max/2到max):
    1. 将[-T, T]映射到INT8的[-128, 127]
    2. 计算量化→反量化后的分布Q
    3. 计算KL(P||Q)
  选择KL最小的T作为该层的量化阈值
  
效果: 比简单MinMax校准精度好1-3%(某些层显著)
```

**为什么选择TensorRT量化？**

| 优势 | 解释 |
|------|------|
| 端到端优化 | 量化+层融合+kernel选择一体完成 |
| 硬件适配 | 针对NVIDIA GPU的INT8 Tensor Core深度优化 |
| 校准便捷 | 只需几百张图片,几分钟完成 |
| 部署一体 | 量化Engine直接运行,无需额外runtime |
| 性能极致 | TensorRT的INT8 kernel经过精心优化 |
| 精度可控 | 可以指定某些层保持FP16(sensitive layers) |

**TensorRT INT8 vs 其他量化方案：**

| 方案 | 精度 | 推理引擎 | 适用场景 |
|------|------|---------|---------|
| TensorRT INT8 | 高(entropy校准) | TensorRT | NVIDIA GPU生产部署 |
| ONNX Runtime INT8 | 中 | ORT | 跨平台 |
| PyTorch Native | 中 | PyTorch | 研究/原型 |
| TVM INT8 | 中 | TVM Runtime | 多硬件 |

---

### Q: CUDA加速图像预处理的具体实现方法和优化手段？

**GPU图像预处理Pipeline：**

```
传统CPU预处理:
  读图(decode) → resize → color_convert → normalize → to_tensor → to_GPU
  瓶颈: resize和normalize在CPU上慢, CPU→GPU传输额外开销

GPU预处理(全在GPU上):
  读图(CPU) → to_GPU(原始bytes) → decode(GPU) → resize → normalize
  所有变换在GPU并行完成, 无中间传输
```

**CUDA Kernel实现示例：**

```cuda
// 融合的resize + normalize kernel:
__global__ void preprocess_kernel(
    const uchar* input,   // 原始图像 [H, W, 3]
    float* output,        // 输出tensor [3, new_H, new_W]
    int src_h, int src_w,
    int dst_h, int dst_w,
    float mean[3], float std[3]
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 目标列
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 目标行
    
    if (x >= dst_w || y >= dst_h) return;
    
    // Bilinear插值计算源坐标
    float src_x = (x + 0.5f) * src_w / dst_w - 0.5f;
    float src_y = (y + 0.5f) * src_h / dst_h - 0.5f;
    
    // 四个邻居坐标
    int x0 = (int)src_x, y0 = (int)src_y;
    float fx = src_x - x0, fy = src_y - y0;
    
    // 双线性插值 + normalize (融合!)
    for (int c = 0; c < 3; c++) {
        float val = (1-fx)*(1-fy)*input[(y0*src_w+x0)*3+c]
                  + fx*(1-fy)*input[(y0*src_w+x0+1)*3+c]
                  + (1-fx)*fy*input[((y0+1)*src_w+x0)*3+c]
                  + fx*fy*input[((y0+1)*src_w+x0+1)*3+c];
        
        // 归一化: (val/255 - mean) / std
        output[c * dst_h * dst_w + y * dst_w + x] = 
            (val / 255.0f - mean[c]) / std[c];
    }
    // 注意: 输出为CHW格式(Tensor格式), 输入为HWC(图像格式)
}
```

**优化手段：**

| 优化 | 原理 | 收益 |
|------|------|------|
| 向量化加载(uchar4) | 一次读4个像素(4字节) | 减少load指令数 |
| 多操作融合 | resize+normalize+HWC→CHW在一个kernel | 避免中间tensor读写 |
| 纹理内存 | 硬件插值器做bilinear(tex2D) | 双线性插值免费 |
| Pinned Memory | cudaMallocHost + 异步传输 | 传输与计算overlap |
| 多Stream | 不同图片在不同stream | 并行处理多张图 |
| 行对齐 | 确保每行起始16字节对齐 | 合并访存 |
| Batch处理 | 一个kernel处理batch张图 | 减少launch开销 |

**NVIDIA DALI vs 手写CUDA：**

```python
# DALI(Data Loading Library): NVIDIA官方GPU预处理库
from nvidia.dali import pipeline_def, fn

@pipeline_def
def my_pipeline():
    images = fn.readers.file(file_root="data/")
    images = fn.decoders.image(images, device="mixed")  # GPU解码
    images = fn.resize(images, size=(224, 224))          # GPU resize
    images = fn.crop_mirror_normalize(images, 
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    return images

# DALI优势: 已高度优化, 支持prefetch, 减少开发工作量
# 手写CUDA: 更灵活, 可融合自定义操作
```

---

### Q: GPU的内存结构：L1 Cache、共享内存、L2 Cache？

**A100 GPU内存层次详细参数：**

```
┌─────────── SM (108个) ────────────┐
│                                    │
│  寄存器文件: 65536 × 32-bit / SM   │  最快, 线程私有
│  带宽: ~19 TB/s                    │  延迟: 1 cycle
│                                    │
│  ┌─ 可配置SRAM (192KB/SM) ─────┐  │
│  │  L1 Data Cache (硬件管理)    │  │  延迟: ~28 cycles
│  │  +                           │  │  带宽: ~19 TB/s
│  │  Shared Memory (程序员管理)  │  │  可配置0/32/64/100/132/164KB
│  └──────────────────────────────┘  │
│                                    │
└────────────────────────────────────┘
           ↓
┌────────── L2 Cache (全GPU共享) ───────┐
│  容量: 40 MB (A100)                    │  延迟: ~200 cycles
│  带宽: ~5 TB/s                         │  所有SM共享
│  新特性: L2 Residency Control          │  可设置persistent区域
└────────────────────────────────────────┘
           ↓
┌────────── HBM2e (全局内存) ───────────┐
│  容量: 80 GB                           │  延迟: ~400ns (~600 cycles)
│  带宽: 2039 GB/s (5个HBM2e stack)     │  所有线程可见
└────────────────────────────────────────┘
```

**L1 Cache vs Shared Memory的配置：**

```cuda
// 设置shared memory优先(需要大量shared memory的kernel):
cudaFuncSetAttribute(kernel, 
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
// 100% → 尽可能多分给shared memory(最大164KB)

// 设置L1优先(依赖硬件cache的kernel):
cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 0);
// 0% → 尽可能多分给L1 cache
```

**L2 Cache Residency Control (Ampere+)：**

```cuda
// 设置某段数据持久驻留在L2中(如权重矩阵):
cudaAccessPolicyWindow policy;
policy.base_ptr = weight_ptr;
policy.num_bytes = weight_size;
policy.hitProp = cudaAccessPropertyPersisting;  // 持久化在L2
policy.missProp = cudaAccessPropertyStreaming;   // miss时流式处理

cudaStreamSetAccessPolicyWindow(stream, &policy);

// 用途: 小权重矩阵(如<40MB)可以常驻L2
// 效果: 后续kernel访问该数据几乎100% L2命中
// 注意: 最多设置L2的一部分为persistent(默认上限为L2的3/4)
```

**编程优化原则：**

| 数据特征 | 推荐存储 | 原因 |
|---------|---------|------|
| 线程私有,频繁使用 | 寄存器 | 最快,编译器自动分配 |
| Block内共享,多次复用 | Shared Memory | 显式管理,19TB/s |
| 只读小数据(<64KB) | 常量内存 | 专用cache,广播优化 |
| 频繁随机访问 | L1/L2(自动cache) | 硬件管理 |
| 小权重(<40MB) | L2 Persistent | 常驻L2,减少HBM访问 |
| 大数据流式访问 | HBM(向量化) | 充分利用2TB/s带宽 |

---

### Q: 共享内存怎么优化？Bank Conflict是什么？

**共享内存优化的完整策略：**

**1. 作为手动管理的Cache（核心用法）：**

```cuda
// 矩阵乘分块: 将全局内存数据加载到共享内存复用
__shared__ float tile_A[TILE][TILE];
__shared__ float tile_B[TILE][TILE];

// 协作加载(所有线程一起加载一个tile)
tile_A[ty][tx] = A[row * K + (t * TILE + tx)];
tile_B[ty][tx] = B[(t * TILE + ty) * N + col];
__syncthreads();

// 复用: 每个元素被TILE个线程读取
for (int k = 0; k < TILE; k++)
    sum += tile_A[ty][k] * tile_B[k][tx];
// tile_A[ty][k]被同列TILE个线程共享, 复用TILE次
```

**2. 避免Bank Conflict：**

```
Bank Conflict定义:
  共享内存 = 32个Bank, 每Bank宽4字节
  同一warp中2+个线程同时访问同一bank的不同地址 → 串行化

检测方法:
  Nsight Compute → Memory → Shared Memory
  看 "Shared Memory Bank Conflicts" 指标

消除方法:
  ├── Padding: s[N][N+1] 错开bank映射
  ├── Swizzle: 地址异或变换
  └── 调整数据布局/访问模式
```

**3. 双缓冲(Double Buffering)：**

```cuda
// 异步加载下一轮数据与计算当前轮重叠
__shared__ float buf[2][TILE][TILE];  // 两个buffer

// 预加载第一轮
load_tile(buf[0], ...);
__syncthreads();

for (int t = 0; t < num_tiles; t++) {
    // 异步加载下一轮到buf[1-current]
    if (t + 1 < num_tiles)
        cp_async_load(buf[(t+1)%2], ...);  // cp.async异步加载
    
    // 计算当前轮(使用buf[current])
    compute(buf[t%2]);
    
    __syncthreads();  // 等待加载和计算都完成
}
// 效果: 加载延迟被计算隐藏
```

**4. 共享内存使用量与Occupancy的权衡：**

```
A100: 每SM 192KB可配置SRAM
  如果kernel用100KB shared memory:
    每SM只能驻留1个block (192/100 = 1)
    可能导致occupancy低
    
  如果kernel用32KB shared memory:
    每SM可驻留多个block
    occupancy高, 延迟隐藏好

Trade-off: 更多shared memory → 更多数据复用 → 但更少active block
  最佳点: 需要实验确定(Occupancy Calculator)
```

---

### Q: ByteTrack的匹配过程？

**ByteTrack的核心创新——利用所有检测框(包括低分)：**

```
传统tracker(如SORT): 只用高置信度检测框
  高分框(>0.7) → 匹配track → 低分框丢弃
  问题: 遮挡时检测分数低→丢弃→丢失ID

ByteTrack: 两次匹配,充分利用低分框
  Step 1: 高分框匹配现有track
  Step 2: 未匹配的track与低分框再匹配
  效果: 遮挡导致的低分框被正确关联→减少ID switch
```

**完整匹配流程：**

```
输入: 
  检测框: D = {d_1, d_2, ..., d_n} 含置信度
  现有Track: T = {t_1, t_2, ..., t_m} 含预测位置(Kalman)

Step 1: 分离高低分检测
  D_high = {d | d.score > thresh_high}    (如>0.6)
  D_low = {d | thresh_low < d.score ≤ thresh_high}  (如0.1-0.6)

Step 2: 第一次匹配(高分框 vs 所有Track)
  Cost矩阵 = IoU(D_high, T_predicted)   // 或结合ReID特征
  匈牙利算法求最优匹配
  → 匹配成功: 更新track (Kalman update)
  → 未匹配的track: T_remain
  → 未匹配的高分框: D_remain

Step 3: 第二次匹配(低分框 vs 未匹配的Track) ← 关键创新!
  Cost矩阵 = IoU(D_low, T_remain)
  匈牙利算法
  → 匹配成功: 更新track (遮挡中的目标被恢复!)
  → 仍未匹配的track: 标记age+1, 超过max_age则删除

Step 4: 初始化新Track
  D_remain中的高分框 → 创建新track

Step 5: Kalman预测
  所有active track做Kalman predict(下一帧位置预测)
```

**为什么两次匹配有效？**

| 场景 | 传统方法 | ByteTrack |
|------|---------|-----------|
| 目标被遮挡50% | 检测分0.3→丢弃→ID丢失 | 低分匹配保持跟踪 |
| 目标模糊 | 检测分0.4→丢弃→新ID | 低分匹配保持旧ID |
| 误检(背景误判) | 可能被匹配 | 第二次匹配IoU阈值高,误检IoU低→不匹配 |

---

### Q: ARM相关的优化了解吗？

**ARM CPU优化的核心技术：**

**1. NEON SIMD指令（128-bit向量）：**

```cpp
#include <arm_neon.h>

// 4个float并行加法:
float32x4_t a = vld1q_f32(ptr_a);   // 加载4个float
float32x4_t b = vld1q_f32(ptr_b);
float32x4_t c = vaddq_f32(a, b);     // 4路并行加法
vst1q_f32(ptr_c, c);                  // 存储结果

// INT8量化推理(16路并行):
int8x16_t va = vld1q_s8(weight);     // 加载16个INT8
int8x16_t vb = vld1q_s8(activation);
int16x8_t vc = vmull_s8(vget_low_s8(va), vget_low_s8(vb));  // 乘+扩展
```

**2. ARM优化vs GPU优化的区别：**

| 维度 | ARM CPU | NVIDIA GPU |
|------|---------|-----------|
| SIMD宽度 | 128bit(NEON) / 512bit(SVE2) | 32线程×32bit = 1024bit |
| 并行度 | 4-8核 | 数千Core |
| 内存层次 | L1(32KB)/L2(512KB)/L3(4MB) | Shared(192KB)/L2(40MB)/HBM(80GB) |
| 带宽 | ~50 GB/s (LPDDR5) | 2000+ GB/s (HBM) |
| 适合场景 | 边缘推理/手机端 | 云端训练/推理 |
| 功耗 | 1-10W | 200-700W |

**3. ARM推理优化工具链：**

| 工具 | 用途 | 特点 |
|------|------|------|
| ARM Compute Library(ACL) | 优化的NN原语 | GEMM/Conv/Pool针对ARM优化 |
| NNAPI | Android标准推理API | 抽象硬件(CPU/GPU/NPU) |
| TFLite | 移动端推理框架 | 支持INT8/FP16, delegate机制 |
| XNNPACK | 移动端高性能kernel | 被TFLite/PyTorch Mobile使用 |
| Qualcomm SNPE/QNN | 高通平台推理 | 支持Hexagon DSP/Adreno GPU |

**4. 移动端推理特殊优化：**
- Im2col-free卷积: 直接卷积避免内存展开(节省内存)
- Winograd: 3×3卷积用Winograd变换减少乘法数
- Per-channel INT8: 移动端量化标准
- 内存复用: 分析tensor生命周期做in-place

---

### Q: 手撕（思路题）：CUDA实现100万个浮点数相加？

（编程题）
