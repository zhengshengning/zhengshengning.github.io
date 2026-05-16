---
title: '沐曦 AI Infra 实习'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: GEMM 优化中 Tiling 块的大小如何选择？

Tiling 是 GEMM 优化的核心策略——将大矩阵乘拆分为多级小块，逐级加载到更快的存储层计算。块大小的选择直接影响性能。

**三级 Tiling 层次**：
```
Global Memory → Block Tile (BM × BN × BK) → Shared Memory
Shared Memory → Thread Tile (TM × TN)     → Registers

C[M×N] = A[M×K] × B[K×N]

外层循环: 将 C 分为 BM×BN 的块，每个 block 负责一块
中层循环: 沿 K 维度分 BK 步迭代（每步加载 A[BM×BK] 和 B[BK×BN] 到 shared）
内层循环: 每个线程计算 TM×TN 个输出（寄存器 tiling）
```

**选择原则及约束条件**：

| 参数 | 约束 | 典型值 | 选择依据 |
|---|---|---|---|
| BM, BN | Shared Memory 容量 | 64-256 | (BM×BK + BK×BN) × sizeof(T) < SM_size |
| BK | 内存事务对齐 | 8-32 | 128B 对齐（float4 访问） |
| TM, TN | 寄存器数量 | 4-8 | TM×TN + 加载缓冲 < 255 寄存器 |
| Block 内线程数 | 硬件限制 | 128-256 | (BM/TM) × (BN/TN) ≤ 1024 |

**Shared Memory 约束的具体计算**：
```
双缓冲 (隐藏加载延迟):
  需要 2 × (BM×BK + BK×BN) × sizeof(T)
  
  例: BM=BN=128, BK=8, FP16
    = 2 × (128×8 + 8×128) × 2 bytes
    = 2 × 4096 bytes = 8 KB  ✓ (A100 最大 164 KB)
  
  例: BM=BN=256, BK=32, FP16
    = 2 × (256×32 + 32×256) × 2 bytes
    = 2 × 32768 bytes = 64 KB  ✓ (但 occupancy 可能下降)
```

**数据复用率分析**：
```
每次从 Global Memory 加载:
  A tile: BM × BK 元素
  B tile: BK × BN 元素
  总加载: BK × (BM + BN) 元素

产出计算量: BM × BN × BK × 2 FLOPs

复用率 = 计算量 / 加载量 = 2×BM×BN×BK / (BK×(BM+BN))
        = 2×BM×BN / (BM+BN)
        
当 BM=BN=128: 复用率 = 2×128×128/256 = 128  (很好)
当 BM=BN=64:  复用率 = 2×64×64/128 = 64     (还行)
当 BM=BN=32:  复用率 = 2×32×32/64 = 32      (较低)
```

**Occupancy 与 Tile 大小的权衡**：
```
情况 1: BM=BN=128, BK=8, 双缓冲 FP16 → 8 KB shared/block
  → SM 可驻留 164/8 = ~16 blocks (受其他因素限制通常 4-8 blocks)
  → Occupancy 高，延迟隐藏好

情况 2: BM=BN=256, BK=32, 双缓冲 FP16 → 64 KB shared/block
  → SM 可驻留 164/64 = 2 blocks
  → Occupancy 低，但数据复用高

最优点: 通常在 occupancy = 25-75% 之间
  "足够的驻留 block 隐藏延迟" 与 "足够大的 tile 提高复用" 的平衡
```

**寄存器 Tile 选择**：
```
每线程计算 TM×TN 个输出:
  需要寄存器: TM×TN (结果) + TM+TN (A/B 加载缓冲) + 辅助变量
  
  TM=TN=8: 64+16+辅助 ≈ 90 寄存器 (可接受)
  TM=TN=16: 256+32+辅助 ≈ 300 寄存器 (超限! 会 spill)
  
  推荐: TM=TN=4~8, 根据 NCU profiling 调整
```

**经验法则总结**：
1. 从 BM=BN=128, BK=8, TM=TN=8 开始
2. 用 NCU 检查 shared memory 和 register 瓶颈
3. 如果 occupancy 太低 → 减小 BM/BN
4. 如果带宽利用率低 → 增大 BK 或增大 BM/BN
5. 最终通过 autotuning 搜索最优组合

---

### Q: Bank Conflict 怎么避免？

**Bank Conflict 的本质**：

Shared Memory 被组织为 **32 个 bank**（NVIDIA GPU），每个 bank 宽 4 字节（32 bits），连续 4 字节地址分配到相邻 bank：

```
地址(byte):  0-3    4-7    8-11   12-15  ...  124-127  128-131  ...
Bank:         0      1      2      3     ...    31       0      ...
```

**冲突规则**：
- 同一 warp 的 32 个线程同时访问 shared memory
- 如果 N 个线程访问同一 bank 的**不同地址** → N-way bank conflict → 串行化 N 次
- 如果多个线程访问同一 bank 的**同一地址** → 广播（broadcast），无冲突
- 目标：32 个线程访问 32 个不同 bank → 1 cycle 完成

**典型冲突场景——GEMM 中列访问**：
```cuda
__shared__ float A[32][32];  // 每行 32×4=128 bytes = 4 轮 bank

// 线程 tid 访问第 col 列:
float val = A[tid][col];  // 所有线程访问不同行的同一列
// A[0][col] → bank = col
// A[1][col] → bank = col (偏移 32×4=128 bytes, 128/4=32 个 bank 后回到 col)
// → 32-way bank conflict! 最差情况
```

**避免方法详解**：

**方法 1: Padding（最简单有效）**：
```cuda
__shared__ float A[32][33];  // 每行多 1 个 float = 4 字节 padding

// 现在地址映射:
// A[0][col] → offset = col × 4, bank = col % 32
// A[1][col] → offset = 33×4 + col×4, bank = (33 + col) % 32 = (1 + col) % 32
// A[2][col] → offset = 66×4 + col×4, bank = (66 + col) % 32 = (2 + col) % 32
// → 每行偏移 1 个 bank，32 行全不冲突!

// 代价: 浪费 32 × 4 = 128 bytes（微不足道）
```

**方法 2: Swizzle 访问模式**：
```cuda
// 通过位运算重映射列索引
__shared__ float A[32][32];
int row = threadIdx.x;
int col = original_col ^ (row & 0x1F);  // XOR 使不同行访问不同 bank

// 更精细的 swizzle（如 cuBLAS 内部使用）:
int swizzled_addr = (addr >> 4) ^ (addr & 0xF);
```

**方法 3: 128-bit 向量化访问**：
```cuda
// 使用 float4 (16 bytes) 加载，每线程占 4 个连续 bank
float4* shared_ptr = reinterpret_cast<float4*>(shared_mem);
float4 data = shared_ptr[threadIdx.x];

// 线程 0: bank 0-3
// 线程 1: bank 4-7
// ...
// 线程 7: bank 28-31
// → 8 线程就覆盖所有 32 bank，天然无冲突（但只能用 8 线程/组）

// 实际: 32 线程 × float4 = 128 个 bank slot = 4 轮无冲突加载
```

**方法 4: 调整数据布局（转置存储）**：
```cuda
// 原始: A[row][col]，列方向访问有冲突
// 转置: A[col][row]，列方向访问变成行方向（天然无冲突）

// 适用场景: 矩阵 B 在 GEMM 中需要列访问
// 解决: 加载到 shared memory 时做转置
__shared__ float B_shared[BK][BN + 1];  // 存储为转置 + padding
B_shared[k][n] = B_global[k * N + n];   // 转置存储
// 之后按行读取: B_shared[k][thread_n] → 行方向，无冲突
```

**检测 Bank Conflict**：
- Nsight Compute 的 Shared Memory 面板显示 bank conflict 统计
- 指标：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared`
- 目标：0 conflict（或接近 0）

---

### Q: 手撕 CUDA：矩阵每一行的 ReduceSum？

（编程题）
