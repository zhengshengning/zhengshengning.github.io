---
title: '英伟达 AI Infra 校招 (2)'
date: 2026-04-16 12:21:26
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: SIMT（Single Instruction Multiple Threads）是什么？

SIMT是NVIDIA GPU独有的执行模型，介于SIMD和MIMD之间，是理解GPU编程的核心概念。

**SIMT vs SIMD对比**：

| 特性 | SIMD (CPU AVX) | SIMT (GPU Warp) |
|------|---------------|-----------------|
| 基本单位 | 数据通道（lane） | 独立线程 |
| 程序计数器 | 共享1个PC | 每线程独立PC（但通常统一） |
| 寄存器 | 共享SIMD寄存器 | 每线程独立寄存器文件 |
| 分支处理 | 不支持（或需向量掩码） | 支持divergence（掩码串行） |
| 编程模型 | 显式向量化（intrinsics） | 标量代码风格，隐式向量化 |
| 内存访问 | 连续或gather/scatter | 每线程独立地址 |

**SIMT工作机制**：
- 32个线程组成一个Warp，通常锁步（lockstep）执行相同指令。
- 当所有线程执行相同路径时，效率最高（等效32-wide SIMD）。
- 当发生分支divergence时：硬件通过**活跃掩码（active mask）**串行执行不同路径——先执行满足条件的线程（其他线程inactive），再执行另一分支。所有路径执行完后，Warp重新收敛到统一PC。

**SIMT对程序员的好处**：
- 可以编写标量风格代码（像写单线程程序），硬件自动将其"向量化"为Warp级别执行。
- 每个线程可以访问不同的内存地址（vs SIMD必须连续/gather模式）。
- 支持线程级的条件分支（尽管有性能代价）。

**SIMT的性能陷阱**：
- Warp Divergence：最坏32个线程走32条路径 → 性能降为1/32。
- 虽然每线程有独立PC，但硬件强制同一Warp内串行执行不同路径。
- Volta架构引入Independent Thread Scheduling：允许线程更灵活的收敛和发散，但基本代价不变。

---

### Q: Occupancy和什么有关，怎么控制？

Occupancy（占用率）= SM上活跃Warp数 / SM理论最大Warp数（如A100为64）。它反映了GPU隐藏延迟的能力。

**影响Occupancy的三大资源因素**：

**1. 寄存器使用量（最常见的限制因素）**：
- A100每SM有65536个32位寄存器。
- 每线程使用的寄存器越多，能同时驻留的线程（Warp）越少。
- 例：每线程用128个寄存器 → 65536/128 = 512线程 = 16 Warp → Occupancy = 16/64 = 25%。
- 控制方法：`__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` 让编译器控制寄存器分配；`-maxrregcount=N` 编译选项全局限制。

**2. 共享内存使用量**：
- A100每SM最大164KB shared memory（配置为prefer shared时）。
- Block使用shared memory越多，SM能驻留的Block越少。
- 例：每Block用82KB shared memory → SM最多驻留2个Block → 如果每Block 256线程则只有512线程 = 16 Warp。
- 控制方法：动态shared memory大小调整、减少不必要的shared memory使用。

**3. Block大小（线程数）**：
- Block中的线程数必须是Warp（32线程）的整数倍。
- SM同时能驻留的Block有上限（如A100最多32个Block/SM）。
- Block太小：可能浪费warp槽位（如每Block 32线程，32个Block = 32 Warp = 50% occupancy）。
- Block太大：如果资源受限可能只能驻留1个Block。

**Occupancy不是唯一目标**：

高Occupancy不等于高性能！关键权衡：
- 更高Occupancy → 更多Warp可切换 → 更好的延迟隐藏。
- 更低Occupancy + 更多寄存器/线程 → 减少register spill到local memory → 减少内存访问。
- GEMM等计算密集kernel在50% occupancy下可能达到峰值性能（每线程有足够寄存器做大tile计算）。

**工具**：CUDA Occupancy Calculator（Excel表或`cudaOccupancyMaxActiveBlocksPerMultiprocessor` API）、Nsight Compute的Occupancy面板。

---

### Q: Bank Conflict的粒度是多少？

**Shared Memory的Bank结构**：

共享内存被物理划分为**32个Bank**，每个Bank宽度为**4字节（32位）**。地址到Bank的映射规则：
```
bank_id = (byte_address / 4) % 32
```

即：连续的4字节分别落在不同的bank中，第0-3字节在bank 0，第4-7字节在bank 1，...，第124-127字节在bank 31，第128-131字节又回到bank 0。

**Bank Conflict的触发条件**：
- 同一Warp中的**不同线程**访问**同一Bank**的**不同地址**（不同行）→ 产生Conflict，需串行化访问。
- N-way conflict = 该bank被N个线程同时访问 → 需要N个serialized周期。

**不产生Conflict的情况**：
1. 每个线程访问不同Bank（理想情况，无conflict）。
2. 多个线程访问**同一地址**（exact same address）→ 硬件广播（Broadcast），无conflict。
3. Compute Capability 3.x+：多个线程访问同一Bank的**同一个32位字**也是broadcast。

**实际示例**：
```cuda
__shared__ float s[32][32];  // 32×32 float数组

// 无conflict: 每线程访问不同列（不同bank）
float val = s[row][threadIdx.x];  // threadIdx.x 0-31对应bank 0-31

// 有conflict: 每线程访问同一列的不同行
float val = s[threadIdx.x][0];  // 所有线程访问bank 0! 32-way conflict

// 解决: padding
__shared__ float s[32][33];  // 每行多一个float，打破bank对齐
float val = s[threadIdx.x][0];  // 现在不同行的第0列落在不同bank
```

**8字节Bank模式（Compute Capability 3.x+）**：
可通过 `cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)` 设置bank宽为8字节，对double类型操作有帮助。

---

### Q: GEMM分块大小受什么影响？

GEMM Tile Size的选择是一个多约束优化问题，需要平衡多个硬件资源限制：

**1. 共享内存容量（最直接的约束）**：
- 需要缓存的数据：A_tile(BM × BK) + B_tile(BK × BN) + 可能的C_tile。
- 约束：(BM×BK + BK×BN) × sizeof(element) ≤ Shared Memory Size
- 例：FP16, BM=BN=128, BK=32 → (128×32 + 32×128) × 2 = 16KB。A100可用164KB → 可以开更大tile或multi-stage buffering。

**2. 寄存器数量（决定每线程计算量）**：
- 每线程负责计算C的一个小块（thread tile），如8×8 = 64个FP32累加器 = 64个寄存器。
- 加上A/B的寄存器buffer和循环变量，总共需要100-200个寄存器/线程。
- 约束：总寄存器 = 每线程寄存器 × Block线程数 ≤ SM寄存器文件（65536）。
- Thread tile越大 → 数据复用率越高（计算访存比提升）→ 但寄存器压力越大。

**3. 计算访存比（Arithmetic Intensity）**：
- 理想tile：让数据在寄存器/shared memory中被充分复用。
- 计算量 = BM × BN × BK × 2 FLOPs。
- 访存量 = (BM×BK + BK×BN) × sizeof。
- Tile AI = 2×BM×BN×BK / [(BM+BN)×BK × sizeof] ≈ BM×BN / [(BM+BN) × sizeof / 2]。
- Tile越大 → AI越高 → 越接近compute-bound → 性能越好（到达峰值）。

**4. Tensor Core的Warp级分块要求**：
- Tensor Core WMMA API要求特定的warp tile维度：如m16n8k16（FP16）、m16n8k8（TF32）。
- CUTLASS中warp tile通常为64×64或32×128，由多个Tensor Core指令组成。
- 不满足对齐要求则无法使用Tensor Core。

**5. 硬件世代差异**：

| GPU | Shared Memory/SM | Registers/SM | 推荐Block Tile |
|-----|-----------------|-------------|---------------|
| V100 | 96KB | 65536 | 128×128, BK=32 |
| A100 | 164KB | 65536 | 128×256或256×128, BK=64 |
| H100 | 228KB | 65536 | 128×256, BK=64, multi-stage |

**典型层次**：Grid tile(问题切分) → Block tile(128×128) → Warp tile(64×32) → Thread tile(8×8) → Tensor Core MMA(16×8×16)。

---

### Q: 使用float4读写全局内存为什么更快？

float4一次读写128位（16字节），本质是利用GPU内存系统的**事务粒度**和**指令效率**特性：

**原因1：减少内存事务数量**

GPU全局内存系统以**32字节**或**128字节段（sector/segment）**为单位传输：
- 使用float（4字节）：warp中32线程发起32次4字节请求，可能产生多个事务。
- 使用float4（16字节）：warp中32线程发起32次16字节请求，如果地址连续对齐，可以完美合并为32×16=512字节=4个128字节事务。
- 如果用float且地址有stride，事务数可能达到32次（每线程独立事务），带宽利用率仅1/32。

**原因2：减少Load/Store指令数量**

```cuda
// 4条load指令
float a = data[tid*4 + 0];
float b = data[tid*4 + 1];
float c = data[tid*4 + 2];
float d = data[tid*4 + 3];

// 1条load指令
float4 vec = reinterpret_cast<float4*>(data)[tid];
```
- 减少3/4的load指令 → 降低指令发射单元压力。
- 减少地址计算指令。
- 编译器可以将float4 load编译为单条LDG.128指令。

**原因3：提高有效带宽利用率**

每次内存事务有固定的开销（地址计算、仲裁、调度）。float4将4次事务的固定开销摊销到1次中，有效数据/总开销比更高。

**使用前提和注意事项**：
- **对齐要求**：数据起始地址必须16字节对齐（`cudaMalloc`分配的地址默认256字节对齐，满足要求）。
- **连续性**：float4假设4个float在内存中连续。如果实际数据有stride则不能直接用float4。
- **维度对齐**：数组长度最好是4的倍数，否则尾部需要特殊处理。
- **寄存器压力**：float4使每线程需要更多寄存器暂存数据。

---

### Q: 一个Block能否被调度到不同的SM上？

**不能**。一个Block在其整个生命周期内只运行在一个SM上，永远不会被迁移到另一个SM。

**原因分析**：

Block内线程共享的硬件资源都是SM-local的：
1. **Shared Memory**：分配在特定SM的物理SRAM上，迁移意味着需要复制这些数据（代价高且破坏原子性）。
2. **同步原语**：`__syncthreads()` 依赖SM内的硬件同步机制（barrier），跨SM无法保证。
3. **寄存器状态**：Block中每个线程的寄存器分配在该SM的寄存器文件上。
4. **Warp调度状态**：Warp的调度信息（PC、stall状态等）维护在SM的调度器中。

**相关事实**：
- 同一Grid中的不同Block可以（且通常会）被调度到不同SM上。
- Block到SM的调度顺序不确定且不可控制（runtime/hardware决定）。
- 一个SM可同时驻留多个Block（受资源限制），这些Block彼此独立执行。
- Block间不能直接通信（只能通过全局内存+原子操作，且无顺序保证）。
- 这种设计使得CUDA程序可以在不同配置的GPU（SM数量不同）上自动scaling。

---

### Q: 常用GPU卡的缓存大小是多少？

| GPU | L1/Shared Memory (per SM) | L2 Cache (全局) | HBM容量 | HBM带宽 |
|-----|--------------------------|----------------|---------|---------|
| V100 | 128KB（共用，可配） | 6MB | 32GB | 900 GB/s |
| A100 | 192KB（共用，可配） | 40MB | 80GB | 2.0 TB/s |
| H100 SXM | 228KB（共用，可配） | 50MB | 80GB | 3.35 TB/s |
| RTX 4090 | 128KB（共用） | 72MB | 24GB | 1.0 TB/s |
| H200 | 228KB | 50MB | 141GB | 4.8 TB/s |
| B200 | 228KB | 50MB | 192GB | 8.0 TB/s |

**设计趋势和工程意义**：
- **L2逐代增大**（6→40→50→72MB）：加速跨SM数据共享、减少HBM压力。A100的40MB L2可持久缓存热点数据（`cudaAccessPolicyWindow`）。
- **L1/SM相对稳定**（128-228KB）：主要受SM面积和功耗约束。Shared Memory部分由程序员显式控制。
- **HBM容量和带宽持续增长**：满足大模型的参数存储和带宽需求。

**配置建议**（A100为例）：
- Shared Memory优先（需要大tile的GEMM kernel）：`cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100)` → 最大164KB Shared + 28KB L1。
- L1优先（大量全局内存随机访问的kernel）：配置更多给L1 cache。

---

### Q: Warp Divergence对性能的影响？

**Warp Divergence是GPU上最常见的性能陷阱之一**，当同一Warp内32个线程走不同分支路径时发生。

**硬件处理机制**：
1. Warp遇到分支指令（if/else）。
2. 评估每个线程的条件。
3. 如果所有线程条件相同（无divergence）→ 正常执行对应分支，无性能损失。
4. 如果有divergence → 硬件序列化执行：
   - 先执行满足条件的线程（其余inactive，通过mask位标记）。
   - 再执行不满足条件的线程。
   - 两个分支都执行完后Warp重新收敛。

**性能影响量化**：
- 2-way divergence（if/else各半）：性能下降50%（两个分支串行执行）。
- 最坏情况：32线程走32条不同路径 → 性能降为1/32。
- 实际影响取决于divergent分支的计算量——如果分支内只有1条指令，影响很小。
- **不同Warp之间的分支不会互相影响**——每个Warp独立调度执行。

**优化策略**：
1. **数据重排**：将相同条件的数据聚集在一起，使同一Warp内线程走相同分支。
2. **无分支代码**：用条件选择指令替代if-else：
   ```cuda
   // 有divergence
   if (x > 0) y = a; else y = b;
   // 无divergence（predicated execution）
   y = (x > 0) ? a : b;  // 编译为select指令，无分支
   ```
3. **Warp级投票函数**：`__ballot_sync(mask, pred)` 获取warp内条件分布，决定是否需要特殊处理。
4. **分支粒度放大**：按warp或block粒度做条件判断而非线程级。

---

### Q: NVIDIA GPU的指令级并行（ILP）是什么？

ILP（Instruction-Level Parallelism）是在**单个线程内**利用多条独立指令同时执行的能力，是TLP（Thread-Level Parallelism）的重要补充。

**GPU上ILP的实现机制**：

**1. 多功能单元并行**：
SM中有多种功能单元（INT、FP32、FP64、Load/Store、Tensor Core等），一条指令在FP32单元执行时，另一条独立指令可以在Load/Store单元执行。

**2. 指令流水线（Pipeline）**：
计算指令是多级流水的（如FP32 mul需要4个cycle），可以在第1个cycle发射后，第2个cycle发射下一条无依赖指令，不需要等前一条完成。

**3. 异步内存操作（Ampere+）**：
`cp.async` 发起后不阻塞后续指令，数据搬运和后续计算可以并行。

**如何增加ILP**：

```cuda
// 低ILP：每条指令依赖上一条
float sum = data[0];
sum += data[1];  // 依赖sum
sum += data[2];  // 依赖sum
sum += data[3];  // 依赖sum

// 高ILP：使用多个独立累加器
float sum0 = data[0], sum1 = data[1], sum2 = data[2], sum3 = data[3];
sum0 += data[4]; sum1 += data[5]; sum2 += data[6]; sum3 += data[7];
float total = sum0 + sum1 + sum2 + sum3;  // 最后合并
```

**ILP与TLP的互补关系**：
- **TLP（Thread-Level Parallelism）**：通过大量Warp切换隐藏延迟。当一个Warp等数据时，切换到另一个Warp。需要高Occupancy。
- **ILP**：在单个线程/Warp内通过独立指令的并行执行提高吞吐。不需要高Occupancy。
- 关系：低Occupancy时ILP更重要（没有足够Warp切换隐藏延迟）；高Occupancy时ILP是额外收益。
- 实践：GEMM kernel通常Occupancy只有25-50%，靠大量ILP（循环展开、多element/thread）达到高性能。

**循环展开是增加ILP最简单的方法**：`#pragma unroll` 将循环体复制多份，暴露独立指令给调度器。

---

### Q: 手撕：CUDA实现矩阵转置？

（编程题）

---

### Q: 手撕：CUDA实现向量外积？

（编程题）
