---
title: '飞腾 AI Infra 校招 二面'
date: 2026-04-16 12:11:10
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 推理优化, 训练优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: 多进程和多线程的差别？

**多进程**和**多线程**是两种并行执行模型，选择取决于隔离性需求和通信开销：

| 特性 | 多进程 | 多线程 |
|------|--------|--------|
| 地址空间 | 独立（互相隔离） | 共享进程地址空间 |
| 通信方式 | IPC：管道/共享内存/socket/消息队列 | 直接读写共享变量（需同步） |
| 创建开销 | 大（~ms 级，需复制页表等） | 小（~us 级，只需分配栈） |
| 上下文切换 | 慢（需切换页表 + flush TLB） | 快（只切换寄存器和栈指针） |
| 稳定性 | 一个进程崩溃不影响其他 | 一个线程崩溃整个进程退出 |
| 调试难度 | 较易（隔离性好） | 较难（竞态/死锁/数据竞争） |
| 资源利用 | 冗余高（每进程独立资源） | 高效（共享代码段/数据段） |

**使用场景选择**：
- **多进程**：需要强隔离（如 Chrome 每个 tab 一个进程防止一个页面崩溃影响其他）、利用多核 CPU 绕过 Python GIL、安全敏感（沙箱隔离）
- **多线程**：需要高频通信（共享内存直接读写）、IO 密集型任务（网络/磁盘等待时让出 CPU）、延迟敏感（线程创建和切换快）

**AI Infra 中的应用**：
- 数据加载：PyTorch DataLoader 使用多进程（num_workers）绕过 GIL，每个 worker 独立做数据预处理
- 推理服务：多线程处理并发请求（共享模型权重，避免多份模型副本占显存）
- 分布式训练：多进程（每个 GPU 一个进程），进程间通过 NCCL 通信

### Q: 算子调优的通用思路：如何针对特定shape超越官方库？

官方库（cuBLAS/cuDNN）为通用 shape 优化，针对特定 shape 的定制 kernel 可以超越它们：

**Step 1: 瓶颈分析**：
- 用 Nsight Compute profiling 判断当前 shape 在官方库下是 compute-bound 还是 memory-bound
- 查看 SOL（Speed of Light）指标：SM 利用率、内存带宽利用率、Tensor Core 利用率
- 确定"距离理论峰值还有多远"以及"差距在哪"

**Step 2: 特化 Tile Size**：
- 官方库通常用启发式选择 tile 大小，对某些 shape 可能选择了次优配置
- 针对具体 M/N/K，遍历搜索最优的 {BM, BN, BK, WM, WN} 组合
- 关键约束：tile 数据量 ≤ shared memory 容量，寄存器使用量 ≤ 线程可用寄存器

**Step 3: 减少冗余计算/边界处理**：
- 如果 shape 恰好是 tile 大小的整数倍，可去掉所有边界检查代码（减少指令数和分支）
- 如果 M/N 是 2 的幂次，可用位运算替代取模
- 固定 shape 编译（template 参数），让编译器做更激进的优化（常量传播/循环展开）

**Step 4: 定制数据布局**：
- 标准 row-major/col-major 可能不是最优。对特定 shape 可以使用 swizzled layout 或 interleaved layout
- 例如：小 M 大 K 场景下，对 A 矩阵做转置存储可改善 GMEM->SMEM 的 coalesced 访问

**Step 5: 极致优化**：
- 手写 PTX/SASS：绕过编译器限制，精确控制寄存器分配和指令调度
- 利用微架构特性：如 Ampere 的异步拷贝（cp.async）、Hopper 的 TMA（Tensor Memory Accelerator）
- Pipeline interleaving：交替执行计算和访存指令，最大化指令级并行

**Step 6: Auto-tuning 框架**：
- 定义参数空间：tile sizes、unroll factors、pipeline stages、向量化宽度
- 搜索策略：网格搜索（小空间）、随机搜索、贝叶斯优化
- 实测性能（而非模型预测），取最优配置
- 工具：CUTLASS profiler、Triton auto-tune、自定义 benchmark script

**实际案例**：针对 LLaMA-7B decode 阶段的 GEMV（M=1, N=4096, K=4096），定制 kernel 可比 cuBLAS 快 20-30%（因为 cuBLAS 的 GEMM kernel 对 M=1 场景有额外开销）。

### Q: 融合算子如何设计？

算子融合（Operator Fusion）是推理优化中最重要的技术之一，将多个独立 kernel 合并为一个：

**融合的收益**：
- 减少 kernel launch 开销（每次 ~5-10us，高频算子积累显著）
- 消除中间 tensor 的 Global Memory 读写（这是最主要收益——memory-bound 算子的瓶颈就是带宽）
- 中间数据保留在寄存器/Shared Memory 中复用
- 减少总显存占用（无需分配中间 tensor）

**设计步骤**：

**1. 分析算子间数据流关系**：
- Producer-Consumer：A 的输出是 B 的输入（如 GEMM -> ReLU -> Add）
- Sibling：多个算子读相同输入（如 Q/K/V 三个线性投影读同一个 x）
- 绘制数据流图，识别可融合的子图

**2. 确定融合模式**：
- **Elementwise 链融合**：连续的逐元素操作（如 Add -> ReLU -> Multiply），最简单也最常见
- **Reduction + 后处理**：如 Softmax（reduction 求 max/sum）+ 元素除法
- **GEMM + Epilogue**：矩阵乘后的 bias add / activation / residual add
- **Attention 内部融合**：QK^T -> Scale -> Mask -> Softmax -> V（FlashAttention）

**3. 资源约束检查**：
- Shared Memory：融合后的 kernel 总共享内存需求不超过 SM 上限（A100: 164KB）
- 寄存器：每线程寄存器不超限（255 个），否则 register spill 性能剧降
- Occupancy：融合后资源需求增加可能降低活跃 warp 数，需在复用率和并行度间权衡

**4. 实现策略**：
- 简单融合：在 GEMM 的 epilogue 阶段（结果从寄存器写出前）执行 bias/activation
- 复杂融合：重新设计并行策略（如 FlashAttention 需要全新的 tiling 和 online softmax 方案）
- Codegen：用 TVM/Triton 等框架自动生成融合 kernel

**常见融合实例**：
- `LayerNorm/RMSNorm + Linear`：读一次数据完成归一化和矩阵乘
- `Linear + Bias + GELU`：GEMM epilogue 中完成偏置和激活
- `Residual Add + LayerNorm`：一次 kernel 完成残差连接和归一化
- `Multi-Head Attention`（FlashAttention）：整个注意力计算融合为一个 kernel

### Q: 混合精度训练相关问题？

混合精度训练（Mixed Precision Training）利用低精度计算加速的同时保持训练精度，是大模型训练的必备技术：

**核心思路**：前向传播和梯度计算用 FP16/BF16（利用 Tensor Core 2x 算力），参数更新和关键计算保持 FP32 精度。

**三大关键技术**：

**1. Loss Scaling（损失缩放）**：
- **问题**：FP16 的最小正规数为 6.1×10^-5，许多梯度值小于此值导致下溢为 0（尤其是深层网络和训练后期）
- **解决**：将 loss 乘以一个大的 scale factor（如 1024-65536），使梯度值被放大到 FP16 可表示的范围
- **动态 Loss Scaling**：初始 scale 很大（65536），如果发现梯度出现 inf/nan 则 scale 减半并跳过该 step；连续 N 步无 overflow 则 scale 翻倍
- **BF16 通常不需要 loss scaling**：BF16 的指数范围与 FP32 相同，不易溢出/下溢

**2. Master Weight（FP32 主权重）**：
- 维护一份 FP32 精度的模型参数副本
- 每步训练：FP16 前向 -> FP16 梯度 -> FP32 梯度（除以 loss scale）-> 更新 FP32 主权重 -> 转换回 FP16 用于下一步前向
- **为什么必要**：FP16 的精度只有 3.3 位有效十进制数，微小的梯度更新（如 lr * gradient = 0.0001 * 0.001 = 0.0000001）可能小于 FP16 的最小精度而被舍入为 0

**3. 精度敏感操作的特殊处理**：
- **Softmax**：输入的 exp 操作对数值范围敏感，需要 FP32 计算
- **LayerNorm/RMSNorm**：归一化中的均值/方差计算用 FP32 累加
- **Loss 计算**：交叉熵中的 log 操作需要 FP32 精度
- **梯度累积**：多 micro-batch 的梯度求和用 FP32 避免误差积累

**显存开销**：混合精度训练实际上增加了显存（需要 FP32 主权重 + FP16 工作副本），但 Tensor Core 的 2x 计算加速足以弥补。Adam 优化器状态：FP32 参数（4B）+ momentum（4B）+ variance（4B）+ FP16 参数（2B）+ FP16 梯度（2B）= 16 bytes/param。

**框架支持**：PyTorch `torch.cuda.amp`（GradScaler + autocast）、DeepSpeed、Megatron-LM 都内置混合精度支持。

### Q: GPU优化方法：Nsight工具链使用、关注的效率指标？

**Nsight Compute**（单 kernel 分析）关注的核心指标：

**1. 计算效率指标**：
- **SM Throughput**（%）：SM 活跃时间占比。低于 60% 说明存在空闲 SM（可能 grid 太小或负载不均）
- **Compute (SM) Throughput**：实际计算吞吐占理论峰值的比例
- **Tensor Core Utilization**：Tensor Core 使用率。GEMM kernel 应 >80%，否则没有有效利用
- **IPC（Instructions Per Cycle）**：衡量指令级并行度

**2. 内存效率指标**：
- **DRAM Throughput**（%）：HBM 带宽利用率。Memory-bound kernel 应 >70%
- **L2 Hit Rate**：L2 缓存命中率。低于 50% 说明数据复用差
- **Shared Memory Throughput**：共享内存带宽利用率
- **Global Load/Store Efficiency**：有效数据 / 实际传输数据的比值。低于 100% 说明存在未对齐或非 coalesced 访问

**3. 执行效率指标**：
- **Occupancy**：活跃 warp 数 / SM 最大 warp 数。通常 >50% 较好，但不是越高越好（高 occupancy 可能减少每线程寄存器降低 ILP）
- **Warp Stall Reasons**：warp 等待原因分析（等内存/等同步/等指令发射）
- **Branch Efficiency**：分支效率，低说明存在 warp divergence

**4. Roofline Model 分析**：
- X 轴：算术强度（FLOPs/Byte），Y 轴：实际 FLOPS
- 判断 kernel 处于计算瓶颈（靠近水平线）还是带宽瓶颈（靠近斜线）
- 指导优化方向：计算瓶颈 -> 用更高效指令/Tensor Core；带宽瓶颈 -> 减少访存/增大复用/压缩数据

**Nsight Systems**（系统级分析）关注：
- **GPU 利用率时间线**：是否有 GPU 空闲间隙（CPU 瓶颈 / kernel launch 延迟）
- **CUDA API 调用**：cudaMemcpy 是否占比过高（H2D/D2H 传输瓶颈）
- **多 stream 并行**：是否有效利用多 stream 实现计算-通信重叠

**优化决策流程**：Nsight Systems 定位热点 kernel -> Nsight Compute 深入分析该 kernel -> 根据 Roofline 位置决定优化方向 -> 实施优化 -> 重新 profile 验证。

### Q: KV Cache原理？FlashAttention内存优化思想？

**KV Cache 原理**：

自回归生成第 t 个 token 时：
- 需要计算 Attention(Q_t, K_{1:t}, V_{1:t})
- 如果不缓存，每步需对所有历史 token 重新计算 K 和 V（重复计算 t-1 个 token 的投影）
- KV Cache：将每步新计算的 K_t, V_t 追加到缓存，下一步直接读取

**显存占用**：2 * n_layers * n_kv_heads * head_dim * seq_len * batch * dtype_bytes。LLaMA-2 7B FP16 单请求 2048 token 约 1GB，长序列（128K）可达 64GB。

**Prefill vs Decode 的不同特征**：
- Prefill：一次性计算完整输入的 Q/K/V，大矩阵乘，compute-bound（充分利用 Tensor Core）
- Decode：每步只有 1 个 query token，GEMV（batch=1 的矩阵向量乘），memory-bound（读取全部 KV Cache 的带宽是瓶颈）

---

**FlashAttention 的内存优化思想**：

**问题**：标准 attention 计算 S=QK^T（N*N 矩阵）需要 O(N^2) 显存物化到 HBM，且 softmax 需要全局 max/sum 信息——看似无法分块。

**核心创新**：

1. **分块计算（Tiling）**：将 Q/K/V 分成小块（tile），每块大小适配 SRAM（~100KB shared memory）
2. **Online Softmax**：利用数学修正因子，允许分块计算 softmax 而无需全局 max：
   - 每处理一个新的 K/V 块，更新 running max 和 running sum
   - 对之前块的输出乘修正因子 exp(old_max - new_max) 进行缩放
3. **不物化 attention 矩阵**：N*N 的 attention score 矩阵从不完整存在于 HBM 中，只在 SRAM 中临时存在一小块

**性能收益**：
- HBM 访问量从 O(N^2) 降为 O(N^2 * d / M)，其中 M 为 SRAM 大小。对 N=4096, d=128, M=100KB，约减少 10x+ 的 HBM 访问
- 显存占用从 O(N^2) 降为 O(N)（只需存 output 和 logsumexp，不存 attention matrix）
- 实际加速：2-4x wall-clock speedup（减少 HBM I/O 是主因）

**FlashAttention-2 改进**：优化 warp 间工作分配，减少非矩阵乘计算的开销，进一步提升 Tensor Core 利用率。FlashAttention-3（Hopper）利用异步拷贝和 warp specialization。

### Q: C++虚函数实现（vtable）、编译四阶段、设计模式？

**虚函数与 vtable 机制**：

C++ 运行时多态通过虚函数表（vtable）实现：

**结构**：
- 每个含虚函数的类有一个 vtable（编译时生成），存放该类所有虚函数的函数指针
- 每个对象持有一个 vptr（虚函数表指针），指向其所属类的 vtable
- vptr 通常在对象内存布局的最前面（偏移 0）

**调用过程**：
```
Base* p = new Derived();
p->virtualFunc();  
// 编译为：(*(p->vptr[func_index]))(p)
// 1. 读对象的 vptr  
// 2. 在 vtable 中按索引找到函数指针  
// 3. 间接调用
```

**性能影响**：
- 额外的间接寻址（1-2 次内存访问查 vtable）
- 阻止内联优化（编译器无法确定调用目标）
- Cache miss：vtable 和目标函数可能不在 cache 中
- 高性能代码（如 CUDA kernel 内部）避免使用虚函数，改用模板（CRTP）实现静态多态

**对象大小影响**：每个含虚函数的对象额外占用 8 bytes（64-bit 系统的 vptr），可能破坏缓存行对齐。

---

**编译四阶段**：

| 阶段 | 输入 | 输出 | 关键操作 |
|------|------|------|---------|
| 预处理（Preprocessing） | .c/.cpp | .i | 宏展开、#include 插入、条件编译 |
| 编译（Compilation） | .i | .s（汇编） | 词法/语法分析、语义分析、IR优化、代码生成 |
| 汇编（Assembly） | .s | .o（目标文件） | 汇编指令转机器码、生成符号表 |
| 链接（Linking） | .o + .a/.so | 可执行文件 | 符号解析、重定位、合并段 |

**关键细节**：
- 编译阶段（第二步）是优化的主战场：O1/O2/O3 优化级别在这里生效
- 静态链接（.a）将库代码嵌入可执行文件；动态链接（.so/.dll）运行时加载
- 链接错误（undefined reference）通常是找不到符号定义——检查库路径和链接顺序

---

**常用设计模式**（AI Infra 相关）：

1. **单例模式（Singleton）**：确保全局唯一实例。应用：CUDA context manager、全局线程池、日志器
2. **工厂模式（Factory）**：封装对象创建逻辑。应用：根据算子类型创建不同的 kernel 实现（如 conv 的 im2col/winograd/fft 选择）
3. **观察者模式（Observer）**：事件通知机制。应用：训练中的 callback（如 loss 变化通知 early stopping）
4. **策略模式（Strategy）**：算法可替换。应用：量化策略（MinMax/Percentile/KL）、调度策略（FIFO/SJF/优先级）的运行时切换
5. **Builder 模式**：分步构建复杂对象。应用：计算图的逐步构建、TensorRT 的 engine builder
