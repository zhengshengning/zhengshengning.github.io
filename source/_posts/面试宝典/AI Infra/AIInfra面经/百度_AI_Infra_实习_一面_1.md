---
title: '百度 AI Infra 实习 一面 (1)'
date: 2026-04-16 12:07:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: FusedAttention优化怎么做的？

FusedAttention 的核心思想是将 Attention 中的多个独立操作融合为**单一 Kernel**，消除中间结果写回全局内存的开销。

**传统 Attention 的问题**：标准实现需要多个 Kernel 串行执行：QK 矩阵乘 -> Scale -> Mask -> Softmax -> AV 矩阵乘。每个 Kernel 之间需要将中间结果（如 N*N 的 attention score 矩阵）写回 HBM 再读入下一个 Kernel。对于长序列（N=4096），这个 N^2 的中间矩阵显存占用巨大且 IO 开销远超计算本身。

**融合策略**：
1. 将上述所有操作合并到一个 Kernel 中，中间结果保留在寄存器或 Shared Memory 中
2. 减少 Kernel Launch 开销（每次 launch 有 ~5-10us 的固定延迟）
3. 避免 N^2 attention 矩阵的全局内存 materialization

**典型实现**：FlashAttention 是最成功的 FusedAttention 实现，通过 tiling + online softmax 实现了完全融合且精确的 attention 计算。xFormers 的 memory_efficient_attention 也是类似思路。

### Q: 介绍FlashAttention？

FlashAttention（Dao et al., 2022）通过**IO 感知的分块计算**彻底优化 Attention 的内存效率。

**核心创新**：
1. **Tiling（分块）**：将 Q/K/V 按块加载到 SRAM（Shared Memory）中计算，块大小根据 SRAM 容量确定（通常 Br=Bc=128）
2. **Online Softmax**：在不知道全行 max/sum 的情况下，通过维护 running statistics 增量计算正确的 softmax
3. **Recomputation**：反向传播时重新计算 attention（而非存储），用 FLOPs 换显存

**性能收益**：
- IO 复杂度从 O(N^2) 降到 O(N^2*d/M)（M 为 SRAM 大小），在 A100 上约快 2-4x
- 显存占用从 O(N^2) 降到 O(N)（不需要存储完整 attention 矩阵）
- 支持更长序列训练（如 16K-128K），不受显存限制

**适用场景**：任何使用标准 self-attention 的模型（BERT/GPT/LLaMA 等）。已成为所有主流框架（PyTorch 2.0+、HuggingFace）的默认 attention 实现。

### Q: FlashAttention的数学推导（Online Softmax）？

Online Softmax 允许分块计算 softmax 而无需一次看到整行数据：

**递推公式**（处理第 j 个 KV 块时）：
1. 计算当前块的局部 max：`m_j = max(K_j * q)`
2. 更新全局 max：`m_new = max(m_old, m_j)`
3. 修正旧的 exp sum：`l_new = l_old * exp(m_old - m_new) + sum(exp(scores_j - m_new))`
4. 修正旧的 output：`O_new = O_old * (l_old * exp(m_old - m_new) / l_new) + (exp(scores_j - m_new) / l_new) * V_j`

**数学正确性**：通过乘以修正因子 `exp(m_old - m_new)` 将之前基于旧 max 的计算结果修正为基于新 max 的等价结果。最终结果与标准 softmax 数值上完全等价（非近似）。

**关键洞察**：softmax(x_i) = exp(x_i - m) / sum(exp(x_j - m))，当 m 改变时只需对已有结果乘一个修正系数即可。

### Q: RMSNorm为什么相比LayerNorm有提升？

RMSNorm（Root Mean Square Normalization）是 LayerNorm 的简化版本，去掉了均值中心化步骤：

**公式对比**：
- LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
- RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma

**为什么有效**：
1. **计算量减少**：不需要计算均值和减去均值的操作。对于 hidden_size=4096 的向量，省去一次 reduction（求 mean）和一次 element-wise 减法
2. **实验验证**：Zhang & Sennrich (2019) 证明均值中心化对训练稳定性的贡献可忽略，归一化的核心价值在于方差缩放
3. **kernel 优化更友好**：操作更少意味着更容易与相邻算子融合（如 RMSNorm + Linear 融合）
4. **去掉 beta 偏置**：LLaMA 等现代模型还去掉了 beta（bias=0），进一步简化

**实际收益**：在 LLaMA 65B 中，每层 RMSNorm 相比 LayerNorm 节省约 5-10% 的归一化计算时间，且训练质量无损。

### Q: 设计一个更灵活有效的显存分配方式（cudaAllocator）？

频繁调用 `cudaMalloc/cudaFree` 效率极低（每次 ~1ms 的 driver 调用），需要自定义内存分配器。

**设计方案**：

1. **内存池化（Pool Allocator）**：启动时预分配大块显存（如一次 cudaMalloc 1GB），后续从池中切分。避免频繁系统调用。

2. **Slab 分级分配**：按大小级别（如 256B/1KB/4KB/64KB/1MB...）维护独立空闲链表。分配时找到最小满足级别的空闲块，O(1) 分配。类似 Linux 的 slab allocator。

3. **延迟释放 + 空闲合并**：释放的块放入空闲池而非立即归还。相邻空闲块可合并（Buddy System）减少碎片。超出水位线时批量归还系统。

4. **流感知复用（Stream-Ordered Allocator）**：不同 CUDA Stream 的生命周期不重叠的块可以复用相同物理地址。通过记录分配/释放的 CUDA Event 判断安全性。PyTorch 的 CUDACachingAllocator 正是此思路。

5. **碎片整理**：定期扫描空闲块分布，必要时通过 cudaMemcpy 紧凑化。但运行时整理有额外拷贝开销。

**参考实现**：PyTorch CUDACachingAllocator、CUDA 11.2+ 的 cudaMallocAsync（内置流序分配器）。

### Q: Llama模型中有几个全连接层？

每个 Transformer Block 中的线性层（以 LLaMA-2 为例）：

**Attention 部分**（4 个）：
- Q 投影：hidden_size -> n_heads * head_dim
- K 投影：hidden_size -> n_kv_heads * head_dim（GQA 时比 Q 小）
- V 投影：hidden_size -> n_kv_heads * head_dim
- O 投影：n_heads * head_dim -> hidden_size

**FFN 部分**（3 个，SwiGLU 结构）：
- gate_proj（W_gate）：hidden_size -> intermediate_size
- up_proj（W_up）：hidden_size -> intermediate_size
- down_proj（W_down）：intermediate_size -> hidden_size

**总计**：每个 Transformer Block 有 **7 个线性层**。此外模型首尾还有 token embedding 层和 LM head（通常共享权重），但不在 Transformer Block 内。

### Q: Llama2的推理流程？每一层有什么算子？

**单层 Transformer Block 的算子序列**：

```
Input x
├── RMSNorm_1(x)
├── Q/K/V Linear (3 or 4 个矩阵乘)
├── RoPE 位置编码 (element-wise 旋转)
├── KV Cache 追加
├── Attention: softmax(QK^T/sqrt(d)) * V (矩阵乘 + softmax + 矩阵乘)
├── O Linear (矩阵乘)
├── Residual Add: x = x + attn_out
├── RMSNorm_2(x)
├── gate = SiLU(Gate Linear(x))  (矩阵乘 + 激活)
├── up = Up Linear(x)  (矩阵乘)
├── element-wise multiply: gate * up
├── Down Linear(gate * up)  (矩阵乘)
└── Residual Add: x = x + ffn_out
```

**最后输出层**：RMSNorm -> LM Head Linear（hidden -> vocab_size）-> Softmax/Sampling

**算子类型统计**：主要是 GEMM（矩阵乘，7 个/层）、element-wise（RoPE/SiLU/乘法/加法）、reduction（RMSNorm/Softmax）。GEMM 占 >90% 的计算量。

### Q: C++11有哪些新特性？

C++11 是现代 C++ 的分水岭，引入了大量核心特性：

**类型系统**：auto 类型推导、decltype、nullptr（替代 NULL）、constexpr 编译期计算

**移动语义**：右值引用（&&）、std::move、移动构造/赋值，避免深拷贝

**智能指针**：unique_ptr/shared_ptr/weak_ptr，替代裸指针管理

**Lambda**：`[capture](params) -> ret { body }`，就地定义匿名函数对象

**并发**：std::thread、std::mutex、std::atomic、std::future/promise、条件变量

**其他**：范围 for（`for(auto& x : container)`）、variadic templates、initializer_list、enum class（强类型枚举）、override/final、static_assert

### Q: C++智能指针？

shared_ptr（引用计数共享所有权，原子操作线程安全）、unique_ptr（独占所有权，零额外开销，默认首选）、weak_ptr（弱引用不增计数，打破循环引用/缓存观察）。详细用法见前面问题。

### Q: unique_ptr如何保证唯一性？

通过**编译期约束**实现：将拷贝构造函数和拷贝赋值运算符声明为 `= delete`。任何尝试拷贝 unique_ptr 的代码都会在编译期报错。

只允许移动操作（移动构造和移动赋值），转移后原 unique_ptr 变为 nullptr。这保证了任何时刻最多只有一个 unique_ptr 拥有资源。

```cpp
unique_ptr<int> p1 = make_unique<int>(42);
// unique_ptr<int> p2 = p1;          // 编译错误！拷贝被delete
unique_ptr<int> p2 = std::move(p1);  // OK，p1变为nullptr
```

### Q: shared_ptr何时析构？

当指向对象的**最后一个 shared_ptr** 销毁或重置时（引用计数 strong_count 降为 0），触发析构并释放对象内存。

**但控制块不一定释放**：控制块中除了 strong_count 还有 weak_count。只有当 weak_count 也降为 0 时（所有 weak_ptr 也销毁），控制块本身才被释放。这意味着如果有 weak_ptr 存活，对象已释放但控制块还在（weak_ptr::lock() 可以安全地检测到对象已死亡）。

**注意**：如果使用 `make_shared`，对象和控制块是一次分配的连续内存——对象析构后整块内存要等 weak_count 归零才释放。这可能导致大对象内存延迟释放的问题。

### Q: 类的成员函数可以当模板吗？

**可以**。类的非虚成员函数可以是模板函数。在使用（调用）时才实例化具体版本。

```cpp
class Parser {
public:
    template<typename T>
    T parse(const string& input) { ... }
};
```

**限制**：虚函数不能是模板。原因是 vtable 的大小需要在编译期确定，而模板可能有无限多个实例化版本，编译器无法为每个可能的实例化都在 vtable 中预留槽位。

### Q: 左值和右值？

- **左值（lvalue）**：有名字、可取地址、生命周期持续的表达式。如变量 `x`、`*ptr`、`arr[i]`。可以出现在赋值号左边。
- **右值（rvalue）**：临时对象、字面量，生命周期即将结束。如 `42`、`x+1`、函数返回的临时对象。
- **右值引用（T&&）**：延长临时对象的生命周期，支持移动语义。`std::move(lvalue)` 将左值转为右值引用。

移动语义的核心价值：对含堆内存的类型（string/vector），移动只需转移内部指针（O(1)），避免深拷贝（O(n)）。

### Q: CUDA有哪几种编程手段？

从高层到底层：

1. **高层库调用**：cuBLAS（矩阵运算）、cuDNN（深度学习）、Thrust（类 STL 并行算法）。最方便但灵活性有限。

2. **CUDA C/C++**：编写 `__global__` kernel 函数，最主流的 GPU 编程方式。开发者控制线程组织、内存层次和同步。

3. **CUDA Graph**：预录制 kernel 执行图，一次提交整个 DAG。减少反复 launch 的 driver 开销（~5us/launch），适合推理等固定执行模式。

4. **PTX 内联汇编**：NVIDIA 的虚拟 ISA，在 CUDA C 中用 `asm()` 嵌入。可使用特殊指令（如 ldmatrix、mma）。

5. **Triton**（新兴）：Python 到 GPU kernel 的编译器，通过 tile 抽象简化 kernel 编写，自动处理内存层次和 tiling。

### Q: Tensor Core和CUDA Core的区别？

| 对比 | CUDA Core | Tensor Core |
|------|-----------|-------------|
| **操作粒度** | 标量 FMA（a*b+c） | 矩阵 MMA（D=A*B+C，如4x4x4） |
| **每周期输出** | 1 个浮点结果 | 一个矩阵块（如 16 个结果） |
| **精度** | 任意（FP64/FP32/FP16/INT） | 混合精度（FP16/BF16/INT8输入，FP32累加） |
| **吞吐量(A100)** | FP32: 19.5 TFLOPS | FP16: 312 TFLOPS (16x) |
| **编程方式** | 普通算术运算 | WMMA/MMA PTX指令 |
| **适用场景** | 通用计算 | 矩阵乘加（GEMM/Attention/Conv） |

**为什么 Tensor Core 更快**：本质上是用更多晶体管面积换取吞吐量——一个 Tensor Core 占用面积远大于一个 CUDA Core，但在矩阵乘场景下 FLOPs/mm^2 更高。大模型训练/推理中 90%+ 的计算是 GEMM，因此 Tensor Core 是性能的决定性因素。

### Q: 手撕：最长连续序列（LeetCode 128）？

（编程题）

### Q: 手撕：至多包含K个不同字符的最长子串（LeetCode 340）？

（编程题）
