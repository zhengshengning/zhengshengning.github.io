---
title: '京东 AI Infra 校招 一面'
date: 2026-04-16 12:06:16
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 大模型推理优化有哪些方法？

大模型推理优化从多个维度入手，核心目标是降低延迟和提高吞吐：

**算子层面**：
- **FlashAttention**：通过 tiling 将 Attention 分块在 SRAM 中计算，避免在 HBM 中存储 N×N 的 attention 矩阵。减少 HBM 读写次数是加速的关键（Attention 本身是 memory-bound），实际加速 2-4x
- **算子融合**：将多个小 kernel 合并为一个大 kernel（如 QKV projection 融合、LayerNorm+Linear 融合），减少 kernel launch 开销（每次 ~5-10μs）和中间 tensor 的 HBM 读写

**内存管理**：
- **PagedAttention**：类似 OS 虚拟内存的 KV Cache 分页管理，将 KV Cache 利用率从传统的 20-40% 提升到接近 100%
- **KV Cache 量化**：FP8/INT8 量化 KV Cache，减少 50% 存储，对精度影响可控

**调度层面**：
- **Continuous Batching**：每个 iteration 动态加入新请求、移除已完成请求，相比 static batching 吞吐提升 2-10x
- **PD 分离**：Prefill（compute-bound）和 Decode（memory-bound）分开部署到不同硬件，各自最大化利用率
- **Chunked Prefill**：长 prompt 分 chunk 执行，chunk 间插入 Decode 请求，平衡延迟和吞吐

**模型压缩**：
- **量化**：INT8（1.5-2x 加速）、INT4（2-3x 加速）、FP8（Hopper 原生支持，算力翻倍）
- **剪枝**：移除不重要的注意力头或 FFN 通道
- **蒸馏**：训练小模型模拟大模型行为

**解码加速**：
- **投机解码（Speculative Decoding）**：小模型快速生成 K 个候选 token，大模型一次并行验证，接受率高时可加速 2-3x
- **并行解码**：Medusa/EAGLE 多头预测，一次生成多个 token

---

### Q: PagedAttention 具体是什么？

PagedAttention 是 vLLM 提出的 KV Cache 管理方案，借鉴操作系统虚拟内存的**分页思想**：

**核心设计**：
- 将 KV Cache 划分为固定大小的 **block**（类似内存页，通常每个 block 存 16 个 token 的 KV）
- 每个请求维护一个 **block table**（类似页表），记录逻辑 block 到物理 block 的映射
- 物理 block 在 GPU 显存中可以不连续，通过 block table 做间接寻址

**工作流程**：
1. 新请求到来时，分配所需的 block（不需要预分配最大序列长度）
2. 生成过程中，当前 block 满了再分配新 block
3. 请求结束后，释放所有 block 回空闲池

**核心优点**：
- **消除碎片**：无需预分配连续空间，block 可以不连续存放。传统方案需要为每个请求预留 max_seq_len 的连续显存，造成大量内部碎片
- **按需分配**：实际用多少分配多少，显存利用率接近 100%（传统方案只有 20-40%）
- **支持 KV Cache 共享**：beam search 中多个 beam 可以共享公共前缀的 block（copy-on-write），只在分歧时拷贝
- **支持 Prefix Caching**：相同 system prompt 的请求共享 KV Cache block

**Attention 计算适配**：PagedAttention kernel 需要根据 block table 索引分散的 KV block，计算效率略低于连续存储（~5% overhead），但整体吞吐因 batch 增大而大幅提升。

---

### Q: FlashAttention 为什么能加速？计算过程是什么？

**加速原因——HBM 带宽是瓶颈**：

标准 Attention 的计算 `O = softmax(QK^T/√d) × V` 需要：
1. 从 HBM 读取 Q, K → 计算 S = QK^T → 写回 HBM（N×N 矩阵）
2. 从 HBM 读取 S → 计算 softmax → 写回 HBM
3. 从 HBM 读取 P, V → 计算 O → 写回 HBM

总 HBM 读写量 ∝ O(N^2)，而 SRAM 只有 ~20 MB，N^2 的 attention 矩阵必须存在 HBM 中。对于 seq_len=4096、head_dim=128，attention 矩阵 = 4096×4096×2bytes = 32 MB/head，远超 SRAM。

FlashAttention 的核心思想：**永远不在 HBM 中物化完整的 N×N attention 矩阵**。

**计算过程（分块 + Online Softmax）**：

```
对 Q 分成 T_Q 个 block（外层循环）：
  对 K/V 分成 T_KV 个 block（内层循环）：
    1. 从 HBM 加载 Q_block, K_block, V_block 到 SRAM
    2. 计算局部 S_block = Q_block × K_block^T
    3. Online Softmax 更新：
       - m_new = max(m_old, rowmax(S_block))  # 更新 running max
       - l_new = l_old × exp(m_old - m_new) + rowsum(exp(S_block - m_new))  # 更新 running sum
       - O_new = O_old × (l_old/l_new) × exp(m_old-m_new) + exp(S_block-m_new) × V_block / l_new
    4. 不写回中间矩阵！
  输出最终 O_block 到 HBM
```

**关键技术**——Online Softmax 使得分块计算可以得到**精确**的 softmax 结果（非近似），只需维护 running max 和 running sum 两个标量。

**效果对比**：
| 指标 | 标准 Attention | FlashAttention |
|---|---|---|
| 显存复杂度 | O(N²) | O(N) |
| HBM 读写量 | O(N²·d) | O(N²·d²/M)（M=SRAM大小）|
| 实际加速 | baseline | 2-4x |
| 适用场景 | 短序列 | 长序列收益更明显 |

---

### Q: PD 分离机制中如何实现调度队列？Chunked Prefill 是什么？

**PD 分离调度架构**：

```
            ┌──────────────┐
            │   调度器      │
            │ (Router)      │
            └──────┬───────┘
           ┌───────┴────────┐
     ┌─────▼─────┐    ┌────▼─────┐
     │ Prefill   │    │  Decode   │
     │ 节点池    │    │  节点池   │
     │(高算力GPU)│    │(高带宽GPU)│
     └─────┬─────┘    └────▲─────┘
           │  KV Cache传输  │
           └────────────────┘
```

**调度实现**：
- 调度器维护两类请求队列：Prefill Queue 和 Decode Queue
- 新请求先进入 Prefill Queue → 分配给负载最低的 Prefill 节点
- Prefill 完成后，KV Cache 通过高速网络（NVLink/RDMA）传输到 Decode 节点
- Decode 节点接收 KV Cache 后将请求加入本地 Decode 批次
- 负载均衡策略：根据各节点的 batch 占用率、显存利用率做加权分配

**Chunked Prefill**：

**问题**：长 prompt（如 8K tokens）的 Prefill 计算可能独占 GPU 数百毫秒，期间所有 Decode 请求被阻塞，导致正在生成的请求延迟飙升（TPOT 恶化）。

**解决方案**：
- 将长 prompt 的 Prefill 分成多个 chunk（如每 chunk 512 tokens）
- 每个 chunk 计算完成后，插入当前 batch 中其他请求的 **Decode token** 一起计算
- 形成交替执行：`[Prefill chunk 1] → [Decode tokens] → [Prefill chunk 2] → [Decode tokens] → ...`

**关键点**：
- 插入的是**其他请求**的 Decode（不是同一请求的），因为当前请求还没 Prefill 完
- chunk 大小的选择需要平衡：chunk 越小 Decode 延迟越低，但 Prefill 效率越低（GEMM 利用率下降）
- 实现需要维护跨 chunk 的中间状态（已计算的 KV Cache）

---

### Q: C++ 的多态如何实现？虚函数表的函数顺序是什么？

**多态的实现机制**：

C++ 运行时多态通过**虚函数（virtual function）+ 动态绑定（dynamic dispatch）**实现。当通过基类指针/引用调用虚函数时，实际调用取决于对象的**真实类型**（而非指针类型）。

**底层实现——虚函数表（vtable）**：

```
对象内存布局：
┌──────────┐
│ vptr ──────→ ┌───────────────────┐
├──────────┤   │ vtable            │
│ 成员变量  │   │ [0] func_A_ptr   │
│ ...       │   │ [1] func_B_ptr   │
└──────────┘   │ [2] func_C_ptr   │
               └───────────────────┘
```

- 每个含虚函数的类有**一个 vtable**（静态数据，编译期生成）
- 每个对象有**一个 vptr**（占 8 字节，在构造函数中设置），指向其所属类的 vtable
- 调用虚函数时：`obj->vptr->vtable[offset](...)`，通过两次间接寻址完成动态分发
- 多态调用的额外开销：一次指针解引用（~1 cycle），几乎可以忽略

**虚函数表中函数顺序**：
1. 按虚函数在类中**首次声明的顺序**排列
2. 派生类的 vtable 结构：
   - 先排基类的虚函数槽位（如果被重写，替换为派生类版本）
   - 再在末尾追加派生类新增的虚函数
3. 多继承时：每个基类子对象有各自的 vptr，指向不同的 vtable 段

```cpp
class Base {
    virtual void f();  // vtable[0]
    virtual void g();  // vtable[1]
};
class Derived : public Base {
    void f() override;      // vtable[0] → Derived::f
    virtual void h();       // vtable[2]（新增）
    // g() 未重写          // vtable[1] → Base::g（继承）
};
```

---

### Q: 手撕：实现快速排序？

（编程题）
