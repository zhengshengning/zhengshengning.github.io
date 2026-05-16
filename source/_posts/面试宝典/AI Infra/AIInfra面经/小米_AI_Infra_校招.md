---
title: '小米 AI Infra 校招'
date: 2026-04-16 12:06:56
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 推理优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 手撕：实现Multi-Head Grouped Attention（MGA/GQA）的init和forward？

（编程题）

---

### Q: Transformer中从输入到输出的完整流程是什么？

以Decoder-only LLM（如Llama）为例，详细描述一次前向传播的完整数据流：

**1. 输入处理**：
- Token Embedding：输入token_ids通过Embedding查表得到向量表示，shape: [batch, seq_len] → [batch, seq_len, hidden_size]。
- 位置编码：RoPE（旋转位置编码）不直接加到embedding上，而是在QK计算时应用旋转矩阵，支持长度外推。

**2. Transformer Block × N层**（每层结构相同）：

```
输入 x
├── RMSNorm(x)                    # Pre-Norm归一化
├── Multi-Head Attention:
│   ├── Q = x_norm × Wq           # QKV线性投影 (GEMM)
│   ├── K = x_norm × Wk
│   ├── V = x_norm × Wv
│   ├── 对Q,K应用RoPE旋转         # 注入位置信息
│   ├── 多头分割: [B,S,H] → [B,num_heads,S,head_dim]
│   ├── Attention = softmax(QK^T/√d_k + causal_mask) × V
│   ├── 多头拼接 + 输出投影 O = concat × Wo
│   └── 结果
├── 残差连接: x = x + Attention_output
├── RMSNorm(x)                    # 第二个归一化
├── FFN (SwiGLU):
│   ├── gate = x_norm × W_gate    # Gate投影
│   ├── up = x_norm × W_up        # Up投影
│   ├── hidden = SiLU(gate) × up  # 门控激活
│   └── down = hidden × W_down    # Down投影回hidden_size
└── 残差连接: x = x + FFN_output
```

**3. 输出层**：
- 最终RMSNorm归一化。
- LM Head线性投影：[batch, seq_len, hidden_size] → [batch, seq_len, vocab_size]。
- Softmax得到词表上的概率分布（训练时用交叉熵loss，推理时sampling/argmax）。

**计算量估算**（对于参数量P的模型）：每个token的前向计算约2P FLOPs，反向约4P FLOPs。

---

### Q: KV Cache有哪些加载方式？

KV Cache的管理和加载方式直接影响推理系统的显存效率、延迟和吞吐，以下是主流方案及其权衡：

**1. 连续内存预分配**：
- 做法：为每个序列预分配 max_seq_len 大小的连续显存。
- 优点：实现简单，内存访问连续（cache友好）。
- 缺点：严重浪费——如果max=4096但实际只生成100个token，浪费97.5%。
- 适用：早期推理框架、固定长度batch。

**2. PagedAttention（分页动态分配）**：
- 做法：KV Cache按固定block（16 tokens）管理，通过block table映射。
- 优点：显存利用率>96%、消除碎片、支持共享。
- 缺点：间接寻址有少量计算开销、block table管理逻辑复杂。
- 代表：vLLM、TensorRT-LLM。

**3. Prefix Caching（前缀共享复用）**：
- 做法：对前缀内容计算hash，相同hash的KV Cache block直接复用，无需重新计算。
- 优点：多请求共享system prompt时节省大量计算和显存。
- 适用：chatbot场景（固定system prompt）、RAG（共享检索前缀）。
- 实现：vLLM的Automatic Prefix Caching (APC)。

**4. Offloading（CPU/磁盘卸载）**：
- 做法：将不活跃请求的KV Cache从GPU卸载到CPU内存或NVMe磁盘。
- 优点：突破GPU显存限制，支持更大并发。
- 缺点：换入时有PCIe带宽延迟（~32GB/s），需要预测性调度。
- 适用：并发请求波动大、部分请求有长等待的场景。

**5. 量化KV Cache**：
- 做法：将KV Cache以INT8/FP8存储，读取时on-the-fly反量化。
- 优点：显存减半（FP16→INT8），同显存支持2倍并发。
- 精度影响：通常对模型质量影响很小（KV的分布相对集中）。
- 代表：vLLM和TensorRT-LLM均支持FP8 KV Cache。

**6. 分布式KV Cache**：
- 做法：KV Cache分布在多卡/多机上，通过高速互联（NVLink/IB）按需获取。
- 适用：超长序列（>128K）、PD分离架构中Prefill机器计算完KV后传输给Decode机器。
- 挑战：传输延迟、同步开销。

---

### Q: PD分离（Prefill-Decode分离）机制是什么？

PD分离将大模型推理的两个阶段部署在**不同的硬件实例**上独立运行和优化，本质是针对两个阶段截然不同的计算特征做专门化部署。

**两阶段的计算特征对比**：

| 特征 | Prefill阶段 | Decode阶段 |
|------|-------------|------------|
| 计算类型 | Compute-bound | Memory-bound |
| 算术强度 | 高（大矩阵乘） | 低（矩阵-向量乘） |
| GPU利用率 | 高（>70%） | 低（<10%） |
| 主要开销 | GEMM计算 | 权重读取 + KV Cache读取 |
| Batch效率 | 天然高效 | 需要大batch填满带宽 |
| 延迟指标 | TTFT（首token时间） | TPOT（每token时间） |

**PD分离架构**：
- **Prefill实例**：配置高算力GPU（如H100），负责处理输入prompt的完整前向计算，生成KV Cache后传输给Decode实例。
- **Decode实例**：配置高带宽内存GPU，负责逐token自回归生成。可以积累大batch提高GPU利用率。
- **KV Cache传输**：通过高速网络（RDMA/NVLink）将KV Cache从Prefill实例传到Decode实例。

**核心优势**：
1. **各自独立扩缩容**：Prefill和Decode实例可根据各自负载独立扩缩，长prompt场景多扩Prefill，高并发场景多扩Decode。
2. **消除相互干扰**：传统混部方案中，长prefill会阻塞同batch中其他请求的decode（排队延迟），分离后彻底消除。
3. **硬件适配**：Prefill可选算力强但显存较小的卡，Decode可选显存大/带宽高的卡。
4. **调度灵活**：Prefill可做大batch并行（高效），Decode独立调度不受prefill延迟影响。

**实现挑战**：Prefill到Decode的KV Cache传输带宽需求大（如70B模型一次prefill 1K token产生约2GB KV Cache），需要高速网络支持。

---

### Q: MTP（Multi-Token Prediction）是什么？

MTP（Multi-Token Prediction）是DeepSeek-V3等模型采用的训练策略，让模型在每个位置**同时预测未来多个token**，而非传统的只预测下一个token。

**传统Next-Token Prediction的局限**：
- 每个位置只提供1 bit的监督信号（下一个token的交叉熵loss）。
- 模型可能学到"短视"策略——只关注局部模式而忽略长程规划。
- 信息密度低：一个训练样本能提供的梯度信号有限。

**MTP实现方式**：
- 在Transformer最后一层后接D个独立的预测头（通常D=2-4），第i个头预测第i+1个未来token。
- 每个预测头共享主干特征（最后一层hidden state），但有独立的线性投影层。
- 训练Loss = 主loss（next token）+ sum(alpha_i × 第i个额外头的loss)，alpha_i通常递减。

**训练收益**：
1. **更丰富的训练信号**：同一个位置提供D+1个token的监督，梯度信号更密集。
2. **加速收敛**：DeepSeek-V3报告MTP使训练收敛速度提升约30%。
3. **促进长程规划**：预测多步迫使模型学习更long-range的依赖关系。
4. **提升模型表示质量**：中间层的hidden state需要编码更多未来信息。

**推理收益（配合投机解码）**：
- 训练好的额外预测头可直接用作Speculative Decoding的draft heads。
- 每步用D个头一次性预测D个候选token，再由主模型批量验证（一次前向验证D个位置）。
- 接受率通常60-80%，等效加速1.5-2.5倍。
- 相比独立draft模型，MTP头与主模型共享参数，对齐度更高、接受率更高。

**类似技术**：Medusa（推理侧增加多个tree-structured预测头）、EAGLE（利用draft model的feature做speculative decoding）。
