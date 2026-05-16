---
title: '字节跳动 AI Infra (2)'
date: 2026-04-16 12:17:12
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 推理优化, 高性能计算, 大厂面经, 面经]
---


<!-- more -->

---

### Q: vLLM与PagedAttention的原理？

**PagedAttention核心原理**：

传统KV Cache管理的内存浪费来源于三点：
1. **过度预留**：按max_seq_len预分配（请求实际可能只用10%）。
2. **内部碎片**：连续分配后请求结束释放，留下不连续空洞。
3. **外部碎片**：空闲空间总量够但不连续，新请求无法使用。

PagedAttention的分页解决方案：
```
物理Block池 (固定大小，如每block 16 tokens的KV):
┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
│ B0 ││ B1 ││ B2 ││ B3 ││ B4 ││ B5 ││ B6 ││ B7 │
└────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘

请求A (已生成40 tokens): Block Table = [B2, B5, B7] (3个block)
请求B (已生成20 tokens): Block Table = [B0, B4]     (2个block)
请求C (Beam 1):         Block Table = [B1, B3, B6] (其中B1,B3来自prefix共享)
请求C (Beam 2):         Block Table = [B1, B3, B6'] (B6'是COW复制)
```

Attention计算时，kernel通过block table查找物理地址：
```cuda
// PagedAttention kernel的核心逻辑
for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int physical_block = block_table[seq_id][block_idx];
    float* k_ptr = k_cache + physical_block * block_size * head_dim;
    float* v_ptr = v_cache + physical_block * block_size * head_dim;
    // 计算attention...
}
```

**vLLM系统架构**：

```
HTTP/gRPC请求 → AsyncLLMEngine
    ├── Tokenizer (分词)
    ├── Scheduler (调度决策)
    │   ├── Prefill队列 (新请求)
    │   ├── Running队列 (正在decode的请求)  
    │   └── Swapped队列 (换出到CPU的请求)
    ├── Block Manager (管理物理block分配)
    └── Worker (执行实际推理)
        ├── Model Runner (前向计算)
        ├── PagedAttention Kernel
        └── Sampler (采样)
```

调度策略：
- 优先处理Running队列中的decode请求（避免TPOT退化）。
- 有空闲block时从Prefill队列拉入新请求。
- 显存不足时将部分请求的KV Cache swap到CPU（FIFO或基于优先级）。
- 抢占（Preemption）：新高优请求可以swap出低优请求。

**性能数据对比**（A100-80GB, LLaMA-13B, ShareGPT trace）：

| 系统 | 吞吐(req/s) | 显存利用率 | 延迟P50 |
|------|------------|-----------|---------|
| HuggingFace | 2.1 | 20-30% | 高 |
| TGI | 8.5 | 50-60% | 中 |
| vLLM | 24.3 | 95%+ | 低 |

vLLM的显著优势主要来自PagedAttention的高显存利用率 + Continuous Batching的调度效率。

### Q: KV Cache原理与优化？

**原理核心**：

Transformer Decoder层中的注意力计算：
```
对第t步:
  Q_t = x_t × W_Q  (当前token的query，维度[1, d])
  K_t = x_t × W_K  (当前token的key)
  V_t = x_t × W_V  (当前token的value)
  
  K_cache = concat(K_cache, K_t)  → [t, d]
  V_cache = concat(V_cache, V_t)  → [t, d]
  
  Attention_t = softmax(Q_t × K_cache^T / sqrt(d)) × V_cache  → [1, d]
```

**为什么K和V可以缓存而Q不行**：
- K_t = x_t × W_K：只依赖token t的embedding，不会因为后续token改变。
- Q_t = x_t × W_Q：同理只依赖当前token，但每步的Q不同（因为当前token不同）。
- 因此历史token的K/V不变可缓存，每步只需计算新token的Q/K/V。

**优化技术全景**：

| 层次 | 技术 | 核心思想 | 效果 |
|------|------|---------|------|
| 内存管理 | PagedAttention | 分页消除碎片 | 利用率100% |
| 架构设计 | GQA(8KV heads) | 减少KV头数 | 显存-8x |
| 架构设计 | MLA(低维投影) | KV压缩到潜在空间 | 显存-16x |
| 数值精度 | KV量化(INT8) | 低精度存储 | 显存-2x |
| 信息筛选 | 稀疏KV(H2O) | 只保留重要token | 显存-5-10x |
| 内容复用 | Prefix Caching | 共享前缀KV | 计算节省 |
| 存储扩展 | Offloading | 卸载到CPU/NVMe | 容量+N倍 |

**MLA（Multi-head Latent Attention，DeepSeek-V2）详解**：

标准GQA：K/V维度 = num_kv_heads × head_dim（如8×128=1024）。
MLA：将KV联合压缩到低维潜在空间（如512维），推理时通过learnable投影恢复。

```
标准: K = X × W_K [d → num_kv_heads × head_dim]
MLA:  c = X × W_down [d → d_compress(512)]  # 压缩
      K = c × W_up_K [d_compress → num_heads × head_dim]  # 恢复（推理时）
```

KV Cache只存压缩后的c（512维 vs 原始1024维），但attention计算时等效于全维度KV。效果：KV Cache减少50-90%，精度无损（训练时学习最优压缩）。

### Q: 推理加速的综合策略有哪些？

将LLM推理加速技术按层次组织，从模型设计到系统部署的全栈优化：

**第1层：模型级优化（改变模型本身）**：

| 技术 | 方法 | 效果 | 代价 |
|------|------|------|------|
| 量化 | W4A16/W8A8/FP8 | 2-4x内存减少，1.5-3x加速 | <1-2% perplexity损失 |
| 剪枝 | 层/头/通道剪枝 | 20-50%参数减少 | 需fine-tune恢复精度 |
| 蒸馏 | 大→小模型知识迁移 | 使用更小模型 | 训练成本 |
| GQA/MQA | 减少KV头数 | KV Cache减少4-64x | 需从头训练 |
| MoE | 稀疏激活 | 活跃参数减少4-8x | 总参数大，显存需求高 |

**第2层：算法级优化（改变计算方式）**：

| 技术 | 方法 | 效果 | 适用阶段 |
|------|------|------|---------|
| FlashAttention | Tiling+Online Softmax | Prefill 2-4x加速 | Prefill |
| FlashDecoding | KV维度Split并行 | Decode长序列加速 | Decode |
| 投机解码 | Draft+Verify | 2-3x生成加速 | Decode |
| 稀疏注意力 | 只计算部分KV | KV访存减少 | Decode |
| Early Exit | 简单token提前退出 | 平均加速1.3-1.5x | Decode |

**第3层：系统级优化（改变调度和管理）**：

| 技术 | 方法 | 效果 | 复杂度 |
|------|------|------|--------|
| Continuous Batching | 迭代级调度 | 吞吐2-3x | 中 |
| PagedAttention | 分页KV管理 | 并发3-5x | 中 |
| PD分离 | Prefill/Decode独立部署 | 利用率+30-50% | 高 |
| 请求调度 | SLA感知优先级 | P99改善 | 中 |
| 抢占/Swap | 不活跃请求卸载 | 处理更多请求 | 高 |

**第4层：硬件级优化（充分利用硬件能力）**：

| 技术 | 方法 | 效果 | 要求 |
|------|------|------|------|
| Tensor Core | FP16/INT8/FP8矩阵指令 | 计算16x加速 | 对齐要求 |
| Tensor Parallel | 多卡拆分降延迟 | 延迟÷N(TP度) | NVLink |
| Pipeline Parallel | 跨节点 | 支持超大模型 | 节点间互联 |
| NVLink/NVSwitch | 高速互联 | 通信开销最小化 | 硬件配置 |
| HBM3E(B200) | 8TB/s带宽 | Decode 4x加速 | 新硬件 |

**第5层：编译优化（自动化提升）**：

| 技术 | 方法 | 效果 | 工具 |
|------|------|------|------|
| 图优化 | 算子融合/常量折叠 | 1.2-2x | TensorRT/torch.compile |
| Kernel生成 | 自动生成优化kernel | 接近手写性能 | Triton/TVM |
| Auto-tune | 搜索最优配置 | 5-20%额外提升 | TensorRT build |
| 量化编译 | INT8/FP8 kernel生成 | 精度+速度 | TensorRT-LLM |

**实际部署中的组合策略（70B模型A100 8卡）**：
```
模型: W8A8量化 + GQA(原生)
算法: FlashAttention + Speculative Decoding
系统: vLLM(PagedAttention + Continuous Batching)
硬件: 8卡TP=8, NVLink互联
编译: TensorRT-LLM with FP8
→ 最终吞吐: ~3000 tokens/s (总), 延迟: TTFT<200ms, TPOT<30ms
```
