---
title: '原粒半导体 AI Infra 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 如何优化CUDA核函数？

CUDA kernel优化遵循"Profile→Analyze→Optimize→Verify"循环，根据瓶颈类型选择不同策略：

**1. 访存优化（解决Memory-bound）**：
- **合并全局内存访问（Coalesced Access）**：确保warp内32个线程访问128字节对齐的连续地址。不对齐/stride访问会触发多次内存事务，带宽利用率可能降至1/32。例如AoS(Array of Struct)改为SoA(Struct of Array)布局。
- **共享内存缓存**：将频繁复用数据从HBM(~400 cycles)加载到Shared Memory(~20 cycles)。典型：GEMM的tile、stencil计算的邻域。
- **向量化读写（float4）**：128位单次传输，减少4倍指令和事务数。要求16字节对齐。
- **避免Bank Conflict**：Shared Memory有32个bank，同warp不同线程访问同bank不同行会串行。解决：padding(数组每行多加1个float)。

**2. 并行度优化**：
- **提高Occupancy**：合理控制每线程寄存器和每Block的Shared Memory使用量。通常128或256线程/Block是好的起点。
- **注意**：Occupancy不是越高越好。有时降低Occupancy换取更多寄存器（减少spill到local memory）反而更快。

**3. 计算优化（解决Compute-bound）**：
- **减少Warp Divergence**：数据重排让同warp线程走相同分支，或用位运算替代条件分支。
- **循环展开**：`#pragma unroll` 消除循环开销、暴露ILP机会。
- **快速数学函数**：`__expf()/__sinf()/__rsqrtf()` 用硬件近似单元，比标准函数快5-10倍（精度损失<2ULP）。

**4. 延迟隐藏**：
- **Double Buffering预取**：计算当前tile时，用cp.async异步加载下一tile。DMA搬运不占计算管线。
- **增加活跃Warp数**：让GPU调度器在Warp stall时有其他就绪Warp可切换。

**5. 算子融合**：
- 多个小kernel合并为一个：省去kernel launch开销（~5us/次）+ 中间tensor不写回HBM再读取。
- 典型：LayerNorm+Add、GEMM+Bias+Activation。

**6. Profiling驱动优化**：
- **Nsight Compute**：分析compute/memory throughput比值定位bound类型，查看stall原因分布。
- **Roofline Model**：可视化kernel离峰值的距离，明确优化方向。

---

### Q: Reduce计算顺序会影响精度吗？

**会**。浮点数运算不满足结合律（(a+b)+c ≠ a+(b+c)），不同的累加顺序会产生不同的舍入误差。

**根本原因**：
- 浮点加法中，较小的数可能被"吞掉"。例如：1e10 + 1.0 - 1e10 = 0.0（而非1.0），因为1.0加到1e10时精度不够表示差异。
- 当大数累积后再加小数，小数的贡献可能完全丢失。

**并行Reduce改变顺序的影响**：
```
串行: sum = a[0] + a[1] + a[2] + a[3] + ... + a[N-1]  (从左到右)
树形: sum = (a[0]+a[1]) + (a[2]+a[3]) + ...  (层次累加)
```
两者结果可能不同！尤其当数值范围跨越多个数量级时差异更大。

**精度敏感度与数据类型的关系**：
- FP64：尾数52位，通常误差可接受。
- FP32：尾数23位，大规模Reduce（如N>10000）误差可能明显。
- FP16/BF16：尾数仅10/7位，**非常敏感**。大规模Reduce几乎必须在FP32下执行。

**缓解方法**：
1. **Kahan Summation（补偿求和）**：维护一个补偿项c记录累积的舍入误差，每次累加时补偿。误差从O(N×eps)降到O(eps)。
2. **混合精度Reduce**：输入FP16/BF16，累加器用FP32，最终结果转回低精度。这是LLM训练中的标准做法。
3. **从小到大排序后累加**：让小数先累加，避免被大数吞掉。但排序本身有开销。
4. **分块Reduce**：先在小块内FP32累加，块间再累加。

**实践中**：对于深度学习，通常要求的精度是"模型收敛+推理质量可接受"，而非bit-exact。因此并行Reduce的非确定性舍入误差通常是可接受的，但需要监控（如loss异常或模型输出NaN）。

---

### Q: 怎么确定计算精度是否满足要求？

精度验证是算子开发和量化部署的必要环节，需要多层次的检验：

**1. 对比参考实现**：
- 参考实现：用FP64或FP32的串行CPU实现（如NumPy、PyTorch CPU）作为ground truth。
- 计算误差指标：
  ```
  绝对误差: abs_err = |actual - reference|
  相对误差: rel_err = |actual - reference| / |reference|
  ```

**2. 容差标准（经验值）**：

| 数据类型 | atol（绝对容差） | rtol（相对容差） | 说明 |
|---------|----------------|----------------|------|
| FP32 | 1e-5 ~ 1e-6 | 1e-5 | PyTorch默认allclose标准 |
| FP16 | 1e-3 | 1e-3 | 受限于半精度表示范围 |
| BF16 | 1e-2 | 1e-2 | 尾数仅7位，精度较低 |
| INT8 | - | - | 看cosine similarity >0.99 |

判断公式：`|actual - expected| <= atol + rtol × |expected|`

**3. 统计分析（不只看单个值）**：
- 最大绝对误差（Max Abs Error）：找到误差最大的元素。
- 平均误差和标准差：了解整体误差分布。
- 百分位误差：P99/P99.9误差值。
- 误差分布直方图：发现是否有少数outlier导致大误差。
- 逐元素对比：用 `torch.testing.assert_close()` 自动报告不匹配位置。

**4. 端到端验证（最终判据）**：
- 模型层面指标：loss值、accuracy、perplexity、BLEU/ROUGE等。
- 单个算子精度达标不代表端到端无问题（误差可能在多层间累积）。
- 量化部署的最终判据：quantized model在评测benchmark上的精度下降是否在可接受范围（如<1%）。

**5. 特殊情况检查**：
- NaN/Inf检测：`torch.isnan(output).any()`。
- 大值/溢出检测：检查输出范围是否在类型表示范围内。
- Subnormal数处理：部分GPU对denormalized数flush to zero，与CPU行为不同。

---

### Q: pytest有哪些常用命令？

pytest是Python最流行的测试框架，核心命令和常用参数：

**基本运行**：
- `pytest test.py`：运行指定文件中的所有测试。
- `pytest test_dir/`：运行目录下所有test_开头的文件。
- `pytest test.py::TestClass::test_method`：运行具体某个测试方法。

**过滤和选择**：
- `pytest -k "keyword"`：运行名称包含keyword的测试（支持and/or/not逻辑）。
- `pytest -m "slow"`：运行标记为@pytest.mark.slow的测试。
- `pytest --co`（collect-only）：只列出会运行的测试，不执行。

**输出控制**：
- `pytest -v`：详细模式，显示每个测试的名称和结果。
- `pytest -s`：不捕获stdout/stderr，显示print输出（默认捕获）。
- `pytest --tb=short`：简化traceback。`--tb=no` 完全不显示。
- `pytest -q`：安静模式，最少输出。

**失败处理**：
- `pytest -x`：遇到第一个失败立即停止。
- `pytest --maxfail=N`：累计N个失败后停止。
- `pytest --lf`（last-failed）：只重跑上次失败的测试。
- `pytest --ff`（failed-first）：先跑上次失败的，再跑其他。
- `pytest --pdb`：测试失败时进入pdb调试器。

**性能和并行**：
- `pytest --durations=10`：显示最慢的10个测试。
- `pytest -n auto`（需要pytest-xdist）：多进程并行执行测试。

**覆盖率**：
- `pytest --cov=module --cov-report=html`：生成代码覆盖率报告。

**AI Infra测试中的常用组合**：
```bash
# 开发中快速验证
pytest tests/test_kernel.py -x -v -s

# CI中完整测试+覆盖率
pytest tests/ --cov=my_module --maxfail=5 -n 4

# 只跑GPU相关测试
pytest -m "gpu" -v --durations=20
```

---

### Q: 常见的非线性算子有哪些？

非线性激活函数是神经网络拟合复杂函数的关键——没有非线性，多层网络等价于单层线性变换。

| 激活函数 | 公式 | 特点 | 典型应用 |
|---------|------|------|---------|
| **ReLU** | max(0, x) | 最简单高效，计算仅1次比较 | CNN、旧模型 |
| **GeLU** | x × Φ(x) ≈ 0.5x(1+tanh(√(2/π)(x+0.044715x³))) | 平滑，保留负值的部分信息 | BERT, GPT, ViT |
| **SiLU/Swish** | x × sigmoid(x) = x / (1+e^(-x)) | 平滑、自门控 | Llama, Qwen FFN |
| **Sigmoid** | 1/(1+e^(-x))，输出(0,1) | 门控信号 | LSTM门、二分类输出 |
| **Tanh** | (e^x-e^(-x))/(e^x+e^(-x))，输出(-1,1) | 零中心 | RNN隐状态 |
| **Softmax** | e^(xi)/Σe^(xj) | 多类归一化 | 注意力权重、分类 |
| **LeakyReLU** | max(0.01x, x) | 避免dead neuron | GAN |
| **SwiGLU** | SiLU(xW_gate) × (xW_up) | 门控线性单元 | Llama/Qwen FFN |

**大模型中的主流选择**：
- FFN激活：**SwiGLU**（SiLU + Gating）成为标准，Llama/Qwen/Mistral均使用。相比GeLU参数量增加50%（多一个gate投影），但效果更好。
- 注意力归一化：**Softmax**（必须用，定义注意力权重的概率分布）。
- 归一化层：虽然RMSNorm/LayerNorm不算"激活函数"，但也是非线性操作。

**CUDA实现注意事项**：
- 这些函数都是逐元素操作（element-wise），天然memory-bound。
- 优化关键不在函数计算本身，而在于**与其他操作融合**——如GEMM+Bias+SiLU融合为一个kernel。
- Softmax需要reduce（求max和sum），不是纯element-wise，需要特殊处理。
