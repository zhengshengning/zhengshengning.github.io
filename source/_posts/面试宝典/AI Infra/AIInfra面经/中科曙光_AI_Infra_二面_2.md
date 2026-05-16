---
title: '中科曙光 AI Infra 二面 (2)'
date: 2026-04-16 11:46:02
categories:
  - [求职面试, 芯片/研究院面经]
tags: [AIInfra, 训练优化, 算子优化, 面经]
---


<!-- more -->

---

### Q: PyTorch如何分析性能？

PyTorch性能分析需要结合系统级和kernel级工具，形成从全局定位到微观优化的完整链路：

**1. torch.profiler（官方推荐，PyTorch 1.8+）**：

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,        # 记录tensor shape
    profile_memory=True,       # 记录显存分配
    with_stack=True            # 记录Python调用栈
) as prof:
    for step, data in enumerate(loader):
        train_step(data)
        prof.step()
```

输出分析：
- **Chrome Trace**：可视化kernel执行时间线，看overlap情况。
- **TensorBoard Plugin**：按op类型统计耗时，显示GPU利用率曲线。
- **表格汇总**：`prof.key_averages().table(sort_by="cuda_time_total")`，按总GPU时间排序各op。

关键指标：GPU Utilization（应>80%）、CUDA Kernel执行占比（vs CPU开销比）、top-N耗时op。

**2. CUDA Events（精确计时）**：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(input)
end.record()
torch.cuda.synchronize()
print(f"Elapsed: {start.elapsed_time(end):.2f} ms")
```

注意：必须`synchronize()`因为CUDA操作异步执行。Events在GPU端记录时间戳，精度约0.5us，比CPU端`time.time()`准确得多（后者受host-device同步影响）。

**3. Nsight Systems（系统级全景）**：

```bash
nsys profile -t cuda,nvtx --stats=true python train.py
```

重点看：
- **GPU Idle gap**：两个kernel之间的间隙是否过大（CPU端Python开销）。
- **数据传输overlap**：H2D/D2H传输是否与计算重叠。
- **多Stream并行**：是否有效利用多个CUDA Stream并行计算。
- **CPU-GPU同步点**：不必要的`synchronize()`导致GPU等待。

典型发现：发现数据预处理是CPU瓶颈，或kernel launch开销占比过高（小kernel太多）。

**4. Nsight Compute（kernel级深度分析）**：

```bash
ncu --set full -o report python inference.py
```

提供每个kernel的详细指标：
- **Occupancy**：实际活跃Warp / 理论最大Warp，目标>50%。
- **Memory Throughput**：实际带宽 vs HBM峰值（A100: 2TB/s）。
- **Compute Throughput**：实际FLOPS vs 峰值。
- **Roofline位置**：直观判断compute-bound还是memory-bound。
- **Stall原因**：Wait Barrier / Long Scoreboard / Memory Throttle等。
- **Warp状态分布**：活跃/等待/空闲的warp比例。

**5. 内存分析**：

```python
# 显存快照
print(torch.cuda.memory_summary())
# 显存分配追踪
torch.cuda.memory._record_memory_history()
# ... 运行代码 ...
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")
# 用PyTorch Memory Visualizer查看
```

关键指标：峰值显存占用、显存碎片率（allocated vs reserved差异）、频繁的alloc/free模式。

**6. 性能分析最佳实践**：
- 先用Nsight Systems看全局，找到瓶颈在哪个阶段（数据加载/前向/反向/通信）。
- 对瓶颈kernel用Nsight Compute深入分析。
- 用torch.profiler的表格快速定位top耗时op。
- 注意warmup：前几步可能触发CUDA context初始化和cuDNN autotuning。

---

### Q: DDP如何优化多机多卡训练？

DDP（DistributedDataParallel）是PyTorch的高效数据并行实现，其优化策略围绕"减少通信开销+最大化计算-通信重叠"展开：

**1. 梯度桶化（Gradient Bucketing）**：

原理：将多个小梯度tensor打包成固定大小的"桶"（bucket），整桶作为一次AllReduce操作执行。

```python
# 桶大小配置（默认25MB）
model = DDP(model, bucket_cap_mb=25)
```

为什么有效：
- AllReduce有固定启动开销（~10us for NCCL），小tensor逐一通信会被启动开销支配。
- 25MB桶在A100 NVLink（600GB/s双向）上传输约42us，启动开销占比合理。
- 桶太大延迟通信启动（等所有梯度计算完），太小则启动开销占比高。

**2. 计算-通信重叠（Overlap）**：

核心机制：反向传播从最后一层向前逐层计算梯度。当一个桶的所有梯度计算完成后，该桶的AllReduce**立即异步开始**，同时前面层继续计算梯度。

```
时间线：
Layer N 反向计算 → [Bucket 1 AllReduce开始]
Layer N-1 反向计算 →          [Bucket 1 AllReduce进行中]
Layer N-2 反向计算 →                    [Bucket 2 AllReduce开始]
...
```

DDP按反向计算顺序（最后层优先）将参数分配到桶中（Bucket 0包含最后层参数，最先完成），确保通信尽早开始。

**3. 梯度压缩**：

```python
# FP16梯度通信
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)

# PowerSGD低秩压缩
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
model.register_comm_hook(
    state=powerSGD_hook.PowerSGDState(process_group=None, matrix_approximation_rank=1),
    hook=powerSGD_hook.powerSGD_hook
)
```

FP16通信将带宽需求减半；PowerSGD用低秩近似压缩梯度（压缩比可达10x），代价是少量精度损失。

**4. 梯度累积跳过同步**：

```python
for i, data in enumerate(loader):
    with model.no_sync() if (i+1) % accum_steps != 0 else nullcontext():
        loss = model(data).loss / accum_steps
        loss.backward()
    if (i+1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

`no_sync()`跳过中间步的AllReduce，只在梯度累积的最后一步通信。将通信频率降为1/accum_steps。

**5. NCCL通信优化**：

- **算法选择**：NCCL根据拓扑自动选择Ring/Tree/Double Binary Tree算法。节点内NVLink用Ring（高带宽），跨节点IB用Tree（低延迟）。
- **环境变量调优**：
  ```bash
  NCCL_ALGO=Ring          # 指定AllReduce算法
  NCCL_MIN_NCHANNELS=4    # 最小通信channel数
  NCCL_BUFFSIZE=8388608   # 8MB NCCL buffer
  ```
- **NVLink vs PCIe**：NVLink带宽600GB/s（A100 8卡），PCIe Gen4仅64GB/s。DDP在NVLink节点内通信开销可忽略。

**6. 静态图优化（PyTorch 2.0+）**：

```python
model = DDP(model, static_graph=True)
```

当模型结构固定（每次前向使用相同的参数子集）时，DDP可以：
- 预计算桶分配，避免每次迭代重新分桶。
- 更激进的通信调度优化。
- 对unused parameters不做额外检查。

---

### Q: 分布式训练中的batch如何设置？

Batch size设置直接影响训练收敛速度、最终精度和硬件利用率，需要综合考虑：

**核心关系**：

```
Global Batch Size = per-GPU Batch Size × GPU数量 × 梯度累积步数
```

**学习率缩放规则**：

| 方法 | 公式 | 适用范围 | 说明 |
|------|------|---------|------|
| 线性缩放 | lr × (global_batch / base_batch) | batch较小时 | batch翻倍lr翻倍 |
| 平方根缩放 | lr × sqrt(global_batch / base_batch) | 大batch | 更保守，Adam常用 |
| LAMB/LARS | 每层自适应 | 超大batch(>32K) | 自动处理不同层的lr |

**Warmup的必要性**：
- 大batch训练初始梯度估计方差大（少量样本的平均），大lr可能导致训练发散。
- Warmup阶段（通常前1000-5000步）从极小lr线性增长到目标lr，让模型参数进入稳定区域。
- 经验公式：warmup步数 ∝ 1/batch_size（batch越大warmup越短，因为每步的有效样本更多）。

**大batch的泛化性问题**：

| Global Batch | 现象 | 原因 | 缓解方法 |
|-------------|------|------|---------|
| 过大(>64K) | 测试loss高于小batch | 收敛到sharp minima | LAMB/LARS优化器 |
| 中等(8K-32K) | 通常可接受 | - | 线性缩放+warmup |
| 较小(<2K) | 训练慢但泛化好 | 梯度噪声有正则化效果 | - |

**实践建议**：

1. **保持Global Batch Size固定**：增加GPU数时，相应减小per-GPU batch或减少梯度累积步数。这样超参不需要重新调整。

2. **Per-GPU batch size的下限**：太小会导致GPU利用率低（compute intensity不够，kernel launch开销占比过高）。典型下限：CNN用32，LLM用1-4（受显存限制）。

3. **大模型训练的典型设置**：
   - GPT-3 175B：global batch = 3.2M tokens（约1500条序列×2K长度），梯度累积约32步。
   - LLaMA-2 70B：global batch = 4M tokens。
   - 随训练进行逐步增大batch（batch size warmup）。

4. **数据并行度与batch的关系**：
   ```
   有效数据并行度 = GPU数 × gradient_accumulation_steps
   通信频率 = 1 / gradient_accumulation_steps
   ```
   梯度累积增加可减少通信频率但增加了等效训练步延迟。

---

### Q: PyTorch图优化和PyTorch 2.0特性？

**PyTorch 2.0核心架构**：

```
Python代码
    ↓ TorchDynamo (图捕获)
FX Graph (中间表示)
    ↓ AOTAutograd (自动微分)
前向+反向 FX Graph
    ↓ TorchInductor (后端代码生成)
Triton Kernel(GPU) / C++ (CPU)
```

**1. TorchDynamo（图捕获层）**：

核心创新：在Python字节码级别拦截执行，识别可以编译的代码段（"graph breaks"分割不可编译部分）。

```python
@torch.compile(mode="reduce-overhead")  # 最大化kernel fusion
def train_step(model, x, y):
    pred = model(x)
    loss = F.cross_entropy(pred, y)
    loss.backward()
    return loss
```

编译模式选择：
- `default`：平衡编译时间和运行加速。
- `reduce-overhead`：最大化fusion，编译慢但运行时消除更多开销。
- `max-autotune`：尝试所有可能的kernel实现并选最快的。

Graph Break处理：遇到无法trace的Python操作（如数据依赖的控制流、print、不支持的API）时产生graph break，break前后各为独立子图分别编译。减少graph break是获得最大加速的关键。

**2. TorchInductor（后端代码生成）**：

将FX Graph编译为高效的平台特定代码：
- **GPU**：生成Triton kernel（自动融合逐元素操作、自动tiling）。
- **CPU**：生成C++/OpenMP代码（利用AVX-512向量化）。
- **自动融合**：连续的pointwise操作（add/mul/relu/dropout）自动融合为单个kernel。
- **内存规划**：分析tensor生命周期，最小化中间buffer分配。

性能数据（torchbench平均）：
- 推理加速：1.3-2.5x（vs eager mode）。
- 训练加速：1.1-1.5x。
- 编译时间：首次约30-120秒（缓存后跳过）。

**3. 动态Shape支持**：

```python
# 标记动态维度
torch._dynamo.mark_dynamic(x, 0)  # batch维度动态
model_compiled = torch.compile(model, dynamic=True)
```

实现机制：
- **Shape Guard**：首次trace记录shape，后续input shape不同时检查是否需要重新编译。
- **Symbolic Shape**：用符号变量表示动态维度（如`s0`），生成的代码使用运行时shape值。
- **Bucket编译**：对常见shape区间各缓存一份编译结果，减少重编译次数。

**4. Scaled Dot Product Attention（SDPA）**：

```python
# 自动选择最优attention实现
F.scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=True)
```

内部自动dispatch到：FlashAttention（CUDA）→ Memory-Efficient Attention（xformers）→ Math实现（fallback）。无需手动安装FlashAttention库。

**5. 图优化Pass**：
- **算子融合**：Conv+BN+ReLU、Linear+Activation、多个pointwise合并。
- **常量折叠**：编译时预计算常量子图。
- **Layout优化**：自动选择NHWC或channels-last格式。
- **内存格式转换**：减少不必要的format转换。

---

### Q: pytest有哪些常用参数？

pytest是Python生态中最流行的测试框架，在AI Infra项目中广泛用于算子正确性验证、性能回归测试：

**运行控制**：

| 参数 | 作用 | 使用场景 |
|------|------|---------|
| `-v` | 详细模式，显示每个测试名称和PASS/FAIL | 开发调试 |
| `-s` | 不捕获stdout/stderr，显示print输出 | 调试中间值 |
| `-x` | 第一个失败即停止 | 快速定位问题 |
| `--maxfail=N` | 累计N个失败后停止 | CI中限制失败输出 |
| `-q` | 安静模式，最少输出 | CI日志精简 |

**测试选择**：

| 参数 | 作用 | 示例 |
|------|------|------|
| `-k "expr"` | 按名称表达式过滤 | `-k "test_gemm and not fp16"` |
| `-m "marker"` | 按标记运行 | `-m "gpu"` 只跑GPU测试 |
| `--co` | 只列出匹配的测试不执行 | 验证过滤是否正确 |
| `FILE::CLASS::METHOD` | 精确指定 | `test_op.py::TestConv::test_3x3` |

**调试支持**：

| 参数 | 作用 | 使用场景 |
|------|------|---------|
| `--pdb` | 失败时进入pdb调试器 | 检查失败时的变量状态 |
| `--tb=short/long/no` | traceback详细程度 | short适合大量失败时的概览 |
| `--lf` | 只重跑上次失败的测试 | 修复后快速验证 |
| `--ff` | 上次失败的优先执行 | 快速确认修复是否有效 |

**并行和性能**：

| 参数 | 作用 | 使用场景 |
|------|------|---------|
| `-n auto`（需pytest-xdist） | 多进程并行执行 | 大量独立测试加速 |
| `--durations=N` | 显示最慢的N个测试 | 优化慢测试 |
| `-p no:warnings` | 禁用warning显示 | 减少输出噪音 |
| `--timeout=N`（需pytest-timeout） | 单测超时限制 | 防止死循环测试 |

**AI Infra项目中的典型组合**：

```bash
# 开发中快速验证单个算子
pytest tests/test_matmul.py -x -v -s -k "test_fp16"

# CI中完整GPU测试+覆盖率
pytest tests/ -m "gpu" --maxfail=10 -n 4 --cov=my_ops --durations=20

# 修复bug后只验证之前失败的
pytest --lf -v

# 性能回归检查
pytest tests/benchmark/ --benchmark-compare --benchmark-min-rounds=5
```

**常用pytest fixture模式（AI算子测试）**：

```python
@pytest.fixture(params=[(128,128), (1024,1024), (4096,4096)])
def matrix_size(request):
    return request.param

@pytest.fixture(params=[torch.float16, torch.float32, torch.bfloat16])
def dtype(request):
    return request.param

def test_gemm_accuracy(matrix_size, dtype):
    M, N = matrix_size
    # ... 测试逻辑
```

---

### Q: PyTorch如何根据YAML注册算子？

PyTorch通过`native_functions.yaml`实现算子的声明式定义和自动代码生成，这是PyTorch算子系统的核心基础设施：

**YAML文件结构（native_functions.yaml）**：

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck
  structured_delegate: add.out
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: add
    SparseCPU: add_sparse
    SparseCUDA: add_sparse_cuda
```

各字段含义：
- `func`：函数签名（名称、参数类型、默认值、返回类型）。
- `dispatch`：不同DispatchKey对应的实现函数名。
- `structured_delegate`：指向out-variant（统一内存管理）。
- `variants`：生成function形式（`torch.add`）和method形式（`tensor.add`）。

**代码生成流程**：

```
native_functions.yaml
       ↓ torchgen (Python代码生成工具)
┌──────────────┬────────────────────┬──────────────────┐
│ RegisterDispatch │ Python Bindings    │ 类型推导代码      │
│ (.cpp注册代码)   │ (torch/_C/...)     │ (meta functions) │
└──────────────┴────────────────────┴──────────────────┘
```

torchgen解析YAML后生成：
1. **C++注册代码**：将各平台实现注册到Dispatcher表中。
2. **Python绑定**：`torch.xxx` Python函数到C++实现的桥接。
3. **类型/Shape推导**：meta function用于在不实际计算的情况下推导输出shape和dtype。
4. **autograd公式**：`derivatives.yaml`定义反向传播规则。

**Dispatch机制（核心路由逻辑）**：

```
torch.add(x, y)
    ↓ Python binding
dispatcher.call(add, x, y)
    ↓ 按优先级匹配DispatchKey
┌─────────────────────────────────────┐
│ Autograd → FuncTorchVmapMode →      │
│ CompositeImplicit → CUDA/CPU/MPS    │
└─────────────────────────────────────┘
```

DispatchKey优先级（高到低）：
- **Autograd**：记录计算图用于反向传播。
- **FuncTorch**：vmap/grad等函数变换。
- **Composite**：设备无关的默认实现。
- **CUDA/CPU/MPS**：具体硬件实现。

**自定义算子注册方式**：

```python
# 方式1：torch.library（推荐，PyTorch 2.0+）
@torch.library.custom_op("mylib::custom_relu", mutates_args=())
def custom_relu(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=0)

@custom_relu.register_fake  # 用于torch.compile
def custom_relu_fake(x):
    return torch.empty_like(x)

# 方式2：C++ TORCH_LIBRARY宏
# TORCH_LIBRARY(mylib, m) {
#   m.def("custom_relu(Tensor x) -> Tensor", custom_relu_impl);
# }
```

**为什么使用YAML声明式定义**：
- 单一数据源（SSOT）：一处定义，自动生成所有平台的注册代码和绑定。
- 可维护性：新增算子只需添加YAML条目+实现函数，无需手动写大量胶水代码。
- 可扩展性：新增平台（如XPU/NPU）只需在dispatch字段添加新key。
- 一致性：自动保证Python API和C++ API的签名一致。

---

### Q: 系统级算子多平台测试的方法？

当算子需要在CPU、NVIDIA GPU、华为NPU等多个平台上正确运行时，测试策略需要系统化设计：

**1. 参考实现对比（Golden Reference）**：

```python
def test_matmul_cross_platform(device, dtype):
    # Golden: FP64 CPU实现（精度最高）
    A_ref = torch.randn(M, K, dtype=torch.float64)
    B_ref = torch.randn(K, N, dtype=torch.float64)
    C_ref = A_ref @ B_ref
    
    # 被测：特定平台实现
    A = A_ref.to(device=device, dtype=dtype)
    B = B_ref.to(device=device, dtype=dtype)
    C = custom_matmul(A, B)
    
    # 对比（容差根据dtype调整）
    C_ref_cast = C_ref.to(dtype=dtype)
    torch.testing.assert_close(C, C_ref_cast.to(device), 
                                atol=TOLERANCE[dtype]['atol'],
                                rtol=TOLERANCE[dtype]['rtol'])
```

为什么用FP64作为参考：FP64尾数52位，远超FP32(23位)/FP16(10位)的精度，其舍入误差相对于低精度可忽略不计。

**2. 数值容差分级标准**：

| 数据类型 | atol | rtol | 额外指标 | 说明 |
|---------|------|------|---------|------|
| FP64 | 1e-10 | 1e-10 | - | 几乎精确 |
| FP32 | 1e-5 | 1e-5 | - | PyTorch默认标准 |
| FP16 | 1e-3 | 1e-3 | - | 受限于半精度精度 |
| BF16 | 1e-2 | 1e-2 | - | 尾数仅7位 |
| INT8 | - | - | cosine_sim > 0.99 | 量化误差用分布度量 |
| FP8 | 5e-2 | 5e-2 | cosine_sim > 0.995 | E4M3/E5M2 |

容差公式：`|actual - expected| <= atol + rtol * |expected|`

**3. 交叉验证策略**：

```
             CPU (FP64 ref)
            /      |       \
     GPU(CUDA)  NPU(Ascend)  CPU(FP32)
        |          |            |
   互相对比结果一致性
```

- 平台两两对比：如果三个平台两两结果一致（在容差内），可信度高。
- 如果某平台与其他两个都不一致，大概率是该平台实现有bug。

**4. 边界和异常测试**：

```python
@pytest.mark.parametrize("edge_case", [
    ("empty", torch.empty(0, 64)),           # 空tensor
    ("single", torch.randn(1, 1)),           # 最小shape
    ("large", torch.randn(8192, 8192)),      # 大shape（可能触发不同kernel路径）
    ("inf", torch.tensor([float('inf')])),    # 无穷值
    ("nan", torch.tensor([float('nan')])),    # NaN传播
    ("subnormal", torch.tensor([1e-38])),     # 非规格化数
    ("max_val", torch.tensor([torch.finfo(torch.float16).max])),  # 类型上界
])
def test_edge_cases(edge_case, device):
    name, tensor = edge_case
    result = my_op(tensor.to(device))
    # 验证不崩溃、NaN传播正确、Inf处理正确
```

**5. CI多平台集成**：

```yaml
# CI Pipeline示例
stages:
  - test_cpu:
      runner: x86_cpu
      script: pytest tests/ -m "not gpu" --tb=short
  - test_gpu:
      runner: nvidia_a100
      script: pytest tests/ -m "gpu" --tb=short
  - test_npu:
      runner: ascend_910b
      script: pytest tests/ -m "npu" --tb=short
  - compare_results:
      script: python compare_cross_platform_results.py
```

**6. 性能回归测试**：

```python
@pytest.mark.benchmark
def test_matmul_perf(benchmark):
    A = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    B = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
    result = benchmark(torch.matmul, A, B)
    # 与baseline对比
    assert benchmark.stats['mean'] < BASELINE_MS * 1.1  # 不超过10%回归
```

记录基准耗时，新版本超过阈值（如10%退化）则CI报警。注意GPU基准测试需要warmup（排除首次kernel编译和context初始化时间）。

**7. 确定性验证**：

多次运行同一输入检查结果是否bit-exact（或在容差内）。非确定性可能来源：
- 并行Reduce的浮点非结合律。
- atomicAdd的非确定性累加顺序。
- cuDNN非确定性算法（需设置`torch.backends.cudnn.deterministic=True`）。
