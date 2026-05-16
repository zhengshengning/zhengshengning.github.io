---
title: '元戎启行 AI Infra 校招 一面 (1)'
date: 2026-04-16 12:19:32
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: 有定义新的MLIR Dialect吗？做项目时有参考torch-mlir吗？

MLIR Dialect是MLIR框架中的**命名空间**（Namespace），定义了一组语义相关的操作（Op）、类型（Type）和属性（Attribute），是实现可复用、可组合编译器的核心抽象。

**自定义Dialect的完整流程**：

1. **TableGen定义（.td文件）**：
```tablegen
def MyDialect : Dialect {
  let name = "my_accel";
  let cppNamespace = "::my_accel";
}

def MatMulOp : Op<MyDialect, "matmul", [NoSideEffect]> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}
```

2. **生成代码**：`mlir-tblgen` 自动生成Op的C++声明、builder、parser/printer。

3. **实现验证和优化**：
   - `verifyInvariants()`：检查操作数类型/形状约束。
   - `fold()`：常量折叠优化。
   - `getCanonicalizationPatterns()`：规范化pattern（如消除恒等变换）。

4. **编写Lowering Pass**：将自定义Op转换到下层Dialect（如lowering到linalg或LLVM dialect）。

**torch-mlir的参考价值**：

torch-mlir是PyTorch到MLIR的桥梁项目，提供了大量可参考的设计模式：
- **动态Shape处理**：使用`!torch.vtensor<[?,?,?],f32>`表示动态维度，通过shape refinement pass逐步确定。
- **Value Tensor语义**：torch dialect中tensor是value语义（immutable），与PyTorch的in-place操作通过copy语义桥接。
- **Op Decomposition策略**：复杂op分解为简单op的组合（如将BatchNorm分解为mean/var/normalize）。
- **Backend Contract**：定义lowering到各backend前需要满足的"契约"（如所有dynamic shape已解析、所有高级op已分解）。

---

### Q: 有考虑动态图（Dynamic Shape）的问题吗？

动态Shape是AI编译器面临的核心挑战之一——tensor维度在编译时未知，只有运行时才能确定。

**为什么动态Shape难处理**：
- 编译优化（如tile大小选择、内存预分配、循环边界）依赖shape信息。
- 不同shape可能需要完全不同的最优kernel（如GEMM的tile策略随矩阵大小变化）。
- 内存管理无法静态规划。

**处理方案**：

**1. 符号化Shape推导（Symbolic Shape Inference）**：
- 用符号变量表示未知维度（如batch=S0, seq_len=S1）。
- 建立shape间的约束关系（如输出shape = f(输入shape)）。
- 能表达shape之间的关系（如"两个tensor的batch维度相同"），利于部分优化。
- 代表：TorchDynamo的symbolic shapes、ONNX的symbolic dims。

**2. Shape Guard + JIT编译**：
- torch.compile的做法：首次运行时trace得到具体shape并编译优化kernel。
- 后续运行时检查shape是否匹配（Shape Guard）——匹配则复用，不匹配则重编译。
- 通过`torch._dynamo.config.dynamic_shapes=True`支持符号化：只要满足约束的shape都能复用同一编译结果。
- 权衡：首次编译开销 vs 后续执行加速。

**3. Padding到固定Shape（Bucket方式）**：
- 将动态维度pad到最近的预定义bucket值（如序列长度pad到[128, 256, 512, 1024, 2048]）。
- 为每个bucket预编译一个优化kernel。
- 运行时选择最接近的bucket。
- 缺点：padding浪费计算和内存；bucket划分需要经验。

**4. MLIR中的处理**：
- 使用 `?` 标注动态维度：`tensor<?x?xf32>`。
- `tensor.dim %t, %idx` 运行时获取具体维度值。
- 优化pass需要区分"静态维度可优化"和"动态维度需保守处理"。
- `shape.assuming` 操作表达"在某shape条件满足时"的优化region。

---

### Q: 转成tosa和tensor后，接着lower到哪些Dialect？

MLIR的Lowering是逐级降低抽象程度的过程，典型路径如下：

```
High-level:  torch dialect / tosa dialect / stablehlo
                     ↓ (decomposition + lowering)
Mid-level:   linalg-on-tensors (结构化计算 + value语义)
                     ↓ (bufferization: tensor → memref)
             linalg-on-memrefs (结构化计算 + 内存引用语义)
                     ↓ (lowering to loops)
Low-level:   affine / scf (仿射/结构化控制流循环)
                     ↓ (lowering)
Target:      ├── LLVM dialect → LLVM IR → 机器码 (CPU)
             ├── gpu dialect → NVVM → PTX → cubin (NVIDIA GPU)
             ├── gpu dialect → ROCDL → AMDGPU ISA (AMD GPU)
             └── 自定义 dialect → 自定义后端代码 (NPU等)
```

**各层级的职责**：
- **linalg**：表达结构化计算（如matmul、conv、generic），便于做tiling/fusion/vectorization变换。
- **affine**：表达仿射循环嵌套（loop nest），支持polyhedral优化（依赖分析、循环变换）。
- **scf**：表达一般的结构化控制流（for/while/if），比affine更通用但优化能力更弱。
- **LLVM dialect**：LLVM IR的MLIR表示形式，可直接转为LLVM IR交给LLVM后端生成目标代码。
- **gpu dialect**：表达GPU编程概念（kernel launch, thread/block ids, shared memory）。

---

### Q: One-shot Bufferization和基于Dialect的Bufferization有什么区别？

Bufferization是将tensor（value语义，immutable）转换为memref（引用语义，mutable memory）的过程，是MLIR从高级表示到可执行代码的关键步骤。

**旧方式：基于Dialect的Bufferization**

- 每个Dialect单独实现自己的bufferize pass（如linalg-bufferize, tensor-bufferize, func-bufferize）。
- 需要手动指定pass的**执行顺序**——先bufferize哪个dialect，后bufferize哪个。
- 各pass独立分析，无法感知其他dialect的op是否会修改同一buffer。
- 结果：容易产生**不必要的内存拷贝**（保守策略，不确定是否安全就插入copy）。

**新方式：One-shot Bufferization**

- **全局分析**：一次性分析整个函数/模块中所有op的内存行为（读/写/别名关系）。
- **最优分配**：通过全局alias分析确定哪些op可以in-place执行（输出复用输入buffer），最小化buffer分配和拷贝。
- **与pass顺序无关**：不依赖各dialect bufferize pass的执行顺序。
- **通过接口驱动**：每个op实现 `BufferizableOpInterface`，声明自己的内存语义（是否写内存、是否aliasing、是否可in-place）。

**对比**：

| 特性 | 基于Dialect | One-shot |
|------|------------|---------|
| 分析范围 | 局部（单dialect） | 全局（跨dialect） |
| 内存拷贝 | 保守，常有冗余copy | 最小化copy |
| Pass依赖 | 顺序敏感 | 顺序无关 |
| 实现方式 | 各dialect独立实现 | 统一接口（BufferizableOpInterface） |
| 目前状态 | 已deprecated | 推荐方式 |

---

### Q: LLVM中的isa和dyn_cast是什么？

LLVM自定义了一套高效的RTTI（Run-Time Type Information）系统，替代C++标准RTTI（dynamic_cast），因为标准RTTI性能开销大且需要虚函数支持。

**三个核心操作**：

```cpp
Value* val = getOperand();

// 1. isa<T>: 类型检查，返回bool
if (isa<BinaryOperator>(val)) {
    // val是BinaryOperator类型
}

// 2. dyn_cast<T>: 检查+转换，失败返回nullptr
if (auto* binOp = dyn_cast<BinaryOperator>(val)) {
    // 安全使用binOp
    binOp->getOpcode();
}

// 3. cast<T>: 直接转换，类型不匹配则assert失败（debug模式）
auto* binOp = cast<BinaryOperator>(val);  // 确定是BinaryOperator时使用
```

**实现原理（不依赖虚函数表）**：

每个类实现一个静态方法 `classof`：
```cpp
class BinaryOperator : public Instruction {
public:
    static bool classof(const Instruction* I) {
        return I->getOpcode() >= firstBinaryOp && I->getOpcode() <= lastBinaryOp;
    }
};
```

- 通过**kind枚举值**（存储在基类中的一个整数字段）判断类型。
- `isa<T>(val)` 最终调用 `T::classof(val)`，只是一次整数比较。
- **性能远优于C++ dynamic_cast**：无需遍历虚函数表继承链，O(1)时间。

**额外变体**：
- `dyn_cast_or_null<T>(val)`：val可能是nullptr时安全使用。
- `cast_or_null<T>(val)`：nullptr时返回nullptr而非assert。
- `isa<T1, T2, T3>(val)`：检查是否是多个类型之一。

---

### Q: 静态图和动态图的区别是什么？

| 特性 | 静态图（Define-and-Run） | 动态图（Define-by-Run） |
|------|------------------------|------------------------|
| 代表 | TF 1.x, TensorRT, MLIR编译 | PyTorch eager, TF 2.x eager |
| 构建方式 | 先完整定义图结构，再运行 | 边执行边构建图 |
| 控制流 | 图中的特殊节点（tf.cond/tf.while） | Python原生if/for/while |
| 调试 | 困难（图已编译，无法逐步调试） | 容易（标准Python调试器） |
| 优化空间 | 大（全局可见，算子融合/内存规划/并行调度） | 小（逐op执行，无全局信息） |
| Kernel Launch | 批量调度（图调度器） | 逐个launch（每个op一次） |
| 动态行为 | 不支持数据依赖的控制流（或需特殊op） | 完全支持Python动态逻辑 |
| 内存管理 | 可全局最优规划（生命周期分析） | 按需分配+GC/引用计数 |

**混合方案（当前主流）**：
- **torch.compile**：用TorchDynamo在Python字节码层面捕获计算图，遇到不可trace的代码（动态控制流）时break graph，分段编译。
- **torch.jit.trace**：记录一次执行的op序列作为静态图。不支持数据依赖控制流。
- **torch.jit.script**：解析Python代码的AST生成图。支持有限的控制流。
- **趋势**：用户侧保持动态图编程体验（易开发），编译器侧自动提取静态子图做优化（高性能）。

---

### Q: MLIR中怎么处理in-place操作？

MLIR通过**Tensor语义 + Bufferization时的alias analysis**来处理in-place操作，这是一个从"纯函数式"到"命令式内存操作"的优雅过渡。

**为什么不在tensor层直接做in-place**：
- Tensor是value语义（类似SSA中的不可变值），不存在mutation的概念。
- `%y = linalg.generic ins(%x) outs(%init)` 产生新tensor `%y`，不修改 `%x`。
- 这种纯函数式语义便于分析和变换（无需考虑alias/side-effect）。

**Bufferization时的in-place分析**：

1. **分析阶段**：对每个op的输入输出，判断是否可以in-place执行（即输出buffer复用输入buffer）。

2. **判据（安全条件）**：
   - 输入tensor在该op之后**不再被任何其他op使用**（last use analysis）。
   - 或者该op只是读取输入不写入（bufferizesToAliasOnly）。
   - 没有其他op依赖该输入tensor的值。

3. **通过BufferizableOpInterface声明语义**：
```cpp
// 每个op实现接口方法
bool bufferizesToMemoryRead(OpOperand& opOperand);  // 该操作数是否被读取
bool bufferizesToMemoryWrite(OpOperand& opOperand); // 该操作数对应的结果是否写入
AliasingOpOperandList getAliasingOpOperands(OpResult result); // 结果alias哪些操作数
```

4. **决策**：
   - 如果in-place安全：输出buffer直接指向输入buffer，无copy。
   - 如果不安全（input后续还有其他use）：必须分配新buffer并copy输入数据。

**示例**：
```mlir
%0 = tensor.insert %val into %t[%i] : tensor<10xf32>
// 如果%t之后不再被使用 → in-place：直接在%t的buffer上写入
// 如果%t之后还有其他use → out-of-place：分配新buffer，copy %t，再写入
```

---

### Q: 手撕：给定一个计算图，计算运行该计算图所需的最小内存？

（编程题）

---

### Q: 手撕：拓扑排序？

（编程题）
