---
title: '蔚来 AI Infra 实习 HR面'
date: 2026-04-16 12:18:56
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: C++11/14/17/20各版本有哪些重要新特性？

**C++11（革命性更新，现代C++起点）：**
- **auto类型推导**：编译器自动推断变量类型，减少冗余声明
- **lambda表达式**：`[captures](params){body}` 定义匿名函数对象
- **右值引用 && 移动语义**：避免不必要的深拷贝，性能关键特性
- **智能指针**：unique_ptr/shared_ptr/weak_ptr，RAII内存管理
- **std::thread**：标准线程库（之前依赖pthread/Win32）
- **range-based for**：`for (auto& x : vec)` 简洁遍历
- **constexpr**：编译时常量表达式求值
- **nullptr**：替代NULL（类型安全）
- **variadic templates**：可变参数模板（enable perfect forwarding）

**C++14（小幅改进）：**
- **泛型lambda**：`[](auto x, auto y) { return x + y; }`
- **返回类型推导**：函数返回类型auto
- **std::make_unique**：安全构造unique_ptr
- **constexpr放宽**：允许循环和局部变量

**C++17（实用特性）：**
- **结构化绑定**：`auto [x, y, z] = tuple;` 解包
- **if constexpr**：编译时条件分支（模板中极其有用）
- **std::optional/variant/any**：更安全的值语义类型
- **折叠表达式**：简化variadic template展开
- **并行STL算法**：`std::sort(std::execution::par, ...)`
- **filesystem库**：标准文件系统操作

**C++20（又一次重大更新）：**
- **Concepts**：约束模板参数（编译错误可读性大幅提升）
- **Ranges**：惰性求值的范围操作（管道式`|`语法）
- **Coroutines**：协程支持（co_await/co_yield/co_return）
- **Modules**：替代头文件的模块系统（加速编译）
- **std::format**：类Python的格式化字符串
- **三路比较<=>**：一个operator实现所有比较运算

**AI Infra开发中常用的特性：**
- 移动语义：大tensor/buffer的所有权转移
- constexpr：编译时计算kernel配置参数
- CRTP(C++11)：静态多态替代虚函数（zero-overhead抽象）
- std::atomic：无锁并发数据结构
- 结构化绑定：解包CUDA kernel的配置参数

### Q: CUDA、TensorRT、TVM各自的定位和区别？

**三者在深度学习部署生态中的层次：**

```
高层(易用)  ←──────────────────────────────→  底层(灵活)
  TensorRT          TVM               CUDA
  (黑盒优化)      (编译器框架)      (手动编程)
```

**详细对比：**

| 维度 | CUDA | TensorRT | TVM |
|------|------|----------|-----|
| 定位 | GPU通用计算平台 | NVIDIA推理优化引擎 | 开源DL编译器框架 |
| 抽象级别 | 线程级编程 | 模型级自动优化 | 张量表达+自动调度 |
| 开发效率 | 低（手写kernel） | 高（自动优化） | 中（需要搜索时间） |
| 性能上限 | 理论100% | ~95%(NVIDIA GPU) | 70-90%(取决于硬件) |
| 硬件支持 | 仅NVIDIA GPU | 仅NVIDIA GPU | 多后端(GPU/CPU/NPU) |
| 灵活性 | 最高（任意操作） | 低（预定义优化） | 中（可自定义调度） |
| 动态shape | 完全支持 | 有限(需profile多shape) | 有限(静态编译) |
| 开源 | SDK开源,驱动闭源 | 闭源核心,开源接口 | 完全开源 |

**典型使用场景：**
- **CUDA**：开发FlashAttention这类需要极致优化的核心算子
- **TensorRT**：部署成熟模型到NVIDIA GPU（如自动驾驶的实时推理）
- **TVM**：需要跨硬件平台部署（同一模型部署到GPU+手机NPU+边缘设备）

**实际工程中的组合使用：**
```
模型训练(PyTorch) → ONNX导出 → TensorRT部署(NVIDIA GPU)
                              → TVM编译(非NVIDIA硬件)
                              → 核心算子用CUDA手写(极致性能需求)
```
