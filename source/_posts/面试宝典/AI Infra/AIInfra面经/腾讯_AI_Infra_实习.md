---
title: '腾讯 AI Infra 实习'
date: 2026-04-16 11:46:18
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: C++指针和引用的区别？

**核心区别总结：**

| 特性 | 指针(Pointer) | 引用(Reference) |
|------|--------------|-----------------|
| 初始化 | 可以不初始化、可以为nullptr | 必须初始化，不可为null |
| 重绑定 | 可以指向不同对象 | 初始化后不可重新绑定 |
| 语法 | 需要`*`解引用、`->`访问成员 | 直接使用（透明的别名） |
| 内存 | 占用指针大小(8字节/64位) | 逻辑上不占独立空间(编译器可能实现为指针) |
| 多级 | 有指针的指针(`int**`) | 没有引用的引用 |
| sizeof | 返回指针本身大小(8) | 返回所引对象大小 |
| 算术 | 支持指针运算(p+1) | 不支持 |
| 数组 | 指针可以遍历数组 | 引用只绑定单个对象 |

**底层实现：** 引用在汇编层面通常就是一个指针，但编译器保证它永远有效（不为null、不悬垂——程序员需确保）。

**使用建议：**
- 函数参数传递：优先用`const T&`（安全，无拷贝开销）
- 需要"可选"语义（可能不存在）：用指针或`std::optional`
- 需要重新绑定/多态容器：用指针（或智能指针）
- 返回值：返回值优先靠RVO/move，不要返回局部变量的引用

---

### Q: C++虚函数的原理？

**实现机制——vtable + vptr：**

```cpp
class Animal {
    virtual void speak();  // vtable slot 0
    virtual void move();   // vtable slot 1
};
class Dog : public Animal {
    void speak() override;  // 覆盖slot 0
    // move()继承自Animal，slot 1不变
};
```

**内存结构：**
```
Dog对象:                  Dog的vtable:
┌─────────┐              ┌──────────────┐
│ vptr ────┼─────────→   │ &Dog::speak  │ slot 0 (重写)
│ 成员...  │              │ &Animal::move│ slot 1 (继承)
└─────────┘              └──────────────┘
```

**虚函数调用过程（通过基类指针）：**
```
Animal* p = new Dog();
p->speak();
// 编译器生成: (*p->vptr[0])()  // 1.读vptr→2.查vtable→3.间接调用
// 实际调用Dog::speak()
```

**性能开销：**
- 一次间接跳转（~2-3 cycles）
- **阻止内联优化**（编译器无法在编译时确定调用目标）
- 在热路径上影响可能显著（如每帧百万次调用的游戏引擎）
- 可用`final`关键字让编译器做去虚拟化(devirtualization)优化

---

### Q: C++调用函数整个压栈过程是怎样的？

**函数调用的完整过程（x86-64为例）：**

```
调用前(Caller):
  1. 前6个整数/指针参数放入寄存器: rdi, rsi, rdx, rcx, r8, r9
  2. 浮点参数放入 xmm0-xmm7
  3. 多余参数从右到左压栈
  4. call指令: 将返回地址(RIP+指令长度)压栈，跳转到函数入口

进入函数(Callee Prologue):
  5. push rbp          ; 保存调用者的栈帧基址
  6. mov rbp, rsp      ; 设置新栈帧基址
  7. sub rsp, N        ; 为局部变量分配空间
  8. [可选] 保存callee-saved寄存器(rbx, r12-r15)

函数执行:
  ...

返回(Callee Epilogue):
  9. mov rsp, rbp      ; 释放局部变量空间
  10. pop rbp          ; 恢复调用者栈帧
  11. ret              ; 弹出返回地址到RIP，跳回调用者
```

**栈帧布局（高地址→低地址）：**
```
┌──────────────────┐ ← 高地址
│ 调用者的栈帧      │
├──────────────────┤
│ 第7+参数(如有)    │  ← 通过rbp+offset访问
│ 返回地址          │
│ 旧的rbp          │  ← rbp指向这里
│ 局部变量1         │  ← rbp-8
│ 局部变量2         │  ← rbp-16
│ ...              │  ← rsp指向栈顶
└──────────────────┘ ← 低地址
```

**现代优化：**
- x86-64调用约定：前6个参数走寄存器，大部分函数不需要压栈参数
- 编译器可能省略`push rbp`(Frame Pointer Omission)用rsp直接偏移访问
- 叶函数(不调用其他函数)可能不需要完整的prologue/epilogue
- RVO/NRVO消除返回值的拷贝

---

### Q: 如何将函数作用域里面的局部变量返回到外部？

**安全的方式：**

**1. 直接return（值语义，最推荐）：**
```cpp
string createString() {
    string s = "hello";  // 局部变量
    return s;  // 编译器做NRVO(Named Return Value Optimization)，零拷贝
}
// NRVO: s直接在调用者的内存位置构造，无拷贝无移动
```

**2. 返回智能指针（堆分配）：**
```cpp
unique_ptr<LargeObject> create() {
    auto obj = make_unique<LargeObject>();
    return obj;  // 移动语义，转移所有权
}
```

**3. 通过输出参数（调用者提供存储）：**
```cpp
void compute(const Input& in, Output& out) {
    out.result = ...;  // 写入调用者提供的对象
}
```

**4. 返回static局部变量的引用：**
```cpp
const Config& getConfig() {
    static Config cfg("settings.json");  // 生命周期为程序全程
    return cfg;  // 安全，但注意线程安全和单例语义
}
```

**绝对不能做的：**
```cpp
int* bad() {
    int x = 42;
    return &x;  // 悬垂指针！函数返回后x已销毁
}
int& alsoBad() {
    int x = 42;
    return x;   // 悬垂引用！同样的问题
}
```

**RVO/NRVO为什么能消除拷贝：**
- 编译器直接在调用者预留的内存位置构造返回对象
- C++17起**保证**copy elision（即使拷贝构造被delete也能编译）
- 效果：`string s = createString()` 中的s直接就是函数内构造的对象，零开销

---

### Q: 如何实现一个事件系统（发布-订阅模式）？

**核心设计：**

```cpp
class EventSystem {
    // 事件类型 → 回调函数列表
    unordered_map<EventType, vector<function<void(const Event&)>>> listeners;
    
public:
    // 注册：订阅特定事件
    size_t subscribe(EventType type, function<void(const Event&)> callback) {
        listeners[type].push_back(std::move(callback));
        return listeners[type].size() - 1;  // 返回ID用于取消
    }
    
    // 发布：触发事件，通知所有订阅者
    void emit(EventType type, const Event& event) {
        if (listeners.count(type)) {
            for (auto& cb : listeners[type])
                cb(event);
        }
    }
    
    // 取消订阅
    void unsubscribe(EventType type, size_t id) { ... }
};
```

**工程中的关键考虑：**

| 问题 | 解决方案 |
|------|---------|
| 订阅者被销毁后悬垂回调 | 用weak_ptr跟踪订阅者存活状态；或要求析构时unsubscribe |
| 线程安全 | 读写锁保护listeners map；或用lock-free queue缓冲事件 |
| emit时修改listeners（迭代中增删） | 延迟处理（标记删除，下次emit前清理） |
| 事件优先级 | 用priority_queue替代vector |
| 性能（大量事件） | 事件池+批量分发；避免频繁heap分配(用小对象优化) |

**与GameObject生命周期绑定：**
```cpp
class Component {
    vector<pair<EventType, size_t>> subscriptions;  // 记录所有订阅
    ~Component() {
        for (auto& [type, id] : subscriptions)
            EventSystem::instance().unsubscribe(type, id);
    }
};
```

---

### Q: UI优化中合批处理怎么做？

合批(Batching)将多个Draw Call合并为一个，减少CPU-GPU通信开销。每次Draw Call的CPU开销约几十微秒（状态切换、命令提交），合批后可将数百个Draw Call压缩为几十个。

**合批方法对比：**

| 方法 | 原理 | 适用场景 | 限制 |
|------|------|---------|------|
| 静态合批 | 相同材质的静态物体Mesh合并 | 不移动的场景物体 | 增加内存，不能动态修改 |
| 动态合批 | 运行时合并小Mesh(顶点<300) | 小型动态物体 | 顶点数限制，CPU开销 |
| GPU Instancing | 同Mesh不同Transform一次绘制 | 大量重复物体(草/树) | 必须相同Mesh |
| 图集(Atlas) | 多纹理合并为一张 | UI/2D游戏 | 纹理大小上限 |
| Indirect Drawing | GPU驱动的绘制（无CPU参与） | 大量不同物体 | 需要现代API(DX12/Vulkan) |

**在AI Infra推理服务中的类比：**
- Dynamic Batching ≈ Continuous Batching（合并多个推理请求为一个batch）
- 减少kernel launch开销 ≈ 减少Draw Call开销
- CUDA Graph ≈ Command Buffer录制（预录制一系列操作一次提交）
