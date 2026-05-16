---
title: '蔚来 AI Infra 实习'
date: 2026-04-16 12:09:23
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 算子优化, 面经]
---


<!-- more -->

---

### Q: Shared Memory Bank Conflict是什么？如何解决？

**Bank的物理结构和映射规则：**

```
NVIDIA GPU共享内存: 32个Bank, 每个Bank宽4字节(32-bit)
地址→Bank映射: bank_id = (byte_address / 4) % 32

物理排列(每Bank每cycle可服务一个请求):
Addr:  0    4    8   ...  124  128  132  ...
Bank:  0    1    2   ...   31    0    1   ...
       ← 128字节一个cycle(32 banks × 4B) →
```

**Bank Conflict发生条件：**

同一warp(32线程)中的多个线程在**同一时钟周期**访问**同一bank的不同地址**时，请求被串行化。

```
无冲突: 32线程访问32个不同bank → 1 cycle完成
2-way冲突: 2个线程访问同一bank不同地址 → 2 cycles
N-way冲突: N个线程冲突 → N cycles (性能下降N倍)
Broadcast: 多线程访问同一bank同一地址 → 1 cycle (硬件广播,不是冲突!)
```

**经典冲突场景：**

```cuda
__shared__ float s[32][32];  // 32×32矩阵

// 场景1: 列访问(32-way冲突!)
float val = s[threadIdx.x][0];
// s[0][0]=bank 0, s[1][0]=(1*32+0)%32=0, s[2][0]=(2*32+0)%32=0...
// 所有线程都访问bank 0的不同地址 → 32倍延迟!

// 场景2: stride=2访问(16-way冲突)
float val = s[0][threadIdx.x * 2];
// thread 0→bank 0, thread 1→bank 2, ..., thread 16→bank 0
// 每个偶数bank被2个线程访问

// 场景3: 正常行访问(无冲突)
float val = s[0][threadIdx.x];
// thread 0→bank 0, thread 1→bank 1, ... → 每个bank一个线程
```

**解决方案：**

| 方案 | 实现 | 适用场景 | 代价 |
|------|------|---------|------|
| Padding | `s[32][32+1]` | 列访问/转置 | 浪费少量shared mem |
| Swizzle | 地址异或变换 | GEMM tile加载 | 索引计算稍复杂 |
| 调整访问模式 | 改变线程-数据映射 | 特定算法 | 可能改变算法逻辑 |
| 128-bit加载 | 使用float4 | 每个线程读4个连续float | 需要数据对齐 |

**Padding详解：**

```cuda
// 列访问32×32矩阵:
__shared__ float s[32][32];     // 同列=同bank → 32-way conflict
__shared__ float s[32][32+1];   // 同列不同bank → 0 conflict!

// 原理:
// s[i][0]地址 = base + i*33*4 → bank = (i*33)%32
// i=0: bank 0, i=1: bank 1, i=2: bank 2, ... 全部不同!
```

### Q: 同一Warp内不同线程的访问约束？

**Warp执行模型的关键约束：**

```
Warp = 32个线程 = GPU执行的最小调度单位
所有32线程同一时刻执行同一条指令(SIMT)

┌─────── Warp (32 threads) ───────┐
│ T0  T1  T2  ...  T31            │
│ 执行同一条指令, 但操作不同数据   │
└─────────────────────────────────┘
```

**全局内存访问约束：**

| 访问模式 | 事务数 | 带宽利用率 | 说明 |
|---------|--------|-----------|------|
| 连续(coalesced) | 1次(128B) | 100% | thread i访问addr+i*4 |
| stride=2 | 2次(128B) | 50% | 跳着访问，浪费一半cache line |
| 随机/分散 | 最多32次 | ~3% | 每线程不同cache line |
| 广播(同一地址) | 1次 | 取决于后续用 | L1 cache命中后广播 |

```cuda
// 最佳: 连续访问(1次事务)
float val = data[blockIdx.x * blockDim.x + threadIdx.x];

// 最差: 随机访问(可能32次事务)
float val = data[random_index[threadIdx.x]];

// AoS vs SoA影响:
struct Particle { float x, y, z, w; };  // AoS
Particle particles[N];
// 访问所有x: particles[tid].x → stride=16字节, 效率25%

float x[N], y[N], z[N], w[N];  // SoA  
// 访问所有x: x[tid] → 连续, 效率100%
```

**共享内存访问约束：**
- 同bank不同地址 → Bank Conflict (串行化)
- 同bank同地址 → Broadcast (无冲突)
- 不同bank → 并行 (理想)

**分支发散约束：**

```cuda
if (threadIdx.x < 16) {
    path_A();  // 前16个线程执行
} else {
    path_B();  // 后16个线程执行
}
// 不是并行! 硬件先执行path_A(后16线程idle), 再执行path_B(前16线程idle)
// 总时间 = time(path_A) + time(path_B), 而非max(A,B)

// Volta+架构(Independent Thread Scheduling):
// 允许更灵活的调度,但分支仍有性能损失
```

### Q: GPU共享内存的广播机制（Broadcast）？

**Broadcast的触发条件和硬件行为：**

```
当warp中多个线程访问共享内存中:
  同一Bank → 同一地址 → 触发Broadcast

硬件行为:
  只执行一次读取 → 结果复制到所有请求该地址的线程
  延迟 = 1次正常共享内存读取 (约5ns)
  不会被计为Bank Conflict!
```

**Broadcast vs Bank Conflict的区别：**

```cuda
__shared__ float s[256];

// Broadcast: 所有线程读同一地址 → 1 cycle
float val = s[0];  // 32个线程全读s[0] → broadcast, 快!

// Bank Conflict: 不同线程读同bank不同地址 → N cycles
float val = s[threadIdx.x * 32];  // thread 0读s[0], thread 1读s[32]
// s[0]和s[32]都在bank 0但地址不同 → 32-way conflict, 慢!
```

**利用Broadcast的实际场景：**

```cuda
// 场景1: 矩阵乘中B矩阵的行广播
// C[i][j] = sum(A[i][k] * B[k][j])
// 同一列的多个线程(不同i)都需要B[k][j]
__shared__ float Bs[TILE][TILE];
// 如果多个线程读Bs的同一元素 → broadcast, 无性能损失

// 场景2: 参数共享
__shared__ float scale;
if (threadIdx.x == 0) scale = compute_scale();
__syncthreads();
float result = data[tid] * scale;  // 所有线程读同一个scale → broadcast

// 场景3: Reduce中广播结果
// warp reduce后结果在lane 0, 需要广播给所有lane:
float sum = warp_reduce(val);
sum = __shfl_sync(0xFFFFFFFF, sum, 0);  // 用shuffle广播更优
// 或通过shared memory:
if (tid == 0) shared_sum = sum;
__syncthreads();
sum = shared_sum;  // broadcast
```

### Q: C++四种Cast转换的区别与应用场景？

| Cast类型 | 安全性 | 检查时机 | 用途 | 典型场景 |
|---------|--------|---------|------|---------|
| static_cast | 中等 | 编译时 | 相关类型转换 | int↔float, 向下转型(无检查) |
| dynamic_cast | 高 | 运行时 | 安全多态转型 | 基类指针→派生类(检查RTTI) |
| const_cast | 低 | 编译时 | 去除/添加const | 兼容不正确的const接口 |
| reinterpret_cast | 最低 | 无 | 位级重解释 | 指针类型强转, 与硬件交互 |

**详细使用示例：**

```cpp
// 1. static_cast — 最常用
float f = static_cast<float>(42);        // int→float
Base* b = static_cast<Base*>(derived);   // 向上转型(安全)
Derived* d = static_cast<Derived*>(b);   // 向下转型(不安全!不检查)
void* v = malloc(100);
int* p = static_cast<int*>(v);          // void*→具体类型

// 2. dynamic_cast — 安全但有开销(需要RTTI)
Base* b = get_object();
if (auto* d = dynamic_cast<Derived*>(b)) {
    d->derived_method();  // 安全: b确实指向Derived
} else {
    // b不是Derived类型, 返回nullptr
}
// 引用版本: dynamic_cast<Derived&>(ref) 失败抛std::bad_cast

// 3. const_cast — 去除const(危险)
void legacy_api(char* s);  // 老API不接受const
const char* msg = "hello";
legacy_api(const_cast<char*>(msg));  // 去除const调用老API
// 注意: 如果msg指向的内存真的是只读的(如字符串字面量), 修改=UB

// 4. reinterpret_cast — CUDA中常用!
float* data = ...;
float4* vec_data = reinterpret_cast<float4*>(data);  // 向量化读取
// 或: char* raw = reinterpret_cast<char*>(&struct_val);  // 序列化
```

**在CUDA开发中的应用：**
- `reinterpret_cast<float4*>`: 向量化加载
- `reinterpret_cast<half2*>`: FP16打包计算
- `static_cast<float>`: 精度转换
- 避免`dynamic_cast`: GPU代码不支持RTTI

### Q: 父类转子类的安全性问题与内存布局约束？

**向下转型(Downcasting)的风险：**

```cpp
class Base { public: virtual void f(); int base_data; };
class Derived : public Base { public: int extra_data; void g(); };

Base* b = new Base();  // 只是一个Base对象
Derived* d = static_cast<Derived*>(b);  // 编译通过!
d->extra_data = 42;  // 未定义行为! 越界写入
d->g();              // 可能读取垃圾数据

// 正确做法:
Base* b2 = new Derived();  // 实际是Derived
Derived* d2 = dynamic_cast<Derived*>(b2);  // 安全: 检查RTTI
if (d2) d2->extra_data = 42;  // 确认后安全访问
```

**内存布局约束：**

```
单继承内存布局:
  Base对象:    [vptr][base_data]          (16 bytes)
  Derived对象: [vptr][base_data][extra_data] (20 bytes)
  
  Base* → Derived*: 指针值不变,但"有效范围"扩大
  如果实际只有16字节, 访问extra_data越界!

多继承内存布局:
  class C : public A, public B {};
  
  C对象: [A子对象: vptr_A + A_data] [B子对象: vptr_B + B_data] [C_data]
  
  C* → A*: 指针不变(A在开头)
  C* → B*: 指针需要偏移! (B不在开头)
  B* → C*: 需要反向偏移(static_cast自动计算)
  
  reinterpret_cast<B*>(c_ptr): 不调整偏移 → 错误!
  static_cast<B*>(c_ptr):      正确调整偏移 → 正确!
```

**安全转型的选择：**

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 确定实际类型(编译时已知) | static_cast | 零运行时开销 |
| 不确定实际类型 | dynamic_cast | 运行时检查RTTI |
| 性能关键路径 | static_cast + assert | 开发时验证，release零开销 |
| 模板/泛型代码 | static_cast | 编译时类型推导已保证安全 |

### Q: 手撕：01背包问题 vs 完全背包问题？

（编程题）
