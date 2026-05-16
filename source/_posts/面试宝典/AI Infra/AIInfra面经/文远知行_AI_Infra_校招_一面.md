---
title: '文远知行 AI Infra 校招 一面'
date: 2026-04-16 12:06:35
categories:
  - [求职面试, 车企/自驾面经]
tags: [AIInfra, 推理优化, 算子优化, 高性能计算, 面经]
---


<!-- more -->

---

### Q: PD分离（Prefill-Decode分离）机制是什么？如何调度两个队列？

**为什么要PD分离？——两个阶段的资源特性完全不同：**

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算特征 | Compute-bound (大矩阵乘) | Memory-bound (逐token生成) |
| 算术强度 | 高 (seq_len个token同时计算) | 低 (1个token × 大batch) |
| GPU利用率 | 高 (Tensor Core忙) | 低 (主要读KV-Cache) |
| 延迟敏感性 | 中 (TTFT可以容忍数百ms) | 高 (TPOT直接影响用户体验) |
| 显存模式 | 临时显存需求大(中间tensor) | KV-Cache持续增长 |

**混合调度的问题：**

```
混合部署(Prefill和Decode在同一GPU):
时间线:
  ─[Decode batch]─[长Prefill]───────────────[Decode batch]─
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
                   Decode被Prefill阻塞! 所有正在生成的请求被卡住
                   用户感知: 输出突然停顿几百ms(P99延迟飙升)
```

**PD分离架构：**

```
                    ┌─────────── Router ───────────┐
                    │  新请求→Prefill池             │
                    │  Prefill完成→传输KV→Decode池  │
                    └────────────┬─────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ↓                                           ↓
┌──── Prefill GPU集群 ────┐              ┌──── Decode GPU集群 ────┐
│ 特点:                    │              │ 特点:                   │
│ - 高算力利用             │  KV传输      │ - 高带宽利用            │
│ - batch可以较小          │ ──────→      │ - 大batch(低延迟)       │
│ - 不影响在线用户         │  NVLink/     │ - CUDA Graph加速        │
│ - 可用INT8/FP8加速      │  RDMA        │ - 持续生成不被打断      │
└──────────────────────────┘              └─────────────────────────┘
```

**调度策略：**

**1. Decode优先策略（保证TPOT）：**

```python
def schedule():
    # 优先级1: Decode请求(已在生成的用户)
    decode_batch = collect_decode_requests()
    if len(decode_batch) > 0:
        execute_decode(decode_batch)  # 必须执行，不能延迟
    
    # 优先级2: Prefill请求(新用户)
    # 只在Decode batch间隙或专用GPU上执行
    if prefill_gpu_available():
        prefill_batch = waiting_queue.pop_batch()
        execute_prefill(prefill_batch)
```

**2. KV-Cache传输优化：**

| 方案 | 延迟 | 带宽需求 | 适用场景 |
|------|------|---------|---------|
| NVLink直传 | ~us级 | 600-900 GB/s | 同节点多卡 |
| RDMA(IB) | ~10us | 200-400 Gb/s | 跨节点 |
| 分层传输 | 首层先传 | 按需 | 超长序列 |
| KV压缩 | 增加计算换带宽 | 减少2-4x | 带宽瓶颈时 |

**3. 防止Prefill饿死：**
- 设置TTFT SLA上限（如2s）
- 预留一定比例的GPU资源给Prefill
- Chunked Prefill：将长Prefill分块穿插在Decode间隙

**实际部署案例：**
- Mooncake(月之暗面): PD分离 + KV-Cache传输优化
- SGLang: 支持PD分离模式
- TensorRT-LLM: 支持配置Prefill/Decode分离

---

### Q: vLLM如何优化显存？

**vLLM显存优化的全方位策略：**

**1. PagedAttention（核心创新）：**

```
传统: 预分配max_seq_len的连续空间
  请求A(实际100 tokens, max=2048): 浪费 1948×kv_size
  10个请求: 浪费 ~95% 显存

PagedAttention: 按需分配16-token Block
  请求A(100 tokens): 分配 ceil(100/16) = 7 blocks
  只浪费最后block的 100%16=4 个空位 → 浪费 <4%
```

**显存利用率量化：**

| 管理方式 | 显存利用率 | 浪费来源 |
|---------|-----------|---------|
| 连续预分配(max_len) | 20-40% | 内部碎片(实际<<max) |
| 连续预分配(预估长度) | 50-70% | 预估不准+外部碎片 |
| PagedAttention | >96% | 仅最后block内部碎片 |

**2. Copy-on-Write（Beam Search/并行采样）：**

```
Beam Search (beam=4):
  传统: 4条beam各自独立存储完整KV → 4x显存
  
  CoW: 共享相同prefix的block(引用计数)
  初始: beam_0..3 共享前缀blocks [ref_count=4]
  分叉时: 只有修改的block才复制
  
  显存节省: 前缀越长节省越多, 理论最大(beam-1)/beam = 75%
```

**3. Prefix Caching：**

```
多个请求共享相同system prompt:
  请求1: [system_prompt(500 tokens)] + [user_query_1]
  请求2: [system_prompt(500 tokens)] + [user_query_2]
  
  传统: 每个请求独立prefill和存储 → 500 tokens KV重复N次
  Prefix Cache: system_prompt的KV只计算存储一次,所有请求共享
  
  节省: N个请求 × 500 tokens × kv_size × (N-1)/N
```

**4. 抢占机制（防OOM）：**

```python
# 显存不足时的应对策略:
if free_blocks < needed_blocks:
    # 策略1: Swap (适合长序列)
    victim = select_lowest_priority_request()
    swap_out_to_cpu(victim.kv_blocks)  # KV搬到CPU内存
    victim.state = SWAPPED
    # 后续显存释放后swap_in恢复
    
    # 策略2: Recompute (适合短序列)
    victim = select_shortest_request()
    release_kv_blocks(victim)
    victim.state = WAITING  # 重新排队prefill
```

**5. 显存预算管理：**

```
GPU 80GB显存分配:
  模型权重(FP16 70B): ~140GB → INT4量化后: ~35GB
  KV-Cache Pool: ~40GB (动态分配给active请求)
  临时计算空间: ~3GB (attention中间结果等)
  系统/框架开销: ~2GB
  
  KV-Cache Pool支持的并发请求数:
  每token KV大小 = 2(KV) × num_layers × num_kv_heads × head_dim × 2bytes(FP16)
  LLaMA-70B: 2 × 80 × 8 × 128 × 2 = 327,680 bytes/token ≈ 320KB/token
  40GB / 320KB = ~125,000 tokens 总预算
  → 125个请求×1000 tokens 或 25个请求×5000 tokens
```

---

### Q: Chunked Prefill是什么？

**问题背景：长Prefill阻塞Decode**

```
用户发送8000 token的长文档:
传统处理: 一次性prefill 8000 tokens → 耗时~200ms
在这200ms内: 所有正在decode的请求被阻塞!

时间线:
  ──[decode]─[─── prefill 8K ───]─[decode]──
              ^^^^^^^^^^^^^^^^^^^
              所有decode用户停顿200ms → P99延迟飙升
```

**Chunked Prefill的解决方案：**

```
将8000 token的prefill分成16个chunk(每chunk 512 tokens):
  
时间线:
  ──[decode+chunk1]─[decode+chunk2]─[decode+chunk3]─...
  
每个step:
  - 执行一个prefill chunk(512 tokens)的计算
  - 同时执行所有decode请求(每个生成1 token)
  - Decode延迟增加很少(chunk较小,不会大幅增加step时间)
```

**关键设计参数：**

| 参数 | 典型值 | 影响 |
|------|--------|------|
| chunk_size | 256-1024 tokens | 越小→decode延迟越低，prefill吞吐越低 |
| max_prefill_tokens/step | 2048-4096 | 限制每step的prefill计算量 |
| prefill_priority | 低于decode | 确保decode不被饿死 |

**性能分析：**

```
假设:
  decode batch: 64个请求, 每个1 token → 64 tokens的GEMM
  prefill chunk: 512 tokens → 512 tokens的GEMM
  
混合后每step: 64 + 512 = 576 tokens的GEMM
  step时间: 从2ms增加到~10ms (不再是200ms的大块!)
  
decode延迟影响:
  原始decode step: 2ms/token
  混合后decode step: ~10ms/token (增加5x, 但仍可接受)
  
  vs 不分chunk: 
  每200ms一次200ms阻塞 → P99=200ms (不可接受)
```

**SarathiServe的核心改进：**
- 证明了chunk大小对TPOT和TTFT的trade-off
- 提出了基于SLO(Service Level Objective)的自适应chunk大小选择
- 在decode延迟约束下最大化prefill吞吐
- 实验显示P99延迟改善3-5倍

---

### Q: 什么是虚拟内存？

**虚拟内存的核心机制：**

```
进程A看到的地址空间:          物理内存:
┌──────────────────┐         ┌────────────────┐
│ 0x0000: code     │ ──页表→ │ 物理帧 #5      │
│ 0x1000: data     │ ──页表→ │ 物理帧 #100    │
│ 0x2000: heap     │ ──页表→ │ 物理帧 #37     │
│ 0x3000: (未映射)  │         │ ...            │
│ ...              │         │ 物理帧 #200    │ ← 进程B的某页
└──────────────────┘         └────────────────┘

进程B看到的地址空间:          
┌──────────────────┐         
│ 0x0000: code     │ ──页表→ │ 物理帧 #200    │
│ 0x1000: data     │ ──页表→ │ 物理帧 #50     │
└──────────────────┘
// A和B用相同虚拟地址, 但映射到不同物理帧 → 隔离!
```

**虚拟内存的四大核心功能：**

| 功能 | 机制 | 收益 |
|------|------|------|
| 进程隔离 | 每个进程独立页表 | 互不影响，安全 |
| 内存超分 | 按需分配+swap | 虚拟空间可超过物理内存 |
| 内存保护 | 页表中权限位(R/W/X) | 防止越权访问 |
| 共享内存 | 多个虚拟页映射同一物理帧 | 高效IPC/共享库 |

**地址翻译过程(x86-64四级页表)：**

```
虚拟地址(48-bit有效):
  [PML4(9bit)][PDPT(9bit)][PD(9bit)][PT(9bit)][Offset(12bit)]
  
翻译: CR3→PML4表→PDPT表→PD表→PT表→物理帧号+Offset
  4次内存访问(无TLB时) → 慢!

TLB加速:
  TLB(Translation Lookaside Buffer): 缓存最近的虚拟→物理映射
  命中率: 通常>99%(利用局部性)
  命中延迟: ~1 cycle
  未命中: 需要page walk(4次内存访问 ~200 cycles)
```

**与GPU显存管理的关系：**
- 传统CUDA: `cudaMalloc`直接分配物理显存地址(无虚拟内存层)
- CUDA Unified Memory: 提供虚拟地址抽象，页面按需在CPU↔GPU迁移
- vLLM PagedAttention: 在应用层模拟虚拟内存分页(逻辑block→物理block)

---

### Q: 进程和线程的区别？操作系统如何调度？

**核心对比：**

| 维度 | 进程 | 线程(LWP) |
|------|------|-----------|
| 地址空间 | 独立(mm_struct) | 共享进程的mm_struct |
| 文件描述符 | 独立 | 共享 |
| 信号处理 | 独立 | 共享(但可独立mask) |
| 创建开销 | ~1ms(fork, COW页表) | ~100us(clone, 共享大部分) |
| 切换开销 | ~5us(TLB flush+页表切换) | ~1us(只切换寄存器/栈) |
| 通信 | IPC(pipe/shm/socket) | 直接共享内存 |
| 崩溃影响 | 独立(段错误不影响其他进程) | 共享(一个线程段错误整个进程崩) |

**Linux CFS(Completely Fair Scheduler)调度原理：**

```
CFS核心: 虚拟运行时间(vruntime)

  每个任务维护vruntime:
    vruntime += 实际运行时间 × (NICE_0_WEIGHT / task_weight)
    
  调度决策: 选择vruntime最小的任务运行(最"欠"CPU的)
  数据结构: 红黑树(按vruntime排序, 选最左节点)
  
  nice值影响:
    nice=0: 权重1024(基准)
    nice=-20: 权重88761(高优先级, vruntime增长慢→获得更多CPU)
    nice=19: 权重15(低优先级, vruntime增长快→获得更少CPU)
```

**调度时机：**

| 触发条件 | 场景 | 行为 |
|---------|------|------|
| 时间片用完 | 运行太久(由CFS period决定) | 检查是否有更小vruntime的任务 |
| 主动让出 | sleep/IO阻塞/yield | 切换到下一个任务 |
| 被抢占 | 高优先级任务就绪 | 设置need_resched标志 |
| 任务结束 | exit/被kill | 切换到下一个 |

**实时调度策略(SCHED_FIFO/SCHED_RR)：**
- 优先级高于所有CFS任务
- SCHED_FIFO: 一直运行直到主动让出
- SCHED_RR: 相同优先级间轮转
- 用于延迟敏感场景(如GPU driver中断处理线程)

---

### Q: TCP/IP分层模型？如何通过IP地址找到目标进行通信？

**TCP/IP四层模型与数据封装：**

```
发送方:                              接收方:
┌──────────────┐                    ┌──────────────┐
│ 应用层        │ HTTP/DNS/gRPC      │ 应用层        │
│ [Data]       │                    │ [Data]       │
├──────────────┤                    ├──────────────┤
│ 传输层        │ TCP/UDP            │ 传输层        │
│ [TCP|Data]   │ 端口号区分进程      │ [TCP|Data]   │
├──────────────┤                    ├──────────────┤
│ 网络层        │ IP/ICMP            │ 网络层        │
│ [IP|TCP|Data]│ IP地址定位主机      │ [IP|TCP|Data]│
├──────────────┤                    ├──────────────┤
│ 链路层        │ Ethernet/WiFi      │ 链路层        │
│ [ETH|IP|TCP|Data|FCS]│ MAC地址定位设备 │            │
└──────┬───────┘                    └──────┬───────┘
       ↓ 物理线路/无线                       ↑
       ─────────────────────────────────────
```

**通过IP地址通信的完整过程：**

```
场景: 主机A(192.168.1.100)要发数据给主机B(10.0.0.50)

Step 1: 应用层DNS解析(如果只有域名)
  DNS查询: www.example.com → 10.0.0.50

Step 2: 判断目标是否在同一子网
  A的IP: 192.168.1.100, 子网掩码: 255.255.255.0
  A的子网: 192.168.1.0/24
  B的IP: 10.0.0.50 → 不在同一子网!
  → 需要通过网关路由

Step 3: 发给默认网关(192.168.1.1)
  ARP: 查询192.168.1.1的MAC地址(已缓存或广播查询)
  封装: [网关MAC | A MAC | IP包(src=A, dst=B) | ...]
  
Step 4: 路由器逐跳转发
  路由器查路由表: 10.0.0.0/24 → 从接口eth2转发
  每一跳: 更换链路层头(MAC地址), IP头不变
  
Step 5: 最后一跳
  到达B所在子网的路由器
  ARP查询B(10.0.0.50)的MAC
  直接投递给B
```

**关键协议的作用：**

| 协议 | 层级 | 作用 | 关键机制 |
|------|------|------|---------|
| ARP | 链路层 | IP→MAC地址解析 | 广播查询, 缓存表 |
| IP | 网络层 | 端到端寻址+路由 | TTL防环, 分片重组 |
| TCP | 传输层 | 可靠有序传输 | 三次握手, 重传, 流控 |
| UDP | 传输层 | 不可靠但快速 | 无连接, 无重传 |
| DNS | 应用层 | 域名→IP解析 | 层次化域名空间, 缓存 |

---

### Q: 手撕：DFS解法+优化为O(NlogN)？

（编程题）
