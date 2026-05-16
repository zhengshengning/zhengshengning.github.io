---
title: '腾讯 TEG AI Infra 一二三面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: SQL中聚合函数和GROUP BY是如何实现的？

**GROUP BY的两种核心实现方式：**

**1. Hash Aggregation（最常用）：**

```
执行流程:
表扫描 → 对每行计算group key的hash
       → 在hash table中查找对应的group
       → 找到: 更新该group的聚合累加器
       → 未找到: 创建新group entry

Hash Table结构:
┌─────────┬─────────────────────────┐
│ Hash Key│ Aggregation State        │
├─────────┼─────────────────────────┤
│ "北京"   │ count=1523, sum=456789  │
│ "上海"   │ count=2341, sum=789012  │
│ "深圳"   │ count=891, sum=234567   │
└─────────┴─────────────────────────┘

优点: O(N)时间复杂度, 适合大量分组
缺点: 内存开销（hash table可能很大），结果无序
```

**2. Sort Aggregation：**

```
执行流程:
表扫描 → 按group key排序 → 顺序扫描，相邻相同key的行聚合

适用场景:
- 数据已按group key有序（如聚簇索引）
- 结果需要有序输出(ORDER BY与GROUP BY相同)
- 分组数很少时排序开销可接受

优点: 结果天然有序，内存开销小（只需维护当前group的状态）
缺点: 排序代价O(N log N)
```

**聚合函数的执行机制：**

| 聚合函数 | 内部状态 | 更新操作 | 最终结果 |
|---------|---------|---------|---------|
| COUNT(*) | int64 counter | counter++ | counter |
| SUM(x) | running_sum | sum += x | sum |
| AVG(x) | sum + count | sum+=x, count++ | sum/count |
| MAX(x) | current_max | max = max(max, x) | max |
| MIN(x) | current_min | min = min(min, x) | min |
| COUNT(DISTINCT x) | HashSet | set.insert(x) | set.size() |

**分布式场景的两阶段聚合：**

```sql
-- 原始查询
SELECT city, SUM(amount) FROM orders GROUP BY city;

-- 分布式执行:
-- Phase 1 (各节点本地聚合):
--   Node 1: GROUP BY city → {北京: 100, 上海: 200}
--   Node 2: GROUP BY city → {北京: 150, 深圳: 300}
-- Phase 2 (全局聚合):
--   Shuffle/合并 → {北京: 250, 上海: 200, 深圳: 300}
```

**注意：** COUNT(DISTINCT)不能简单分阶段合并（各节点的distinct集合可能有重叠），需要用HyperLogLog近似或全局shuffle后去重。

---

### Q: SQL中JOIN是如何实现的？有什么优化方法？

**三种JOIN实现算法详解：**

**1. Nested Loop Join (NLJ):**

```
for each row r in outer_table:      // 外表(驱动表)
    for each row s in inner_table:  // 内表(被驱动表)
        if r.key == s.key:
            emit(r, s)

复杂度: O(M × N)
优化: Index Nested Loop Join
  → 内表的join key有索引时, 内层改为index lookup: O(M × logN)
适用: 外表很小(几行) + 内表有索引
```

**2. Hash Join（等值连接最高效）:**

```
// Build阶段: 对小表建hash table
build_table = smaller_table
hash_table = {}
for each row r in build_table:
    hash_table[hash(r.key)].append(r)

// Probe阶段: 扫描大表做探测
for each row s in probe_table:
    bucket = hash_table[hash(s.key)]
    for each matching row r in bucket:
        emit(r, s)

复杂度: O(M + N)
内存: O(min(M,N)) — build阶段需要内存容纳小表的hash table
溢出处理: Grace Hash Join — hash table放不下时分区写磁盘
```

**3. Sort-Merge Join:**

```
// Sort阶段: 两表分别按join key排序
sort(table_A, by=key)
sort(table_B, by=key)

// Merge阶段: 双指针归并
i, j = 0, 0
while i < |A| and j < |B|:
    if A[i].key == B[j].key: emit match, advance both
    elif A[i].key < B[j].key: i++
    else: j++

复杂度: O(M logM + N logN + M + N)
优点: 如果数据已有序(索引/上一步排序结果)，只需O(M+N)
适用: 非等值连接(range join)、数据已有序
```

**优化器如何选择JOIN策略：**

| 条件 | 选择的算法 | 原因 |
|------|-----------|------|
| 小表(<1000行) + 大表有索引 | Index NLJ | 索引查找快 |
| 等值连接 + 内存足够 | Hash Join | O(M+N)最快 |
| 数据已有序 | Sort-Merge | 免排序 |
| 非等值连接(range) | Sort-Merge或NLJ | Hash无法处理范围 |
| 内存不足 | Grace Hash / External Sort-Merge | 分区溢出 |

**高级JOIN优化：**
- **Join Reorder**：多表JOIN时优化连接顺序（NP-hard问题，优化器用动态规划或贪心）
- **Bloom Filter**：probe前先用bloom filter快速过滤不可能匹配的行
- **Partition-wise Join**：分区表之间只需对应分区做join
- **Semi-Join Reduction**：分布式场景先传输distinct key列表过滤

---

### Q: 数据库优化器的实现原理？如何选择执行计划？

**优化器的完整工作流程：**

```
SQL语句
  ↓ Parser (语法分析)
AST (抽象语法树)
  ↓ Binder (语义分析: 解析表名、列名、类型检查)
逻辑计划 (关系代数表达式)
  ↓ Logical Optimizer (规则优化: 谓词下推、列裁剪等)
优化后的逻辑计划
  ↓ Physical Planner (枚举物理实现 + Cost Model选择)
物理执行计划
  ↓ Executor (执行)
结果集
```

**Cost Model的组成：**

```
Cost = IO_Cost + CPU_Cost + Network_Cost(分布式)

IO_Cost:
  - 顺序扫描: pages × seq_page_cost
  - 随机读取: rows × random_page_cost (通常 4× seq_page_cost)
  
CPU_Cost:
  - 行处理: rows × cpu_tuple_cost
  - 索引比较: rows × cpu_index_cost
  - 表达式计算: rows × cpu_operator_cost

关键输入 — 统计信息:
  - 表行数(pg_class.reltuples)
  - 列基数(n_distinct)
  - 值分布直方图(most_common_vals + histogram_bounds)
  - 空值比例(null_frac)
  - 平均列宽(avg_width)
```

**选择率(Selectivity)估算：**

```sql
WHERE age > 30
-- 选择率 = (max - 30) / (max - min) (均匀假设)
-- 或用直方图更精确估算

WHERE city = '北京'
-- 选择率 = mcv_freq('北京') (如果在most common values中)
-- 或 = (1 - sum(mcv_freqs)) / n_distinct (其他值均匀分配)

WHERE age > 30 AND city = '北京'
-- 独立假设: sel = sel(age>30) × sel(city='北京')
-- (可能不准确, 相关性导致估算偏差)
```

**物理计划枚举：**

| 逻辑算子 | 可选物理实现 |
|---------|------------|
| Scan | SeqScan, IndexScan, IndexOnlyScan, BitmapScan |
| Join | NestedLoop, HashJoin, MergeJoin |
| Aggregate | HashAggregate, SortAggregate |
| Sort | External Sort, In-memory Sort, Index利用有序性 |

**如何看执行计划(MySQL/PostgreSQL):**

```sql
EXPLAIN ANALYZE SELECT * FROM orders 
JOIN customers ON orders.cust_id = customers.id 
WHERE orders.amount > 1000;

-- 输出:
-- Hash Join (cost=... rows=1234 actual time=...)
--   Hash Cond: (orders.cust_id = customers.id)
--   -> Seq Scan on orders (cost=... rows=5000)
--        Filter: (amount > 1000)  Rows Removed: 95000
--   -> Hash (cost=... rows=10000)
--        -> Seq Scan on customers
```

---

### Q: Buffer Pool中脏页如何管理？数据丢失如何解决？

**Buffer Pool脏页管理机制：**

```
Buffer Pool结构:
┌──────────────────────────────────────────┐
│  Page 1 (clean) │ Page 2 (dirty) │ ...  │
│  Page N (dirty) │ Page M (clean) │ ...  │
└──────────────────────────────────────────┘
         ↕ LRU链表管理 ↕
     
脏页标记: 每个buffer frame有dirty bit
  - 修改页时: 标记dirty = true
  - 刷盘后: dirty = false
```

**脏页刷盘策略（以InnoDB为例）：**

| 策略 | 触发条件 | 说明 |
|------|---------|------|
| LRU淘汰 | Free list为空需要新页 | 淘汰LRU尾部页，脏页先刷盘再淘汰 |
| Checkpoint | 定期或redo log空间不足 | 批量刷脏页，推进LSN |
| Flush List | 后台线程(page cleaner) | 按oldest_modification排序刷最旧的脏页 |
| 紧急刷盘 | redo log即将写满 | 同步刷脏页直到释放足够redo空间 |

**WAL保证数据不丢失：**

```
Write-Ahead Logging原则:
  修改数据前，先将修改内容写入redo log并确保持久化

事务提交流程:
1. 修改Buffer Pool中的页 (内存中修改, 快)
2. 生成redo log record写入Log Buffer
3. commit时: Log Buffer fsync到磁盘 (顺序IO, 快)
4. 返回commit成功给用户
5. 脏页可以延后刷盘 (随时可以从redo log恢复)

崩溃恢复:
1. 找到最后一个checkpoint的LSN
2. 从该LSN开始重放redo log (前滚, redo)
3. 对未提交事务回滚 (用undo log)
```

**关键概念：**

| 概念 | 含义 |
|------|------|
| LSN (Log Sequence Number) | redo log中每条记录的唯一递增编号 |
| Checkpoint | 已刷盘的脏页对应的最大LSN，恢复只需从此处开始 |
| Doublewrite Buffer | 防止partial page write导致数据损坏 |
| fsync | 确保数据从OS缓存写到磁盘持久化 |

---

### Q: LRU算法有什么缺陷？如何改进？

**LRU的经典问题：**

**问题1: 预读失效(Prefetch Pollution)**

```
正常访问模式: 热页 [A, B, C, D] 在LRU前部

预读触发: 读取page X, OS预读了X+1, X+2, X+3到LRU头部
LRU变化: [X+3, X+2, X+1, X, A, B, C, D]
                                     ↑ 热页被挤到尾部
如果X+1/X+2/X+3从未被真正访问 → 白白占据热位置
```

**问题2: 缓存污染(Buffer Pool Pollution)**

```
全表扫描(100万行):
  顺序读取每个page，每个page只用一次
  LRU: [page_1M, ..., page_2, page_1, 之前的热页全被挤出]
  
结果: 一次全表扫描把整个buffer pool的热数据冲走
      后续正常查询全部cache miss → 性能骤降
```

**改进方案：**

**1. MySQL InnoDB的分段LRU (Young/Old)：**

```
Buffer Pool LRU链表:
┌──── Young区 (5/8) ────┐┌──── Old区 (3/8) ────┐
│ 真正的热页             ││ 新进入/待观察的页     │
│ (频繁访问才能待在这)   ││ (首次访问进入这里)    │
└────────────────────────┘└─────────────────────┘
       ↑                          ↑
  被再次访问且距上次访问>1s      新页进入点(midpoint insertion)
  才从Old提升到Young

全表扫描时: 页进入Old区 → 短时间内不会被再次访问 
           → 不会提升到Young → 被快速淘汰
           → 不影响Young区的热页！
```

**2. LRU-K (数据库学术界方案)：**

```
LRU-1: 标准LRU, 一次访问就进入缓存
LRU-2: 页被访问2次才进入正式缓存, 第一次只进入历史队列
LRU-K: 页被访问K次才进入, 用第K次的访问时间排序

优点: 有效过滤一次性访问
缺点: 需要维护访问历史，复杂度高
```

**3. CLOCK算法（Linux页面置换）：**

```
环形缓冲 + reference bit:
- 页被访问时: reference bit = 1
- 需要淘汰时: 指针扫描
  - bit=1: 改为0 (给第二次机会), 跳过
  - bit=0: 淘汰该页

优点: 近似LRU效果，开销低(不需要维护链表)
```

**4. ARC (Adaptive Replacement Cache):**

```
自适应在LRU和LFU之间平衡:
- T1: 最近访问一次的页（LRU顺序）
- T2: 最近访问多次的页（LFU顺序）
- 根据命中模式动态调整T1/T2的大小比例
- ZFS和一些数据库使用
```

---

### Q: 并发B+树如何实现？

**Latch Crabbing（蟹行协议）——最经典方案：**

```
查找(读操作):
  1. 加Root的S-latch(共享锁)
  2. 进入子节点, 加子节点S-latch
  3. 释放父节点S-latch (已经不需要了)
  4. 重复直到叶节点

插入(写操作):
  1. 加Root的X-latch(排他锁)
  2. 进入子节点, 加子节点X-latch
  3. 如果子节点"安全"(不会分裂: 当前key数 < max-1):
     → 释放所有祖先节点的X-latch (它们不会被修改)
  4. 如果子节点"不安全":
     → 保持所有祖先的锁 (可能需要分裂传播到父节点)

"安全"判定:
  插入安全: 节点未满 (不会分裂)
  删除安全: 节点超过半满 (不会合并)
```

**Latch Crabbing的问题和优化：**

| 方案 | 思路 | 优缺点 |
|------|------|--------|
| 基本Crabbing | 写操作自顶向下加X-latch | Root锁竞争严重(高并发瓶颈) |
| 乐观Latch | 写时先乐观用S-latch下行，到叶节点才加X-latch | 减少上层锁竞争，分裂时需要重试 |
| B-link Tree | 每个节点加right-link指针 | 分裂时只锁当前节点(半个节点通过right-link可达) |
| Latch-Free B-Tree | CAS + epoch-based reclamation | 最高并发，实现复杂 |

**B-link Tree的巧妙设计：**

```
标准B+树分裂需要同时锁父和子:
  [Parent: ..., 20, ...]
       ↓
  [Child: 1,3,5,7,9,12,15,18] → 满了需要分裂

B-link Tree: 节点带right pointer
  分裂步骤:
  1. 只锁Child，分裂为 [1,3,5,7] →right→ [9,12,15,18]
  2. 释放Child锁
  3. 锁Parent，插入新key和指针
  
  如果步骤2-3之间有读者:
  读者到达旧Child，发现要找的key > high_key
  → 跟随right pointer到新节点
  → 无需锁父节点也能找到数据！
```

---

### Q: 火山模型和其他执行模型的区别？

**三种数据库执行模型对比：**

**1. 火山模型 (Volcano/Iterator/Tuple-at-a-time):**

```cpp
class Operator {
    virtual void open() = 0;
    virtual Tuple* next() = 0;  // 每次返回一行
    virtual void close() = 0;
};

// 执行: 自顶向下拉取
// Project → Filter → SeqScan
auto* project = new ProjectOp(new FilterOp(new SeqScanOp("table")));
project->open();
while (auto* tuple = project->next()) {  // 逐行拉取
    output(tuple);  // 每次一个虚函数调用链
}
```

| 优点 | 缺点 |
|------|------|
| 通用：任意算子可组合 | 虚函数调用开销（每行多次） |
| 内存友好（一次一行） | CPU cache不友好（数据不连续） |
| 实现简单 | 解释执行开销大 |

**2. 物化模型 (Materialization/Operator-at-a-time):**

```cpp
// 每个算子一次处理所有数据，输出完整结果集
vector<Tuple> SeqScan::execute() {
    return read_all_from_table();  // 全部读出
}
vector<Tuple> Filter::execute() {
    auto input = child->execute();     // 子算子先全部算完
    return filter(input, predicate);   // 再一次性过滤
}
```

| 优点 | 缺点 |
|------|------|
| 无虚函数调用开销 | 中间结果可能很大(内存问题) |
| 可批量优化(SIMD) | 不适合大数据量 |

**3. 向量化模型 (Vectorized/Batch-at-a-time) — 现代OLAP主流:**

```cpp
class Operator {
    virtual ColumnBatch* next_batch() = 0;  // 每次返回一批(如1024行)
};

// 每次处理1024行:
// - 摊销虚函数调用开销(1次/1024行 vs 1次/行)
// - 列式内存布局 → SIMD友好
// - batch大小适合L1/L2 cache
```

| 维度 | 火山模型 | 向量化模型 | 加速比 |
|------|---------|-----------|--------|
| 虚函数开销/行 | 1次 | 1/1024次 | 1024x |
| SIMD利用 | 无 | 列式batch天然适合 | 4-8x |
| Cache利用 | 差(行式散乱) | 好(列连续) | 2-5x |
| 总体性能 | baseline | 10-100x(OLAP) | - |

**典型数据库的选择：**
- 火山模型：MySQL、PostgreSQL（传统OLTP）
- 向量化模型：ClickHouse、DuckDB、Velox（现代OLAP）
- JIT编译：Hyper、MemSQL（编译执行计划为机器码，消除解释开销）

---

### Q: 行存和列存的应用场景？

**核心设计差异：**

```
行存(Row Store):                    列存(Column Store):
┌─────────────────────────┐         ┌──────┬───────┬────────┐
│ id=1, name="张", age=25 │         │ id列  │ name列│ age列  │
│ id=2, name="李", age=30 │         │  1    │  张   │  25    │
│ id=3, name="王", age=28 │         │  2    │  李   │  30    │
└─────────────────────────┘         │  3    │  王   │  28    │
                                    └──────┴───────┴────────┘
一行数据物理连续                     一列数据物理连续
```

**性能对比：**

| 操作 | 行存 | 列存 | 原因 |
|------|------|------|------|
| 单行读取(SELECT * WHERE id=1) | 快（一次IO读完整行） | 慢（需读多个列文件） | 行连续 vs 列分散 |
| 单行写入(INSERT/UPDATE) | 快（写入一个位置） | 慢（修改多个列文件） | 一个 vs 多个位置 |
| 聚合查询(SELECT AVG(age)) | 慢（读整行但只用1列） | 快（只读age列） | 列裁剪 |
| 扫描分析(WHERE age>25) | 慢（读大量无关数据） | 快（只扫age列） | 数据量减少 |
| 压缩率 | 低（混合类型） | 高（同类型连续） | 同类型更易压缩 |
| SIMD加速 | 困难 | 容易（列连续处理） | 数据布局对齐 |

**典型应用场景：**

| 场景 | 推荐存储 | 代表系统 |
|------|---------|---------|
| OLTP(事务处理) | 行存 | MySQL, PostgreSQL |
| OLAP(分析) | 列存 | ClickHouse, Snowflake |
| 时序数据 | 列存 | InfluxDB, TimescaleDB |
| 数据湖 | 列存文件格式 | Parquet, ORC |
| 混合负载(HTAP) | 行列混合 | TiDB(行+TiFlash列), SAP HANA |
| AI特征存储 | 列存 | 快速读取特征子集 |

**列存的压缩优势：**
- 同一列数据类型相同，pattern重复 → Run-Length Encoding, Dictionary Encoding有效
- 典型压缩率：列存5-10x vs 行存2-3x
- 压缩后IO更少 → 进一步加速分析查询

---

### Q: Linux中断处理模块如何实现？CPU如何处理多个中断？

**中断处理的完整流程：**

```
硬件设备产生中断信号
     ↓
中断控制器(APIC/GIC)接收
     ↓ (IRQ编号 + 优先级仲裁)
CPU响应中断:
  1. 关闭本CPU中断(cli)
  2. 保存当前上下文(寄存器压栈)
  3. 跳转到中断向量表[IRQ]对应的handler
     ↓
┌──────── 上半部(Top Half) ────────┐
│  硬中断处理(irq_handler):         │
│  - 读取硬件状态/数据              │
│  - ACK中断控制器                  │
│  - 标记下半部工作                  │
│  - 执行时间: 必须<100us           │
└──────────────────────────────────┘
     ↓ (开中断, 恢复执行)
┌──────── 下半部(Bottom Half) ──────┐
│  延后执行的耗时工作:               │
│  - Softirq: 高频(如网络包处理)    │
│  - Tasklet: 基于softirq的封装     │
│  - Workqueue: 可睡眠(如磁盘IO)    │
│  - Threaded IRQ: 中断线程化       │
└──────────────────────────────────┘
```

**上下半部的对比：**

| 维度 | 上半部(硬中断) | 下半部(软中断/workqueue) |
|------|--------------|----------------------|
| 执行时机 | 立即 | 延后(软中断在硬中断返回后, WQ在内核线程中) |
| 可否睡眠 | 不可 | Softirq不可, Workqueue可以 |
| 抢占 | 不可被同级抢占 | 可被硬中断抢占 |
| 执行时间 | 极短(<100us) | 可较长 |
| 典型操作 | 读硬件寄存器、ACK | 数据处理、协议栈 |

**多中断处理机制：**

```
场景: CPU正在处理IRQ_A时，IRQ_B到来

处理策略:
1. 同优先级/低优先级: 排队等待(中断被关闭或标记pending)
2. 高优先级: 
   - x86: 处理硬中断时关闭同级中断，不支持嵌套
   - ARM GIC: 支持优先级抢占(高优先级中断可打断低优先级handler)

中断亲和性(Affinity):
  /proc/irq/N/smp_affinity → 指定哪些CPU处理该中断
  用途: 网卡中断绑定到特定核，避免cache bounce

多队列网卡(MSI-X):
  一块网卡有多个中断向量(如64个)
  每个队列的中断分发到不同CPU核 → 并行处理网络包
```

**NAPI(网络中断优化)：**

```
高速网络场景: 每秒百万包，每包一个中断 → CPU被中断淹没

NAPI解决方案: 中断+轮询混合模式
1. 首个包到达: 触发硬中断
2. 硬中断handler: 关闭网卡中断, 启动轮询(poll)
3. 软中断中: 批量从网卡buffer读取数据包(一次处理多个)
4. 读完后: 重新开启网卡中断

效果: 高负载时接近轮询效率，低负载时保持中断的低延迟
```

---

### Q: 进程和线程的区别？用户态线程的问题？

**进程 vs 线程的本质区别：**

| 维度 | 进程 | 线程 |
|------|------|------|
| 定义 | 资源分配的基本单位 | CPU调度的基本单位 |
| 地址空间 | 独立(4GB/128TB虚拟空间) | 共享进程地址空间 |
| 创建开销 | 大(~1ms, fork需复制页表/fd表) | 小(~10-100us, 共享一切) |
| 上下文切换 | 慢(需切换页表/刷TLB/切换寄存器) | 快(只切换寄存器和栈指针) |
| 通信 | 需IPC(pipe/socket/shm) | 直接访问共享内存 |
| 隔离性 | 强(一个崩不影响另一个) | 弱(一个线程段错误整个进程崩) |
| 内核表示(Linux) | task_struct + 独立mm_struct | task_struct + 共享mm_struct |

**Linux中进程和线程的统一表示：**

```c
// Linux中线程就是共享资源的进程(LWP, Light Weight Process)
// clone()系统调用通过flags控制共享什么:
clone(CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | ...)
//    ^共享内存  ^共享文件系统 ^共享fd表    ^共享信号处理
// 全共享 = 线程; 全不共享 = 进程

// pthread_create内部就是调用clone(CLONE_VM|...)
// fork内部是clone(不带CLONE_VM)，子进程得到独立地址空间(COW)
```

**用户态线程(Green Thread/Coroutine)的问题：**

| 问题 | 原因 | 后果 |
|------|------|------|
| 阻塞IO阻塞整个进程 | 内核只看到一个线程(1:N模型) | 一个协程read()阻塞，所有协程停止 |
| 无法利用多核 | 所有协程在一个内核线程上 | CPU密集任务无并行 |
| 无抢占式调度 | 用户态调度器无硬件timer中断 | 一个协程死循环则饿死其他协程 |
| 系统调用问题 | 系统调用是线程粒度的 | page fault等阻塞所有协程 |

**现代解决方案(M:N模型)：**

```
Go goroutine / Rust tokio / Java虚拟线程:
  M个用户态协程 映射到 N个内核线程

Go的GMP模型:
  G (Goroutine): 用户态协程(2KB栈)
  M (Machine): 内核线程
  P (Processor): 逻辑处理器(通常=CPU核数)

  G → P → M → CPU核
  
  关键设计:
  - G阻塞IO时: runtime将G挂起, P切换到另一个G(非阻塞)
  - G计算密集: sysmon检测长时间不让出的G，强制抢占(1.14+)
  - 工作窃取: P空闲时从其他P的队列偷G执行
```

---

### Q: 用户态向内核态切换的过程？

**切换的触发方式和完整过程：**

**触发方式：**

| 触发方式 | 示例 | 特点 |
|---------|------|------|
| 系统调用(syscall) | read(), write(), mmap() | 用户主动请求内核服务 |
| 中断(interrupt) | 定时器中断、网卡中断 | 外部设备异步通知 |
| 异常(exception) | 页错误(page fault)、除零、段错误 | CPU执行时检测到的错误 |

**x86-64 syscall指令的完整过程：**

```
用户态(Ring 3):
  mov rax, __NR_read   ; 系统调用号
  mov rdi, fd          ; 参数1
  mov rsi, buf         ; 参数2
  mov rdx, count       ; 参数3
  syscall              ; 触发切换!

CPU硬件自动完成(~100 cycles):
  1. RCX = 保存用户态RIP(返回地址)
  2. R11 = 保存用户态RFLAGS
  3. RIP = MSR_LSTAR(内核入口地址, 即syscall handler)
  4. CPL = 0 (Ring3→Ring0, 权限提升)
  5. 段选择器变为内核段
  6. 关闭中断(IF=0)
  
内核入口(entry_SYSCALL_64):
  ; 切换到内核栈
  movq %rsp, PER_CPU(rsp_scratch)  ; 保存用户栈
  movq PER_CPU(kernel_stack), %rsp  ; 加载内核栈(从TSS)
  
  ; 保存用户态寄存器到内核栈(pt_regs结构)
  pushq %rcx  ; 用户RIP
  pushq %r11  ; 用户RFLAGS
  pushq %rdi, %rsi, %rdx, ... ; 保存所有寄存器
  
  ; 根据rax系统调用号查sys_call_table
  call sys_call_table[rax]  ; 执行实际的内核函数
  
  ; 返回用户态
  sysretq
    ; RIP = RCX(之前保存的返回地址)
    ; RFLAGS = R11
    ; CPL = 3 (Ring0→Ring3)
```

**完整开销分析：**

| 操作 | 开销 | 说明 |
|------|------|------|
| syscall/sysret指令 | ~50-100 cycles | 硬件特权级切换 |
| 保存/恢复寄存器 | ~50 cycles | 压栈/弹栈操作 |
| 栈切换 | ~20 cycles | 用户栈→内核栈 |
| TLB/Cache影响 | 0-数千cycles | 内核代码可能导致cache eviction |
| KPTI(Meltdown缓解) | +~500 cycles | 需要切换页表(隔离内核映射) |
| **总计** | **~200-1000 cycles (~100-500ns)** | 取决于硬件和安全缓解措施 |

**为什么vDSO可以避免切换？**

```
vDSO (Virtual Dynamic Shared Object):
  内核将某些只读数据映射到用户空间(如时间戳)
  
  gettimeofday() 不需要真正进入内核:
  - 内核定期更新共享页中的时间值
  - 用户态直接读取共享页 → 无syscall开销
  
  类似: clock_gettime(), getcpu()
```

---

### Q: 手撕：实现一个基于内存的文件系统（字典树变体）？

（编程题）

---

### Q: 手撕：快慢指针找中点+反转链表？

（编程题）

---

### Q: 手撕：字符串大写转小写后按字典序排序（不用库函数）？

（编程题）
