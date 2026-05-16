---
title: '美团 AI Infra 校招 一面'
date: 2026-04-15 12:00:00
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 手撕：删除链表中的重复节点 II？

（编程题）

### Q: HashMap 引入红黑树的目的？为什么不一开始就用红黑树？

**引入红黑树的动机**：

Java 8 的 HashMap 在桶内链表长度超过阈值（默认 8）时，将链表转为红黑树。核心目的是**防止 hash 冲突攻击导致的性能退化**。

**问题场景分析**：
```
正常情况: hash 分布均匀，每个桶 0-2 个元素
  查找: O(1) 平均

恶意攻击/极端情况: 大量 key hash 到同一桶
  链表退化: O(n) 查找 → n=10000 时性能崩溃
  红黑树: O(log n) 查找 → n=10000 时仅 ~13 次比较
```

**为什么不一开始就用红黑树**：

1. **内存开销差异巨大**：
   ```
   链表节点: next 指针 (8 bytes) + key/value/hash
   TreeNode: left + right + parent + color (32+ bytes) + 继承 Node
   ```
   红黑树节点内存是链表的 **2-3 倍**

2. **短链表下性能对比**：
   - 链表长度 1-6 时，顺序遍历比树查找**更快**（CPU cache 友好，无分支预测开销）
   - 红黑树查找需要比较 + 左右分支跳转（cache miss 概率高）
   - 实测：长度 < 8 时链表平均查找时间 < 红黑树

3. **维护成本**：
   - 红黑树的插入/删除需要旋转（rotation）和重染色（recoloring）
   - 平均每次插入 0.6 次旋转，删除 0.8 次旋转
   - 链表的插入/删除是 O(1) 的简单指针操作

4. **实际统计**：大多数 bucket 在正常负载因子（0.75）下只有 0-2 个元素，使用链表的桶 > 99%

**阈值为什么是 8**：
- 根据泊松分布，负载因子 0.75 时，单桶 ≥ 8 个元素的概率 < 0.00000006
- 这意味着正常使用中几乎不会触发树化
- 阈值 8 在"不浪费内存"和"防止退化"之间取得平衡

### Q: 用二叉排序树或 AVL 代替红黑树可以吗？

**BST（无平衡保证）**——不可以：
- 极端情况退化为链表：如果 hash 值的自然序是有序的，BST 退化为 O(n)
- 根本没有解决链表的退化问题
- 唯一比链表好的点是可以二分，但前提是平衡

**AVL 树**——可以，但红黑树更优的理由：

| 维度 | AVL | 红黑树 |
|---|---|---|
| 平衡严格度 | 严格平衡（左右高度差 ≤ 1） | 近似平衡（最长路径 ≤ 2×最短） |
| 查找性能 | 略优（树高更矮） | 略差（树高最多 2×log(n)） |
| 插入旋转次数 | 最多 2 次 | 最多 2 次 |
| **删除旋转次数** | **最多 O(log n) 次** | **最多 3 次** |
| 适用场景 | 查找密集、修改少 | 增删频繁 |

**关键差异在删除操作**：
- HashMap 的使用模式：频繁 put/remove（增删密集）
- AVL 删除后可能需要从删除点一路旋转到根（O(log n) 次旋转）
- 红黑树删除最多 3 次旋转 + 若干次重染色（O(1) 结构调整）
- 在 HashMap 这种增删频繁的场景下，红黑树的综合效率更高

**B+ 树呢？**——过重：
- B+ 树适合磁盘 I/O 优化（减少访问次数），内存场景无此需求
- 节点结构复杂，空间开销大
- HashMap 的单桶元素量有限（通常 < 100），用 B+ 树大材小用

### Q: 线程池的核心参数？

线程池（以 Java ThreadPoolExecutor 为例）通过参数化配置实现灵活的线程管理策略：

**六大核心参数**：

```java
new ThreadPoolExecutor(
    corePoolSize,      // 核心线程数
    maximumPoolSize,   // 最大线程数
    keepAliveTime,     // 非核心线程存活时间
    TimeUnit.SECONDS,  // 时间单位
    workQueue,         // 任务队列
    threadFactory,     // 线程工厂
    handler            // 拒绝策略
);
```

**参数详解与设计哲学**：

| 参数 | 含义 | 设计考量 |
|---|---|---|
| corePoolSize | 长期保持存活的线程数，即使空闲也不销毁 | CPU 密集型：= CPU 核心数；IO 密集型：= 2× CPU 核心数 |
| maximumPoolSize | 允许的最大线程数（含核心线程） | 防止线程无限创建导致 OOM |
| keepAliveTime | 非核心线程空闲超时后销毁 | 应对流量突刺后回收资源 |
| workQueue | 任务等待队列 | 有界队列防 OOM，无界队列注意堆积 |
| threadFactory | 创建线程的工厂 | 自定义命名（便于调试）、设置优先级、设置 daemon |
| handler | 队列和线程都满时的拒绝策略 | 保护系统不被压垮 |

**任务提交流程**：
```
新任务提交
    ↓
当前线程数 < corePoolSize?
    → Yes: 创建新核心线程执行
    → No ↓
workQueue 未满?
    → Yes: 放入队列等待
    → No ↓
当前线程数 < maximumPoolSize?
    → Yes: 创建非核心线程执行
    → No ↓
执行拒绝策略
```

**四种内置拒绝策略**：
- **AbortPolicy**（默认）：抛出 RejectedExecutionException
- **CallerRunsPolicy**：由提交任务的线程自己执行（天然限流）
- **DiscardPolicy**：静默丢弃
- **DiscardOldestPolicy**：丢弃队列头部最旧的任务

**常见配置模板**：
```
CPU 密集型（如矩阵计算）:
  core = max = CPU 核心数, queue = 小有界队列

IO 密集型（如 HTTP 调用）:
  core = CPU × 2, max = CPU × 4, queue = 中等有界队列

高吞吐消息处理:
  core = max = 固定值, queue = 大有界队列, 拒绝策略 = CallerRuns
```

### Q: synchronized 和基于 AQS 的锁有哪些区别？

两者都是 Java 中实现互斥的机制，但设计层次和能力有本质区别：

| 维度 | synchronized | AQS (ReentrantLock 等) |
|---|---|---|
| **实现层面** | JVM 内置（monitorenter/monitorexit 字节码） | Java 代码层面（CAS + CLH 队列） |
| **锁的获取** | 隐式（进入代码块自动获取） | 显式（手动 lock()/unlock()） |
| **公平性** | 非公平（无法配置） | 支持公平/非公平模式 |
| **可中断性** | 不可中断（线程在等待锁时无法被中断） | lockInterruptibly() 支持中断 |
| **超时获取** | 不支持 | tryLock(timeout) 支持 |
| **条件变量** | 只有一个隐含的 wait set | 支持多个 Condition（精细唤醒） |
| **锁状态查询** | 不支持 | isLocked()、getQueueLength() |
| **性能** | JDK 6+ 优化后（偏向锁→轻量锁→重量锁）轻度竞争接近无锁 | 竞争激烈时队列管理更高效 |
| **死锁检测** | JVM 内置支持 | 需手动实现 |

**synchronized 的锁升级路径**（JDK 6+）：
```
无锁 → 偏向锁（单线程反复进入） → 轻量锁（CAS 竞争） → 重量锁（OS mutex）
              ↓                        ↓                     ↓
        开销: ~1ns               开销: ~20ns          开销: ~10μs（涉及 syscall）
```

**AQS 的内部原理**：
```
核心: volatile int state + CLH 双向队列

获取锁:
  1. CAS 修改 state (0→1): 成功则获取
  2. 失败: 创建 Node 加入 CLH 队列尾部
  3. 自旋或 park 等待前驱节点释放

释放锁:
  1. 修改 state (1→0)
  2. unpark 队列中下一个等待者
```

**选择建议**：
- 简单互斥 + 不需要高级特性 → synchronized（代码简洁、不会忘记释放）
- 需要超时、中断、公平性、多 Condition → ReentrantLock
- 读多写少场景 → ReentrantReadWriteLock
- 高并发计数器 → 不用锁，用 AtomicLong 或 LongAdder

### Q: 如何理解最左前缀原则？联合索引 (a,b,c) 下 a>1 and c=1 的索引使用情况？

**最左前缀原则的本质**：

联合索引 (a,b,c) 在 B+ 树中按 **(a, b, c) 的字典序** 组织数据：
```
B+ 树中数据排列顺序:
  (1, 1, 1) < (1, 1, 2) < (1, 2, 1) < (2, 1, 1) < (2, 1, 3) < (3, 1, 1)
  
  先按 a 排序 → a 相同时按 b 排序 → a,b 都相同时按 c 排序
```

**关键推论**：
- 知道 a 的值/范围 → 可以定位 B+ 树上 a 对应的区域
- 在 a 确定的情况下，b 是有序的 → 可以继续二分
- 但如果 a 是范围查询，则不同 a 值下 b 的全局顺序不确定 → b 无法使用索引

**分析 `a>1 AND c=1`**：

```
索引 (a,b,c) 的 B+ 树:
  a=2: (2,1,1), (2,1,3), (2,2,1), (2,3,2)
  a=3: (3,1,2), (3,2,1), (3,2,3)
  a=5: (5,1,1), (5,2,2)
  
  a>1 可以利用索引做范围扫描 ✓
  但在 a>1 的结果中, c 的值是: 1,3,1,2,2,1,3,1,2 — 无序!
  所以 c=1 无法通过索引过滤 ✗
```

**执行过程**：
1. 利用索引中 a 字段做范围扫描（type = range）
2. 对于每条索引记录，c=1 的过滤有两种处理：
   - **回表过滤**：读取完整行数据后在 server 层过滤
   - **ICP（Index Condition Pushdown）**：MySQL 5.6+ 的优化，在存储引擎层直接用索引中 c 的值过滤（减少回表次数）

**ICP 优化的效果**：
```
无 ICP: 索引扫描 a>1 的所有记录 → 全部回表 → server 层过滤 c=1
有 ICP: 索引扫描 a>1 的记录 → 在索引中检查 c=1 → 只有满足条件的才回表
```
ICP 不改变索引的使用方式（仍然只用 a 字段定位），但减少了不必要的回表 I/O。

**各种查询条件的索引利用情况**：

| 查询条件 | 利用索引字段 | 原因 |
|---|---|---|
| a=1 AND b=2 AND c=3 | a, b, c（全部） | 等值逐级匹配 |
| a=1 AND b>2 AND c=3 | a, b | b 是范围，c 无序 |
| a>1 AND b=2 | a | a 是范围，b 无序 |
| b=2 AND c=3 | 无（全表扫描） | 缺少最左字段 a |
| a=1 AND c=3 | a（跳过 b） | b 不确定时 c 无序，但 ICP 可优化 |

### Q: 线上出现慢查询如何定位和排查？

**系统化排查流程**：

```
Step 1: 发现 → 慢查询日志 / 监控告警
Step 2: 定位 → EXPLAIN 分析执行计划
Step 3: 诊断 → 确认根因（索引/锁/数据量）
Step 4: 优化 → 针对性方案
Step 5: 验证 → 线上灰度观察
```

**Step 1：开启慢查询日志并发现问题**：
```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 1;  -- 超过 1 秒记录
SET GLOBAL log_queries_not_using_indexes = ON;  -- 记录未使用索引的查询

-- 或通过 pt-query-digest 分析慢查询日志
-- pt-query-digest /var/log/mysql/slow.log
```

**Step 2：EXPLAIN 关键指标**：

| EXPLAIN 字段 | 关注点 | 问题信号 |
|---|---|---|
| type | 访问类型 | ALL（全表扫描）、index（全索引扫描） |
| key | 实际使用的索引 | NULL（未使用索引） |
| rows | 预估扫描行数 | 远大于结果集行数 |
| Extra | 额外信息 | Using filesort、Using temporary |
| filtered | 过滤比例 | 过低说明扫描了大量无用行 |

**Step 3：常见根因分析**：

| 根因类别 | 具体原因 | 诊断方法 |
|---|---|---|
| 索引缺失/失效 | 没建索引、函数运算导致索引失效、类型转换 | EXPLAIN type=ALL, key=NULL |
| 大表全扫描 | 数据量大且无合适索引 | rows >> 实际需要的行数 |
| 锁等待 | 行锁/表锁/MDL 锁 | SHOW PROCESSLIST 看 State |
| 复杂 JOIN | 多表 JOIN 无合适索引 | EXPLAIN 中多个 ALL |
| 子查询 | 相关子查询每行执行一次 | Extra: Dependent subquery |
| 排序/分组 | filesort 处理大数据量 | Extra: Using filesort |
| 深分页 | LIMIT 1000000, 10 | 扫描前 100 万行后丢弃 |

**Step 4：优化方案**：
```sql
-- 1. 添加合适索引
ALTER TABLE t ADD INDEX idx_abc (a, b, c);

-- 2. 重写 SQL（避免函数套在索引列上）
-- Bad:  WHERE YEAR(create_time) = 2024
-- Good: WHERE create_time >= '2024-01-01' AND create_time < '2025-01-01'

-- 3. 深分页优化（游标法）
-- Bad:  SELECT * FROM t LIMIT 1000000, 10
-- Good: SELECT * FROM t WHERE id > last_seen_id LIMIT 10

-- 4. 子查询改 JOIN
-- Bad:  SELECT * FROM t WHERE id IN (SELECT id FROM sub WHERE ...)
-- Good: SELECT t.* FROM t JOIN sub ON t.id = sub.id WHERE ...

-- 5. 覆盖索引（避免回表）
-- 创建包含查询所有字段的联合索引
```

### Q: 手撕：二维矩阵行有序列有序，查找某个值？

（编程题）
