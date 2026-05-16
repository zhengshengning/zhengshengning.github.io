---
title: '百度 AI Infra 校招'
date: 2026-04-16 12:11:37
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: etcd和Redis索引实现的区别？为什么有区别？

| 对比 | etcd | Redis |
|------|------|-------|
| **索引结构** | B+ 树（boltdb）+ MVCC | 哈希表 + 跳表 + 多种数据结构 |
| **设计目标** | 强一致性、可靠的配置存储 | 高吞吐、低延迟的缓存/数据库 |
| **一致性** | Raft 协议保证线性一致性 | 异步复制，最终一致性 |
| **持久化** | 每次写入持久化到磁盘（WAL） | 可选 RDB/AOF，默认内存为主 |
| **数据规模** | 建议 <数GB（配置/元数据） | 可达数百 GB（内存限制） |
| **典型操作** | Watch/CAS/Lease/事务 | GET/SET/LPUSH/ZADD/Pub-Sub |

**区别的根本原因**：
- etcd 追求一致性（CP）：Raft 共识协议保证所有节点看到相同数据，代价是写入延迟高（需要多数节点确认）。B+ 树适合范围扫描（如 Watch prefix）。
- Redis 追求性能（AP 倾向）：单线程模型避免锁开销，内存操作保证微秒级延迟。哈希表 O(1) 查找适合缓存场景。

---

### Q: 操作系统的文件管理？

文件系统是 OS 中负责数据持久化组织的核心子系统：

**核心功能**：
- **文件组织**：目录树结构（/root/dir/file），路径名到 inode 的映射
- **存储分配**：将逻辑文件映射到物理磁盘块。策略：连续分配（简单但碎片）、链式分配（无碎片但随机慢）、索引分配（inode 直接/间接指针，ext4 采用）
- **空闲管理**：位图（bitmap，ext4）或空闲链表追踪可用磁盘块
- **权限控制**：Unix 权限位（rwx for user/group/other）+ POSIX ACL
- **缓存**：Page Cache 缓存磁盘数据到内存，大幅减少 IO

**VFS 层**：Linux 虚拟文件系统抽象不同文件系统的统一接口（inode/dentry/super_block），用户态 open/read/write 通过 VFS 路由到具体文件系统实现（ext4/XFS/NFS/procfs）。

---

### Q: Kafka如何实现顺序消费？如何做到消息不丢失？顺序消费如何写文件？

**顺序消费保证**：
Kafka 的顺序性在 **Partition** 粒度内保证。同一 Partition 内消息严格有序（基于 offset）。需要保序的消息必须发到同一 Partition（通过 key hash 路由）。跨 Partition 无顺序保证。

**消息不丢失**（端到端）：
- **Producer**：设置 `acks=all`（等所有 ISR 副本写入确认）、`retries=MAX`、`enable.idempotence=true`（幂等性防重复）
- **Broker**：`replication.factor >= 3`（多副本）、`min.insync.replicas >= 2`（至少 2 个副本同步写入）
- **Consumer**：手动提交 offset（`enable.auto.commit=false`），确保处理完成后再 commit。失败时不提交，下次从旧 offset 重新消费。

**顺序写文件**：
Kafka 采用**日志结构存储**——消息以追加写（append-only）方式顺序写入 Segment 文件（默认 1GB/段）。利用 OS Page Cache 和顺序 IO 的特性实现极高吞吐（GB/s 级别）。磁盘顺序写速度接近内存随机写。

---

### Q: cgroup和namespace原理？

两者是 Linux 容器化的两大内核基础设施：

**Namespace（隔离——看到什么）**：
让进程"以为"自己独占系统资源。内核为每种资源维护独立的命名空间实例。当前支持 8 种：PID（进程号）、Network（网络栈）、Mount（文件系统挂载）、UTS（主机名）、IPC（信号量/消息队列）、User（UID映射）、Cgroup、Time。

通过 `clone()` 传入 CLONE_NEW* 标志创建新命名空间。

**Cgroup（资源管控——能用多少）**：
层级化的进程资源管理机制。通过 cgroupfs（`/sys/fs/cgroup/`）配置：
- CPU：`cpu.cfs_quota_us/cpu.cfs_period_us`（如 100000/100000 = 1 核）、`cpu.shares`（相对权重）
- Memory：`memory.limit_in_bytes`（硬限制，超出 OOM Kill）、`memory.soft_limit_in_bytes`（软限制，回收优先级）
- IO：`blkio.throttle.read_bps_device`（磁盘带宽限制）
- PID：`pids.max`（防 fork bomb）

Cgroup v2（统一层级）已成为主流，相比 v1 简化了多控制器管理。

---

### Q: K8s Pod创建过程？CSI和CNI在哪里调用的？

Pod 从创建到运行的完整流程：

1. **kubectl/API 调用** -> API Server 验证+准入控制 -> Pod spec 写入 etcd
2. **Scheduler** watch 到未调度 Pod -> 根据资源需求/亲和性/污点选择最优 Node -> 更新 Pod 的 `spec.nodeName`
3. **Kubelet**（目标 Node）watch 到分配给自己的 Pod：
   - 调用 **CRI**（Container Runtime Interface）创建 sandbox 容器和应用容器
   - **CNI 调用时机**：sandbox 容器创建后、应用容器启动前。Kubelet 调用 CNI 插件（如 Calico/Cilium）为 Pod 分配 IP、创建 veth pair、配置路由规则
   - **CSI 调用时机**：Pod spec 中声明了 PVC 时。Kubelet 通过 CSI Driver 的 NodeStageVolume/NodePublishVolume 接口挂载存储卷到容器内路径
4. 容器启动 -> readiness probe 通过 -> Pod 变为 Ready

---

### Q: K8s中request和limit的实现原理？

request 和 limit 通过 Linux **Cgroup** 机制实现资源管控：

**Request（请求量）**：
- 作用：调度依据。Scheduler 确保 Node 上所有 Pod 的 request 总和不超过 Node 可分配资源
- CPU 实现：对应 cgroup 的 `cpu.shares`（相对权重）。request=500m -> shares=512，保证在竞争时获得约 0.5 核的时间片
- Memory 实现：不直接对应 cgroup 参数，仅作为调度约束。但影响 OOM score（request 低的 Pod 优先被 Kill）

**Limit（限制量）**：
- 作用：硬性上限，运行时强制执行
- CPU 实现：对应 cgroup 的 `cpu.cfs_quota_us / cpu.cfs_period_us`。limit=2000m -> quota=200000/period=100000（最多用 2 核）。超限被 throttle（不是 Kill）
- Memory 实现：对应 cgroup 的 `memory.limit_in_bytes`。超限触发 OOM Kill（容器被杀重启）

**最佳实践**：CPU 可以不设 limit（超额使用不伤害）；Memory 必须设 limit（超额使用导致 OOM 影响稳定性）。request 设为实际使用量的 P95，limit 设为 P99 或 2x request。
