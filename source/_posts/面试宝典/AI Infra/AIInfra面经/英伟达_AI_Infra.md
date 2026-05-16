---
title: '英伟达 AI Infra'
date: 2026-04-16 12:18:32
categories:
  - [求职面试, 知名科技公司面经]
tags: [AIInfra, 算子优化, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 优化CUDA内核时通常从哪些方面入手？

CUDA内核优化是一个**Profiling驱动**的迭代过程，遵循"测量→分析→优化→验证"循环：

**Step 1：Profiling定位瓶颈**

使用Nsight Compute（NCU）分析kernel是compute-bound还是memory-bound：
- 查看**Compute Throughput**（实际FLOPS / 峰值FLOPS）和**Memory Throughput**（实际带宽 / 峰值带宽）。
- 如果Memory Throughput高但Compute低 → Memory-bound，优先优化访存。
- 如果Compute Throughput高但Memory低 → Compute-bound，优先优化计算。
- Roofline图直观显示kernel落在哪个区域。

**Step 2：针对性优化策略**

**Memory-bound kernel优化**：
- 合并全局内存访问：确保warp内线程访问连续地址（coalesced），一次128字节事务完成。
- 向量化读写：float4一次搬运16字节，减少事务数和指令数。
- 共享内存缓存：将热数据从HBM（400 cycle延迟）搬到Shared Memory（20-30 cycle），多次复用。
- 算子融合：消除中间tensor的HBM写回→读取开销。
- 只读缓存路径：`__ldg()` / `const __restrict__` 利用L1只读缓存。

**Compute-bound kernel优化**：
- Tensor Core利用：FP16/BF16/INT8/FP8矩阵乘，吞吐是CUDA Core的8-16倍。
- 减少Warp Divergence：保证warp内线程走统一路径。
- 指令级并行（ILP）：循环展开、多累加器，让独立指令并行执行。
- 快速数学函数：`__expf()` 牺牲精度换速度（误差<2 ULP）。

**Step 3：并行度调优**
- 提高Occupancy：但注意Occupancy不是越高越好，有时更多寄存器/线程带来的数据复用超过并行度收益。
- Block大小选择：通常128或256线程。太小浪费warp槽，太大可能受资源限制。

**Step 4：延迟隐藏**
- Double Buffering：cp.async加载下一批数据，同时计算当前数据。
- 增加每线程工作量：grid-stride loop让每线程处理多个元素。

**Step 5：验证**
- 性能对比（speedup vs baseline）。
- 精度验证（与参考实现对比误差）。
- 不同输入规模的robustness测试。

### Q: 手撕：时钟指针夹角计算？

（编程题）

### Q: 手撕：多线程顺序打印？

（编程题）

### Q: 如果时钟要支持毫秒级精度，算法该如何调整？

核心算法不变——仍然是分别计算时针和分针的绝对角度再取差值，只需**扩展输入精度和时间基准**：

**精确的角速度**：
- 分针：每60秒转一圈（360度），即每毫秒转 360 / (60×1000) = 0.006度/ms
- 时针：每12小时转一圈，即每毫秒转 360 / (12×60×60×1000) = 0.0000083333度/ms

**算法步骤**：
1. 将输入时间解析为距12:00:00.000的总毫秒数 `T_ms = h*3600000 + m*60000 + s*1000 + ms`
2. 分针角度 = T_ms × 0.006 (对360取模)
3. 时针角度 = T_ms × 0.0000083333... (对360取模)
4. 夹角 = |分针角度 - 时针角度|
5. 如果夹角 > 180，取 360 - 夹角（取较小角）

**精度注意事项**：
- 必须使用 `double` 类型避免float的精度不足（float只有约7位有效数字，计算12小时内的毫秒级角度会丢失精度）。
- 或者使用整数运算：将角度乘以一个大因子（如3600000）避免浮点，最后再除回来。

### Q: 多线程方案中，如果某个线程异常退出，如何保证系统不卡死？

线程异常退出如果持有锁或其他同步原语的"占有权"，会导致其他线程永久等待（死锁/活锁），需要多层防护：

**1. 超时机制（最基本的防护）**：
```cpp
// 不要无限等待，使用带超时的等待
auto status = sem.wait_for(std::chrono::seconds(5));
if (status == std::cv_status::timeout) {
    // 超时处理：跳过当前线程、报告异常、触发恢复
}
```
- `sem_timedwait`（POSIX）、`std::condition_variable::wait_for`（C++11）。
- 超时后进入异常处理路径而非永久阻塞。

**2. Watchdog看门狗线程**：
- 每个工作线程定期更新心跳计数器（原子变量）。
- 独立的Watchdog线程定期检查所有心跳——如果某线程心跳长时间未更新，判定其异常退出。
- Watchdog触发恢复动作：释放异常线程持有的资源、重启替代线程、通知上层。

**3. 异常捕获与资源释放**：
```cpp
void worker() {
    try {
        // 使用RAII guard确保资源释放
        std::lock_guard<std::mutex> lock(mtx);
        // ... 工作逻辑
    } catch (...) {
        // 异常时RAII自动释放锁
        // 额外清理：通知其他线程、设置错误标志
        error_flag.store(true);
        cv.notify_all();  // 唤醒所有等待者
    }
}
```
- RAII是关键：`lock_guard`/`unique_lock` 保证即使异常也能释放锁。

**4. 健康检查与自动重启**：
- 使用 `pthread_kill(thread, 0)` 检查线程是否存活（0号信号不杀线程，只检查存在性）。
- 检测到退出后创建替代线程接管任务。

**5. 优雅降级设计**：
- 系统设计为N-1个线程也能工作的模式。
- 使用任务队列而非固定线程分配：线程异常退出后其未完成任务留在队列中，由其他线程pickup。
- 生产者-消费者模式天然具备这种容错能力。
