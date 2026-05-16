---
title: '阿里巴巴 控股集团 AI Infra 实习 一面'
date: 2026-04-16 12:13:07
categories:
  - [求职面试, 大厂面经]
tags: [AIInfra, 大厂面经, 面经]
---


<!-- more -->

---

### Q: 开发一个MCP（Model Context Protocol）如何评测效果？怎么迭代？

MCP（Model Context Protocol）为LLM提供标准化的外部工具/数据接口。评测和迭代需要从多个维度系统化进行：

**评测维度**：

**1. 功能正确性（最基础）**
- **工具调用成功率**：LLM生成的调用参数能否被MCP服务正确解析和执行
  - 参数类型正确率（如期望int传了string）
  - 参数值有效率（如传了不存在的文件路径）
  - 典型指标：目标>95%成功率
- **返回结果准确性**：MCP返回的数据是否正确、完整、格式规范
- **边界测试**：空输入、超长输入、特殊字符、并发调用

**2. 端到端效果（核心指标）**
- **任务完成率**：集成到Agent后，使用该MCP tool的任务完成率 vs 不使用时的对比
- **选择正确性**：LLM在多个tool中能否正确选择该MCP（tool selection accuracy）
- **多步推理成功率**：需要多次调用MCP协作完成的复杂任务

**3. 性能指标**
- **调用延迟**：P50/P95/P99延迟（通常要求P99 < 3s）
- **并发承载**：最大QPS、在高并发下的延迟退化
- **资源消耗**：内存/CPU占用
- **超时/重试率**：网络抖动下的稳定性

**4. 鲁棒性**
- **异常输入处理**：畸形参数不应导致服务crash
- **错误信息质量**：返回的错误信息是否能帮助LLM修正调用
- **幂等性**：重复调用是否安全
- **降级策略**：依赖服务不可用时的优雅处理

**迭代方式**：

```
收集失败case → 分类分析 → 针对性优化 → A/B测试验证 → 上线
```

**失败case分类与应对**：

| 失败类型 | 原因 | 优化手段 |
|---------|------|---------|
| 工具不被选择 | tool description描述不清 | 优化description，增加usage examples |
| 参数错误 | schema定义不明确 | 完善parameter description/enum约束 |
| 结果解析失败 | 返回格式不符合LLM期望 | 结构化输出+添加解释性字段 |
| 多步逻辑失败 | 中间状态处理不当 | 添加状态查询接口/增加错误恢复逻辑 |
| 超时 | 后端处理慢 | 异步化+进度回调/缓存热点数据 |

**关键迭代实践**：
- 维护**golden test set**：覆盖各种场景的标准测试集，每次修改后回归验证
- **Shadow mode**：新版本先上shadow模式，对比新旧版本在真实流量上的表现
- **Tool description A/B测试**：微调description措辞对LLM选择行为影响巨大（一个词的改变可能影响10%+的选择率）

### Q: 如何对一个Agent系统做链路追踪和可观测性？

Agent系统的非确定性使得可观测性比传统微服务更加关键——你需要理解LLM"为什么"做出某个决策。

**1. 结构化日志（Structured Logging）**

每一步Agent的Decision-Action-Observation需要记录：
```json
{
  "trace_id": "req-abc123",
  "step": 3,
  "timestamp": "2024-01-01T12:00:00Z",
  "phase": "action",
  "model": "gpt-4",
  "input_tokens": 2048,
  "output_tokens": 156,
  "thought": "用户需要查询订单状态，我需要调用order_query工具",
  "tool_called": "order_query",
  "tool_params": {"order_id": "12345"},
  "tool_result": {"status": "shipped", "tracking": "SF123"},
  "latency_ms": 1230,
  "cost_usd": 0.012
}
```

**2. Trace ID与链路串联**
- 为每个用户请求分配唯一Trace ID
- 串联完整推理链路：用户输入 → 意图识别 → 规划 → [工具调用1 → 工具调用2 → ...] → 汇总 → 输出
- 支持层级关系：主Agent调用子Agent时，子Agent的trace作为子span
- 实现：OpenTelemetry集成，支持span/trace标准

**3. 专业平台可视化**

| 平台 | 特点 | 适用场景 |
|------|------|---------|
| LangSmith | LangChain官方，深度集成 | LangChain项目 |
| Phoenix (Arize) | 开源，支持Trace+Eval | 通用Agent系统 |
| Langfuse | 开源自托管，轻量 | 隐私要求高 |
| Weights & Biases Traces | ML实验管理集成 | 研发阶段 |

可视化内容：推理链路图、token消耗热力图、延迟瀑布图、工具调用序列时间线

**4. 指标监控（Metrics）**

| 指标类别 | 具体指标 | 告警阈值 |
|---------|---------|---------|
| 延迟 | 单步延迟、端到端延迟 | P99 > 10s |
| 步数 | 完成任务的平均步数 | > 10步（可能死循环） |
| Token消耗 | 每请求总token数 | > 50K（成本异常） |
| 工具调用 | 调用成功率、每请求调用次数 | 成功率 < 90% |
| 质量 | 用户满意度、任务完成率 | 完成率下降5% |

**5. 异常检测与告警**
- **死循环检测**：同一工具被连续调用>3次且参数相同→强制中断
- **幻觉告警**：调用不存在的工具名、生成无效的API参数→标记并报告
- **成本异常**：单请求token消耗超过阈值→可能是无限推理循环
- **输出质量监控**：通过规则/小模型检测输出是否符合格式约束

**6. 实践建议**
- 开发阶段：所有中间步骤verbose打印，方便debug
- 生产阶段：采样记录（1-10%流量全量trace），异常全量记录
- 回放能力：能够基于trace日志重放完整推理过程（用于排查问题）

### Q: Agent对话记录怎么保存？用什么数据结构？

**数据结构设计**：

**1. 消息列表（最基础，适合简单Agent）**
```python
@dataclass
class Message:
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str       # 消息内容
    name: str = None   # 工具名称（role=tool时）
    tool_calls: List[ToolCall] = None  # 工具调用请求
    metadata: Dict = None  # 时间戳、token数、cost等

# 整个对话就是有序列表
conversation: List[Message] = []
```
- 优点：简单直观、与API格式一致
- 缺点：线性结构不支持分支/回溯

**2. 树形结构（适合探索性Agent）**
```python
@dataclass
class ConversationNode:
    message: Message
    parent: Optional['ConversationNode']
    children: List['ConversationNode']  # 分支对话
    is_active: bool  # 当前活跃路径

class ConversationTree:
    root: ConversationNode
    active_path: List[ConversationNode]  # 当前选定路径
```
- 适用：Tree-of-Thought推理、多方案探索后选择、允许回退到某个节点重试
- 实例：ChatGPT的"regenerate response"本质就是树分支

**3. 带索引的向量存储（适合长期记忆检索）**
```python
class MemoryStore:
    vector_db: VectorDatabase  # embedding检索
    structured_db: Dict        # 结构化数据（用户偏好、实体）
    
    def add(self, message: Message):
        embedding = encode(message.content)
        self.vector_db.insert(embedding, metadata=message)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Message]:
        return self.vector_db.search(encode(query), top_k)
```
- 适用：历史对话很长、需要按语义检索相关历史

**4. 分层存储（生产级Agent的最佳实践）**
```
┌─ 工作记忆（In-context）─────────────────────────────┐
│  最近3-5轮完整对话 + 系统消息                          │
│  存储位置：直接在prompt中                              │
└──────────────────────────────────────────────────────┘
         ↓ 超出窗口时压缩
┌─ 会话记忆（Session Cache）──────────────────────────┐
│  当前会话的完整历史（可能20-50轮）                     │
│  存储位置：Redis / 内存缓存（TTL=会话超时时间）         │
└──────────────────────────────────────────────────────┘
         ↓ 会话结束后持久化
┌─ 长期记忆（Persistent）─────────────────────────────┐
│  历史会话摘要 + 用户画像 + 学到的偏好                   │
│  存储位置：PostgreSQL + Vector DB                     │
└──────────────────────────────────────────────────────┘
```

**实现选择**：

| 存储方案 | 适用场景 | 延迟 | 容量 |
|---------|---------|------|------|
| 内存 List | 单次会话、原型开发 | ~0ms | 受限于RAM |
| Redis | 多实例共享会话状态 | ~1ms | GB级 |
| PostgreSQL | 持久化历史、结构化查询 | ~5ms | TB级 |
| Vector DB (Milvus) | 语义检索历史 | ~10ms | 十亿级向量 |
| 文件系统 (JSON/SQLite) | 简单应用、本地Agent | ~1ms | GB级 |

### Q: 了解哪些Agent开发框架？

**主流Agent框架对比**：

**1. LangChain / LangGraph**
- **定位**：最流行的LLM应用开发框架
- **核心能力**：Chain（链式调用）、Agent（工具调用循环）、RAG Pipeline
- **LangGraph**：LangChain的扩展，支持有状态的多步Agent workflow（基于图的状态机）
- **优点**：生态丰富（集成200+ tools）、文档完善、社区活跃
- **缺点**：抽象层次多导致debug困难、性能开销、版本迭代快不稳定
- **适用**：快速原型开发、标准RAG/Agent场景

**2. CrewAI**
- **定位**：多Agent协作框架
- **核心能力**：定义Agent角色（role+goal+backstory）→ 分配Task → 协作完成
- **优点**：多Agent交互设计直观、支持委托（delegation）、角色扮演效果好
- **缺点**：底层依赖LangChain、定制化有限
- **适用**：需要多角色协作的场景（如"产品经理+工程师+测试"协作开发）

**3. AutoGen（Microsoft）**
- **定位**：多Agent对话框架，支持人机协作
- **核心能力**：Agent间消息传递、人类反馈接入、代码执行沙箱
- **优点**：人类可以随时介入Agent对话、支持group chat模式
- **缺点**：设计偏学术、生产化支持较弱
- **适用**：研究性项目、需要人机协同的场景

**4. Dify**
- **定位**：低代码Agent/RAG平台
- **核心能力**：可视化workflow编排、内置RAG pipeline、API即服务
- **优点**：上手门槛低（拖拽式）、内置向量数据库和模型管理
- **缺点**：复杂逻辑表达受限、深度定制困难
- **适用**：非技术团队快速搭建AI应用、企业内部知识库

**5. Claude Code / OpenAI Agents SDK**
- **定位**：原生tool use + Agent能力
- **核心能力**：
  - 原生Function Calling（不需要额外prompt engineering）
  - 结构化输出（Structured Output）
  - 多工具并行调用
- **优点**：与模型能力最紧密集成、延迟低、可靠性高
- **缺点**：绑定特定模型提供商
- **适用**：对可靠性要求高的生产系统

**6. Semantic Kernel（Microsoft）**
- **定位**：企业级AI编排框架
- **核心能力**：Plugin系统（类似MCP的tool定义）、Memory管理、Planner
- **优点**：C#/.NET/Python/Java多语言支持、企业级特性（认证/授权/审计）
- **缺点**：学习曲线较陡、社区不如LangChain活跃
- **适用**：.NET技术栈企业、需要企业级治理

**选择建议**：

| 需求 | 推荐框架 |
|------|---------|
| 快速原型/POC | LangChain/LangGraph |
| 多Agent协作 | CrewAI / AutoGen |
| 生产部署（可靠性优先） | 直接用SDK (OpenAI/Claude) + 自建 |
| 非技术团队 | Dify |
| 企业级.NET项目 | Semantic Kernel |
| 极致可控性 | 不用框架，自建Agent loop |

### Q: 手撕：最长有效括号？

（编程题）
