---
title: 从零构建Claude多智能体协作系统
date: 2026-03-11 10:00:00
categories: AI编程
tags: [Claude, 多智能体, Agent, opencode, 智能体协作]
---

本文基于 `precision-alignment-agent` 项目实践经验，详细介绍如何从零构建 opencode/claude 多智能体协作系统，涵盖核心概念、技术栈选择、项目结构设计、智能体定义、工作流设计等完整内容。

**可以直接用这篇文章作为大模型的输入，让大模型参考这篇文章生成一个多智能体系统！**

<!-- more -->

## 目录

1. [核心概念理解](#1-核心概念理解)
2. [技术栈选择](#2-技术栈选择)
3. [项目结构设计](#3-项目结构设计)
4. [智能体定义方法](#4-智能体定义方法)
5. [工作流设计](#5-工作流设计)
6. [命令行工具搭建](#6-命令行工具搭建)
7. [知识库系统](#7-知识库系统)
8. [逐步实现指南](#8-逐步实现指南)
9. [调试与迭代](#9-调试与迭代)
10. [最佳实践总结](#10-最佳实践总结)

---

## 1. 核心概念理解

### 1.1 什么是多智能体系统？

```
┌─────────────────────────────────────────────────────────────────┐
│                     多智能体协作模型                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   传统方式:  人类 ←→ 单一 AI                                       │
│                                                                 │
│   多智能体:  人类 ←→ 编排者(Orchestrator)                          │
│                         ├── 专家智能体 A (只读分析)                │
│                         ├── 专家智能体 B (代码修改)                │
│                         ├── 专家智能体 C (测试执行)                │
│                         └── 专家智能体 D (最终审查)                │
│                                                                 │
│   核心优势:                                                      │
│     • 职责分离 → 减少错误                                         │
│     • 专业化 → 每个智能体专注一件事                                 │
│     • 可并行 → 独立任务同时执行                                    │
│     • 可控制 → 精细的权限管理                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 扁平编排 vs 层级编排

| 模式 | 特点 | 适用场景 |
|------|------|---------|
| **扁平编排** | 单一主智能体直接调度所有子智能体 | 流程清晰、阶段明确的任务 |
| **层级编排** | 多层中间智能体，形成树状结构 | 复杂决策、需要多层抽象的任务 |

**建议初学者从扁平编排开始**，更容易理解和调试。

### 1.3 智能体的核心属性

每个智能体需要定义：

```yaml
name: 智能体名称
description: 职责描述
role: primary | subagent  # 主智能体还是子智能体

model:
  tier: reasoning | coding  # 推理型还是编码型
  temperature: 0.0-1.0      # 创造性程度

capabilities:              # 能力/权限列表
  - read                   # 读取文件
  - write                  # 写入文件
  - bash                   # 执行命令
  - delegate: [...]        # 可调用的子智能体

skills: []                 # 可使用的技能
```

---

## 2. 技术栈选择

### 2.1 核心工具

| 工具 | 用途 | 安装方式 |
|------|------|---------|
| **just** | 任务运行器（类似 Make） | `curl -fsSL https://just.systems/install.sh \| bash` |
| **uv** | Python 包/环境管理 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **bun** | JavaScript 运行时 | `curl -fsSL https://bun.sh/install \| bash` |
| **agent-caster** | 智能体定义适配器 | `uvx agent-caster` |

### 2.2 AI Coding Agent 平台

| 平台 | 特点 | 配置方式 |
|------|------|---------|
| **opencode-ai** | 开源、可定制 | `bun install -g opencode-ai` |
| **Claude Code** | Anthropic 官方 | 使用 `.claude/` 配置 |
| **Cursor** | IDE 集成 | 使用 `.cursor/` 配置 |

### 2.3 推荐的项目依赖结构

```toml
# refit.toml - 智能体适配器配置
[project]
agents_dir = "roles"   # 智能体定义的源目录

[targets.opencode]             # 目标平台: opencode
enabled = true
output_dir = "."

[targets.opencode.model_map]   # 模型映射
reasoning = "claude-opus"      # 推理任务用的模型
coding = "claude-sonnet"       # 编码任务用的模型

[targets.claude]               # 目标平台: claude
enabled = true
output_dir = "."
```

---

## 3. 项目结构设计

### 3.1 推荐的目录结构

```
my-agent-project/
├── roles/                      # 【源】智能体定义（平台无关）
│   ├── main-orchestrator.md    # 主编排智能体
│   ├── analyzer.md             # 分析智能体
│   ├── executor.md             # 执行智能体
│   └── reviewer.md             # 审查智能体
│
├── .opencode/                  # 【生成】opencode 平台配置
│   └── agents/*.md             # 由 agent-caster 自动生成
│
├── .claude/                    # 【生成】claude 平台配置
│   └── agents/*.md
│
├── knowledge/                  # 【人工】领域知识库
│   ├── commons/                # 通用知识
│   └── domain-specific/        # 领域特定知识
│
├── skills/                     # 【可选】自定义技能
│   └── my-skill/
│       └── SKILL.md
│
├── .gitignore                  # 忽略生成文件
├── Justfile                    # 命令定义
├── refit.toml                  # 适配器配置
├── README.md                   # 项目说明
└── install.sh                  # 一键安装脚本
```

### 3.2 `.gitignore` 配置

```gitignore
# 生成的平台配置（不提交）
.opencode/
.claude/
.cursor/
opencode.json

# 运行时数据
.pda/
.paa/
*.log

# 编辑器
.vscode/
.idea/
```

---

## 4. 智能体定义方法

### 4.1 智能体定义文件格式

每个智能体使用 **YAML frontmatter + Markdown prompt** 格式：

```markdown
---
name: my-analyzer
description: >
  分析代码并生成报告的智能体。只读权限，不修改代码。
role: subagent

model:
  tier: reasoning
  temperature: 0.1

capabilities:
  - read
  - write-report
  - web-read
---

# My Analyzer

你是一个代码分析智能体，专门负责...

## 输入要求

- **target_path**: 要分析的代码路径
- **analysis_type**: 分析类型

## 输出格式

你的报告需要包含：
1. **概述**: 简要描述...
2. **详细分析**: ...
3. **建议**: ...

## 约束

- 只读：不修改任何代码
- 报告写入: `.sessions/{task_id}/analyzer/`
```

### 4.2 主编排智能体模板

```markdown
---
name: main-orchestrator
description: >
  主编排智能体。负责计划、协调和决策，将具体工作委托给子智能体。
role: primary

model:
  tier: coding
  temperature: 0.2

capabilities:
  - read
  - write-report
  - delegate:
      - analyzer
      - executor
      - reviewer
---

# Main Orchestrator

你是主编排智能体。你的职责是**计划、协调和委托**，而不是直接执行。

## 架构

你 (Orchestrator)
  ├── @analyzer    分析任务（只读）
  ├── @executor    执行任务（写）
  └── @reviewer    审查任务（只读）

## 工作流程

### Phase 1: 分析
1. 调用 @analyzer 分析目标
2. 读取分析报告

### Phase 2: 计划
1. 基于分析报告制定计划
2. 定义成功标准

### Phase 3: 执行循环（最多 N 次迭代）
1. 调用 @executor 执行具体任务
2. 评估结果
3. 若不满足，调整后重试

### Phase 4: 审查
1. 调用 @reviewer 进行最终审查
2. 生成报告或输出

## 委托规则

| 动作 | 委托给 |
|------|--------|
| 分析代码/数据 | @analyzer |
| 修改文件/执行操作 | @executor |
| 最终验证 | @reviewer |

## 你可以直接做的事

- 读取会话文件做决策
- 写入计划和上下文文件
- 评估子智能体结果并决定下一步
```

### 4.3 子智能体模板

```markdown
---
name: executor
description: >
  执行具体任务的智能体。可以修改文件、运行命令。
role: subagent

model:
  tier: coding
  temperature: 0.1

capabilities:
  - read
  - write
  - bash:
      - "make*"
      - "npm*"
      - "git add*"
      - "git commit*"
---

# Executor

执行 Orchestrator 分配的具体任务。

## 职责范围

- **可以做**: 修改代码、运行构建、执行测试
- **不可以做**: 推送代码、创建 PR（由 Reviewer 处理）

## 工作流程

1. **接收指令**: Orchestrator 会告诉你具体要修改什么
2. **执行修改**: 精确修改指定文件
3. **验证**: 运行基础测试确认修改有效
4. **报告**: 将结果写入 `.sessions/{task_id}/executor/`

## 约束

- 每次只处理一个明确的任务
- 修改前先确认理解正确
- 遇到复杂问题时报告而非猜测
```

---

## 5. 工作流设计

### 5.1 设计工作流的原则

```
┌─────────────────────────────────────────────────────────────────┐
│                      工作流设计原则                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 单一职责                                                     │
│     每个智能体只做一类事情                                        │
│                                                                 │
│  2. 最小权限                                                     │
│     只给必要的权限（只读 vs 可写 vs 可执行）                       │
│                                                                 │
│  3. 明确边界                                                     │
│     清晰定义什么可以做、什么不可以做                               │
│                                                                 │
│  4. 可验证                                                       │
│     每个阶段都有明确的成功/失败标准                               │
│                                                                 │
│  5. 可回滚                                                       │
│     失败时能够恢复到已知良好状态                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 典型工作流模式

**模式 A：线性流水线**
```
Input → Analyze → Plan → Execute → Validate → Output
```

**模式 B：迭代循环**
```
Input → Analyze → Plan → [Execute → Validate → Assess]* → Output
                              ↑_________↓ (失败则重试)
```

**模式 C：并行分析**
```
              ┌→ Analyzer A ─┐
Input → Fork ─┼→ Analyzer B ─┼→ Merge → Plan → Execute → Output
              └→ Analyzer C ─┘
```

### 5.3 实际案例

**Debug Agent 工作流：**
```
Phase 1 (顺序)    Phase 2           Phase 3 (循环)        Phase 4
┌──────────┐    ┌─────────┐    ┌───────────────────┐   ┌──────────┐
│Reproducer│ →  │ Analyzer│ →  │ Fixer             │ → │Documenter│
└──────────┘    └─────────┘    │    ↓              │   └──────────┘
                               │ Tester            │
                               │    ↓ (循环≤5次)   │
                               └───────────────────┘
```

---

## 6. 命令行工具搭建

### 6.1 Justfile 基础结构

```just
# Justfile - 任务定义文件

# 默认任务：显示帮助
default:
    @just --list

# ════════════════════════════════════════════════════════════════
# 安装与设置
# ════════════════════════════════════════════════════════════════

# 安装所有依赖
setup:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    curl -fsSL https://bun.sh/install | bash
    bun install -g opencode-ai
    @echo "Setup complete!"

# ════════════════════════════════════════════════════════════════
# 智能体管理
# ════════════════════════════════════════════════════════════════

# 从源定义生成平台配置
adapt:
    uvx agent-caster cast

# ════════════════════════════════════════════════════════════════
# 工作流启动
# ════════════════════════════════════════════════════════════════

# 启动主工作流
start task_name additional_info="":
    #!/usr/bin/env bash
    set -euo pipefail
    
    just adapt
    
    AGENT="main-orchestrator"
    PROMPT="Start workflow for {{ task_name }}. Additional: {{ additional_info }}"
    
    opencode --agent "$AGENT" --prompt "$PROMPT"

# ════════════════════════════════════════════════════════════════
# Agentic Commands - 供智能体使用的命令
# ════════════════════════════════════════════════════════════════

# 运行测试
agentic-run-tests TEST_PATH:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running tests: {{ TEST_PATH }}"
    # 你的测试命令

# 构建项目
agentic-build:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building project..."
    # 你的构建命令
```

### 6.2 命令命名约定

| 前缀 | 用途 | 示例 |
|------|------|------|
| 无前缀 | 人类使用的命令 | `just setup`, `just start` |
| `agentic-` | 智能体使用的命令 | `just agentic-build` |
| `_` | 内部辅助命令 | `just _helper` |

---

## 7. 知识库系统

### 7.1 知识库设计

```
knowledge/
├── README.md                  # 知识库说明
├── commons/                   # 通用知识
│   ├── best-practices.md      # 最佳实践
│   └── common-patterns.md     # 常见模式
└── domain/                    # 领域特定知识
    ├── topic-a.md
    └── topic-b.md
```

### 7.2 知识文件格式

```markdown
---
title: 知识点标题
category: commons | domain
created_at: 2026-01-01
updated_at: 2026-03-09
tags: [tag1, tag2]
target_agents: [analyzer, executor]
summary: 一句话概述
---

## 概述

详细说明...

## 使用场景

何时使用...

## 示例

代码或配置示例...

## 相关知识

- 链接到其他文档
```

### 7.3 两种知识库的区别

| 类型 | 目录 | 权限 | 内容 |
|------|------|------|------|
| **人工知识库** | `knowledge/` | 智能体只读 | 最佳实践、通用规则 |
| **自动知识库** | `.sessions/` | 智能体读写 | 任务执行记录、中间结果 |

---

## 8. 逐步实现指南

### 阶段 1：最小可行版本（1-2天）

**目标**：让一个简单的双智能体系统跑起来

```
步骤 1: 创建项目结构
  └── mkdir -p my-agent-project/roles
  └── cd my-agent-project

步骤 2: 定义两个智能体
  └── 主智能体 (orchestrator.md)
  └── 执行智能体 (executor.md)

步骤 3: 配置适配器
  └── 创建 refit.toml

步骤 4: 编写 Justfile
  └── setup, adapt, start 命令

步骤 5: 测试运行
  └── just adapt
  └── just start "test task"
```

### 阶段 2：增加专业化智能体（3-5天）

**目标**：根据你的领域添加专业智能体

```
步骤 1: 分析你的工作流需要哪些专业角色
  └── 分析/研究型？执行/构建型？验证/审查型？

步骤 2: 为每个角色创建智能体定义
  └── 明确职责、权限、输入输出

步骤 3: 更新主编排智能体
  └── 添加委托规则

步骤 4: 添加对应的 agentic-* 命令

步骤 5: 迭代测试
```

### 阶段 3：知识库与技能（1周+）

**目标**：构建领域知识和可复用技能

```
步骤 1: 整理领域知识
  └── 创建 knowledge/ 目录结构
  └── 编写核心知识文档

步骤 2: 创建自定义技能（可选）
  └── skills/my-skill/SKILL.md

步骤 3: 在智能体定义中引用

步骤 4: 测试知识查询效果
```

---

## 9. 调试与迭代

### 9.1 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 智能体不执行任务 | prompt 不够明确 | 添加具体示例 |
| 子智能体未被调用 | 委托规则不清晰 | 明确列出委托表 |
| 权限错误 | capabilities 配置不足 | 添加需要的权限 |
| 工作流中断 | 缺少错误处理 | 添加失败回退逻辑 |

### 9.2 日志与追踪

```bash
# 建议在工作流中生成会话报告
.sessions/
└── {task_id}/
    ├── context.md           # 任务上下文
    ├── orchestrator/        # 主智能体日志
    ├── analyzer/            # 分析智能体报告
    ├── executor/            # 执行智能体报告
    └── reviewer/            # 审查智能体报告
```

### 9.3 迭代改进策略

```
1. 先跑通 → 让工作流能完整执行
2. 再改准 → 提高智能体输出质量
3. 后优化 → 减少迭代次数、提升效率
```

---

## 10. 最佳实践总结

### 应该做的

1. **从简单开始** - 先用 2-3 个智能体验证概念
2. **明确职责边界** - 每个智能体只做一件事
3. **最小权限原则** - 只给必要的 capabilities
4. **充分的 prompt** - 包含示例、约束、输出格式
5. **生成会话报告** - 便于调试和审计
6. **版本控制智能体定义** - 源文件提交，生成文件忽略

### 应该避免的

1. **一个智能体做所有事** - 失去了多智能体的意义
2. **模糊的指令** - "优化代码" vs "将此函数的时间复杂度从 O(n²) 改为 O(n)"
3. **缺少验证环节** - 执行后必须有检查
4. **手动编辑生成文件** - 会被覆盖
5. **忽略错误处理** - 定义失败时的行为

---

## 快速参考卡片

### 创建新项目

```bash
mkdir my-agent && cd my-agent
mkdir -p roles knowledge/commons

# 创建配置文件
cat > refit.toml << 'EOF'
[project]
agents_dir = "roles"

[targets.opencode]
enabled = true
output_dir = "."
EOF

# 创建 Justfile
cat > Justfile << 'EOF'
default:
    @just --list

adapt:
    uvx agent-caster cast

start task:
    just adapt
    opencode --agent "orchestrator" --prompt "{{ task }}"
EOF
```

### 智能体定义速查

```yaml
# 主智能体
role: primary
capabilities: [read, write-report, delegate: [...]]

# 只读子智能体
role: subagent
capabilities: [read, write-report]

# 可写子智能体
role: subagent
capabilities: [read, write]

# 可执行子智能体
role: subagent
capabilities: [read, write-report, bash: [...]]
```

---

*文档生成时间：2026-03-09*
