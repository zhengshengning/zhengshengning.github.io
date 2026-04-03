---
title: ClaudeCode使用指南
date: 2026-02-09 16:20:00
categories:
  - [路飞玩AI, AI编程]
tags: [AI编程, ClaudeCode, VS Code, 代码助手]
---

Claude Code 是 Anthropic 推出的**命令行 AI 编程助手**——你可以把它理解成一个住在终端里的"AI 程序员搭档"，你用自然语言告诉它想做什么，它就能直接读代码、改代码、跑命令，帮你完成从写代码到调试的整个流程。

<!-- more -->

## ⚙️ 安装

Claude Code 以 npm 包的形式分发，需要 Node.js 18+ 环境：

```bash
# 全局安装
npm install -g @anthropic-ai/claude-code

# 验证安装
claude --version
```

安装后在终端进入你的项目目录，直接运行 `claude` 即可启动交互式对话。

## ⚙️ 配置

### API 密钥

Claude Code 需要 Anthropic API 密钥。首次运行时会引导你完成认证，也可以通过环境变量提前配置：

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 模型选择

默认使用 Claude Sonnet，可以通过 `/model` 命令切换模型：

```
/model claude-sonnet-4-20250514
/model claude-opus-4-20250514
```

### 项目级配置

在项目根目录创建 `CLAUDE.md` 文件来为 Claude Code 提供项目上下文——这相当于给你的 AI 搭档一份"项目说明书"，让它更好地理解项目的技术栈、代码规范和注意事项：

```markdown
# 项目说明

这是一个 Python 后端项目，使用 Flask 框架。

## 💻 代码规范
- 使用 type hints
- 函数必须有 docstring
- 测试使用 pytest

## 🏗️ 项目结构
- src/ - 源代码
- tests/ - 测试代码
```

## ✨ 核心功能

### 代码生成与编辑

直接用自然语言描述需求，Claude Code 会自动读取相关文件、生成或修改代码：

```
> 给 user_service.py 添加一个分页查询用户的方法
> 把这个同步函数改成 async 版本
> 创建一个 REST API 的 CRUD 模板
```

### 代码理解与解释

Claude Code 能阅读整个代码库，帮你理解复杂逻辑：

```
> 解释一下 auth 模块的登录流程
> 这个递归函数的时间复杂度是多少？
> 梳理一下请求从 controller 到 database 的完整链路
```

### Bug 诊断与修复

```
> 这个测试为什么失败了？帮我修复
> 分析一下为什么这个 API 会返回 500 错误
> 检查这段代码有没有内存泄漏的风险
```

### 测试生成

```
> 为 UserService 类生成单元测试
> 补充边界条件的测试用例
> 给这个 API endpoint 写集成测试
```

### Git 操作

Claude Code 可以直接帮你完成 Git 工作流：

```
> 把当前的改动提交，写一个合适的 commit message
> 创建一个 PR，描述这次的功能改动
> 帮我 review 一下 PR #42 的代码
```

## ⌨️ 常用命令

| 命令 | 功能 |
|------|------|
| `claude` | 启动交互式对话 |
| `claude "问题"` | 单次提问模式 |
| `claude -p "问题"` | 非交互式（管道友好）模式 |
| `/model` | 切换模型 |
| `/compact` | 压缩上下文，释放 token 空间 |
| `/clear` | 清空对话历史 |
| `/cost` | 查看当前会话的 token 消耗 |
| `/help` | 查看所有可用命令 |
| `Esc` | 中断当前操作 |

## 🔌 MCP：给 AI 插上"外挂"

**MCP（Model Context Protocol）** 是让 AI 能"插上各种外挂"的标准接口——就像 USB 让电脑能连接各种外设一样，MCP 让 Claude Code 能连接数据库、调用 API、操作第三方服务。

通过配置 MCP Server，你可以让 Claude Code 获得额外的能力，比如：

- 查询 PostgreSQL 数据库
- 调用 Jira / Linear 等项目管理工具
- 读写 Notion 文档
- 操作 AWS / GCP 等云服务

配置方式（在 `.claude/settings.json` 中添加）：

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

## 🪝 Hooks：自动化你的工作流

**Hooks** 是在特定事件发生时自动执行的脚本——就像"门铃响了自动开灯"一样，当 Claude Code 执行某些操作时会自动触发你预设的脚本。

常见的 Hook 使用场景：

- 每次编辑文件后自动运行 linter
- 提交代码前自动跑测试
- 通知你 Claude Code 完成了耗时任务

在 `.claude/settings.json` 中配置：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "npx prettier --write $CLAUDE_FILE_PATH"
      }
    ]
  }
}
```

## 🎓 Skills：教 Claude 新本领

**Skills** 是教 Claude Code 新本领的"技能卡"——装上就能解锁新能力。Skills 本质上是 Markdown 文件，定义了 Claude Code 在特定场景下应该如何操作。

你可以通过斜杠命令 `/` 来触发已安装的 Skill，比如 `/commit` 会按照预定义的规范帮你生成提交信息。

## 🖥️ IDE 集成

Claude Code 除了在终端独立使用，还可以集成到 VS Code 和 JetBrains IDE 中：

### VS Code

安装官方扩展后，可以在 VS Code 内直接使用 Claude Code 的全部功能，享受与编辑器的深度集成（如内联 diff、文件跳转等）。

### JetBrains

JetBrains 插件同样提供了集成体验，支持 IntelliJ IDEA、PyCharm、WebStorm 等全系列 IDE。

## 💡 实用技巧

1. **善用 CLAUDE.md**：把项目的关键信息写在 `CLAUDE.md` 里，Claude Code 每次启动都会自动读取，省去反复解释项目背景的麻烦。

2. **用 `/compact` 管理上下文**：长对话中 token 会逐渐耗尽，及时使用 `/compact` 压缩上下文可以节省费用并保持对话质量。

3. **管道组合**：Claude Code 支持 Unix 管道，可以与其他命令行工具组合使用：
   ```bash
   cat error.log | claude -p "分析这个错误日志，找出根因"
   git diff | claude -p "review 这些改动，指出潜在问题"
   ```

4. **权限控制**：Claude Code 在执行文件修改、命令执行等操作前会请求你的确认，你可以根据需要设置自动批准规则来提升效率。

5. **多文件操作**：Claude Code 可以同时理解和修改多个文件，适合跨模块的重构任务。

## 🎯 检验标准与进阶方向

### 自我检验清单

学完本文后，你应该能做到以下几点：

- [ ] 能独立安装 Claude Code 并完成 API 密钥配置
- [ ] 能使用自然语言指令让 Claude Code 生成、修改和解释代码
- [ ] 能编写 `CLAUDE.md` 文件为项目提供上下文信息
- [ ] 能理解 MCP 的作用并为 Claude Code 配置 MCP Server
- [ ] 能配置 Hooks 实现文件编辑后自动格式化等自动化流程
- [ ] 能使用管道模式将 Claude Code 与其他命令行工具组合使用
- [ ] 能通过 `/compact`、`/model` 等命令管理对话上下文和模型选择
- [ ] 能使用 Claude Code 完成 Git 提交、PR 创建等版本控制操作

### 进阶方向

| 方向 | 说明 | 推荐资料 |
|------|------|----------|
| MCP Server 开发 | 自己编写 MCP Server，让 Claude Code 连接内部系统 | [MCP 官方文档](https://modelcontextprotocol.io/introduction) |
| 自定义 Slash Commands | 创建团队专属的 Skills，标准化工作流 | [Claude Code Skills 文档](https://docs.anthropic.com/en/docs/claude-code/skills) |
| CI/CD 集成 | 在 GitHub Actions 等流水线中使用 Claude Code 做自动化 code review | [Claude Code GitHub Actions](https://docs.anthropic.com/en/docs/claude-code/github-actions) |
| Hooks 高级用法 | 结合 Pre/Post Hook 实现复杂的自动化质量保障流程 | [Claude Code Hooks 文档](https://docs.anthropic.com/en/docs/claude-code/hooks) |
| 多 Agent 协作 | 使用 Claude Code 的 sub-agent 能力并行处理复杂任务 | [Claude Code Agent 文档](https://docs.anthropic.com/en/docs/claude-code/sub-agents) |

## 📚 参考资料

- [Claude Code 官方文档](https://docs.anthropic.com/en/docs/claude-code/overview) - 最权威的使用指南，涵盖安装、配置、功能详解
- [Claude Code GitHub 仓库](https://github.com/anthropics/claude-code) - 源码、Issue 讨论与最新更新
- [Anthropic API 文档](https://docs.anthropic.com/en/api/getting-started) - API 密钥申请与管理
- [Model Context Protocol 官网](https://modelcontextprotocol.io/) - MCP 协议规范与 Server 生态
- [Claude Code VS Code 扩展](https://marketplace.visualstudio.com/items?itemName=anthropic.claude-code) - VS Code 集成插件
- [Anthropic 官方博客](https://www.anthropic.com/engineering) - Claude Code 最新功能与最佳实践分享