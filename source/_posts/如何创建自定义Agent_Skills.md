---
title: 如何创建自定义Agent Skills
date: 2026-02-12 10:55:00
categories: AI编程
tags: [AI编程, Agent Skills, Claude]
---

Custom Skills（自定义技能）让你可以通过特定于你的组织或个人工作方式的专业知识和工作流来增强 Claude。本文介绍了如何创建、构建和测试你自己的 Skills。

<!-- more -->

Skills 可以简单到只有几行指令，也可以复杂到包含可执行代码的多文件包。最好的 Skills：

*   解决特定的、可重复的任务
*   有 Claude 可以遵循的清晰指令
*   在有帮助时包含示例
*   定义了何时应该使用它们
*   专注于一个工作流，而不是试图做所有事情

## 创建 Skill.md 文件

每个 Skill 由一个目录组成，其中至少包含一个 `Skill.md` 文件，这是 Skill 的核心。该文件必须以 YAML frontmatter 开头，以保存 `name` 和 `description` 字段，这是必需的元数据。它还可以包含其他元数据、给 Claude 的指令或参考文件、可执行脚本或工具。

### 必需的元数据字段

`name`: 你的 Skill 的易于理解的名称（最多 64 个字符）

*   示例: Brand Guidelines

`description`: 对 Skill 做什么以及何时使用的清晰描述。

*   这至关重要——Claude 使用此描述来确定何时调用你的 Skill（最多 200 个字符）。
*   示例: Apply Acme Corp brand guidelines to presentations and documents, including official colors, fonts, and logo usage.

### 可选的元数据字段

`dependencies`: 你的 Skill 所需的软件包。

*   示例: python>=3.8, pandas>=1.5.0

`Skill.md` 文件中的元数据作为渐进式披露系统的第一层，提供足够的信息让 Claude 知道何时应该使用该 Skill，而无需加载所有内容。

### Markdown Body

Markdown body 是元数据之后的第二层细节，因此如果需要在读取元数据后，Claude 将访问此内容。根据你的任务，Claude 可以访问 `Skill.md` 文件并使用该 Skill。

### 示例 Skill.md

Brand Guidelines Skill

```markdown
## Metadata
name: Brand Guidelines
description: Apply Acme Corp brand guidelines to all presentations and documents
## Overview
This Skill provides Acme Corp's official brand guidelines for creating consistent, professional materials. When creating presentations, documents, or marketing materials, apply these standards to ensure all outputs match Acme's visual identity. Claude should reference these guidelines whenever creating external-facing materials or documents that represent Acme Corp.
## Brand Colors
Our official brand colors are:
- Primary: #FF6B35 (Coral)
- Secondary: #004E89 (Navy Blue)
- Accent: #F7B801 (Gold)
- Neutral: #2E2E2E (Charcoal)
## Typography
Headers: Montserrat Bold
Body text: Open Sans Regular
Size guidelines:
- H1: 32pt
- H2: 24pt
- Body: 11pt
## Logo Usage
Always use the full-color logo on light backgrounds. Use the white logo on dark backgrounds. Maintain minimum spacing of 0.5 inches around the logo.
## When to Apply
Apply these guidelines whenever creating:
- PowerPoint presentations
- Word documents for external sharing
- Marketing materials
- Reports for clients
## Resources
See the resources folder for logo files and font downloads.
```

## 添加 Resources

如果你有太多信息无法添加到单个 [Skill.md](http://skill.md) 文件中（例如，仅适用于特定场景的部分），你可以通过在你的 Skill 目录中添加文件来添加更多内容。例如，将包含补充和参考信息的 `REFERENCE.md` 文件添加到你的 Skill 目录。在 [Skill.md](http://skill.md) 中引用它将帮助 Claude 决定在执行 Skill 时是否需要访问该资源。

## 添加 Scripts

对于更高级的 Skills，将可执行代码文件附加到 [Skill.md](http://skill.md)，允许 Claude 运行代码。例如，我们的文档 skills 使用以下编程语言和包：

*   Python (pandas, numpy, matplotlib)
*   JavaScript/ [Node.js](http://node.js)
*   帮助文件编辑的包
*   visualization tools（可视化工具）

注意: Claude 和 Claude Code 在加载 Skills 时可以从标准存储库（Python PyPI, JavaScript npm）安装包。使用 API Skills 时无法在运行时安装额外的包——所有依赖项必须预先安装在容器中。

## 打包你的 Skill

一旦你的 Skill 文件夹完成：

1.  确保文件夹名称与你的 Skill 名称匹配。
2.  创建文件夹的 ZIP 文件。
3.  ZIP 应包含 Skill 文件夹作为其根目录（不是子文件夹）。

正确的结构:

```text
my-Skill.zip
└── my-Skill/
    ├── Skill.md
    └── resources/
```

不正确的结构:

```text
my-Skill.zip
└── (files directly in ZIP root)
```

## 测试你的 Skill

### 上传之前

1.  审查你的 `Skill.md` 清晰度
2.  检查描述是否准确反映了 Claude 何时应该使用该 Skill
3.  验证所有引用的文件是否存在于正确的位置
4.  使用示例提示进行测试，以确保 Claude 适当地调用它

### 上传到 Claude 之后

1.  在 [Settings > Capabilities](https://claude.ai/settings/capabilities) 中启用 Skill。
2.  尝试几个应该触发它的不同提示
3.  审查 Claude 的思维过程以确认它正在加载 Skill
4.  如果 Claude 在预期时没有使用它，请迭代描述

Team 和 Enterprise 计划的注意事项: 要使 skill 对你组织中的所有用户可用，请参阅 Provisioning and managing Skills for your organization。

## 最佳实践

**保持专注**: 为不同的工作流创建单独的 Skills。多个专注的 Skills 比一个大型 Skill 组合得更好。

**编写清晰的描述**: Claude 使用描述来决定何时调用你的 Skill。具体说明它何时适用。

**从简单开始**: 在添加复杂的 scripts 之前，先从 Markdown 中的基本指令开始。你可以随时扩展 Skill。

**使用示例**: 在你的 `Skill.md` 文件中包含输入和输出示例，以帮助 Claude 理解成功是什么样子的。

**增量测试**: 在每次重大更改后进行测试，而不是一次性构建复杂的 Skill。

**Skills 可以相互构建**: 虽然 Skills 不能显式引用其他 Skills，但 Claude 可以自动一起使用多个 Skills。这种可组合性是 Skills 功能最强大的部分之一。

**审查开放 Agent Skills 规范**: 遵循 [agentskills.io](http://agentskills.io) 上的指南，这样你创建的 skills 就可以在采用该标准的跨平台上工作。

有关 skill 创建的更深入指南，请参阅我们 Claude Docs 中的 [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)。

## 安全考虑

*   在向你的 `Skill.md` 文件添加 scripts 时要谨慎。
*   不要硬编码敏感信息（API 密钥，密码）。
*   在启用之前审查你下载的任何 Skills。
*   使用适当的 MCP 连接进行外部服务访问。

## 参考示例 Skills

访问我们在 GitHub 上的存储库以获取可以用作模板的示例 Skills: [https://github.com/anthropics/skills/tree/main/skills](https://github.com/anthropics/skills/tree/main/skills)。

## 相关文章

*   [What are Skills?](https://support.claude.com/en/articles/12512176-what-are-skills)
*   [Using Skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
*   [Teach Claude your way of working using skills](https://support.claude.com/en/articles/12580051-teach-claude-your-way-of-working-using-skills)
*   [How to create a skill with Claude through conversation](https://support.claude.com/en/articles/12599426-how-to-create-a-skill-with-claude-through-conversation)
*   [Claude for Financial Services Skills](https://support.claude.com/en/articles/12663107-claude-for-financial-services-skills)

来源: https://support.claude.com/en/articles/12512198-how-to-create-custom-skills