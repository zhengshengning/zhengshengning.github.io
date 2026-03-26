---
title: OpenCode入门使用指南
date: 2026-02-09 16:15:00
categories:
  - [战胜玩AI, AI编程]
tags: [AI编程, OpenCode, 终端工具]
---

OpenCode 是一个开源的 AI 编程智能体，提供基于终端的界面、桌面应用程序或 IDE 扩展。本文介绍 OpenCode 的安装、配置和使用方法。

<!-- more -->

# 简介 | OpenCode

来源: https://opencode.ai/docs/

# 简介

开始使用 OpenCode。

[OpenCode](/) 是一个开源的 AI 编程智能体。它提供基于终端的界面、桌面应用程序或 IDE 扩展。

> 白话理解：你可以把 OpenCode 想象成一个住在终端里的"AI 结对编程伙伴"。你用自然语言告诉它想做什么，它就能帮你读代码、写代码、改代码，就像一个随叫随到的高级程序员坐在你旁边一样。

![OpenCode TUI with the opencode theme](/docs/_astro/screenshot.CQjBbRyJ_1dLadc.webp)

让我们开始吧。

#### 先决条件

要在终端中使用 OpenCode，你需要：

1.  一个现代终端模拟器，例如：
    *   [WezTerm](https://wezterm.org) ，跨平台
    *   [Alacritty](https://alacritty.org) ，跨平台
    *   [Ghostty](https://ghostty.org) ，Linux 和 macOS
    *   [Kitty](https://sw.kovidgoyal.net/kitty/) ，Linux 和 macOS
2.  你想使用的 LLM 提供商的 API 密钥。

> 白话理解：API 密钥就像一把"通行证钥匙"，你拿着它才能调用大模型的能力。每个提供商（比如 OpenAI、Anthropic）会给你发一串字符串，OpenCode 拿着这串字符串去帮你跟大模型对话。

## 安装

安装 OpenCode 最简单的方法是通过安装脚本。

终端窗口

```bash
curl -fsSL https://opencode.ai/install | bash
```

你也可以使用以下命令安装：

*   使用 Node.js
    *   [npm](#tab-panel-0)
    *   [Bun](#tab-panel-1)
    *   [pnpm](#tab-panel-2)
    *   [Yarn](#tab-panel-3)

终端窗口

```bash
npm install -g opencode-ai
```

终端窗口

```bash
bun install -g opencode-ai
```

终端窗口

```bash
pnpm install -g opencode-ai
```

终端窗口

```bash
yarn global add opencode-ai
```

*   在 macOS 和 Linux 上使用 Homebrew

终端窗口

```bash
brew install anomalyco/tap/opencode
```

我们建议使用 OpenCode tap 以获取最新版本。官方的 `brew install opencode` formula 由 Homebrew 团队维护，更新频率较低。

*   在 Arch Linux 上使用 Paru

终端窗口

```bash
paru -S opencode-bin
```

#### Windows

推荐：使用 WSL

为了在 Windows 上获得最佳体验，我们建议使用 [Windows Subsystem for Linux (WSL)](/docs/windows-wsl)。它提供了更好的性能，并完全兼容 OpenCode 的功能。

*   使用 Chocolatey

终端窗口

```bash
choco install opencode
```

*   使用 Scoop

终端窗口

```bash
scoop install opencode
```

*   使用 NPM

终端窗口

```bash
npm install -g opencode-ai
```

*   使用 Mise

终端窗口

```bash
mise use -g github:anomalyco/opencode
```

*   使用 Docker

终端窗口

```bash
docker run -it --rm ghcr.io/anomalyco/opencode
```

目前正在开发对使用 Bun 在 Windows 上安装 OpenCode 的支持。

你也可以从 [Releases](https://github.com/anomalyco/opencode/releases) 下载二进制文件。

## 配置

使用 OpenCode，你可以通过配置 API 密钥来使用任何 LLM 提供商。

如果你是第一次使用 LLM 提供商，我们建议使用 [OpenCode Zen](/docs/zen)。
这是 OpenCode 团队测试和验证过的精选模型列表。

1.  在 TUI 中运行 `/connect` 命令，选择 opencode，然后前往 [opencode.ai/auth](https://opencode.ai/auth)。

```bash
/connect
```

2.  登录，添加你的账单详细信息，并复制你的 API 密钥。
3.  粘贴你的 API 密钥。

```
> API key... enter
```

或者，你可以选择其他提供商之一。[了解更多](/docs/providers#directory)。

## 初始化

现在你已经配置了提供商，可以导航到你想处理的项目。

终端窗口

```bash
cd /path/to/project
```

然后运行 OpenCode。

终端窗口

```bash
opencode
```

接下来，通过运行以下命令为项目初始化 OpenCode。

```bash
/init
```

这将让 OpenCode 分析你的项目，并在项目根目录创建一个 `AGENTS.md` 文件。

> 白话理解：`AGENTS.md` 就像你给新来的同事写的一份"项目说明书"。OpenCode 每次开工前都会先读这份说明书，了解项目的目录结构、技术栈和编码规范，这样它才能写出符合项目风格的代码。

提示：
你应该将项目的 `AGENTS.md` 文件提交到 Git。

这有助于 OpenCode 理解项目结构和使用的编码模式。

## 用法

现在你准备好使用 OpenCode 来处理你的项目了。随意问它任何问题！

如果你是第一次使用 AI 编程智能体，这里有一些可能对你有帮助的例子。

### 提问

你可以让 OpenCode 向你解释代码库。

提示：
使用 `@` 键在项目中模糊搜索文件。

```
How is authentication handled in @packages/functions/src/api/index.ts
```

如果代码库中有一部分不是你编写的，这会很有帮助。

### 添加功能

你可以让 OpenCode 为你的项目添加新功能。不过我们建议先让它制定一个计划。

1.  创建计划

OpenCode 有一个 **计划模式**，它禁用其进行更改的能力，而是建议它将如何实现该功能。

> 白话理解：计划模式就像装修前先画设计图。OpenCode 在这个模式下只会"纸上谈兵"，告诉你它打算怎么改，不会动你一行代码。等你看完方案觉得没问题了，再切回构建模式让它真正动手。

使用 `Tab` 键切换到它。你会在右下角看到一个指示器。

```
<TAB>
```

现在让我们描述我们要它做什么。

```
When a user deletes a note, we'd like to flag it as deleted in the database.
Then create a screen that shows all the recently deleted notes.
From this screen, the user can undelete a note or permanently delete it.
```

你要给 OpenCode 足够的细节来理解你想要什么。把它当作是在跟你团队里的初级开发人员交谈。

提示：
给 OpenCode 充足的上下文和例子，以帮助它理解你想要什么。

2.  迭代计划

一旦它给你一个计划，你可以给它反馈或添加更多细节。

```
We'd like to design this new screen using a design I've used before.
[Image #1] Take a look at this image and use it as a reference.
```

提示：
将图片拖放到终端中以将其添加到提示中。

OpenCode 可以扫描你给它的任何图片并将它们添加到提示中。你可以通过将图片拖放到终端来做到这一点。

3.  构建功能

一旦你对计划感到满意，再次按 `Tab` 键切换回 **构建模式**。

```
<TAB>
```

然后让它进行更改。

```
Sounds good! Go ahead and make the changes.
```

### 进行更改

对于更直接的更改，你可以要求 OpenCode 直接构建它，而无需先审查计划。

```
We need to add authentication to the /settings route. Take a look at how this is
handled in the /notes route in @packages/functions/src/notes.ts and implement
the same logic in @packages/functions/src/settings.ts
```

你要确保提供足够多的细节，以便 OpenCode 做出正确的更改。

### 撤销更改

假设你让 OpenCode 做一些更改。

```
Can you refactor the function in @packages/functions/src/api/index.ts?
```

但是你意识到这不是你想要的。你可以使用 `/undo` 命令 **撤销** 更改。

> 白话理解：`/undo` 和 `/redo` 就像文档编辑器里的"撤销"和"重做"按钮。AI 改坏了？一个 `/undo` 回到改之前；后悔撤销了？一个 `/redo` 再恢复回来。多次 `/undo` 还能一步步往回退，像时光机一样。

```bash
/undo
```

OpenCode 现在将恢复你所做的更改并再次显示你的原始消息。

```
Can you refactor the function in @packages/functions/src/api/index.ts?
```

从这里你可以调整提示并让 OpenCode 重试。

提示：
你可以多次运行 `/undo` 来撤销多个更改。

或者你可以使用 `/redo` 命令 **重做** 更改。

```bash
/redo
```

## 分享

你与 OpenCode 的对话可以 [与你的团队分享](/docs/share)。

```bash
/share
```

这将创建当前对话的链接并将其复制到你的剪贴板。

注意：
对话默认不共享。

这里有一个与 OpenCode 的 [对话示例](https://opencode.ai/s/4XP1fce5)。

## 自定义

就是这样！你现在是使用 OpenCode 的专家了。

为了让它更适合你，我们建议 [选择一个主题](/docs/themes)、[自定义键绑定](/docs/keybinds)、[配置代码格式化程序](/docs/formatters)、[创建自定义命令](/docs/commands)，或者摆弄 [OpenCode 配置](/docs/config)。

## 检验标准与进阶方向

### 自我检验清单

学完本文后，你可以用以下问题检验自己的掌握程度：

- [ ] 能在自己的操作系统上独立完成 OpenCode 的安装（至少掌握一种安装方式）
- [ ] 能配置至少一个 LLM 提供商的 API 密钥，并成功连接
- [ ] 能使用 `/init` 命令为项目生成 `AGENTS.md` 文件，并理解其作用
- [ ] 能使用 `@` 符号引用项目中的文件，向 OpenCode 提出代码相关问题
- [ ] 能熟练使用 `Tab` 键在计划模式和构建模式之间切换，先审查方案再执行
- [ ] 能使用 `/undo` 和 `/redo` 命令管理 OpenCode 所做的代码变更
- [ ] 能使用 `/share` 命令将对话分享给团队成员
- [ ] 能编写足够详细的自然语言提示，使 OpenCode 准确理解需求并生成正确的代码

### 进阶方向

| 方向 | 说明 | 推荐资源 |
|------|------|----------|
| 提供商配置 | 学习对接更多 LLM 提供商（OpenAI、Anthropic、本地模型等），根据任务类型选择最合适的模型 | [OpenCode Providers 文档](https://opencode.ai/docs/providers) |
| 主题与键绑定 | 自定义 OpenCode 的外观和快捷键，打造个性化的开发环境 | [Themes 文档](https://opencode.ai/docs/themes)、[Keybinds 文档](https://opencode.ai/docs/keybinds) |
| 自定义命令 | 创建可复用的自定义命令，将常用操作封装为一键触发的工作流 | [Commands 文档](https://opencode.ai/docs/commands) |
| 代码格式化 | 配置代码格式化程序，确保 OpenCode 生成的代码符合项目规范 | [Formatters 文档](https://opencode.ai/docs/formatters) |
| MCP 集成 | 通过 Model Context Protocol 扩展 OpenCode 的能力，接入数据库、API 等外部工具 | [MCP 文档](https://opencode.ai/docs/mcp) |
| 提示工程 | 学习如何编写更精准的提示词，提升 AI 生成代码的质量和准确性 | [OpenCode 最佳实践](https://opencode.ai/docs/) |

## 参考资料

- [OpenCode 官方文档](https://opencode.ai/docs/) - 官方完整文档，涵盖安装、配置、使用的方方面面
- [OpenCode GitHub 仓库](https://github.com/anomalyco/opencode) - 源代码、Issue 追踪和 Release 下载
- [OpenCode Zen](https://opencode.ai/docs/zen) - OpenCode 团队精选的模型列表，适合初次使用者
- [OpenCode Providers 目录](https://opencode.ai/docs/providers#directory) - 支持的 LLM 提供商完整列表
- [OpenCode 对话分享示例](https://opencode.ai/s/4XP1fce5) - 一个真实的 OpenCode 对话示例