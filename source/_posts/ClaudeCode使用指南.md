---
title: ClaudeCode使用指南
date: 2026-02-09 16:20:00
categories: AI编程
tags: [AI编程, ClaudeCode, VS Code, 代码助手]
---

ClaudeCode 是 Anthropic 的 VS Code AI 编程助手。

<!-- more -->

## 安装

1. 打开 VS Code 扩展市场 (Ctrl+Shift+X)
2. 搜索 "Claude Dev"
3. 点击安装

## 配置

VS Code 设置中添加：

```json
{
  "claude.apiKey": "your-anthropic-api-key",
  "claude.model": "claude-3-5-sonnet-20241022",
  "claude.maxTokens": 4096,
  "claude.autoSave": true
}
```

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+L | Claude 侧边栏 |
| Ctrl+K | 内联编辑 |
| Ctrl+I | 代码解释 |

## 主要功能

- **代码生成**：描述需求自动生成代码
- **代码解释**：选中代码获取解释
- **Bug修复**：描述问题自动修复
- **测试生成**：自动生成单元测试
- **代码重构**：优化代码结构

## 使用示例

在侧边栏输入：
- "创建一个 Flask API"
- "为这个函数添加错误处理"
- "解释这段代码的作用"
- "生成单元测试"

## 工作区配置

项目根目录创建 `.vscode/settings.json`：

```json
{
  "claude.contextFiles": ["src/**/*.py"],
  "claude.ignoreFiles": ["node_modules/**"],
  "claude.customInstructions": "Python后端项目"
}
```

## 总结

ClaudeCode 深度集成 VS Code，适合图形界面用户。