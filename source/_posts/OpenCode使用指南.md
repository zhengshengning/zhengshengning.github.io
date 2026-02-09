---
title: OpenCode使用指南
date: 2026-02-09 16:15:00
categories: AI编程
tags: [AI编程, OpenCode, 终端工具]
---

OpenCode 是开源的终端 AI 编程助手。

<!-- more -->

## 安装

```bash
go install github.com/opencode-ai/opencode@latest
# 或
pip install opencode-cli
```

## 配置

创建 `~/.opencode/config.yaml`：

```yaml
provider: openai
openai:
  api_key: your-key
  model: gpt-4
anthropic:
  api_key: your-key
  model: claude-3-opus
settings:
  max_tokens: 4096
```

## 使用

```bash
opencode                     # 交互模式
opencode "问题"              # 直接提问
opencode --file main.py "解释" # 带文件
```

## 命令

| 命令 | 说明 |
|------|------|
| /help | 帮助 |
| /model | 切换模型 |
| /file | 添加文件 |
| /quit | 退出 |