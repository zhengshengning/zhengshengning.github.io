---
title: 使用uv构建Python命令行工具：从代码到全局命令
date: 2026-03-25 10:00:00
categories:
  - [编程技能包, Python]
tags: [Python, uv, CLI, Click, 命令行工具]
---

在 Python 开发中，我们经常需要创建一些命令行工具来提高工作效率。本文将手把手教你如何使用现代 Python 打包工具 **uv**，将一个简单的 Python 模块构建成可以通过 `nzs-tool myhello Alice` 这种形式调用的全局命令行工具。

<!-- more -->

## 为什么选择 uv？

`uv` 是 Astral 团队（ruff 的开发者）推出的新一代 Python 包管理工具，它比传统的 `pip` 快 10-100 倍，并且原生支持 `pyproject.toml` 标准。使用 uv 管理 Python 项目，不仅能获得极快的安装速度，还能享受更现代化的开发体验。

## 最终效果预览

完成本教程后，你将能够：
1. 编写 Python 命令行工具代码
2. 通过 uv 安装到本地环境
3. 在终端直接运行 `nzs-tool myhello Alice` 这样的命令

```
$ nzs-tool myhello Alice
Hello, Alice!

$ nzs-tool mygoodbye Bob
Goodbye, Bob!

$ nzs-tool --help
Usage: nzs-tool [OPTIONS] COMMAND [ARGS]...

  nzs-tool - 多功能命令行工具

Options:
  --help  Show this message and exit.

Commands:
  mygoodbye   向 NAME 告别
  myhello     向 NAME 问好
```

## 第一步：项目初始化

首先，创建项目目录并初始化 uv 项目。

```bash
# 创建项目目录
mkdir nzs-tool
cd nzs-tool

# 初始化 uv 项目（会生成 pyproject.toml）
uv init
```

执行 `uv init` 后，uv 会自动创建一个 `pyproject.toml` 文件，包含基本的项目配置。

## 第二步：规划项目结构

采用 `src` 布局是 Python 社区的推荐实践，可以避免导入时的一些常见问题。

```
nzs-tool/
├── pyproject.toml
├── README.md
└── src/
    └── nzs_tool/          # 注意：包名使用下划线
        ├── __init__.py
        └── cli.py
```

创建这些目录和文件：

```bash
mkdir -p src/nzs_tool
touch src/nzs_tool/__init__.py
touch src/nzs_tool/cli.py
```

## 第三步：编写命令行工具代码

我们将使用 **Click** 库来构建命令行接口。Click 是一个优雅的 Python CLI 库，通过装饰器就能轻松创建复杂的命令行工具。

首先，添加 Click 依赖：

```bash
uv add click
```

然后在 `src/nzs_tool/cli.py` 中编写代码：

```python
import click

@click.group()
def cli():
    """nzs-tool - 多功能命令行工具"""
    pass

@cli.command(name="myhello")
@click.argument("name")
def hello(name):
    """向 NAME 问好"""
    click.echo(f"Hello, {name}!")

@cli.command(name="mygoodbye")
@click.argument("name")
def goodbye(name):
    """向 NAME 告别"""
    click.echo(f"Goodbye, {name}!")

# 可以继续添加更多子命令
# @cli.command()
# def another_command():
#     """另一个功能"""
#     pass

if __name__ == "__main__":
    cli()  # 方便直接调试运行
```

**代码解析：**
- `@click.group()` 创建一个命令组，支持子命令
- `@cli.command()` 注册子命令，`name` 参数指定命令名称
- `@click.argument()` 定义位置参数（如 `myhello` 后面的名字）
- `click.echo()` 是跨平台的打印函数，比 `print` 更强大

## 第四步：配置 pyproject.toml

这是最关键的一步。我们需要在 `pyproject.toml` 中配置两件事：
1. 项目的元数据（名称、版本、依赖等）
2. 命令行入口点（告诉 Python 如何调用我们的命令）

编辑 `pyproject.toml` 文件：

```toml
[project]
name = "nzs-tool"
version = "0.1.0"
description = "A sample CLI tool with subcommands"
readme = "README.md"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
license = { text = "MIT" }
requires-python = ">=3.8"
dependencies = [
    "click>=8.0"
]

[project.scripts]
# 格式：命令名称 = "模块路径:函数名"
nzs-tool = "nzs_tool.cli:cli"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

**关键点解析：**
- `[project.scripts]` 是配置命令行入口的地方
- `nzs-tool = "nzs_tool.cli:cli"` 表示：
  - 创建名为 `nzs-tool` 的命令
  - 调用 `nzs_tool.cli` 模块中的 `cli` 函数
- `cli` 函数就是我们之前在 `cli.py` 中用 `@click.group()` 装饰的函数

## 第五步：安装工具

现在，我们可以将工具安装到本地环境中。uv 提供了两种安装方式：

### 方式一：可编辑安装（推荐开发阶段）

```bash
uv pip install -e .
```

`-e` 参数表示可编辑安装，对源代码的修改会立即生效，无需重新安装，非常适合开发调试。

### 方式二：直接安装

```bash
uv pip install .
```

这种方式会将当前代码打包并安装，后续修改需要重新安装才能生效。

安装完成后，可以验证是否安装成功：

```bash
uv pip list | grep nzs-tool
# 应该显示：nzs-tool 0.1.0
```

## 第六步：运行命令

安装成功后，你就可以在终端直接运行命令了！

```bash
# 运行子命令
nzs-tool myhello Alice
# 输出: Hello, Alice!

nzs-tool mygoodbye Bob
# 输出: Goodbye, Bob!

# 查看帮助
nzs-tool --help
# 显示主命令帮助，列出所有子命令

nzs-tool myhello --help
# 显示 myhello 子命令的帮助
```

如果你使用的是虚拟环境但没有激活，可以使用 `uv run` 来执行：

```bash
uv run nzs-tool myhello Alice
```

`uv run` 会自动识别当前项目环境，非常方便。

## 进阶：添加更多功能

### 1. 添加选项参数

Click 支持各种参数类型，包括选项（option）：

```python
@cli.command()
@click.option("--count", "-c", default=1, help="重复次数")
@click.argument("name")
def repeat(name, count):
    """重复说 hello"""
    for i in range(count):
        click.echo(f"Hello, {name}! (次数: {i+1})")
```

使用方式：
```bash
nzs-tool repeat Alice --count 3
# 或简写
nzs-tool repeat Alice -c 3
```

### 2. 添加配置文件支持

你可以让工具读取配置文件，实现更灵活的功能：

```python
import json
from pathlib import Path

def load_config():
    config_path = Path.home() / ".nzs-tool.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

@cli.command()
def config():
    """显示当前配置"""
    config = load_config()
    click.echo(json.dumps(config, indent=2))
```

### 3. 添加颜色输出

Click 内置了颜色支持：

```python
@cli.command()
@click.option("--color", is_flag=True, help="使用彩色输出")
def status(color):
    if color:
        click.secho("✓ 操作成功", fg="green")
        click.secho("✗ 操作失败", fg="red")
    else:
        click.echo("操作成功")
        click.echo("操作失败")
```

## 调试技巧

### 直接运行 Python 文件

在开发过程中，可以直接运行 Python 文件来调试：

```bash
python src/nzs_tool/cli.py myhello Alice
```

这需要你在 `cli.py` 文件末尾添加 `if __name__ == "__main__": cli()` 代码（我们已经在代码中添加了）。

### 使用 uv run 运行未安装的命令

即使没有安装，也可以使用 `uv run` 直接运行：

```bash
uv run python src/nzs_tool/cli.py myhello Alice
```

### 查看入口点脚本位置

安装后，可以查看生成的入口点脚本：

```bash
# 查看脚本位置
which nzs-tool

# 查看脚本内容（Unix/Linux/macOS）
cat $(which nzs-tool)
```

## 发布到 PyPI（可选）

如果你想和全世界分享你的工具，可以发布到 PyPI：

### 1. 构建分发包

```bash
uv build
```

这会在 `dist/` 目录下生成 `.tar.gz` 源码包和 `.whl` 二进制包。

### 2. 发布

```bash
# 首次发布需要配置 PyPI token
uv publish
```

发布后，任何人都可以通过以下命令安装：

```bash
uv pip install nzs-tool
# 或
pip install nzs-tool
```

## 常见问题

### Q1: 命令找不到怎么办？

检查虚拟环境是否激活，或者使用 `uv run nzs-tool`。

### Q2: 修改代码后命令没有变化？

如果使用可编辑安装（`-e`），修改应该立即生效。如果使用普通安装，需要执行 `uv pip install .`。

### Q3: Windows 上运行报错？

Click 和 uv 都完美支持 Windows。如果遇到路径问题，确保在命令提示符或 PowerShell 中运行，而不是在 WSL 中。

### Q4: 如何添加依赖？

使用 `uv add package-name` 添加依赖，会自动更新 `pyproject.toml` 和 `uv.lock`。

## 总结

通过本教程，我们完成了一个完整的 Python 命令行工具开发流程：

1. 使用 uv 初始化项目
2. 编写 Click 命令行代码
3. 配置 pyproject.toml 的入口点
4. 通过 uv 安装到本地
5. 在终端直接调用命令

这种开发模式的优势：
- **开发效率高**：uv 的极速安装和可编辑模式
- **用户体验好**：用户只需一个命令就能安装和运行
- **标准化**：完全遵循 Python 打包规范（PEP 621）
- **易扩展**：可以轻松添加更多子命令和功能

现在，你已经掌握了使用 uv 构建 Python 命令行工具的全部技巧。快去创建属于你自己的命令行工具吧！

## 参考资料

- [uv 官方文档](https://docs.astral.sh/uv/)
- [Click 官方文档](https://click.palletsprojects.com/)
- [Python 打包用户指南](https://packaging.python.org/)
- [PEP 621 - 使用 pyproject.toml 存储项目元数据](https://peps.python.org/pep-0621/)
