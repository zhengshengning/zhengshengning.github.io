---
title: 构建并发布Python包到PyPI完整指南
date: 2026-03-11 10:00:00
categories: 编程技能包
tags: [Python, PyPI, 包管理, GitHub Actions, CI/CD, uv, hatch]
---

本文将详细介绍如何构建并发布一个 Python 包到 Python Package Index (PyPI)，涵盖从手动发布到 CI 自动化发布的完整流程。

<!-- more -->

## 目录

1. [项目结构规划](#一、项目结构规划)
2. [编写 pyproject.toml](#二、编写-pyproject-toml)
3. [安装构建工具](#三、安装构建工具)
4. [本地构建 Python 包](#四、本地构建-Python-包)
5. [手动发布到 PyPI](#五、手动发布到-PyPI)
6. [测试安装](#六、测试安装)
7. [GitHub Actions 自动发布](#七、GitHub-Actions-自动发布)
8. [PyPI Trusted Publisher 配置](#八、PyPI-Trusted-Publisher-配置)
9. [进阶：使用 hatch-vcs 自动管理版本](#九、进阶：使用-hatch-vcs-自动管理版本)
10. [完整流程总结](#十、完整流程总结)

---

## 一、项目结构规划

推荐使用 `src/` 布局，这是现代 Python 项目的最佳实践：

```
my-package/
│
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── hello.py
│
├── tests/
│   └── test_hello.py
│
├── .github/
│   └── workflows/
│       └── publish.yml
│
├── pyproject.toml
├── README.md
└── LICENSE
```

示例代码 `src/my_package/hello.py`：

```python
def say_hello():
    print("Hello from my package!")
```

`src/my_package/__init__.py`：

```python
from my_package.hello import say_hello

__all__ = ["say_hello"]
```

**为什么使用 src/ 布局？**

- 避免本地模块与安装后的模块混淆
- 强制在测试时使用已安装的包
- 防止意外导入未打包的模块

---

## 二、编写 pyproject.toml

`pyproject.toml` 是 Python 包的核心配置文件，遵循 PEP 517/518/621 规范。

### 2.1 基础配置（使用 uv 构建）

```toml
[build-system]
requires = ["uv"]
build-backend = "uv.build"

[project]
name = "my-package-demo"
version = "0.1.0"
description = "A demo Python package"
readme = "README.md"
requires-python = ">=3.9"
license = "MIT"
authors = [
    { name = "Your Name", email = "you@example.com" }
]

dependencies = []

[project.urls]
Homepage = "https://github.com/username/my-package"
Repository = "https://github.com/username/my-package"
```

### 2.2 进阶配置（使用 hatchling + hatch-vcs）

使用 `hatchling` + `hatch-vcs` 实现更强大的构建和版本管理：

```toml
[build-system]
requires = ["hatch-vcs", "hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-package"
description = "A demo Python package with advanced features"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
authors = [{ name = "Your Name", email = "you@example.com" }]
dependencies = [
    "pydantic>=2.0",
    "typer>=0.15.0",
]
dynamic = ["version"]  # 版本号从 git tag 自动获取

# CLI 入口点
[project.scripts]
my-cli = "my_package.cli:app"

# 插件入口点（可选）
[project.entry-points."my_package.plugins"]
default = "my_package.plugins.default:DefaultPlugin"

[project.urls]
Homepage = "https://github.com/username/my-package"
Repository = "https://github.com/username/my-package"

# hatch 构建配置
[tool.hatch.build.targets.wheel]
packages = ["src/my_package"]

# 从 git tag 获取版本
[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/my_package/_version.py"
```

### 2.3 关键字段说明

| 字段 | 作用 |
|------|------|
| `name` | PyPI 包名（用户 `pip install` 时使用） |
| `version` | 版本号（或设为 `dynamic` 自动生成） |
| `dependencies` | 运行时依赖 |
| `requires-python` | 支持的 Python 版本 |
| `project.scripts` | 安装后可用的命令行工具 |
| `project.entry-points` | 插件系统入口点 |

---

## 三、安装构建工具

### 3.1 安装 uv（推荐）

**Linux / macOS：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows：**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**验证安装：**

```bash
uv --version
```

### 3.2 安装 hatch（可选）

如果使用 hatchling 构建：

```bash
pip install hatch
```

---

## 四、本地构建 Python 包

进入项目目录执行构建：

```bash
cd my-package
uv build
```

构建成功后会生成 `dist/` 目录：

```
dist/
├── my_package_demo-0.1.0.tar.gz      # 源码包
└── my_package_demo-0.1.0-py3-none-any.whl  # wheel 二进制包
```

| 文件类型 | 说明 |
|----------|------|
| `.tar.gz` | 源码分发包，安装时需要构建 |
| `.whl` | wheel 包，预构建的二进制格式，安装更快 |

---

## 五、手动发布到 PyPI

### 5.1 安装上传工具

```bash
pip install twine
```

### 5.2 上传到 PyPI

```bash
twine upload dist/*
```

系统会提示输入凭据：

```
username: __token__
password: pypi-xxxx  # 在 PyPI 账号中生成的 API Token
```

### 5.3 获取 PyPI Token

1. 登录 [https://pypi.org](https://pypi.org)
2. 进入 Account Settings → API tokens
3. 创建新 Token，选择作用域（整个账号或特定项目）
4. 复制 Token（以 `pypi-` 开头）

发布成功后可在以下地址查看：

```
https://pypi.org/project/my-package-demo/
```

---

## 六、测试安装

从 PyPI 安装并测试：

```bash
pip install my-package-demo
```

验证：

```python
from my_package.hello import say_hello
say_hello()
# 输出: Hello from my package!
```

---

## 七、GitHub Actions 自动发布

手动发布容易出错，生产环境推荐使用 CI 自动化。以下是推荐的配置：

### 7.1 创建 workflow 文件

`.github/workflows/publish.yml`：

```yaml
name: "CI - Publish"

on:
  push:
    tags: ["v*"]  # 仅在推送 v 开头的 tag 时触发
  workflow_dispatch:  # 允许手动触发

permissions:
  contents: read
  id-token: write  # OIDC 认证必需

jobs:
  build:
    name: Build Distribution
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          python-version: "3.13"

      - name: Build package
        run: uv build

      - name: Upload distribution
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    name: Publish To PyPI
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/my-package
    steps:
      - name: Download distribution
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish package
        uses: pypa/gh-action-pypi-publish@release/v1
```

### 7.2 workflow 结构说明

```
触发条件
   ↓
build job（构建）
   ├── checkout 代码
   ├── 安装 uv
   ├── uv build 构建包
   └── 上传 artifact
   ↓
publish job（发布）
   ├── 下载 artifact
   └── 发布到 PyPI
```

**关键设计：**

1. **build 与 publish 分离** - 构建产物通过 artifact 传递，职责清晰
2. **tag 触发** - 只有推送版本 tag 才触发发布
3. **OIDC 认证** - 无需存储 PyPI Token

---

## 八、PyPI Trusted Publisher 配置

PyPI Trusted Publisher 使用 OIDC（OpenID Connect）认证，无需在 GitHub 存储 API Token，更安全。

### 8.1 配置步骤

1. 登录 PyPI：[https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/)

2. 点击 "Add a new pending publisher"（新项目）或在项目设置中添加

3. 填写信息：
   - **Owner**: GitHub 用户名或组织名
   - **Repository name**: 仓库名
   - **Workflow name**: `publish.yml`（与你的 workflow 文件名一致）
   - **Environment name**: `pypi`（与 workflow 中的 environment 一致）

4. 保存配置

### 8.2 确保 workflow 配置正确

```yaml
permissions:
  id-token: write  # 必须有这个权限

jobs:
  publish:
    environment:
      name: pypi  # 必须与 PyPI 配置的 environment 一致
```

---

## 九、进阶：使用 hatch-vcs 自动管理版本

手动维护版本号容易遗忘或出错。推荐使用 `hatch-vcs` 从 git tag 自动获取版本。

### 9.1 配置 pyproject.toml

```toml
[build-system]
requires = ["hatch-vcs", "hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-package"
dynamic = ["version"]  # 版本号动态生成

[tool.hatch.version]
source = "vcs"  # 从版本控制系统获取

[tool.hatch.build.hooks.vcs]
version-file = "src/my_package/_version.py"  # 生成版本文件
```

### 9.2 工作流程

```
git tag v1.2.3
     ↓
构建时自动读取 tag
     ↓
版本号设为 1.2.3
     ↓
生成 _version.py 文件
```

**优点：**

- 版本号与 git tag 强绑定，不会出错
- 无需手动修改代码中的版本号
- 代码中可通过 `_version.py` 访问版本

---

## 十、完整流程总结

### 10.1 手动发布流程

```
创建项目结构（src/ 布局）
          ↓
编写 pyproject.toml
          ↓
uv build（构建）
          ↓
生成 dist/（.tar.gz + .whl）
          ↓
twine upload dist/*
          ↓
发布到 PyPI
```

### 10.2 自动发布流程

```
Developer
    ↓
git tag v1.0.0 && git push origin v1.0.0
    ↓
GitHub Actions 触发
    ↓
uv build 构建
    ↓
OIDC 认证
    ↓
自动发布到 PyPI
    ↓
用户: pip install my-package
```

### 10.3 推荐实践清单

| 实践 | 说明 |
|------|------|
| 使用 `src/` 布局 | 避免导入混淆 |
| 使用 tag 触发发布 | 版本控制清晰 |
| build 与 publish 分离 | 职责明确，易于调试 |
| 使用 Trusted Publisher | 无需存储 Token，更安全 |
| 使用 hatch-vcs | 自动管理版本号 |
| 添加测试 | 确保发布前质量 |

---

## 附录：常用命令速查

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 构建包
uv build

# 手动上传
twine upload dist/*

# 创建并推送 tag
git tag v1.0.0
git push origin v1.0.0

# 本地测试安装
pip install dist/*.whl

# 从 PyPI 安装
pip install my-package-demo
```

---

## 参考资源

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [uv 文档](https://docs.astral.sh/uv/)
- [Hatch 文档](https://hatch.pypa.io/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)
