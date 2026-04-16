---
name: web-content-crawler
description: >
  使用 Playwright 无头浏览器批量爬取网页内容，支持动态 JS 渲染页面，将结果保存为 Hexo 博客 Markdown 文件。
  已适配平台：牛客网(nowcoder)、知乎(zhihu)、CSDN、小红书(xiaohongshu)，支持自定义新平台。
  触发场景：(1) 用户说"爬取"、"抓取"、"crawl"、"批量获取"网页内容
  (2) 用户提供一批链接要求获取其内容并保存
  (3) 用户要求从牛客/知乎/CSDN/小红书等平台批量下载帖子/文章
  (4) 用户要求将网页内容转为 Markdown 文件
---

# Web Content Crawler

使用 Playwright 无头浏览器批量爬取动态网页内容，保存为 Hexo Markdown 文件。

## 前置依赖

- Python 3.8+
- `playwright` Python 包（`pip install playwright`）
- 系统 Google Chrome 浏览器（脚本优先使用系统 Chrome，无需 `playwright install`）

## 工作流程

### 1. 确认爬取需求

明确以下信息（缺失则询问用户）：
- **目标平台**：nowcoder / zhihu / csdn / xiaohongshu / generic
- **链接来源**：用户提供的 Markdown 文件、文本列表、或 JSON 文件
- **输出目录**：保存爬取结果的路径
- **分类和标签**：Hexo front-matter 中的 categories 和 tags

### 2. 准备链接列表

从用户提供的材料中提取链接，生成 JSON 格式：

```json
[
  {"id": "001", "title": "文章标题", "url": "https://..."},
  {"id": "002", "title": "另一篇", "url": "https://..."}
]
```

若用户提供的是 Markdown 文件，用正则 `\[([^\]]+)\]\((https?://[^)]+)\)` 提取标题和 URL。

### 3. 执行爬取

运行爬取脚本：

```bash
python3 scripts/crawl.py \
  --platform nowcoder \
  --links /tmp/links.json \
  --output /path/to/output/ \
  --categories "求职面试, AI Infra" \
  --tags "面经, 牛客" \
  --retries 3
```

脚本自动处理：
- 用系统 Chrome 进行无头浏览器渲染
- 平台特定的 CSS 选择器提取正文
- 失败自动重试（指数退避）
- 每个请求间有延迟（避免被封）

### 4. 检查结果

- 验证文件数量与链接数量一致
- 抽查 2-3 个文件确认内容质量（无导航栏噪音）
- 若仍有失败，可单独重试失败链接

## 平台适配

已支持的平台配置在 `scripts/crawl.py` 的 `PLATFORM_CONFIG` 字典中。

添加新平台时：
1. 运行 `scripts/inspect_selectors.py <url>` 检查目标页面 DOM 结构
2. 在 `PLATFORM_CONFIG` 中添加新平台配置
3. 更新 `references/platform_selectors.md` 记录选择器

各平台选择器详情见 [references/platform_selectors.md](references/platform_selectors.md)。

## 常见问题处理

**网络超时/断连**：脚本内置重试机制。若大量失败，等网络恢复后提取失败 ID 列表单独重试。

**内容为空或太短**：可能是选择器失效。用 `inspect_selectors.py` 重新检查 DOM 结构，更新选择器。

**Playwright 浏览器未安装**：脚本优先使用系统 `channel="chrome"`。若系统无 Chrome，运行 `playwright install chromium`。

**反爬/需登录**：增大 `wait_time`，设置合理 User-Agent。部分平台需登录才能看完整内容，这种情况需提示用户。
