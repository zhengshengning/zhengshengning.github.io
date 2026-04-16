# 平台选择器参考

各平台已验证的 CSS 选择器和注意事项。当需要添加新平台或已有选择器失效时参考此文件。

## 牛客网 (nowcoder)

**URL 模式**:
- `/feed/main/detail/{id}` — 动态帖子
- `/discuss/{id}` — 讨论帖

**已验证选择器** (2026-04):
| 选择器 | 说明 | 适用页面 |
|--------|------|----------|
| `.feed-content-text` | 帖子正文（最精准） | /feed/ |
| `.nc-post-content` | 讨论帖正文 | /discuss/ |
| `.post-content-box` | 讨论帖含标题 | /discuss/ |
| `.main-content-container` | 整个内容区（含用户信息） | 通用 fallback |

**注意事项**:
- 牛客使用动态 JS 渲染，必须用浏览器爬取
- `wait_until="domcontentloaded"` + 2s 延迟即可，`networkidle` 容易超时
- 网络不稳定时需重试机制，建议 3 次重试 + 指数退避

---

## 知乎 (zhihu)

**URL 模式**:
- `/question/{qid}/answer/{aid}` — 回答
- `/p/{id}` — 专栏文章

**建议选择器**:
| 选择器 | 说明 |
|--------|------|
| `.RichContent-inner` | 回答/文章正文 |
| `.Post-RichTextContainer` | 专栏文章 |
| `.Post-RichText` | 专栏文章备选 |
| `article.Post-Main` | 整篇文章 |

**注意事项**:
- 知乎有反爬检测，建议设置合理的 User-Agent 和请求间隔
- 未登录状态下部分内容可能被折叠

---

## CSDN

**URL 模式**:
- `/article/details/{id}` — 博客文章

**建议选择器**:
| 选择器 | 说明 |
|--------|------|
| `#content_views` | 文章正文（最精准） |
| `.article_content` | 文章内容区 |
| `.blog-content-box` | 整个博客内容框 |
| `.htmledit_views` | HTML 编辑器内容 |

**注意事项**:
- CSDN 未登录会有"登录后查看全文"弹窗，可能影响爬取
- 部分文章需要关注才能查看全文

---

## 小红书 (xiaohongshu)

**URL 模式**:
- `/explore/{id}` — 笔记页
- `/discovery/item/{id}` — 笔记页

**建议选择器**:
| 选择器 | 说明 |
|--------|------|
| `#detail-desc .note-text` | 笔记正文 |
| `.note-content` | 笔记内容区 |
| `#detail-desc` | 详情区域 |

**注意事项**:
- 小红书反爬较严，建议增大请求间隔（3-5s）
- 以图片为主的内容 inner_text 可能较短
- 未登录可能看不到完整内容

---

## 添加新平台

1. 使用 `inspect_selectors.py` 检查目标页面 DOM 结构
2. 找到内容区域的精准选择器（优先选择最小且最精准的）
3. 在 `crawl.py` 的 `PLATFORM_CONFIG` 中添加配置
4. 更新本文件记录新平台的选择器和注意事项
