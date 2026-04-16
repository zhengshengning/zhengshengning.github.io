#!/usr/bin/env python3
"""
Universal web content crawler using Playwright (headless browser).
Supports dynamic JS-rendered pages (牛客, 知乎, 小红书, CSDN, etc.)

Usage:
    python3 crawl.py --platform nowcoder --links links.json --output ./output
    python3 crawl.py --platform zhihu --links links.json --output ./output
    python3 crawl.py --platform csdn --links links.json --output ./output
    python3 crawl.py --platform xiaohongshu --links links.json --output ./output
    python3 crawl.py --platform generic --links links.json --output ./output

links.json format:
[
  {"id": "001", "title": "文章标题", "url": "https://..."},
  {"id": "002", "title": "另一篇", "url": "https://..."}
]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed. Run: pip install playwright")
    sys.exit(1)


# ============================================================
# Platform-specific selectors and extraction logic
# ============================================================

PLATFORM_CONFIG = {
    "nowcoder": {
        "name": "牛客网",
        "wait_selector": ".feed-content-text, .nc-post-content, .post-content-box",
        "content_selectors": [
            ".feed-content-text",
            ".nc-post-content",
            ".post-content-box",
            ".main-content-container",
        ],
        "wait_time": 2,
        "wait_until": "domcontentloaded",
    },
    "zhihu": {
        "name": "知乎",
        "wait_selector": ".RichContent-inner, .Post-RichText, .AnswerItem",
        "content_selectors": [
            ".RichContent-inner",
            ".Post-RichTextContainer",
            ".Post-RichText",
            ".QuestionAnswer-content",
            "article.Post-Main",
            ".AnswerItem .RichContent",
        ],
        "wait_time": 3,
        "wait_until": "domcontentloaded",
    },
    "csdn": {
        "name": "CSDN",
        "wait_selector": "#content_views, .article_content, .blog-content-box",
        "content_selectors": [
            "#content_views",
            ".article_content",
            "#article_content",
            ".blog-content-box",
            ".htmledit_views",
        ],
        "wait_time": 2,
        "wait_until": "domcontentloaded",
    },
    "xiaohongshu": {
        "name": "小红书",
        "wait_selector": ".note-content, .content, #detail-desc",
        "content_selectors": [
            "#detail-desc .note-text",
            ".note-content",
            "#detail-desc",
            ".content",
            ".desc",
        ],
        "wait_time": 4,
        "wait_until": "domcontentloaded",
    },
    "generic": {
        "name": "通用",
        "wait_selector": "article, main, .content, .post, .entry-content",
        "content_selectors": [
            "article",
            "main",
            ".content",
            ".post-content",
            ".entry-content",
            ".article-content",
            "#content",
        ],
        "wait_time": 3,
        "wait_until": "domcontentloaded",
    },
}


def sanitize_filename(name: str) -> str:
    """Remove characters problematic in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = name.strip('_')
    return name[:100]  # Limit length


def extract_content(page, platform: str) -> str:
    """Extract main content from page using platform-specific selectors."""
    config = PLATFORM_CONFIG[platform]

    try:
        page.wait_for_selector(config["wait_selector"], timeout=10000)
    except Exception:
        pass

    time.sleep(config["wait_time"])

    for sel in config["content_selectors"]:
        el = page.query_selector(sel)
        if el:
            text = el.inner_text()
            if text and len(text.strip()) > 20:
                return text.strip()

    # Fallback: body text
    body_text = page.inner_text('body')
    return body_text.strip() if body_text else ""


def generate_markdown(item: dict, content: str, output_dir: str, platform_name: str,
                      categories: str, tags: str) -> str:
    """Generate a markdown file for the crawled content."""
    idx = item["id"]
    title = item["title"]
    url = item["url"]
    safe_title = sanitize_filename(title)
    filename = f"{idx}_{safe_title}.md"
    filepath = os.path.join(output_dir, filename)

    # Escape single quotes in title for YAML
    yaml_title = title.replace("'", "''")

    md = f"""---
title: '{yaml_title}'
date: {time.strftime('%Y-%m-%d %H:%M:%S')}
categories:
  - [{categories}]
tags: [{tags}]
---

# {title}

<!-- more -->

> 原文链接: {url}
> 来源: {platform_name}

---

{content}
"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md)
    return filepath


def crawl(links: list, platform: str, output_dir: str,
          categories: str, tags: str, max_retries: int = 3) -> dict:
    """Crawl all links and save as markdown files."""
    config = PLATFORM_CONFIG[platform]
    os.makedirs(output_dir, exist_ok=True)
    results = {"success": [], "failed": []}

    with sync_playwright() as p:
        # Try system Chrome first, fall back to bundled Chromium
        try:
            browser = p.chromium.launch(headless=True, channel="chrome")
        except Exception:
            browser = p.chromium.launch(headless=True)

        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        total = len(links)
        for i, item in enumerate(links):
            idx = item["id"]
            title = item["title"]
            url = item["url"]
            success = False

            for attempt in range(1, max_retries + 1):
                print(f"[{idx}/{total:03d}] {'(retry '+str(attempt)+') ' if attempt > 1 else ''}{title}")
                try:
                    page.goto(url, wait_until=config["wait_until"], timeout=30000)
                    content = extract_content(page, platform)

                    if content and len(content.strip()) > 30:
                        filepath = generate_markdown(
                            item, content, output_dir, config["name"],
                            categories, tags
                        )
                        print(f"  -> OK: {os.path.basename(filepath)}")
                        results["success"].append(idx)
                        success = True
                        break
                    else:
                        print(f"  -> Content too short, retrying...")
                except Exception as e:
                    print(f"  -> Error: {e}")

                if attempt < max_retries:
                    wait = 5 * attempt
                    print(f"  -> Waiting {wait}s...")
                    time.sleep(wait)

            if not success:
                print(f"  -> FAILED after {max_retries} attempts")
                generate_markdown(
                    item, "(内容获取失败，请手动访问原文链接查看)",
                    output_dir, config["name"], categories, tags
                )
                results["failed"].append(idx)

            time.sleep(1)  # Polite delay

        browser.close()

    print(f"\n=== Done: {len(results['success'])} success, {len(results['failed'])} failed ===")
    if results['failed']:
        print(f"Failed IDs: {results['failed']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Crawl web content using headless browser")
    parser.add_argument("--platform", required=True,
                        choices=list(PLATFORM_CONFIG.keys()),
                        help="Target platform")
    parser.add_argument("--links", required=True,
                        help="Path to JSON file with links")
    parser.add_argument("--output", required=True,
                        help="Output directory for markdown files")
    parser.add_argument("--categories", default="未分类",
                        help="Hexo categories (comma-separated)")
    parser.add_argument("--tags", default="爬取",
                        help="Hexo tags (comma-separated)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Max retries per link (default: 3)")
    args = parser.parse_args()

    with open(args.links, 'r', encoding='utf-8') as f:
        links = json.load(f)

    print(f"Platform: {PLATFORM_CONFIG[args.platform]['name']}")
    print(f"Links: {len(links)}")
    print(f"Output: {args.output}")
    print()

    crawl(links, args.platform, args.output,
          args.categories, args.tags, args.retries)


if __name__ == '__main__':
    main()
