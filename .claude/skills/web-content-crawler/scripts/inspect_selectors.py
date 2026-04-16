#!/usr/bin/env python3
"""
Inspect a webpage's DOM structure to discover content selectors.
Use this when adding support for a new platform.

Usage:
    python3 inspect_selectors.py <url>
"""

import sys
import time
from playwright.sync_api import sync_playwright

KEYWORDS = ['post', 'topic', 'content', 'detail', 'article', 'feed',
            'rich', 'discuss', 'body', 'text', 'note', 'entry', 'desc',
            'main', 'answer', 'blog']


def inspect(url: str):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, channel="chrome")
        except Exception:
            browser = p.chromium.launch(headless=True)

        page = browser.new_page()
        print(f"Loading: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(4)

        elements = page.query_selector_all('[class]')
        seen = set()
        results = []

        for el in elements:
            cls = el.get_attribute('class') or ''
            for c in cls.split():
                if any(kw in c.lower() for kw in KEYWORDS):
                    if c not in seen:
                        seen.add(c)
                        text = el.inner_text() or ''
                        preview = text[:80].replace('\n', ' ')
                        results.append((c, len(text), preview))

        results.sort(key=lambda x: -x[1])
        print(f"\nFound {len(results)} content-related classes:\n")
        print(f"{'CLASS':<45} {'CHARS':>6}  PREVIEW")
        print("-" * 100)
        for cls, length, preview in results[:30]:
            print(f".{cls:<44} {length:>6}  {preview}")

        browser.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <url>")
        sys.exit(1)
    inspect(sys.argv[1])
