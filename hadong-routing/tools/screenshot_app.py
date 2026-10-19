"""Chụp ảnh màn hình ứng dụng Streamlit (cần Streamlit đang chạy và Playwright + Microsoft Edge/Chrome).

python tools/screenshot_app.py --url "http://localhost:8501/?o=100&d=2500&h=8" --out reports/figures/app_screenshot.png
"""
from __future__ import annotations

import argparse

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", default="msedge", help="msedge | chrome")
    ap.add_argument("--wait-ms", type=int, default=25000)
    args = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1700, "height": 1150}, color_scheme="light")
        page.goto(args.url, wait_until="networkidle", timeout=120000)
        page.wait_for_selector("iframe", timeout=120000)          # bản đồ folium
        page.wait_for_selector("[data-testid='stDataFrame']", timeout=120000)
        page.wait_for_timeout(args.wait_ms)                        # chờ tile bản đồ và biểu đồ
        page.screenshot(path=args.out, full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
