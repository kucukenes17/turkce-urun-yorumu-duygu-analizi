"""Çalışan Gradio demosunun aydınlık + karanlık ekran görüntülerini kaydeder.

Önce demoyu başlatın (python app.py), sonra:
    python scripts/screenshot_demo.py

Çıktılar: docs/images/demo-light.png, docs/images/demo-dark.png
"""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860"
TEXT = "Kötü diyemem, fiyatına göre gayet iyi."
OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 880},
                                device_scale_factor=2)
        page.goto(URL, wait_until="networkidle")
        page.fill("textarea", TEXT)
        page.get_by_role("button", name="Analiz et").click()
        time.sleep(3)
        page.screenshot(path=str(OUT / "demo-light.png"))
        print("kaydedildi:", OUT / "demo-light.png")

        page.click("#theme-toggle-btn")
        time.sleep(1)
        page.screenshot(path=str(OUT / "demo-dark.png"))
        print("kaydedildi:", OUT / "demo-dark.png")

        browser.close()


if __name__ == "__main__":
    main()
