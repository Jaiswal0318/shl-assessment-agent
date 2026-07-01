"""
SHL Product Catalog Scraper (Playwright).

Scrapes Individual Test Solutions from the SHL product catalog and saves
structured JSON for downstream embedding and FAISS indexing.

Usage:
    python -m app.scraper
    python -m app.scraper --skip-details
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOGUE_BASE = "https://www.shl.com/solutions/products/product-catalog/"
VALID_TYPES = {"A", "B", "C", "D", "E", "K", "P", "S"}

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "shl_product_catalog.json"


def _extract_page_assessments(page) -> list[dict]:
    """Extract assessment rows from the current catalog page."""
    items: list[dict] = []
    links = page.locator('a[href*="/product-catalog/view/"]')
    count = links.count()

    seen_urls: set[str] = set()
    for i in range(count):
        link = links.nth(i)
        href = link.get_attribute("href") or ""
        name = (link.inner_text() or "").strip()
        if not href or not name:
            continue

        full_url = urljoin(BASE_URL, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        row = link.locator("xpath=ancestor::tr[1]")
        test_types: list[str] = []
        remote = "no"
        adaptive = "no"

        if row.count() > 0:
            row_text = row.inner_text()
            for code in VALID_TYPES:
                if re.search(rf"\b{code}\b", row_text):
                    test_types.append(code)
            if "Yes" in row_text or row.locator("span.catalogue__circle, i").count() > 0:
                remote = "yes"

        slug = full_url.rstrip("/").split("/")[-1]
        items.append(
            {
                "entity_id": slug,
                "name": name,
                "link": full_url,
                "keys": _types_to_keys(test_types),
                "description": "",
                "job_levels": [],
                "languages": ["English (US)"],
                "duration": "",
                "duration_raw": "",
                "remote": remote,
                "adaptive": adaptive,
                "status": "ok",
            }
        )

    return items


def _types_to_keys(test_types: list[str]) -> list[str]:
    mapping = {
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "A": "Ability & Aptitude",
        "S": "Simulations",
        "B": "Biodata & Situational Judgment",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
    }
    return [mapping[t] for t in test_types if t in mapping]


def scrape_catalogue(max_pages: int = 40) -> list[dict]:
    """Scrape all Individual Test Solutions using Playwright."""
    assessments: list[dict] = []
    seen_urls: set[str] = set()
    consecutive_empty = 0

    logger.info("Starting Playwright scrape: %s", CATALOGUE_BASE)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )

        for page_num in range(max_pages):
            start = page_num * 12
            url = f"{CATALOGUE_BASE}?start={start}&type=1"
            logger.info("Page %d: %s", page_num + 1, url)

            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
                page_items = _extract_page_assessments(page)
            except Exception as exc:
                logger.warning("Failed page %d: %s", page_num + 1, exc)
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                continue

            if not page_items:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                continue

            consecutive_empty = 0
            new_count = 0
            for item in page_items:
                if item["link"] in seen_urls:
                    continue
                seen_urls.add(item["link"])
                assessments.append(item)
                new_count += 1

            logger.info("Found %d new assessments (total %d)", new_count, len(assessments))
            time.sleep(1.5)

        browser.close()

    return assessments


def enrich_details(assessments: list[dict]) -> list[dict]:
    """Visit individual assessment pages for descriptions and duration."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        for idx, item in enumerate(assessments, 1):
            if item.get("description"):
                continue
            try:
                page.goto(item["link"], wait_until="domcontentloaded", timeout=30000)
                body = page.inner_text("body")

                desc = ""
                for selector in [".product-catalogue__description", ".description", "main p"]:
                    loc = page.locator(selector).first
                    if loc.count() > 0:
                        text = loc.inner_text().strip()
                        if len(text) > 20:
                            desc = text[:500]
                            break

                if not desc:
                    paragraphs = page.locator("p")
                    for i in range(min(paragraphs.count(), 5)):
                        text = paragraphs.nth(i).inner_text().strip()
                        if len(text) > 30:
                            desc = text[:500]
                            break

                item["description"] = desc

                duration_match = re.search(r"(\d+)\s*(?:min|minute)", body, re.I)
                if duration_match:
                    minutes = duration_match.group(1)
                    item["duration"] = f"{minutes} minutes"
                    item["duration_raw"] = minutes

                logger.info("[%d/%d] %s", idx, len(assessments), item["name"])
                time.sleep(0.5)
            except Exception as exc:
                logger.warning("Detail fetch failed for %s: %s", item["name"], exc)

        browser.close()

    return assessments


def save_catalog(assessments: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d assessments to %s", len(assessments), OUTPUT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHL catalog scraper (Playwright)")
    parser.add_argument("--skip-details", action="store_true", help="Skip per-assessment detail pages")
    parser.add_argument("--max-pages", type=int, default=40)
    args = parser.parse_args()

    assessments = scrape_catalogue(max_pages=args.max_pages)
    if not args.skip_details and assessments:
        assessments = enrich_details(assessments)

    save_catalog(assessments)
    logger.info("Scrape complete: %d assessments", len(assessments))


if __name__ == "__main__":
    main()
