import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import json

import httpx
import trafilatura

SITEMAP_FILE = "sitemap.xml"  # path to your local sitemap XML
START = 1199  # 0-indexed URL position to start sampling from
N = 8  # number of URLs to sample
OUTPUT_FILE = "sample_output.json"  # output file for extracted data


def parse_sitemap(path: str, start: int, limit: int) -> List[str]:
    tree = ET.parse(path)
    root = tree.getroot()

    # Handle the default sitemap namespace
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls: List[str] = []
    seen = 0
    for url_node in root.findall("sm:url", ns):
        loc = url_node.find("sm:loc", ns)
        if loc is not None and loc.text:
            if seen < start:
                seen += 1
                continue
            urls.append(loc.text.strip())
            seen += 1
        if len(urls) >= limit:
            break

    return urls


def extract_main_text(html: str, url: str) -> str:
    text = trafilatura.extract(
        html,
        url=url,
        include_tables=True,
        include_comments=False,
    )
    return text or ""


def main() -> List[Dict[str, Any]]:
    urls = parse_sitemap(SITEMAP_FILE, START, N)
    print(f"Parsed {len(urls)} URLs from {SITEMAP_FILE} (start={START}, n={N})\n")

    pages_data: List[Dict[str, Any]] = []

    headers = {
        "User-Agent": "IRCC-Assistant-Indexer/0.1 (+contact: you@example.com)"
    }

    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        for i, url in enumerate(urls, start=1):
            print(f"[{i}/{len(urls)}] Fetching: {url}")
            try:
                r = client.get(url)
                r.raise_for_status()

                text = extract_main_text(r.text, url)

                pages_data.append({
                    "url": url,
                    "status": r.status_code,
                    "chars": len(text),
                    "text": text,
                })

            except Exception as e:
                pages_data.append({
                    "url": url,
                    "error": str(e),
                    "text": "",
                })

    # Quick preview
    print("\n--- Preview ---")
    for page in pages_data:
        print("=" * 90)
        print(page["url"])
        if page.get("error"):
            print(f"ERROR: {page['error']}")
        else:
            preview = (page["text"][:900] + "…") if len(page["text"]) > 900 else page["text"]
            print(preview)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved output to {OUTPUT_FILE}")

    return pages_data


if __name__ == "__main__":
    main()