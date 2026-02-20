import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import trafilatura


SITEMAP_FILE = "sitemap.xml"
START = 0  # 0-indexed URL position to start from (after filtering)
N_URLS: Optional[int] = None  # set to None to parse until end of sitemap
MIN_TEXT_CHARS = 300
OUTPUT_FILE = "extracted_pages.jsonl"
BASE_PATH = (
    "/en/immigration-refugees-citizenship/corporate/publications-manuals/"
    "operational-bulletins-manuals"
)
SUB_ROUTE_FILTER: Optional[str] = "temporary-residents"  # e.g. "temporary-residents"; None disables


def is_in_scope(url: str, base_path: str, sub_route: Optional[str]) -> bool:
    path = urlparse(url).path.rstrip("/")
    normalized_base = base_path.rstrip("/")

    if not path.startswith(normalized_base + "/"):
        return False

    if not sub_route:
        return True

    normalized_sub = sub_route.strip("/")
    sub_prefix = f"{normalized_base}/{normalized_sub}"
    return (
        path == sub_prefix
        or path == sub_prefix + ".html"
        or path == sub_prefix + ".htm"
        or path.startswith(sub_prefix + "/")
    )


def parse_sitemap(path: str, start: int, limit: Optional[int]) -> List[str]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls: List[str] = []
    seen = 0
    for url_node in root.findall("sm:url", ns):
        loc = url_node.find("sm:loc", ns)
        if loc is None or not loc.text:
            continue

        url = loc.text.strip()
        if "expired" in url.lower():
            continue
        if not is_in_scope(url, BASE_PATH, SUB_ROUTE_FILTER):
            continue

        if seen < start:
            seen += 1
            continue

        urls.append(url)
        seen += 1

        if limit is not None and len(urls) >= limit:
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
    urls = parse_sitemap(SITEMAP_FILE, START, N_URLS)
    n_label = "all" if N_URLS is None else N_URLS
    sub_label = SUB_ROUTE_FILTER if SUB_ROUTE_FILTER else "all"
    print(
        f"Parsed {len(urls)} URLs from sitemap "
        f"(start={START}, n={n_label}, sub_route={sub_label}).\n"
    )

    headers = {"User-Agent": "IRCC-Assistant-Indexer/0.1 (+contact: you@example.com)"}

    rows: List[Dict[str, Any]] = []
    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        for i, url in enumerate(urls, start=1):
            print(f"[{i}/{len(urls)}] Fetching: {url}")
            row: Dict[str, Any] = {"url": url, "source": "canada.ca"}
            try:
                response = client.get(url)
                response.raise_for_status()

                text = extract_main_text(response.text, url)
                row.update(
                    {
                        "status": response.status_code,
                        "chars": len(text),
                        "text": text,
                        "skip_chunking": len(text.strip()) < MIN_TEXT_CHARS,
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "error": str(exc),
                        "status": None,
                        "chars": 0,
                        "text": "",
                        "skip_chunking": True,
                    }
                )

            rows.append(row)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    skipped = sum(1 for r in rows if r.get("skip_chunking"))
    print(f"\nSaved {len(rows)} extracted pages to {OUTPUT_FILE} (skip_chunking={skipped}).")

    return rows


if __name__ == "__main__":
    main()
