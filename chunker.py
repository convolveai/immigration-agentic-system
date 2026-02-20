import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import json

import httpx
import trafilatura

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter


SITEMAP_FILE = "sitemap.xml"
START = 1400  # 0-indexed URL position to start from (after filtering)
N_URLS: Optional[int] = None  # set to None to parse until end of sitemap
OUTPUT_FILE = "chunks_output.json"

# Chunking knobs (tune later)
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64


def parse_sitemap(path: str, start: int, limit: Optional[int]) -> List[str]:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls: List[str] = []
    seen = 0
    for url_node in root.findall("sm:url", ns):
        loc = url_node.find("sm:loc", ns)
        if loc is not None and loc.text:
            url = loc.text.strip()
            if "expired" in url.lower():
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


def main() -> Dict[str, Any]:
    urls = parse_sitemap(SITEMAP_FILE, START, N_URLS)
    n_label = "all" if N_URLS is None else N_URLS
    print(f"Parsed {len(urls)} URLs from sitemap (start={START}, n={n_label}).\n")

    headers = {"User-Agent": "IRCC-Assistant-Indexer/0.1 (+contact: you@example.com)"}

    docs: List[Document] = []
    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        for i, url in enumerate(urls, start=1):
            print(f"[{i}/{len(urls)}] Fetching: {url}")
            r = client.get(url)
            r.raise_for_status()

            text = extract_main_text(r.text, url)

            # Skip empty / super short pages
            if len(text.strip()) < 300:
                print(f"  -> Skipping (too short): {len(text)} chars")
                continue

            # Store as a LlamaIndex Document with metadata
            docs.append(
                Document(
                    text=text,
                    metadata={
                        "url": url,
                        "source": "canada.ca",
                    },
                )
            )

    print(f"\nBuilt {len(docs)} documents. Chunking...\n")

    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    # Convert docs -> chunks (nodes)
    nodes = splitter.get_nodes_from_documents(docs)

    # Store chunks in memory (plain Python objects)
    chunks: List[Dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        chunks.append(
            {
                "chunk_id": f"chunk_{idx}",
                "text": node.get_content(),
                "metadata": node.metadata,  # includes url/source
            }
        )

    print(f"Created {len(chunks)} chunks total.\n")

    # Preview a few chunks
    for c in chunks[:5]:
        print("=" * 90)
        print("URL:", c["metadata"]["url"])
        print(c["text"][:700] + ("…" if len(c["text"]) > 700 else ""))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved output to {OUTPUT_FILE}")

    return {"docs": docs, "chunks": chunks}


if __name__ == "__main__":
    main()