import json
from typing import Any, Dict, List

from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter


INPUT_FILE = "extracted_pages.jsonl"
OUTPUT_FILE = "chunks.jsonl"
MIN_TEXT_CHARS = 300

# Chunking knobs
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64


def load_extracted_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def should_chunk(row: Dict[str, Any]) -> bool:
    if row.get("error"):
        return False
    if row.get("skip_chunking"):
        return False
    text = row.get("text", "")
    return isinstance(text, str) and len(text.strip()) >= MIN_TEXT_CHARS


def main() -> List[Dict[str, Any]]:
    rows = load_extracted_rows(INPUT_FILE)
    print(f"Loaded {len(rows)} extracted rows from {INPUT_FILE}.")

    docs: List[Document] = []
    for row in rows:
        if not should_chunk(row):
            continue

        docs.append(
            Document(
                text=row["text"],
                metadata={
                    "url": row.get("url"),
                    "source": row.get("source", "canada.ca"),
                    "status": row.get("status"),
                    "chars": row.get("chars"),
                },
            )
        )

    print(f"Chunking {len(docs)} documents...")

    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    nodes = splitter.get_nodes_from_documents(docs)

    chunks: List[Dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        chunks.append(
            {
                "chunk_id": f"chunk_{idx}",
                "text": node.get_content(),
                "metadata": node.metadata,
            }
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to {OUTPUT_FILE}.")
    return chunks


if __name__ == "__main__":
    main()
