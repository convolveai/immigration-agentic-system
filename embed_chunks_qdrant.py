import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


INPUT_FILE = "chunks.jsonl"
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 200
RECREATE_COLLECTION = True  # set True to drop/recreate collection before upsert


def load_dotenv_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            if isinstance(text, str) and text.strip():
                chunks.append(row)
    return chunks


def batched(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
) -> None:
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    if exists and recreate:
        client.delete_collection(collection_name=collection_name)
        exists = False

    if exists:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def build_points(
    chunks: List[Dict[str, Any]],
    vectors: List[List[float]],
) -> List[PointStruct]:
    points: List[PointStruct] = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        metadata = chunk.get("metadata", {}) or {}
        url = metadata.get("url")
        source = metadata.get("source")
        status = metadata.get("status")
        chars = metadata.get("chars")
        chunk_id = chunk.get("chunk_id")

        payload = {
            "text": chunk.get("text", ""),
            "metadata": {
                "url": url,
                "source": source,
                "status": status,
                "chars": chars,
                "chunk_id": chunk_id,
            },
        }

        raw_id = chunk.get("chunk_id")
        point_id: int | str
        if isinstance(raw_id, int) and raw_id >= 0:
            point_id = raw_id
        elif isinstance(raw_id, str):
            if raw_id.isdigit():
                point_id = int(raw_id)
            elif raw_id.startswith("chunk_") and raw_id[6:].isdigit():
                point_id = int(raw_id[6:])
            else:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
        else:
            fallback = f"chunk-{idx}-{url or ''}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, fallback))

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    return points


def main() -> None:
    load_dotenv_file()

    qdrant_url = require_env("QDRANT_URL")
    qdrant_api_key = require_env("QDRANT_API_KEY")
    qdrant_collection = require_env("QDRANT_COLLECTION")
    require_env("OPENAI_API_KEY")

    chunks = load_chunks(INPUT_FILE)
    if not chunks:
        print(f"No chunks found in {INPUT_FILE}. Nothing to embed.")
        return

    print(f"Loaded {len(chunks)} chunks from {INPUT_FILE}.")

    openai_client = OpenAI()
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    all_vectors: List[List[float]] = []
    chunk_batches = batched(chunks, EMBED_BATCH_SIZE)
    for i, batch in enumerate(chunk_batches, start=1):
        texts = [item["text"] for item in batch]
        response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = [item.embedding for item in response.data]
        all_vectors.extend(vectors)
        print(f"Embedded batch {i}/{len(chunk_batches)} ({len(batch)} chunks)")

    ensure_collection(
        client=qdrant_client,
        collection_name=qdrant_collection,
        vector_size=len(all_vectors[0]),
        recreate=RECREATE_COLLECTION,
    )

    points = build_points(chunks, all_vectors)
    point_batches = batched(points, UPSERT_BATCH_SIZE)
    for i, batch in enumerate(point_batches, start=1):
        qdrant_client.upsert(collection_name=qdrant_collection, points=batch)
        print(f"Upserted batch {i}/{len(point_batches)} ({len(batch)} points)")

    print(
        f"Done. Embedded and upserted {len(points)} chunks into collection '{qdrant_collection}'."
    )


if __name__ == "__main__":
    main()
