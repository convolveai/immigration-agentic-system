import argparse
import os
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
DEFAULT_TOP_K = 5


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


def build_context(docs: List) -> str:
    context_blocks: List[str] = []
    for i, doc in enumerate(docs, start=1):
        url = (doc.metadata or {}).get("url", "unknown")
        context_blocks.append(f"[Doc {i}] URL: {url}\n{doc.page_content}")
    return "\n\n".join(context_blocks)


def run_query(question: str, top_k: int) -> None:
    load_dotenv_file()

    qdrant_url = require_env("QDRANT_URL")
    qdrant_api_key = require_env("QDRANT_API_KEY")
    qdrant_collection = require_env("QDRANT_COLLECTION")
    require_env("OPENAI_API_KEY")

    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=qdrant_collection,
        embedding=embeddings,
        content_payload_key="text",
        metadata_payload_key="metadata",
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    docs = retriever.invoke(question)

    if not docs:
        print("No matching documents found.")
        return

    context = build_context(docs)
    prompt = (
        "You are a helpful assistant answering questions using only the provided context. "
        "If the answer is not in context, say you don't have enough information.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Provide a concise answer and include source URLs at the end."
    )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    response = llm.invoke(prompt)

    print("\n=== Answer ===\n")
    print(response.content)

    print("\n=== Retrieved Sources ===")
    seen = set()
    for doc in docs:
        url = (doc.metadata or {}).get("url")
        if url and url not in seen:
            seen.add(url)
            print(f"- {url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple LangChain RAG retrieval over Qdrant."
    )
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Top-k results")
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    run_query(question=question, top_k=args.k)


if __name__ == "__main__":
    main()
