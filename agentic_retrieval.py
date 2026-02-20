import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
TOP_K = 5
PERMANENT_RESIDENCE_COLLECTION = "permanent-residence-v1"


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


def build_agent():
    load_dotenv_file()

    qdrant_url = require_env("QDRANT_URL")
    qdrant_api_key = require_env("QDRANT_API_KEY")
    qdrant_collection = require_env("QDRANT_COLLECTION")
    require_env("OPENAI_API_KEY")

    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    temporary_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=qdrant_collection,
        embedding=embeddings,
        content_payload_key="text",
        metadata_payload_key="metadata",
    )

    temporary_retriever = temporary_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    temporary_residence_tool = create_retriever_tool(
        retriever=temporary_retriever,
        name="temporary_residence_information_tool",
        description=(
            "Use this tool to answer questions about IRCC operational manuals "
            "for temporary residents."
        ),
    )

    permanent_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=PERMANENT_RESIDENCE_COLLECTION,
        embedding=embeddings,
        content_payload_key="text",
        metadata_payload_key="metadata",
    )

    permanent_retriever = permanent_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    permanent_residence_tool = create_retriever_tool(
        retriever=permanent_retriever,
        name="permanent_residence_information_tool",
        description=(
            "Use this tool to answer questions about IRCC operational manuals "
            "for permanent residence."
        ),
    )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    return create_agent(
        model=llm,
        tools=[temporary_residence_tool, permanent_residence_tool],
        system_prompt=(
            "You are an assistant for IRCC temporary and permanent residence policy questions. "
            "Use the most relevant tool for the user's question. "
            "If information is missing, say you don't know."
        ),
    )


def main() -> None:
    agent = build_agent()

    while True:
        question = input("\nQuestion (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        print("\n=== Agent Updates ===")
        messages = []
        final_answer = None
        seen_tool_calls = set()
        seen_tool_results = set()
        for update in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="updates",
        ):
            if not isinstance(update, dict):
                continue

            for node_name, node_state in update.items():
                print(f"[{node_name}] update")
                if isinstance(node_state, dict) and "messages" in node_state:
                    messages = node_state.get("messages", messages)

                    for message in messages:
                        if isinstance(message, AIMessage):
                            for tool_call in message.tool_calls or []:
                                tool_call_id = tool_call.get("id") or "unknown"
                                if tool_call_id in seen_tool_calls:
                                    continue
                                seen_tool_calls.add(tool_call_id)

                                print(
                                    f"[tool_call_start] name={tool_call.get('name')} id={tool_call_id}"
                                )
                                print(f"[tool_call_args] {tool_call.get('args')}")

                            if message.content and not (message.tool_calls or []):
                                final_answer = message.content

                        elif isinstance(message, ToolMessage):
                            result_id = message.tool_call_id or str(id(message))
                            if result_id in seen_tool_results:
                                continue
                            seen_tool_results.add(result_id)

                            content = message.content if isinstance(message.content, str) else str(message.content)
                            preview = content[:1200]
                            suffix = "…" if len(content) > 1200 else ""
                            print(f"[tool_result] id={result_id}")
                            print(f"{preview}{suffix}")

        if not messages:
            print("\nNo response returned.")
            continue

        print("\n=== Answer ===\n")
        print(final_answer if final_answer else messages[-1].content)


if __name__ == "__main__":
    main()
