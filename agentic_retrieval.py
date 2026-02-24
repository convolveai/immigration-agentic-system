import json
import os
import requests
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import create_retriever_tool, tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openrouter import ChatOpenRouter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "x-ai/grok-4.1-fast"
TOP_K = 5
PERMANENT_RESIDENCE_COLLECTION = "permanent-residence-v1"
TEMPORARY_RESIDENCE_COLLECTION = "temporary-residents-v1"
AGENT_LOG_FILE = Path(__file__).resolve().parent / "agent_updates.log"


def _serialize_for_log(obj: object) -> object:
    """Convert LangChain messages and other objects to JSON-serializable form."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(type(obj).__name__)


def _log_raw_update(log_path: Path, update: object) -> None:
    """Append one raw stream update as indented JSON."""
    try:
        payload = json.dumps(
            update,
            default=_serialize_for_log,
            ensure_ascii=False,
            indent=2,
        )
    except (TypeError, ValueError):
        payload = json.dumps(
            {"raw_repr": repr(update)},
            ensure_ascii=False,
            indent=2,
        )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(payload + "\n\n")


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

def extract_response_text(data: dict) -> str:
    for item in data.get("output") or []:
        if item.get("type") != "message":
            continue
        parts = []
        for block in item.get("content") or []:
            if block.get("type") == "output_text" and "text" in block:
                parts.append(block["text"])
        if parts:
            return "".join(parts)
    return ""

@tool
def perform_ai_web_search(question: str) -> str:
    """Search the web using AI for IRCC and government information. Use only official sources."""
    
    load_dotenv_file()
    url = 'https://openrouter.ai/api/v1/responses'
    headers = {
        "Authorization": f'Bearer {require_env("OPENROUTER_API_KEY")}',
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHAT_MODEL,
        "instructions": "You are a web search assistant for Canadian immigration (IRCC) and government information. Use only official sources (e.g. canada.ca, government domains). Respond with a clear, focused breakdown of what you found: key points, eligibility or requirements if relevant, and source links. Do not speculate; if the answer is not in official sources, say so.",
        "input": question,
        "plugins": [{"id": "web", "max_results": 3}],
        "max_output_tokens": 9000,
        "reasoning": {
            "effort": "low"
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    text = extract_response_text(data)
    return text
        

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_agent():
    load_dotenv_file()

    qdrant_url = require_env("QDRANT_URL")
    qdrant_api_key = require_env("QDRANT_API_KEY")
    require_env("OPENAI_API_KEY")

    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    temporary_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=TEMPORARY_RESIDENCE_COLLECTION,
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

    # llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    llm = ChatOpenRouter(
        model=CHAT_MODEL,
        temperature=0,
    )

    return create_agent(
        model=llm,
        tools=[temporary_residence_tool, permanent_residence_tool, perform_ai_web_search],
        system_prompt=(
            "You are an assistant for IRCC temporary and permanent residence policy questions. "
            "Use web search first to find official government/IRCC information. "
            "If the web search results do NOT contain an answer from official sources (e.g. no canada.ca / government sources, or unclear/off-topic), "
            "you MUST search the knowledge base using the temporary residence and/or permanent residence information tools. "
            "Do NOT use or rely on answers from the web search when they are not from official sources—use the knowledge base instead. "
            "If neither official web results nor the knowledge base gives enough to answer confidently, say you don't know."
        ),
    )


def main() -> None:
    agent = build_agent()
    AGENT_LOG_FILE.write_text("", encoding="utf-8")

    while True:
        question = input("\nQuestion (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        print("\n=== Agent Updates ===")
        with open(AGENT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"event": "question", "question": question},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n\n"
            )
        messages = []
        final_answer = None
        seen_tool_calls = set()
        seen_tool_results = set()
        for update in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="updates",
        ):
            _log_raw_update(AGENT_LOG_FILE, update)

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
