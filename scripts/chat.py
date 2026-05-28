"""Start an interactive RAG chat session over the saved FAISS index."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_project.llm.llm_client import LLMClientError  # noqa: E402
from rag_project.pipelines.chat_pipeline import ChatSession  # noqa: E402


def main() -> int:
    index_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "vector_store"
    session = ChatSession.from_faiss_index(index_dir=index_dir)

    print("RAG chat mode. Type /exit or /quit to stop.")
    while True:
        question = input("you> ").strip()
        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("bye")
            return 0
        if not question:
            continue

        try:
            response = session.ask(question)
        except LLMClientError as exc:
            print(f"\nassistant> LLM error: {exc}\n")
            continue
        print(f"\nassistant> {response.answer}\n")
        if response.contexts:
            sources = [context.metadata.get("relative_path", context.id) for context in response.contexts]
            print("sources: " + ", ".join(sources) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
