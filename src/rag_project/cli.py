"""Command-line interface for the local RAG project."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag_project.llm.llm_client import LLMClientError
from rag_project.logging_setup import setup_logging
from rag_project.pipelines.chat_pipeline import ChatSession
from rag_project.pipelines.ingest_pipeline import ingest_to_faiss
from rag_project.pipelines.search_pipeline import search_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load documents and save a FAISS index.")
    ingest_parser.add_argument("--data", type=Path, default=Path("data/raw"))
    ingest_parser.add_argument("--index", type=Path, default=Path("data/vector_store"))

    search_parser = subparsers.add_parser("search", help="Search the saved FAISS index.")
    search_parser.add_argument("question")
    search_parser.add_argument("--index", type=Path, default=Path("data/vector_store"))
    search_parser.add_argument("--top-k", type=int, default=5)

    chat_parser = subparsers.add_parser("chat", help="Interactive chat over the saved FAISS index.")
    chat_parser.add_argument("--index", type=Path, default=Path("data/vector_store"))
    chat_parser.add_argument("--top-k", type=int, default=None)
    chat_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    setup_logging(
        console_level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    )

    if args.command == "ingest":
        indexed_count = ingest_to_faiss(input_dir=args.data, index_dir=args.index)
        print(f"Indexed {indexed_count} chunks from {args.data} into {args.index}.")
        return 0

    if args.command == "search":
        results = search_index(args.question, index_dir=args.index, top_k=args.top_k)
        for number, result in enumerate(results, start=1):
            source = result.document.metadata.get("relative_path", result.document.id)
            print(f"[{number}] score={result.score:.4f} source={source}")
            print(result.document.text[:700].strip())
            print()
        return 0

    if args.command == "chat":
        session = ChatSession.from_faiss_index(index_dir=args.index)
        print("RAG chat mode. Type /exit or /quit to stop.")
        while True:
            question = input("you> ").strip()
            if question.lower() in {"/exit", "/quit", "exit", "quit"}:
                print("bye")
                return 0
            if not question:
                continue

            try:
                response = session.ask(question, top_k=args.top_k)
            except LLMClientError as exc:
                print(f"\nassistant> LLM error: {exc}\n")
                continue
            print(f"\nassistant> {response.answer}\n")
            if response.contexts:
                sources = [
                    context.metadata.get("relative_path", context.id)
                    for context in response.contexts
                ]
                print("sources: " + ", ".join(sources) + "\n")

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
