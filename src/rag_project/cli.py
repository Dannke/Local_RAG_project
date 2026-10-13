"""Command-line interface for the local RAG project."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from rag_project.application.chat import ChatSession
from rag_project.bootstrap import create_chat_session, ingest_to_faiss, search_index
from rag_project.domain.errors import LLMClientError
from rag_project.domain.models import SearchResult
from rag_project.logging_setup import setup_logging

EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit"})


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(
        console_level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING
    )
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


def print_search_results(results: Sequence[SearchResult]) -> None:
    for number, result in enumerate(results, start=1):
        source = result.document.relative_path or result.document.id
        print(f"[{number}] score={result.score:.4f} source={source}")
        print(result.document.text[:700].strip())
        print()


def run_chat_loop(session: ChatSession, top_k: int | None = None) -> int:
    print("RAG chat mode. Type /exit or /quit to stop.")
    while True:
        question = input("you> ").strip()
        if question.lower() in EXIT_COMMANDS:
            print("bye")
            return 0
        if not question:
            continue

        try:
            response = session.ask(question, top_k=top_k)
        except LLMClientError as exc:
            print(f"\nassistant> LLM error: {exc}\n")
            continue
        print(f"\nassistant> {response.answer}\n")
        if response.contexts:
            sources = [context.relative_path or context.id for context in response.contexts]
            print("sources: " + ", ".join(sources) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load documents and save a FAISS index.")
    ingest_parser.add_argument("--data", type=Path, default=Path("data/raw"))
    ingest_parser.add_argument("--index", type=Path, default=Path("data/vector_store"))
    ingest_parser.set_defaults(handler=_run_ingest)

    search_parser = subparsers.add_parser("search", help="Search the saved FAISS index.")
    search_parser.add_argument("question")
    search_parser.add_argument("--index", type=Path, default=Path("data/vector_store"))
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.set_defaults(handler=_run_search)

    chat_parser = subparsers.add_parser("chat", help="Interactive chat over the saved FAISS index.")
    chat_parser.add_argument("--index", type=Path, default=Path("data/vector_store"))
    chat_parser.add_argument("--top-k", type=int, default=None)
    chat_parser.add_argument("--verbose", action="store_true")
    chat_parser.set_defaults(handler=_run_chat)

    return parser


def _run_ingest(args: argparse.Namespace) -> int:
    indexed_count = ingest_to_faiss(input_dir=args.data, index_dir=args.index)
    print(f"Indexed {indexed_count} chunks from {args.data} into {args.index}.")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    print_search_results(search_index(args.question, index_dir=args.index, top_k=args.top_k))
    return 0


def _run_chat(args: argparse.Namespace) -> int:
    return run_chat_loop(create_chat_session(index_dir=args.index), top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
