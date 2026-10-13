import json
import tempfile
import unittest
from pathlib import Path

from rag_project.domain.models import Document, SearchResult
from rag_project.infrastructure.chat_store import (
    create_chat,
    get_chat_sources_path,
    load_search_results,
    save_search_results,
)


def sample_results() -> list[SearchResult]:
    return [
        SearchResult(
            document=Document(
                id="docs/manual.pdf:page-0001:0000",
                text="Первая страница руководства.",
                metadata={
                    "relative_path": "docs/manual.pdf",
                    "file_type": "pdf",
                    "page": 1,
                    "page_label": "1",
                    "chunk_index": 0,
                },
            ),
            score=0.8734,
        ),
        SearchResult(
            document=Document(id="notes.txt:0000", text="Plain text note.", metadata={}),
            score=-0.25,
        ),
    ]


class SearchResultsRoundTripTests(unittest.TestCase):
    def test_search_results_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            chat = create_chat(Path(temp_dir) / "chats", title="Sources")
            results = sample_results()

            save_search_results(chat.path, results)
            loaded = load_search_results(chat.path)

            self.assertEqual(loaded, results)
            self.assertTrue(all(isinstance(result, SearchResult) for result in loaded))

    def test_fresh_chat_has_no_search_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            chat = create_chat(Path(temp_dir) / "chats")

            self.assertEqual(load_search_results(chat.path), [])

    def test_saving_an_empty_list_replaces_previous_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            chat = create_chat(Path(temp_dir) / "chats")
            save_search_results(chat.path, sample_results())

            save_search_results(chat.path, [])

            self.assertEqual(load_search_results(chat.path), [])

    def test_missing_sources_file_yields_no_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            chat = create_chat(Path(temp_dir) / "chats")
            get_chat_sources_path(chat.path).unlink()

            self.assertEqual(load_search_results(chat.path), [])


class MalformedSourcesTests(unittest.TestCase):
    def _write_sources(self, chat_dir: Path, raw: str) -> None:
        get_chat_sources_path(chat_dir).write_text(raw, encoding="utf-8")

    def test_malformed_entries_are_skipped(self) -> None:
        payload = [
            {"document": {"id": "valid", "text": "kept", "metadata": {"page": 2}}, "score": 0.5},
            "not a dict",
            42,
            None,
            {"score": 0.9},
            {"document": "not a dict", "score": 0.9},
            {"document": {"id": 7, "text": "id is not a string"}, "score": 0.9},
            {"document": {"id": "no-text"}, "score": 0.9},
            {"document": {"id": "bad-meta", "text": "kept", "metadata": ["x"]}, "score": "high"},
            {"document": {"id": "no-score", "text": "kept"}},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            chat = create_chat(Path(temp_dir) / "chats")
            self._write_sources(chat.path, json.dumps(payload))

            loaded = load_search_results(chat.path)

            self.assertEqual(
                loaded,
                [
                    SearchResult(Document(id="valid", text="kept", metadata={"page": 2}), 0.5),
                    SearchResult(Document(id="bad-meta", text="kept", metadata={}), 0.0),
                    SearchResult(Document(id="no-score", text="kept", metadata={}), 0.0),
                ],
            )

    def test_non_list_or_corrupt_payload_yields_no_results(self) -> None:
        for raw in ('{"document": {"id": "a", "text": "b"}}', "{not json", "null", ""):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as temp_dir:
                chat = create_chat(Path(temp_dir) / "chats")
                self._write_sources(chat.path, raw)

                self.assertEqual(load_search_results(chat.path), [])
