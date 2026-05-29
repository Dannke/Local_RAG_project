from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from rag_project.chat_store import (
    chats_root,
    create_chat,
    delete_chat,
    get_chat_documents_dir,
    get_chat_index_dir,
    initialize_chats,
    list_chats,
    load_messages,
    rename_chat,
    save_messages,
)


class ChatStoreTest(unittest.TestCase):
    def test_initialize_creates_first_chat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chats"

            chats = initialize_chats(root)

            self.assertEqual(len(chats), 1)
            self.assertEqual(chats[0].id, "chat_001")
            self.assertTrue((root / "chat_001" / "meta.json").exists())
            self.assertTrue((root / "chat_001" / "messages.json").exists())
            self.assertTrue(get_chat_documents_dir(chats[0].path).is_dir())
            self.assertTrue(get_chat_index_dir(chats[0].path).is_dir())

    def test_create_chat_uses_unique_incrementing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chats"

            first = create_chat(root, title="First")
            second = create_chat(root, title="Second")

            self.assertEqual(first.id, "chat_001")
            self.assertEqual(second.id, "chat_002")
            self.assertEqual([chat.id for chat in list_chats(root)], ["chat_001", "chat_002"])

    def test_meta_round_trip_and_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chats"
            now = datetime(2026, 5, 29, 10, 15, tzinfo=timezone.utc)

            chat = create_chat(root, now=now)
            renamed = rename_chat(root, chat.id, "Project notes")
            loaded = list_chats(root)[0]

            self.assertEqual(chat.title, "Chat 29.05 10:15")
            self.assertEqual(renamed.title, "Project notes")
            self.assertEqual(loaded.title, "Project notes")
            self.assertEqual(loaded.created_at, chat.created_at)

    def test_messages_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chats"
            chat = create_chat(root)
            messages = [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]

            save_messages(chat.path, messages)

            self.assertEqual(load_messages(chat.path), messages)

    def test_delete_chat_removes_directory_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chats"
            chat = create_chat(root)

            delete_chat(root, chat.id)

            self.assertFalse(chat.path.exists())
            self.assertEqual(list_chats(root), [])

    def test_delete_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chats"

            with self.assertRaises(RuntimeError):
                delete_chat(root, "../outside")

    def test_app_can_create_new_chat_after_deleting_last(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "chats"
            chat = initialize_chats(root)[0]
            delete_chat(root, chat.id)

            chats = initialize_chats(root)

            self.assertEqual(len(chats), 1)
            self.assertEqual(chats[0].id, "chat_001")

    def test_chats_root_points_inside_project_data(self) -> None:
        self.assertEqual(
            chats_root(Path("project")).as_posix(),
            "project/data/chats",
        )


if __name__ == "__main__":
    unittest.main()
