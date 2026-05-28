import tempfile
import unittest

from rag_project.models import Document
from rag_project.vectorstores.faiss_store import FaissVectorStore


class FaissVectorStoreTests(unittest.TestCase):
    def test_save_load_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(temp_dir)
            store.add(
                [
                    Document(id="a", text="alpha document"),
                    Document(id="b", text="beta document"),
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            )
            store.save()

            loaded = FaissVectorStore(temp_dir)
            loaded.load()
            results = loaded.search([1.0, 0.0], top_k=1)

            self.assertEqual(loaded.count(), 2)
            self.assertEqual(results[0].document.id, "a")

    def test_load_from_disk_classmethod(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FaissVectorStore(temp_dir)
            store.add([Document(id="a", text="alpha document")], [[1.0, 0.0]])
            store.save_to_disk()

            loaded = FaissVectorStore.load_from_disk(temp_dir)

            self.assertEqual(loaded.count(), 1)
            self.assertEqual(loaded.search([1.0, 0.0], top_k=1)[0].document.text, "alpha document")
