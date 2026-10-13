import tempfile
import unittest
from pathlib import Path

from rag_project.domain.manifest import IndexManifest, with_chunk_ids
from rag_project.infrastructure.document_directory import scan_source_files
from rag_project.infrastructure.index_storage import FaissIndexRepository


class IndexManifestTests(unittest.TestCase):
    def test_scan_source_files_hashes_supported_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "doc.txt").write_text("hello", encoding="utf-8")
            (root / "ignore.tmp").write_text("skip", encoding="utf-8")

            records = scan_source_files(root)

            self.assertEqual(list(records), ["doc.txt"])
            self.assertEqual(records["doc.txt"].size, 5)

    def test_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "doc.txt").write_text("hello", encoding="utf-8")
            record = scan_source_files(root)["doc.txt"]
            manifest = IndexManifest(files={"doc.txt": with_chunk_ids(record, ["chunk-1"])})

            repository = FaissIndexRepository(root / "index")
            repository.save_manifest(manifest)
            loaded = repository.load_manifest()

            self.assertEqual(loaded.files["doc.txt"].chunk_ids, ["chunk-1"])
