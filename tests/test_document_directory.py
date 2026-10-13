import hashlib
import tempfile
import unittest
from pathlib import Path

from rag_project.infrastructure.document_directory import (
    DocumentDirectory,
    UnsupportedDocumentError,
)


def write(root: Path, relative_path: str, content: str = "content") -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class SaveTests(unittest.TestCase):
    def test_save_keeps_only_the_base_name(self) -> None:
        for filename in ("nested/dir/report.md", "../escape.txt", "./plain.txt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "documents"
                directory = DocumentDirectory(root)

                name = directory.save(filename, b"payload")

                self.assertEqual(name, Path(filename).name)
                self.assertEqual((root / name).read_bytes(), b"payload")
                self.assertEqual([path.name for path in root.iterdir()], [name])
                self.assertEqual(sorted(p.name for p in Path(temp_dir).iterdir()), ["documents"])

    def test_save_creates_root_and_accepts_any_extension_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "new" / "documents"

            name = DocumentDirectory(root).save("NOTES.TXT", b"upper")

            self.assertEqual(name, "NOTES.TXT")
            self.assertEqual((root / "NOTES.TXT").read_bytes(), b"upper")

    def test_save_rejects_unsupported_extensions(self) -> None:
        for filename in ("image.png", "setup.exe", "README", "archive.txt.zip", ""):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "documents"

                with self.assertRaises(UnsupportedDocumentError):
                    DocumentDirectory(root).save(filename, b"data")

                self.assertFalse(root.exists())

    def test_unsupported_document_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(UnsupportedDocumentError, ValueError))


class ListFilesTests(unittest.TestCase):
    def test_list_files_returns_supported_files_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative_path in ("a.txt", "b.MD", "sub/c.pdf", "sub/deep/d.docx"):
                write(root, relative_path)
            for relative_path in ("e.png", "no_extension", "sub/f.doc"):
                write(root, relative_path)
            (root / "folder.txt").mkdir()  # a directory with a supported suffix
            directory = DocumentDirectory(root)

            listed = sorted(directory.relative_path(path) for path in directory.list_files())

            self.assertEqual(listed, ["a.txt", "b.MD", "sub/c.pdf", "sub/deep/d.docx"])

    def test_list_files_of_missing_root_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(DocumentDirectory(Path(temp_dir) / "missing").list_files(), [])


class DeleteAndClearTests(unittest.TestCase):
    def test_delete_removes_a_file_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = write(root, "sub/a.txt")
            keep = write(root, "b.txt")

            DocumentDirectory(root).delete(target)

            self.assertFalse(target.exists())
            self.assertTrue(keep.exists())

    def test_delete_refuses_paths_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "documents"
            root.mkdir()
            outside = write(Path(temp_dir), "outside.txt")
            directory = DocumentDirectory(root)

            for path in (outside, root / ".." / "outside.txt"):
                with self.subTest(path=str(path)):
                    with self.assertRaises(RuntimeError):
                        directory.delete(path)
                    self.assertTrue(outside.exists())

    def test_delete_of_missing_file_inside_root_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            DocumentDirectory(temp_dir).delete(Path(temp_dir) / "missing.txt")

    def test_clear_removes_supported_files_and_returns_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write(root, "a.txt")
            write(root, "sub/b.md")
            write(root, "sub/c.pdf")
            keep = write(root, "keep.png")
            directory = DocumentDirectory(root)

            self.assertEqual(directory.clear(), 3)

            self.assertEqual(directory.list_files(), [])
            self.assertTrue(keep.exists())
            self.assertEqual(directory.clear(), 0)

    def test_clear_of_missing_root_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(DocumentDirectory(Path(temp_dir) / "missing").clear(), 0)


class ScanAndLoadTests(unittest.TestCase):
    def test_scan_fingerprints_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a_path = write(root, "a.txt", "hello")
            write(root, "sub/b.md", "# Title\n\nBody")
            write(root, "skip.png", "not a document")

            records = DocumentDirectory(root).scan()

            self.assertEqual(list(records), ["a.txt", "sub/b.md"])
            a_record = records["a.txt"]
            self.assertEqual(a_record.path, "a.txt")
            self.assertEqual(a_record.sha256, hashlib.sha256(a_path.read_bytes()).hexdigest())
            self.assertEqual(a_record.size, a_path.stat().st_size)
            self.assertEqual(a_record.mtime_ns, a_path.stat().st_mtime_ns)
            self.assertEqual(a_record.chunk_ids, [])
            self.assertEqual(records["sub/b.md"].path, "sub/b.md")

    def test_scan_hash_tracks_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = DocumentDirectory(root)
            write(root, "a.txt", "first version")
            before = directory.scan()["a.txt"].sha256

            write(root, "a.txt", "second version")

            self.assertNotEqual(directory.scan()["a.txt"].sha256, before)

    def test_scan_of_missing_root_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(DocumentDirectory(Path(temp_dir) / "missing").scan(), {})

    def test_load_returns_one_document_for_a_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = write(root, "sub/b.md", "# Title\n\nMarkdown body.")

            documents = DocumentDirectory(root).load("sub/b.md")

            self.assertEqual(len(documents), 1)
            document = documents[0]
            self.assertEqual(document.id, "sub/b.md")
            self.assertEqual(document.text, "# Title\n\nMarkdown body.")
            self.assertEqual(document.metadata["relative_path"], "sub/b.md")
            self.assertEqual(document.metadata["file_type"], "md")
            self.assertEqual(document.metadata["source"], str(path))

    def test_load_of_blank_file_returns_no_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write(Path(temp_dir), "blank.txt", "  \n\n ")

            self.assertEqual(DocumentDirectory(temp_dir).load("blank.txt"), [])

    def test_load_all_loads_every_text_file_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write(root, "b.md", "markdown")
            write(root, "sub/c.txt", "nested text")
            write(root, "a.txt", "text")
            write(root, "blank.txt", "   ")
            write(root, "skip.png", "image")

            documents = DocumentDirectory(root).load_all()

            # .txt files are loaded before .md files, each group sorted by path.
            self.assertEqual([doc.id for doc in documents], ["a.txt", "sub/c.txt", "b.md"])
            self.assertEqual([doc.text for doc in documents], ["text", "nested text", "markdown"])

    # Regression: on POSIX (e.g. the Docker image) rglob("*.txt") is
    # case-sensitive, so "NOTES.TXT" was scanned into the manifest but never
    # indexed. Loading must ignore extension case like scanning does.
    def test_load_all_loads_every_scanned_file_regardless_of_extension_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write(root, "NOTES.TXT", "upper-case extension")
            write(root, "Guide.Md", "mixed-case extension")
            directory = DocumentDirectory(root)

            scanned = sorted(directory.scan())
            loaded = sorted(doc.metadata["relative_path"] for doc in directory.load_all())

            self.assertEqual(scanned, ["Guide.Md", "NOTES.TXT"])
            self.assertEqual(loaded, scanned)
