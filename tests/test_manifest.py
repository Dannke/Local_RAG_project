import unittest

from rag_project.domain.manifest import (
    IndexManifest,
    ManifestDiff,
    SourceFileRecord,
    chunk_ids_by_source,
)
from rag_project.domain.models import Document


def record(
    path: str,
    sha256: str = "sha-1",
    chunk_ids: list[str] | None = None,
    size: int = 10,
    mtime_ns: int = 100,
) -> SourceFileRecord:
    return SourceFileRecord(
        path=path, sha256=sha256, size=size, mtime_ns=mtime_ns, chunk_ids=chunk_ids or []
    )


def chunk(path: str, index: int) -> Document:
    return Document(
        id=f"{path}:{index:04d}", text=f"chunk {index} of {path}", metadata={"relative_path": path}
    )


class ManifestDiffTests(unittest.TestCase):
    def test_identical_sources_have_an_empty_diff(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt"), "b.md": record("b.md")})

        diff = manifest.diff({"a.txt": record("a.txt"), "b.md": record("b.md")})

        self.assertEqual(diff, ManifestDiff(added=[], changed=[], deleted=frozenset()))
        self.assertFalse(diff.requires_rebuild)

    def test_new_files_are_added_in_current_order_without_rebuild(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt")})
        current = {"z.txt": record("z.txt"), "a.txt": record("a.txt"), "m.md": record("m.md")}

        diff = manifest.diff(current)

        self.assertEqual(diff.added, ["z.txt", "m.md"])
        self.assertEqual(diff.changed, [])
        self.assertEqual(diff.deleted, frozenset())
        self.assertFalse(diff.requires_rebuild)

    def test_everything_is_added_for_an_empty_manifest(self) -> None:
        diff = IndexManifest.empty().diff({"a.txt": record("a.txt"), "b.md": record("b.md")})

        self.assertEqual(diff.added, ["a.txt", "b.md"])
        self.assertFalse(diff.requires_rebuild)

    def test_changed_content_hash_requires_rebuild(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt", sha256="old")})

        diff = manifest.diff({"a.txt": record("a.txt", sha256="new")})

        self.assertEqual(diff.changed, ["a.txt"])
        self.assertEqual(diff.added, [])
        self.assertTrue(diff.requires_rebuild)

    def test_size_mtime_or_chunk_ids_alone_are_not_a_change(self) -> None:
        manifest = IndexManifest(
            files={"a.txt": record("a.txt", size=1, mtime_ns=1, chunk_ids=["a.txt:0000"])}
        )

        diff = manifest.diff({"a.txt": record("a.txt", size=2, mtime_ns=2)})

        self.assertEqual(diff.changed, [])
        self.assertFalse(diff.requires_rebuild)

    def test_deleted_file_requires_rebuild(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt"), "b.md": record("b.md")})

        diff = manifest.diff({"a.txt": record("a.txt")})

        self.assertEqual(diff.deleted, frozenset({"b.md"}))
        self.assertTrue(diff.requires_rebuild)

    def test_mixed_diff_reports_each_kind(self) -> None:
        manifest = IndexManifest(
            files={
                "keep.txt": record("keep.txt"),
                "edit.txt": record("edit.txt"),
                "gone.txt": record("gone.txt"),
            }
        )
        current = {
            "keep.txt": record("keep.txt"),
            "edit.txt": record("edit.txt", sha256="sha-2"),
            "new.txt": record("new.txt"),
        }

        diff = manifest.diff(current)

        self.assertEqual(diff.added, ["new.txt"])
        self.assertEqual(diff.changed, ["edit.txt"])
        self.assertEqual(diff.deleted, frozenset({"gone.txt"}))
        self.assertTrue(diff.requires_rebuild)

    def test_requires_rebuild_only_for_changed_or_deleted(self) -> None:
        only_added = ManifestDiff(added=["a"], changed=[], deleted=frozenset())
        only_changed = ManifestDiff(added=[], changed=["a"], deleted=frozenset())
        only_deleted = ManifestDiff(added=[], changed=[], deleted=frozenset({"a"}))

        self.assertFalse(only_added.requires_rebuild)
        self.assertTrue(only_changed.requires_rebuild)
        self.assertTrue(only_deleted.requires_rebuild)


class FromChunksTests(unittest.TestCase):
    def test_every_scanned_file_is_included_with_its_chunk_ids(self) -> None:
        records = {
            "a.txt": record("a.txt", sha256="sha-a", size=3, mtime_ns=7),
            "b.md": record("b.md"),
            "empty.txt": record("empty.txt"),
        }
        chunks = [chunk("a.txt", 0), chunk("b.md", 0), chunk("a.txt", 1)]

        manifest = IndexManifest.from_chunks(records, chunks)

        self.assertEqual(set(manifest.files), {"a.txt", "b.md", "empty.txt"})
        self.assertEqual(manifest.files["a.txt"].chunk_ids, ["a.txt:0000", "a.txt:0001"])
        self.assertEqual(manifest.files["b.md"].chunk_ids, ["b.md:0000"])
        self.assertEqual(manifest.files["empty.txt"].chunk_ids, [])
        expected_a = record(
            "a.txt", sha256="sha-a", size=3, mtime_ns=7, chunk_ids=["a.txt:0000", "a.txt:0001"]
        )
        self.assertEqual(manifest.files["a.txt"], expected_a)
        self.assertEqual(manifest.version, 1)

    def test_input_records_are_not_mutated(self) -> None:
        records = {"a.txt": record("a.txt")}

        IndexManifest.from_chunks(records, [chunk("a.txt", 0)])

        self.assertEqual(records["a.txt"].chunk_ids, [])

    def test_chunks_of_unscanned_or_unknown_sources_are_ignored(self) -> None:
        records = {"a.txt": record("a.txt")}
        chunks = [
            chunk("a.txt", 0),
            chunk("not-scanned.txt", 0),
            Document(id="orphan:0000", text="no source metadata"),
        ]

        manifest = IndexManifest.from_chunks(records, chunks)

        self.assertEqual(set(manifest.files), {"a.txt"})
        self.assertEqual(manifest.files["a.txt"].chunk_ids, ["a.txt:0000"])


class WithAppendedTests(unittest.TestCase):
    def test_appends_new_sources_and_keeps_existing_ones(self) -> None:
        existing = record("a.txt", chunk_ids=["a.txt:0000"])
        manifest = IndexManifest(files={"a.txt": existing})
        records = {"a.txt": record("a.txt"), "b.md": record("b.md", sha256="sha-b")}

        appended = manifest.with_appended(records, ["b.md"], [chunk("b.md", 0), chunk("b.md", 1)])

        self.assertEqual(set(appended.files), {"a.txt", "b.md"})
        self.assertEqual(appended.files["a.txt"], existing)
        self.assertEqual(appended.files["b.md"].chunk_ids, ["b.md:0000", "b.md:0001"])
        self.assertEqual(appended.files["b.md"].sha256, "sha-b")

    def test_original_manifest_is_unchanged(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt", chunk_ids=["a.txt:0000"])})

        manifest.with_appended({"b.md": record("b.md")}, ["b.md"], [chunk("b.md", 0)])

        self.assertEqual(set(manifest.files), {"a.txt"})

    def test_records_added_files_without_chunks(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt", chunk_ids=["a.txt:0000"])})

        appended = manifest.with_appended({"blank.txt": record("blank.txt")}, ["blank.txt"], [])

        self.assertEqual(appended.files["blank.txt"].chunk_ids, [])

    def test_manifest_version_is_preserved(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt")}, version=2)

        self.assertEqual(
            manifest.with_appended({"b.md": record("b.md")}, ["b.md"], [chunk("b.md", 0)]).version,
            2,
        )
        self.assertEqual(manifest.without("a.txt").version, 2)


class WithoutTests(unittest.TestCase):
    def test_removes_one_path(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt"), "b.md": record("b.md")})

        remaining = manifest.without("a.txt")

        self.assertEqual(set(remaining.files), {"b.md"})
        self.assertEqual(set(manifest.files), {"a.txt", "b.md"})

    def test_unknown_path_is_a_noop(self) -> None:
        manifest = IndexManifest(files={"a.txt": record("a.txt")})

        self.assertEqual(manifest.without("missing.txt").files, manifest.files)


class ChunkIdsBySourceTests(unittest.TestCase):
    def test_groups_chunk_ids_by_relative_path_preserving_order(self) -> None:
        grouped = chunk_ids_by_source([chunk("a.txt", 0), chunk("b.md", 0), chunk("a.txt", 1)])

        self.assertEqual(grouped, {"a.txt": ["a.txt:0000", "a.txt:0001"], "b.md": ["b.md:0000"]})
        self.assertIs(type(grouped), dict)

    def test_skips_chunks_without_a_relative_path(self) -> None:
        chunks = [
            Document(id="x:0000", text="no metadata"),
            Document(id="y:0000", text="empty path", metadata={"relative_path": ""}),
            Document(id="z:0000", text="null path", metadata={"relative_path": None}),
            chunk("a.txt", 0),
        ]

        self.assertEqual(chunk_ids_by_source(chunks), {"a.txt": ["a.txt:0000"]})

    def test_empty_input_yields_empty_mapping(self) -> None:
        self.assertEqual(chunk_ids_by_source([]), {})
