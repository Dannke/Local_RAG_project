import unittest

from rag_project.ui.formatting import (
    citation_filename,
    citation_preview,
    file_icon,
    file_type,
    human_size,
    relevance_for,
)

EXCELLENT = ("Excellent Match", "#66bb6a")
GOOD = ("Good Match", "#9ccc65")
MODERATE = ("Moderate Match", "#ffa726")
WEAK = ("Weak Match", "#ef5350")


class FileLabelTests(unittest.TestCase):
    def test_file_icon_is_case_insensitive(self) -> None:
        for extension in (".pdf", ".PDF", ".Pdf"):
            with self.subTest(extension=extension):
                self.assertEqual(file_icon(extension), "📄")
        self.assertEqual(file_icon(".DOCX"), "📝")
        self.assertEqual(file_icon(".md"), "📖")

    def test_file_icon_falls_back_for_unknown_extensions(self) -> None:
        for extension in (".xyz", "", "pdf"):
            with self.subTest(extension=extension):
                self.assertEqual(file_icon(extension), "📎")

    def test_file_type_is_case_insensitive(self) -> None:
        self.assertEqual(file_type(".MD"), "Markdown")
        self.assertEqual(file_type(".txt"), "Text")
        self.assertEqual(file_type(".Docx"), "DOCX")
        self.assertEqual(file_type(".PDF"), "PDF")

    def test_file_type_falls_back_for_unknown_extensions(self) -> None:
        for extension in (".png", "", "md"):
            with self.subTest(extension=extension):
                self.assertEqual(file_type(extension), "File")


class HumanSizeTests(unittest.TestCase):
    def test_human_size_boundaries(self) -> None:
        cases = [
            (0, "0 B"),
            (1023, "1023 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1024 * 1024, "1.0 MB"),
            (int(5.5 * 1024 * 1024), "5.5 MB"),
            (3 * 1024**3, "3072.0 MB"),
        ]
        for size, expected in cases:
            with self.subTest(size=size):
                self.assertEqual(human_size(size), expected)


class RelevanceTests(unittest.TestCase):
    def test_tiers_switch_at_thresholds(self) -> None:
        cases = [
            (0.95, (95, *EXCELLENT)),
            (0.8, (80, *EXCELLENT)),
            (0.79, (79, *GOOD)),
            (0.6, (60, *GOOD)),
            (0.59, (59, *MODERATE)),
            (0.4, (40, *MODERATE)),
            (0.39, (39, *WEAK)),
            (0.0, (0, *WEAK)),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(relevance_for(score), expected)

    def test_out_of_range_scores_are_clamped(self) -> None:
        self.assertEqual(relevance_for(1.7), (100, *EXCELLENT))
        self.assertEqual(relevance_for(-0.3), (0, *WEAK))
        self.assertEqual(relevance_for(-5), (0, *WEAK))

    def test_integer_scores_are_accepted(self) -> None:
        self.assertEqual(relevance_for(1), (100, *EXCELLENT))
        self.assertEqual(relevance_for(0), (0, *WEAK))

    def test_unusable_scores_return_none(self) -> None:
        for score in (None, float("nan"), float("inf"), float("-inf"), "abc", object(), [0.5]):
            with self.subTest(score=score):
                self.assertIsNone(relevance_for(score))


class CitationTests(unittest.TestCase):
    def test_citation_filename_uses_the_base_name(self) -> None:
        self.assertEqual(citation_filename("docs/sub/manual.pdf"), "manual.pdf")
        self.assertEqual(citation_filename("manual.pdf"), "manual.pdf")

    def test_citation_filename_falls_back_for_empty_source(self) -> None:
        self.assertEqual(citation_filename(""), "Unknown source")

    def test_citation_preview_collapses_whitespace(self) -> None:
        self.assertEqual(citation_preview("  first\n\nsecond\t third  "), "first second third")

    def test_citation_preview_of_empty_text_is_empty(self) -> None:
        self.assertEqual(citation_preview(""), "")
        self.assertEqual(citation_preview("   \n "), "")

    def test_citation_preview_keeps_text_within_limit(self) -> None:
        self.assertEqual(citation_preview("a" * 10, limit=10), "a" * 10)

    def test_citation_preview_truncates_with_ellipsis(self) -> None:
        self.assertEqual(citation_preview("abcdefghij klm", limit=10), "abcdefghij…")
        # Trailing whitespace at the cut is dropped before the ellipsis.
        self.assertEqual(citation_preview("abcde fghij", limit=6), "abcde…")

    def test_citation_preview_default_limit(self) -> None:
        preview = citation_preview("word " * 100)

        self.assertTrue(preview.endswith("…"))
        self.assertLessEqual(len(preview), 261)
        self.assertTrue(preview.startswith("word word"))
