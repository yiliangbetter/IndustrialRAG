"""Regression tests for BatchParser.filter_supported_files recursion.

Folder ingest must include nested manuals when recursive=True and must not
walk subdirectories when recursive=False. A glob/rglob swap silently drops
or double-counts plant documents.
"""

from pathlib import Path

from raganything.batch_parser import BatchParser


def _tree(tmp_path: Path) -> None:
    (tmp_path / "root.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.md").write_bytes(b"# notes")
    nested = tmp_path / "line" / "unit"
    nested.mkdir(parents=True)
    (nested / "nested.pdf").write_bytes(b"%PDF")
    (nested / "skip.csv").write_bytes(b"a,b\n")
    (tmp_path / "readme.py").write_bytes(b"print('no')")


class TestBatchParserRecursiveFilter:
    def test_recursive_includes_nested_supported_files(self, tmp_path):
        _tree(tmp_path)
        parser = BatchParser(parser_type="mineru", skip_installation_check=True)

        found = parser.filter_supported_files([str(tmp_path)], recursive=True)
        names = {Path(path).name for path in found}

        assert names == {"root.pdf", "notes.md", "nested.pdf"}
        assert "skip.csv" not in names
        assert "readme.py" not in names

    def test_non_recursive_stays_in_top_directory(self, tmp_path):
        _tree(tmp_path)
        parser = BatchParser(parser_type="mineru", skip_installation_check=True)

        found = parser.filter_supported_files([str(tmp_path)], recursive=False)
        names = {Path(path).name for path in found}

        assert names == {"root.pdf", "notes.md"}
        assert "nested.pdf" not in names

    def test_direct_file_and_missing_path(self, tmp_path):
        pdf = tmp_path / "only.pdf"
        pdf.write_bytes(b"%PDF")
        parser = BatchParser(parser_type="mineru", skip_installation_check=True)

        found = parser.filter_supported_files(
            [str(pdf), str(tmp_path / "missing-dir")], recursive=True
        )
        assert found == [str(pdf)]
