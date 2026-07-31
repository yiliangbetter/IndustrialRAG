"""Regression tests for BatchMixin file-extension wrappers."""

from pathlib import Path

from raganything.batch import BatchMixin
from raganything.config import RAGAnythingConfig


class DummyBatch(BatchMixin):
    def __init__(self, config: RAGAnythingConfig):
        self.config = config


def test_get_supported_file_extensions_includes_common_types():
    batch = DummyBatch(RAGAnythingConfig(parser="mineru"))

    exts = batch.get_supported_file_extensions()

    assert ".pdf" in exts
    assert ".png" in exts
    assert ".docx" in exts
    assert ".txt" in exts


def test_filter_supported_files_uses_config_recursive_default(tmp_path):
    (tmp_path / "root.pdf").write_bytes(b"%PDF")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.png").write_bytes(b"png")
    (tmp_path / "skip.csv").write_text("a,b\n", encoding="utf-8")

    recursive_batch = DummyBatch(
        RAGAnythingConfig(parser="mineru", recursive_folder_processing=True)
    )
    recursive_hits = {
        Path(p).name
        for p in recursive_batch.filter_supported_files([str(tmp_path)])
    }
    assert "root.pdf" in recursive_hits
    assert "child.png" in recursive_hits
    assert "skip.csv" not in recursive_hits

    flat_batch = DummyBatch(
        RAGAnythingConfig(parser="mineru", recursive_folder_processing=False)
    )
    flat_hits = {
        Path(p).name for p in flat_batch.filter_supported_files([str(tmp_path)])
    }
    assert "root.pdf" in flat_hits
    assert "child.png" not in flat_hits


def test_filter_supported_files_explicit_recursive_overrides_config(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.docx").write_bytes(b"docx")

    batch = DummyBatch(
        RAGAnythingConfig(parser="mineru", recursive_folder_processing=False)
    )

    hits = batch.filter_supported_files([str(tmp_path)], recursive=True)
    assert any(Path(p).name == "child.docx" for p in hits)
