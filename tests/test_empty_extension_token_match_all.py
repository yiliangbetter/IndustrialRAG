"""Empty extension tokens make folder ingest match every path.

``SUPPORTED_FILE_EXTENSIONS`` is split on commas and each token is stripped.
A trailing comma leaves ``""``. ``process_folder_complete`` concatenates that
token onto ``*`` / ``**/*``, so the batch selects extensionless files, other
types, and directory entries, and it selects real PDFs a second time.
"""

from collections import Counter
from pathlib import Path

import pytest

from raganything.batch import BatchMixin
from raganything.config import RAGAnythingConfig


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_batch(tmp_path, extensions):
    class DummyBatch(BatchMixin):
        pass

    dummy = DummyBatch()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": str(tmp_path / "parsed"),
            "parse_method": "auto",
            "supported_file_extensions": extensions,
            "recursive_folder_processing": True,
            "max_concurrent_files": 4,
            "parser": "mineru",
        },
    )()
    dummy.process_calls = []

    async def _ensure():
        return {"success": True}

    async def _process(file_path, **kwargs):
        dummy.process_calls.append({"file_path": file_path, **kwargs})

    dummy._ensure_lightrag_initialized = _ensure
    dummy.process_document_complete = _process
    return dummy


def _write_tree(folder: Path) -> None:
    nested = folder / "line-a"
    nested.mkdir(parents=True)
    (folder / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    (folder / "notes.txt").write_text("shift notes")
    (folder / "README").write_text("no extension")
    (nested / "child.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "readme.md").write_text("nested notes")


def _names(dummy) -> Counter:
    return Counter(Path(call["file_path"]).name for call in dummy.process_calls)


def test_trailing_comma_keeps_empty_extension_token(monkeypatch):
    monkeypatch.setenv("SUPPORTED_FILE_EXTENSIONS", ".pdf,")

    config = RAGAnythingConfig()

    assert config.supported_file_extensions == [".pdf", ""]


@pytest.mark.asyncio
async def test_empty_extension_token_ingests_every_path_and_duplicates_pdfs(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SUPPORTED_FILE_EXTENSIONS", ".pdf,")
    extensions = RAGAnythingConfig().supported_file_extensions
    folder = tmp_path / "docs"
    _write_tree(folder)
    dummy = _make_batch(tmp_path, extensions)

    await dummy.process_folder_complete(str(folder))

    assert _names(dummy) == Counter(
        {
            "manual.pdf": 2,
            "child.pdf": 2,
            "notes.txt": 1,
            "README": 1,
            "readme.md": 1,
            "line-a": 1,
        }
    )
    directory_calls = [
        Path(call["file_path"])
        for call in dummy.process_calls
        if Path(call["file_path"]).name == "line-a"
    ]
    assert len(directory_calls) == 1
    assert directory_calls[0].is_dir()


@pytest.mark.asyncio
async def test_pdf_only_extensions_skip_unrelated_paths(tmp_path):
    folder = tmp_path / "docs"
    _write_tree(folder)
    dummy = _make_batch(tmp_path, [".pdf"])

    await dummy.process_folder_complete(str(folder))

    assert _names(dummy) == Counter({"manual.pdf": 1, "child.pdf": 1})
