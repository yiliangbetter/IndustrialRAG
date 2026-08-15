"""Folder-batch ingest edges on BatchMixin.process_folder_complete.

Industrial runs feed directories of manuals. Init failure, missing folders,
empty matches, per-file errors, and nested output remapping must not silently
skip or abort the whole batch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from raganything.batch import BatchMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.infos = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, *args, **kwargs):
        pass


def _make_batch(
    tmp_path,
    *,
    init_result=None,
    process_impl=None,
    recursive=True,
    extensions=None,
):
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
            "supported_file_extensions": extensions or [".pdf", ".docx"],
            "recursive_folder_processing": recursive,
            "max_concurrent_files": 2,
            "parser": "mineru",
        },
    )()
    dummy._init_result = init_result if init_result is not None else {"success": True}
    dummy.process_calls = []

    async def _ensure():
        return dummy._init_result

    async def _process(file_path, **kwargs):
        dummy.process_calls.append({"file_path": file_path, **kwargs})
        if process_impl is not None:
            return await process_impl(file_path, **kwargs)
        return None

    dummy._ensure_lightrag_initialized = _ensure
    dummy.process_document_complete = _process
    return dummy


@pytest.mark.asyncio
async def test_process_folder_complete_raises_when_lightrag_init_fails(tmp_path):
    dummy = _make_batch(tmp_path, init_result={"success": False, "error": "no embed"})
    folder = tmp_path / "docs"
    folder.mkdir()
    with pytest.raises(RuntimeError, match="LightRAG initialization failed"):
        await dummy.process_folder_complete(str(folder))
    assert dummy.process_calls == []


@pytest.mark.asyncio
async def test_process_folder_complete_raises_when_folder_missing(tmp_path):
    dummy = _make_batch(tmp_path)
    with pytest.raises(FileNotFoundError, match="Folder not found"):
        await dummy.process_folder_complete(str(tmp_path / "missing"))


@pytest.mark.asyncio
async def test_process_folder_complete_returns_when_no_supported_files(tmp_path):
    dummy = _make_batch(tmp_path)
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "notes.txt").write_text("ignore me")
    await dummy.process_folder_complete(str(folder))
    assert dummy.process_calls == []
    assert any("No supported files" in w for w in dummy.logger.warnings)


@pytest.mark.asyncio
async def test_process_folder_complete_processes_files_and_isolates_failures(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    good = folder / "ok.pdf"
    bad = folder / "bad.pdf"
    good.write_bytes(b"%PDF-1.4\n")
    bad.write_bytes(b"%PDF-1.4\n")

    async def process_impl(file_path, **kwargs):
        if Path(file_path).name == "bad.pdf":
            raise ValueError("parse exploded")
        return None

    dummy = _make_batch(tmp_path, process_impl=process_impl)
    await dummy.process_folder_complete(
        str(folder),
        output_dir=str(tmp_path / "out"),
        display_stats=True,
        recursive=False,
    )

    processed_names = sorted(Path(c["file_path"]).name for c in dummy.process_calls)
    assert processed_names == ["bad.pdf", "ok.pdf"]
    assert any("parse exploded" in e for e in dummy.logger.errors)
    assert any("Successful: 1" in i for i in dummy.logger.infos)
    assert any("Failed: 1" in i for i in dummy.logger.infos)


@pytest.mark.asyncio
async def test_process_folder_complete_remaps_nested_output_and_file_name(tmp_path):
    folder = tmp_path / "docs"
    nested = folder / "line-a"
    nested.mkdir(parents=True)
    top = folder / "root.pdf"
    child = nested / "child.pdf"
    top.write_bytes(b"%PDF-1.4\n")
    child.write_bytes(b"%PDF-1.4\n")

    dummy = _make_batch(tmp_path)
    output_dir = tmp_path / "out"
    await dummy.process_folder_complete(
        str(folder),
        output_dir=str(output_dir),
        parse_method="ocr",
        recursive=True,
        display_stats=False,
    )

    by_name = {Path(c["file_path"]).name: c for c in dummy.process_calls}
    assert set(by_name) == {"root.pdf", "child.pdf"}

    root_call = by_name["root.pdf"]
    assert root_call["output_dir"] == str(output_dir)
    assert root_call["file_name"] is None
    assert root_call["parse_method"] == "ocr"

    child_call = by_name["child.pdf"]
    assert child_call["output_dir"] == str(output_dir / "line-a")
    assert child_call["file_name"] == str(Path("line-a") / "child.pdf")


@pytest.mark.asyncio
async def test_process_folder_complete_non_recursive_skips_nested(tmp_path):
    folder = tmp_path / "docs"
    nested = folder / "line-a"
    nested.mkdir(parents=True)
    (folder / "root.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "child.pdf").write_bytes(b"%PDF-1.4\n")

    dummy = _make_batch(tmp_path, recursive=False)
    await dummy.process_folder_complete(str(folder), recursive=False)

    names = [Path(c["file_path"]).name for c in dummy.process_calls]
    assert names == ["root.pdf"]
