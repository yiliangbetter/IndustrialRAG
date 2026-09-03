"""Folder ingest must inherit config when optional args are omitted.

Batch jobs call process_folder_complete with only a folder path. If omitted
recursive/extensions/output_dir/parse_method stop reading config, nested
manuals are skipped or parsed with the wrong method into the wrong directory.
"""

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


def _make_batch(tmp_path):
    class DummyBatch(BatchMixin):
        pass

    dummy = DummyBatch()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": str(tmp_path / "parsed"),
            "parse_method": "ocr",
            "supported_file_extensions": [".pdf"],
            "recursive_folder_processing": True,
            "max_concurrent_files": 2,
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


@pytest.mark.asyncio
async def test_omitted_args_use_config_recursive_extensions_and_parse_method(tmp_path):
    folder = tmp_path / "docs"
    nested = folder / "line-a"
    nested.mkdir(parents=True)
    (folder / "root.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "child.pdf").write_bytes(b"%PDF-1.4\n")
    (folder / "notes.txt").write_text("ignore")
    (folder / "slide.docx").write_bytes(b"PK")

    dummy = _make_batch(tmp_path)
    await dummy.process_folder_complete(str(folder))

    by_name = {Path(c["file_path"]).name: c for c in dummy.process_calls}
    assert set(by_name) == {"root.pdf", "child.pdf"}

    parsed = Path(dummy.config.parser_output_dir)
    root_call = by_name["root.pdf"]
    assert root_call["output_dir"] == str(parsed)
    assert root_call["parse_method"] == "ocr"
    assert root_call["file_name"] is None

    child_call = by_name["child.pdf"]
    assert child_call["output_dir"] == str(parsed / "line-a")
    assert child_call["parse_method"] == "ocr"
    assert child_call["file_name"] == str(Path("line-a") / "child.pdf")


@pytest.mark.asyncio
async def test_config_non_recursive_skips_nested_when_arg_omitted(tmp_path):
    folder = tmp_path / "docs"
    nested = folder / "line-a"
    nested.mkdir(parents=True)
    (folder / "root.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "child.pdf").write_bytes(b"%PDF-1.4\n")

    dummy = _make_batch(tmp_path)
    dummy.config.recursive_folder_processing = False

    await dummy.process_folder_complete(str(folder))

    names = [Path(c["file_path"]).name for c in dummy.process_calls]
    assert names == ["root.pdf"]
