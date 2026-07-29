"""Regression tests for BatchMixin.process_folder_complete."""

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


def _make_batch(tmp_path, processed=None, fail_paths=None):
    class DummyBatch(BatchMixin):
        pass

    batch = DummyBatch()
    batch.logger = FakeLogger()
    batch.config = type(
        "Config",
        (),
        {
            "parser_output_dir": str(tmp_path / "out"),
            "parse_method": "auto",
            "supported_file_extensions": [".pdf", ".txt"],
            "recursive_folder_processing": True,
            "max_concurrent_files": 2,
        },
    )()

    processed = processed if processed is not None else []
    fail_paths = set(fail_paths or [])

    async def fake_ensure():
        return {"success": True}

    async def fake_process(
        file_path,
        output_dir=None,
        parse_method=None,
        split_by_character=None,
        split_by_character_only=False,
        file_name=None,
        **kwargs,
    ):
        if str(file_path) in fail_paths or file_path in fail_paths:
            raise RuntimeError(f"fail:{file_path}")
        processed.append(
            {
                "file_path": str(file_path),
                "output_dir": output_dir,
                "parse_method": parse_method,
                "file_name": file_name,
            }
        )

    batch._ensure_lightrag_initialized = fake_ensure
    batch.process_document_complete = fake_process
    return batch, processed


@pytest.mark.asyncio
async def test_process_folder_complete_missing_folder_raises(tmp_path):
    batch, _ = _make_batch(tmp_path)
    with pytest.raises(FileNotFoundError, match="Folder not found"):
        await batch.process_folder_complete(str(tmp_path / "missing"))


@pytest.mark.asyncio
async def test_process_folder_complete_init_failure_raises(tmp_path):
    batch, _ = _make_batch(tmp_path)

    async def fail_init():
        return {"success": False, "error": "no llm"}

    batch._ensure_lightrag_initialized = fail_init
    folder = tmp_path / "docs"
    folder.mkdir()

    with pytest.raises(RuntimeError, match="LightRAG initialization failed"):
        await batch.process_folder_complete(str(folder))


@pytest.mark.asyncio
async def test_process_folder_complete_empty_folder_warns(tmp_path):
    batch, processed = _make_batch(tmp_path)
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "notes.md").write_text("ignored extension")

    await batch.process_folder_complete(str(folder))

    assert processed == []
    assert any("No supported files found" in w for w in batch.logger.warnings)


@pytest.mark.asyncio
async def test_process_folder_complete_discovers_files_and_preserves_subdir_paths(
    tmp_path,
):
    batch, processed = _make_batch(tmp_path)
    folder = tmp_path / "docs"
    nested = folder / "sub"
    nested.mkdir(parents=True)
    root_pdf = folder / "a.pdf"
    nested_txt = nested / "b.txt"
    root_pdf.write_bytes(b"%PDF-1.4\n")
    nested_txt.write_text("hello")

    out = tmp_path / "parsed"
    await batch.process_folder_complete(
        str(folder),
        output_dir=str(out),
        parse_method="ocr",
        recursive=True,
        max_workers=2,
        display_stats=True,
    )

    by_path = {p["file_path"]: p for p in processed}
    assert set(by_path) == {str(root_pdf), str(nested_txt)}

    assert by_path[str(root_pdf)]["output_dir"] == str(out)
    assert by_path[str(root_pdf)]["file_name"] is None
    assert by_path[str(root_pdf)]["parse_method"] == "ocr"

    assert by_path[str(nested_txt)]["output_dir"] == str(out / "sub")
    assert by_path[str(nested_txt)]["file_name"] == "sub/b.txt"
    assert by_path[str(nested_txt)]["parse_method"] == "ocr"

    assert any("Successful: 2 files" in msg for msg in batch.logger.infos)


@pytest.mark.asyncio
async def test_process_folder_complete_non_recursive_skips_nested(tmp_path):
    batch, processed = _make_batch(tmp_path)
    folder = tmp_path / "docs"
    nested = folder / "sub"
    nested.mkdir(parents=True)
    (folder / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "b.txt").write_text("nested")

    await batch.process_folder_complete(
        str(folder),
        recursive=False,
        file_extensions=[".pdf", ".txt"],
        display_stats=False,
    )

    assert [p["file_path"] for p in processed] == [str(folder / "a.pdf")]


@pytest.mark.asyncio
async def test_process_folder_complete_records_per_file_failures(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    ok = folder / "ok.pdf"
    bad = folder / "bad.pdf"
    ok.write_bytes(b"%PDF-1.4\nok")
    bad.write_bytes(b"%PDF-1.4\nbad")

    batch, processed = _make_batch(tmp_path, fail_paths={str(bad)})

    await batch.process_folder_complete(
        str(folder),
        file_extensions=[".pdf"],
        recursive=False,
        display_stats=True,
    )

    assert [p["file_path"] for p in processed] == [str(ok)]
    assert any("Successful: 1 files" in msg for msg in batch.logger.infos)
    assert any("Failed: 1 files" in msg for msg in batch.logger.infos)
    assert any(str(bad) in err for err in batch.logger.errors)
