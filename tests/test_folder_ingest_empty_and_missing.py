"""Folder ingest must fail closed before it creates parse output.

``process_folder_complete`` initializes LightRAG, then selects files.
A missing upload folder and a folder with no supported files must not
create the parse output directory or call document ingest. The batch
summary is a separate switch from content-block statistics.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_batch_mixin():
    """Load BatchMixin without importing raganything or the parser stack."""
    pkg_name = "_raganything_batch_empty_folder_1118"
    if pkg_name not in sys.modules:
        package = types.ModuleType(pkg_name)
        package.__path__ = [str(REPO_ROOT / "raganything")]
        sys.modules[pkg_name] = package

        batch_parser = types.ModuleType(f"{pkg_name}.batch_parser")
        batch_parser.BatchParser = object
        batch_parser.BatchProcessingResult = object
        sys.modules[f"{pkg_name}.batch_parser"] = batch_parser

        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.batch",
            REPO_ROOT / "raganything" / "batch.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return sys.modules[f"{pkg_name}.batch"].BatchMixin


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, message, *args, **kwargs):
        self.infos.append(str(message))

    def warning(self, message, *args, **kwargs):
        self.warnings.append(str(message))

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))

    def debug(self, message, *args, **kwargs):
        pass


_SUCCESS = object()


def _make_processor(batch_mixin, tmp_path, init_result=_SUCCESS):
    class Dummy(batch_mixin):
        pass

    dummy = Dummy()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": str(tmp_path / "parsed"),
            "parse_method": "auto",
            "display_content_stats": False,
            "supported_file_extensions": [".pdf"],
            "recursive_folder_processing": True,
            "max_concurrent_files": 1,
        },
    )()
    dummy.process_calls = []
    dummy.init_calls = 0
    if init_result is _SUCCESS:
        init_result = {"success": True}

    async def fake_ensure():
        dummy.init_calls += 1
        return init_result

    async def fake_process(file_path, **kwargs):
        dummy.process_calls.append((file_path, kwargs))

    dummy._ensure_lightrag_initialized = fake_ensure
    dummy.process_document_complete = fake_process
    return dummy


@pytest.mark.asyncio
async def test_unsupported_folder_does_not_create_output_or_ingest(tmp_path):
    batch_mixin = _load_batch_mixin()
    folder = tmp_path / "uploads"
    folder.mkdir()
    (folder / "notes.txt").write_text("not a manual", encoding="utf-8")
    output_dir = tmp_path / "out" / "parsed"
    dummy = _make_processor(batch_mixin, tmp_path)

    await dummy.process_folder_complete(
        str(folder),
        output_dir=str(output_dir),
        file_extensions=[".pdf"],
        recursive=True,
    )

    assert dummy.init_calls == 1
    assert dummy.process_calls == []
    assert not output_dir.exists()
    assert dummy.logger.errors == []
    assert any(
        "No supported files found" in message and str(folder) in message
        for message in dummy.logger.warnings
    )
    assert not any("Processing complete!" in message for message in dummy.logger.infos)


@pytest.mark.asyncio
async def test_missing_folder_raises_after_init_without_creating_output(tmp_path):
    batch_mixin = _load_batch_mixin()
    missing = tmp_path / "missing-uploads"
    output_dir = tmp_path / "out" / "parsed"
    dummy = _make_processor(batch_mixin, tmp_path)

    with pytest.raises(FileNotFoundError, match="Folder not found"):
        await dummy.process_folder_complete(
            str(missing),
            output_dir=str(output_dir),
        )

    assert dummy.init_calls == 1
    assert dummy.process_calls == []
    assert not output_dir.exists()
    assert dummy.logger.warnings == []


@pytest.mark.asyncio
async def test_init_failure_is_not_replaced_by_a_missing_folder(tmp_path):
    batch_mixin = _load_batch_mixin()
    missing = tmp_path / "missing-uploads"
    dummy = _make_processor(
        batch_mixin,
        tmp_path,
        init_result={"success": False, "error": "index down"},
    )

    with pytest.raises(RuntimeError, match="index down"):
        await dummy.process_folder_complete(str(missing))

    assert dummy.init_calls == 1
    assert dummy.process_calls == []
    assert not (tmp_path / "parsed").exists()


@pytest.mark.asyncio
async def test_init_without_result_reports_unknown_error(tmp_path):
    batch_mixin = _load_batch_mixin()
    folder = tmp_path / "uploads"
    folder.mkdir()
    (folder / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    dummy = _make_processor(batch_mixin, tmp_path, init_result=None)

    with pytest.raises(RuntimeError, match="unknown error"):
        await dummy.process_folder_complete(
            str(folder),
            output_dir=str(tmp_path / "out"),
        )

    assert dummy.init_calls == 1
    assert dummy.process_calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_omitted_display_stats_still_logs_folder_summary(tmp_path):
    batch_mixin = _load_batch_mixin()
    folder = tmp_path / "uploads"
    folder.mkdir()
    (folder / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "out"
    dummy = _make_processor(batch_mixin, tmp_path)

    await dummy.process_folder_complete(str(folder), output_dir=str(output_dir))

    assert [Path(path).name for path, _kwargs in dummy.process_calls] == ["manual.pdf"]
    assert output_dir.is_dir()
    assert any("Processing complete!" in message for message in dummy.logger.infos)
    assert any("Successful: 1" in message for message in dummy.logger.infos)
    assert any("Failed: 0" in message for message in dummy.logger.infos)

    dummy.logger = FakeLogger()
    dummy.process_calls.clear()
    await dummy.process_folder_complete(
        str(folder),
        output_dir=str(output_dir),
        display_stats=False,
    )

    assert [Path(path).name for path, _kwargs in dummy.process_calls] == ["manual.pdf"]
    assert any("Found 1 files" in message for message in dummy.logger.infos)
    assert not any("Processing complete!" in message for message in dummy.logger.infos)
    assert not any("Successful:" in message for message in dummy.logger.infos)
    assert not any("Failed:" in message for message in dummy.logger.infos)
