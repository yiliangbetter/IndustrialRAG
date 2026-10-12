"""Deeply nested folder ingest must keep relative citations and output dirs.

``process_folder_complete`` treats a file as nested when its path relative to
the upload folder has more than one parent. A single subdirectory
(``line-a/child.pdf``) is already covered. Manual trees are often
``machine/section/spec.pdf``. A check that only matches exactly one
subdirectory would store that spec beside the folder-root PDF and drop the
relative citation.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_batch_mixin():
    """Load BatchMixin without importing raganything/__init__.py."""
    pkg_name = "_raganything_batch_deep_nest_da91"
    if pkg_name not in sys.modules:
        package = types.ModuleType(pkg_name)
        package.__path__ = [str(REPO_ROOT / "raganything")]
        sys.modules[pkg_name] = package

        def load(module_name: str, filename: str):
            spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.{module_name}",
                REPO_ROOT / "raganything" / filename,
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

        load("parser", "parser.py")
        load("batch_parser", "batch_parser.py")
        load("batch", "batch.py")
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


def _make_processor(batch_mixin, tmp_path):
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
            "supported_file_extensions": [".pdf"],
            "recursive_folder_processing": True,
            "max_concurrent_files": 2,
        },
    )()
    dummy.process_calls = []

    async def fake_ensure():
        return {"success": True}

    async def fake_process(file_path, **kwargs):
        if Path(file_path).name == "broken.pdf":
            raise RuntimeError("parse exploded")
        dummy.process_calls.append((Path(file_path).name, kwargs))

    dummy._ensure_lightrag_initialized = fake_ensure
    dummy.process_document_complete = fake_process
    return dummy


@pytest.mark.asyncio
async def test_deeply_nested_file_keeps_relative_citation_and_output_dir(tmp_path):
    batch_mixin = _load_batch_mixin()
    folder = tmp_path / "uploads"
    nested = folder / "machine" / "section"
    nested.mkdir(parents=True)
    (folder / "cover.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "spec.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "notes.txt").write_text("not a pdf", encoding="utf-8")

    dummy = _make_processor(batch_mixin, tmp_path)
    output_dir = tmp_path / "out"

    await dummy.process_folder_complete(
        str(folder),
        output_dir=str(output_dir),
        parse_method="ocr",
        display_stats=False,
        file_extensions=[".pdf"],
        recursive=True,
        max_workers=2,
    )

    by_name = {name: kwargs for name, kwargs in dummy.process_calls}
    assert set(by_name) == {"cover.pdf", "spec.pdf"}

    assert by_name["cover.pdf"]["file_name"] is None
    assert by_name["cover.pdf"]["output_dir"] == str(output_dir)
    assert by_name["cover.pdf"]["parse_method"] == "ocr"

    relative_spec = Path("machine") / "section" / "spec.pdf"
    assert by_name["spec.pdf"]["file_name"] == str(relative_spec)
    assert by_name["spec.pdf"]["output_dir"] == str(output_dir / "machine" / "section")
    assert by_name["spec.pdf"]["parse_method"] == "ocr"
    assert output_dir.is_dir()


@pytest.mark.asyncio
async def test_deeply_nested_failure_does_not_drop_sibling_files(tmp_path):
    batch_mixin = _load_batch_mixin()
    folder = tmp_path / "uploads"
    nested = folder / "machine" / "section"
    nested.mkdir(parents=True)
    (folder / "cover.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "spec.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "broken.pdf").write_bytes(b"%PDF-1.4\n")

    dummy = _make_processor(batch_mixin, tmp_path)

    await dummy.process_folder_complete(
        str(folder),
        output_dir=str(tmp_path / "out"),
        display_stats=True,
        file_extensions=[".pdf"],
        recursive=True,
        max_workers=1,
    )

    processed = {name for name, _kwargs in dummy.process_calls}
    assert processed == {"cover.pdf", "spec.pdf"}
    assert any(
        "broken.pdf" in message and "parse exploded" in message
        for message in dummy.logger.errors
    )
    assert any("Successful: 2" in message for message in dummy.logger.infos)
    assert any("Failed: 1" in message for message in dummy.logger.infos)
