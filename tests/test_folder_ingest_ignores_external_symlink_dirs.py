"""Folder ingest must not follow a directory symlink outside the upload tree.

``process_folder_complete`` selects files with ``Path.glob``. A symlinked
subdirectory whose target sits outside the folder would otherwise pull manuals
from another plant, another user's drop, or a secret mount into the same RAG
batch. Real nested files must still be ingested, and an in-tree alias must not
process the same manual twice.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def _load_batch_mixin():
    """Load BatchMixin without importing the raganything package.

    Package import pulls LightRAG. Folder selection only needs the mixin.
    """
    package_name = "_raganything_folder_symlink_fd71"
    root = Path(__file__).resolve().parents[1] / "raganything"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    def load(subname: str, filename: str):
        full_name = f"{package_name}.{subname}"
        spec = importlib.util.spec_from_file_location(full_name, root / filename)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[full_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    load("parser", "parser.py")
    load("batch_parser", "batch_parser.py")
    return load("batch", "batch.py").BatchMixin


BatchMixin = _load_batch_mixin()


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


def _make_batch(tmp_path: Path):
    class DummyBatch(BatchMixin):
        pass

    dummy = DummyBatch()
    dummy.logger = _Logger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser_output_dir": str(tmp_path / "parsed"),
            "parse_method": "auto",
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
        dummy.process_calls.append(file_path)
        return None

    dummy._ensure_lightrag_initialized = _ensure
    dummy.process_document_complete = _process
    return dummy


def test_recursive_folder_ingest_skips_external_symlink_dirs(tmp_path):
    folder = tmp_path / "docs"
    nested = folder / "line-a"
    nested.mkdir(parents=True)
    (folder / "local.pdf").write_bytes(b"%PDF-1.4\n")
    (nested / "child.pdf").write_bytes(b"%PDF-1.4\n")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.pdf").write_bytes(b"%PDF-1.4\n")
    (folder / "escape").symlink_to(outside, target_is_directory=True)
    # Alias of a real subfolder. Following it would ingest child.pdf twice.
    (folder / "alias").symlink_to(nested, target_is_directory=True)

    dummy = _make_batch(tmp_path)
    asyncio.run(
        dummy.process_folder_complete(
            str(folder),
            output_dir=str(tmp_path / "out"),
            recursive=True,
            display_stats=False,
        )
    )

    processed = sorted(Path(path).resolve() for path in dummy.process_calls)
    assert processed == sorted(
        [
            (folder / "local.pdf").resolve(),
            (nested / "child.pdf").resolve(),
        ]
    )
    assert all("escape" not in Path(path).parts for path in dummy.process_calls)
    assert all("alias" not in Path(path).parts for path in dummy.process_calls)
    assert all(Path(path).name != "secret.pdf" for path in dummy.process_calls)
