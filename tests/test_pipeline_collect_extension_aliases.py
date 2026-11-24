"""Pipeline file selection must treat aliased extension tokens as one suffix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"


def _load_pipeline():
    name = "rag_pipeline_collect_ext_alias_1124"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_files_collapses_aliased_pdf_tokens_and_skips_directories(tmp_path):
    pipeline = _load_pipeline()
    folder = tmp_path / "manuals"
    nested = folder / "sub"
    nested.mkdir(parents=True)
    manual = folder / "manual.pdf"
    notes = folder / "notes.txt"
    nested_pdf = nested / "nested.pdf"
    manual.write_bytes(b"%PDF")
    notes.write_bytes(b"notes")
    nested_pdf.write_bytes(b"%PDF")
    # A directory whose name ends in .pdf must not be selected.
    (folder / "decoy.pdf").mkdir()

    found = pipeline._collect_files(
        folder, ["PDF", ".pdf", "  .pdf  ", " pdf "], recursive=False
    )

    assert found == [manual.resolve()]

    both = pipeline._collect_files(folder, ["txt", ".PDF"], recursive=True)
    assert both == sorted([manual.resolve(), notes.resolve(), nested_pdf.resolve()])
    assert both.count(manual.resolve()) == 1
