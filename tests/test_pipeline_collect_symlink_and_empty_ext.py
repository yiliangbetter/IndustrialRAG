"""Pipeline file selection must stay inside the upload tree and the extension list.

``scripts/rag_pipeline_parse_graph_chat.py`` is the production parse-to-graph
path. ``_collect_files`` uses ``Path.glob``. A symlinked subdirectory must not
pull manuals from outside the folder, and an in-tree alias must not ingest the
same nested manual twice.

A blank ``SUPPORTED_FILE_EXTENSIONS`` token is normalized to ``"."``. That is a
trailing-dot suffix, not the match-all pattern used by
``process_folder_complete``. Extensionless files, other types, and directories
must stay out of the batch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_collect_symlink_empty_ext", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tree(folder: Path) -> Path:
    nested = folder / "line-a"
    nested.mkdir(parents=True)
    (folder / "manual.pdf").write_bytes(b"%PDF-1.4\n")
    (folder / "notes.txt").write_text("shift notes")
    (folder / "README").write_text("no extension")
    (folder / "trailing.").write_text("dot")
    (nested / "child.pdf").write_bytes(b"%PDF-1.4\n")

    outside = folder.parent / "outside"
    outside.mkdir()
    (outside / "secret.pdf").write_bytes(b"%PDF-1.4\n")
    (folder / "escape").symlink_to(outside, target_is_directory=True)
    (folder / "alias").symlink_to(nested, target_is_directory=True)
    return outside


def test_collect_files_skips_external_and_alias_symlink_dirs(pipeline, tmp_path):
    folder = tmp_path / "docs"
    _write_tree(folder)

    found = pipeline._collect_files(folder, [".pdf", "PDF"], recursive=True)

    assert found == sorted(
        [
            (folder / "manual.pdf").resolve(),
            (folder / "line-a" / "child.pdf").resolve(),
        ]
    )
    assert all("escape" not in path.parts for path in found)
    assert all("alias" not in path.parts for path in found)
    assert all(path.name != "secret.pdf" for path in found)


def test_collect_files_non_recursive_stays_at_top_level(pipeline, tmp_path):
    folder = tmp_path / "docs"
    _write_tree(folder)

    found = pipeline._collect_files(folder, ["pdf"], recursive=False)

    assert found == [(folder / "manual.pdf").resolve()]


def test_blank_extension_token_is_not_match_all(pipeline, tmp_path):
    folder = tmp_path / "docs"
    _write_tree(folder)

    found = pipeline._collect_files(folder, [".pdf", "", "  "], recursive=True)
    names = [path.name for path in found]

    assert names == ["child.pdf", "manual.pdf", "trailing."]
    assert "notes.txt" not in names
    assert "README" not in names
    assert "secret.pdf" not in names
    assert all(path.is_file() for path in found)
