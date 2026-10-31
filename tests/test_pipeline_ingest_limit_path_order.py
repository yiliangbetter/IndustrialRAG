"""Pipeline --limit keeps the sorted path prefix, not creation or basename order.

``_collect_files`` sorts resolved paths, and ``_ingest_folder`` then slices
that list when ``limit > 0``. Top-level ``m.pdf`` sorts before nested
``sub/b.pdf`` because the full path is compared, even though basename ``b``
comes first and the nested file is created first. ``--limit 1`` must ingest
``m.pdf`` only. ``--limit 2`` must ingest ``m.pdf`` and then ``sub/b.pdf``,
leaving later ``z.pdf`` and a ``.txt`` sibling out.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_ingest_limit_order_1031",
        REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _ListLogger:
    def info(self, msg, *args):
        pass

    def error(self, msg, *args):
        pass


class _RecordingRAG:
    def __init__(self):
        self.parses = []
        self.inserts = []

    async def parse_document(self, file_path, **kwargs):
        name = Path(file_path).name
        self.parses.append({"name": name, "file_path": file_path, **kwargs})
        return [{"type": "text", "text": name}], f"doc-{name}"

    async def insert_content_list(self, content_list, **kwargs):
        self.inserts.append({"content_list": content_list, **kwargs})


def _layout(tmp_path: Path) -> tuple[Path, Path]:
    folder = tmp_path / "docs"
    (folder / "sub").mkdir(parents=True)
    # Creation order disagrees with resolved-path order.
    (folder / "z.pdf").write_bytes(b"%PDF-z")
    (folder / "sub" / "b.pdf").write_bytes(b"%PDF-b")
    (folder / "m.pdf").write_bytes(b"%PDF-m")
    (folder / "notes.txt").write_text("skip", encoding="utf-8")
    return folder.resolve(), (tmp_path / "parsed").resolve()


def _ingest(pipeline, rag, folder, parser_output, limit):
    config = SimpleNamespace(
        supported_file_extensions=[".pdf"],
        display_content_stats=False,
    )
    return pipeline._ingest_folder(
        rag,
        config,
        _ListLogger(),
        input_folder=folder,
        parser_output_dir=parser_output,
        parse_method="ocr",
        parse_extra={"lang": "ch"},
        recursive=True,
        limit=limit,
        skip_multimodal=True,
    )


@pytest.mark.asyncio
async def test_limit_one_selects_resolved_path_order_not_basename(pipeline, tmp_path):
    folder, parser_output = _layout(tmp_path)
    rag = _RecordingRAG()

    ok, fail = await _ingest(pipeline, rag, folder, parser_output, limit=1)

    assert (ok, fail) == (1, 0)
    assert [item["name"] for item in rag.parses] == ["m.pdf"]
    assert [item["file_path"] for item in rag.inserts] == ["m.pdf"]
    assert rag.inserts[0]["doc_id"] == "doc-m.pdf"
    assert rag.inserts[0]["skip_multimodal_processing"] is True
    assert rag.parses[0]["parse_method"] == "ocr"
    assert rag.parses[0]["lang"] == "ch"
    assert rag.parses[0]["output_dir"] == str(parser_output)
    assert not (parser_output / "sub").exists()


@pytest.mark.asyncio
async def test_limit_two_keeps_the_sorted_prefix_including_nested(pipeline, tmp_path):
    folder, parser_output = _layout(tmp_path)
    rag = _RecordingRAG()

    ok, fail = await _ingest(pipeline, rag, folder, parser_output, limit=2)

    assert (ok, fail) == (2, 0)
    assert [item["name"] for item in rag.parses] == ["m.pdf", "b.pdf"]
    assert [item["file_path"] for item in rag.inserts] == ["m.pdf", "sub/b.pdf"]
    assert rag.parses[1]["output_dir"] == str(parser_output / "sub")
    assert rag.inserts[1]["doc_id"] == "doc-b.pdf"
    assert "notes.txt" not in [item["name"] for item in rag.parses]
    assert "z.pdf" not in [item["name"] for item in rag.parses]
