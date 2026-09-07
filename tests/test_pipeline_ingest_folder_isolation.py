"""Pipeline folder ingest must isolate per-file failures and preserve citations.

scripts/rag_pipeline_parse_graph_chat.py is the production parse→graph path.
A bad PDF must not abort sibling manuals; nested files must keep relative
citations and nested parser output dirs; --limit and --skip-multimodal must
reach insert_content_list.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    path = REPO_ROOT / "scripts" / "rag_pipeline_parse_graph_chat.py"
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline_parse_graph_chat_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.errors = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))


class FakeRAG:
    def __init__(self, fail_names=frozenset()):
        self.fail_names = set(fail_names)
        self.parse_calls = []
        self.insert_calls = []

    async def parse_document(
        self,
        file_path,
        output_dir=None,
        parse_method=None,
        display_stats=None,
        **kwargs,
    ):
        self.parse_calls.append(
            {
                "file_path": file_path,
                "output_dir": output_dir,
                "parse_method": parse_method,
                "kwargs": kwargs,
            }
        )
        name = Path(file_path).name
        if name in self.fail_names:
            raise RuntimeError(f"parse failed: {name}")
        doc_id = f"doc-{Path(file_path).stem}"
        return [{"type": "text", "text": f"body of {name}"}], doc_id

    async def insert_content_list(
        self,
        content_list,
        file_path=None,
        doc_id=None,
        skip_multimodal_processing=False,
        **kwargs,
    ):
        self.insert_calls.append(
            {
                "content_list": content_list,
                "file_path": file_path,
                "doc_id": doc_id,
                "skip_multimodal_processing": skip_multimodal_processing,
            }
        )


def _config():
    return SimpleNamespace(
        supported_file_extensions=[".pdf"],
        display_content_stats=False,
    )


@pytest.mark.asyncio
async def test_ingest_folder_continues_after_one_parse_failure(pipeline, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "ok.pdf").write_bytes(b"%PDF")
    (folder / "bad.pdf").write_bytes(b"%PDF")
    out = tmp_path / "out"
    rag = FakeRAG(fail_names={"bad.pdf"})
    logger = FakeLogger()

    ok, fail = await pipeline._ingest_folder(
        rag,
        _config(),
        logger,
        input_folder=folder,
        parser_output_dir=out,
        parse_method="auto",
        parse_extra={"lang": "ch"},
        recursive=True,
        limit=0,
        skip_multimodal=True,
    )

    assert ok == 1
    assert fail == 1
    assert len(rag.insert_calls) == 1
    assert rag.insert_calls[0]["file_path"] == "ok.pdf"
    assert rag.insert_calls[0]["doc_id"] == "doc-ok"
    assert any("bad.pdf" in err for err in logger.errors)


@pytest.mark.asyncio
async def test_ingest_folder_nested_output_forwards_skip_and_parse_id(
    pipeline, tmp_path
):
    folder = tmp_path / "docs"
    nested = folder / "plant-a"
    nested.mkdir(parents=True)
    (nested / "manual.pdf").write_bytes(b"%PDF")
    out = tmp_path / "parse-out"
    rag = FakeRAG()

    ok, fail = await pipeline._ingest_folder(
        rag,
        _config(),
        FakeLogger(),
        input_folder=folder,
        parser_output_dir=out,
        parse_method="ocr",
        parse_extra={"backend": "pipeline"},
        recursive=True,
        limit=0,
        skip_multimodal=True,
    )

    assert (ok, fail) == (1, 0)
    assert len(rag.parse_calls) == 1
    parse = rag.parse_calls[0]
    assert Path(parse["file_path"]).name == "manual.pdf"
    assert Path(parse["output_dir"]) == out / "plant-a"
    assert parse["parse_method"] == "ocr"
    assert parse["kwargs"]["backend"] == "pipeline"
    assert (out / "plant-a").is_dir()

    insert = rag.insert_calls[0]
    assert insert["file_path"] == "plant-a/manual.pdf"
    assert insert["doc_id"] == "doc-manual"
    assert insert["skip_multimodal_processing"] is True


@pytest.mark.asyncio
async def test_ingest_folder_limit_truncates_and_can_keep_multimodal(
    pipeline, tmp_path
):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"%PDF")
    (folder / "b.pdf").write_bytes(b"%PDF")
    rag = FakeRAG()

    ok, fail = await pipeline._ingest_folder(
        rag,
        _config(),
        FakeLogger(),
        input_folder=folder,
        parser_output_dir=tmp_path / "out",
        parse_method="auto",
        parse_extra={},
        recursive=False,
        limit=1,
        skip_multimodal=False,
    )

    assert (ok, fail) == (1, 0)
    assert len(rag.parse_calls) == 1
    assert len(rag.insert_calls) == 1
    assert rag.insert_calls[0]["skip_multimodal_processing"] is False
