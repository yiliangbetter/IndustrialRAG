"""Citation identity follows use_full_path and does not rewrite the path.

Two machine manuals that share a basename must not overwrite each other's
failed-document row, and a nested relative path must stay intact when full
paths are enabled. Basename mode is what makes ``manual.pdf`` collide.
An explicit file_name is the citation even when the flag says otherwise.
"""

import os

import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class StatusStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _processor(use_full_path):
    processor = ProcessorMixin()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "use_full_path": use_full_path,
            "allow_embedding_only_ingestion": False,
        },
    )()
    processor.lightrag = type("LightRAG", (), {"doc_status": StatusStorage()})()
    return processor


@pytest.mark.asyncio
async def test_failed_status_key_collapses_to_basename():
    processor = _processor(use_full_path=False)
    errors = iter(("green8 index down", "blue9 index down"))

    async def fail_init():
        return {"success": False, "error": next(errors)}

    processor._ensure_lightrag_initialized = fail_init

    green = "models/green8/manual.pdf"
    blue = "/var/manuals/blue9/manual.pdf"
    assert await processor.process_document_complete_lightrag_api(green) is False
    assert await processor.process_document_complete_lightrag_api(blue) is False

    records = processor.lightrag.doc_status.records
    assert list(records) == ["doc-pre-manual.pdf"]
    row = records["doc-pre-manual.pdf"]
    assert row["status"] == DocStatus.FAILED
    assert row["file_path"] == "manual.pdf"
    assert row["error_msg"] == "blue9 index down"
    assert "green8" not in row["file_path"]
    assert processor.lightrag.doc_status.index_done_calls == 2


@pytest.mark.asyncio
async def test_failed_status_key_keeps_the_given_path_string():
    processor = _processor(use_full_path=True)
    errors = iter(("green8 index down", "blue9 index down", "windows path down"))

    async def fail_init():
        return {"success": False, "error": next(errors)}

    processor._ensure_lightrag_initialized = fail_init

    green = "models/green8/../green8/manual.pdf"
    blue = "/var/manuals/blue9/manual.pdf"
    windows = "models\\blue9\\manual.pdf"
    assert await processor.process_document_complete_lightrag_api(green) is False
    assert await processor.process_document_complete_lightrag_api(blue) is False
    assert await processor.process_document_complete_lightrag_api(windows) is False

    records = processor.lightrag.doc_status.records
    assert records[f"doc-pre-{green}"]["file_path"] == green
    assert records[f"doc-pre-{green}"]["error_msg"] == "green8 index down"
    assert records[f"doc-pre-{green}"]["status"] == DocStatus.FAILED
    assert os.path.isabs(records[f"doc-pre-{green}"]["file_path"]) is False
    assert records[f"doc-pre-{blue}"]["file_path"] == blue
    assert records[f"doc-pre-{blue}"]["error_msg"] == "blue9 index down"
    assert records[f"doc-pre-{windows}"]["file_path"] == windows
    assert records[f"doc-pre-{windows}"]["error_msg"] == "windows path down"
    assert "doc-pre-manual.pdf" not in records


@pytest.mark.asyncio
async def test_text_insert_citation_follows_flag_unless_file_name_is_set(monkeypatch):
    captured = []
    parsed = []

    async def fake_insert(lightrag, input, file_paths=None, **kwargs):
        captured.append(
            {
                "file_paths": file_paths,
                "text": input,
                "ids": kwargs.get("ids"),
            }
        )

    monkeypatch.setattr("raganything.processor.insert_text_content", fake_insert)

    async def ingest(use_full_path, file_path, file_name=None):
        processor = _processor(use_full_path)

        async def ready():
            return {"success": True}

        async def parse(path, output_dir, parse_method, display_stats, **kwargs):
            parsed.append(
                {
                    "path": path,
                    "output_dir": output_dir,
                    "parse_method": parse_method,
                    "display_stats": display_stats,
                }
            )
            return (
                [{"type": "text", "text": "Torque 12 N·m", "page_idx": 0}],
                "doc-from-parse",
            )

        async def mark(doc_id):
            captured.append({"marked": doc_id})

        processor._ensure_lightrag_initialized = ready
        processor.parse_document = parse
        processor._mark_multimodal_processing_complete = mark
        await processor.process_document_complete(
            file_path,
            output_dir="parsed",
            parse_method="ocr",
            display_stats=False,
            doc_id="doc-1",
            file_name=file_name,
        )

    nested = "models/green8/manual.pdf"
    await ingest(False, nested)
    await ingest(True, nested)
    await ingest(False, "/var/manuals/other/manual.pdf", file_name=nested)
    await ingest(True, "/var/manuals/other/manual.pdf", file_name=nested)

    assert parsed == [
        {
            "path": nested,
            "output_dir": "parsed",
            "parse_method": "ocr",
            "display_stats": False,
        },
        {
            "path": nested,
            "output_dir": "parsed",
            "parse_method": "ocr",
            "display_stats": False,
        },
        {
            "path": "/var/manuals/other/manual.pdf",
            "output_dir": "parsed",
            "parse_method": "ocr",
            "display_stats": False,
        },
        {
            "path": "/var/manuals/other/manual.pdf",
            "output_dir": "parsed",
            "parse_method": "ocr",
            "display_stats": False,
        },
    ]
    assert [item["file_paths"] for item in captured if "file_paths" in item] == [
        "manual.pdf",
        nested,
        nested,
        nested,
    ]
    assert all(item["text"] == "Torque 12 N·m" for item in captured if "text" in item)
    assert all(item["ids"] == "doc-1" for item in captured if "ids" in item)
    assert [item["marked"] for item in captured if "marked" in item] == [
        "doc-1",
        "doc-1",
        "doc-1",
        "doc-1",
    ]
