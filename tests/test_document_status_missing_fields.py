"""Status queries must tolerate rows that omit chunk and timestamp fields."""

from __future__ import annotations

import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class _Logger:
    def error(self, *args, **kwargs):
        return None


class _DocStatus:
    def __init__(self, record):
        self.record = record

    async def get_by_id(self, doc_id):
        return self.record


def _processor(record):
    processor = ProcessorMixin()
    processor.logger = _Logger()
    processor.lightrag = type("LightRAG", (), {})()
    processor.lightrag.doc_status = _DocStatus(record)
    return processor


@pytest.mark.asyncio
async def test_status_query_fills_omitted_chunk_fields():
    record = {"scheme_name": "plant-a"}
    status = await _processor(record).get_document_processing_status("doc-1")

    assert status == {
        "exists": True,
        "text_processed": False,
        "multimodal_processed": False,
        "fully_processed": False,
        "chunks_count": 0,
        "chunks_list": [],
        "status": "",
        "updated_at": "",
        "raw_status": record,
    }


@pytest.mark.asyncio
async def test_string_processed_without_multimodal_flag_is_not_complete():
    record = {"status": "processed"}
    status = await _processor(record).get_document_processing_status("doc-1")

    assert status["exists"] is True
    assert status["text_processed"] is True
    assert status["status"] == DocStatus.PROCESSED
    assert status["multimodal_processed"] is False
    assert status["fully_processed"] is False
    assert status["chunks_count"] == 0
    assert status["chunks_list"] == []
    assert status["updated_at"] == ""
    assert status["raw_status"] is record


@pytest.mark.asyncio
async def test_string_processed_and_multimodal_flag_is_complete_without_counts():
    record = {"status": "processed", "multimodal_processed": True}
    status = await _processor(record).get_document_processing_status("doc-1")

    assert status["text_processed"] is True
    assert status["multimodal_processed"] is True
    assert status["fully_processed"] is True
    assert status["chunks_count"] == 0
    assert status["chunks_list"] == []
    assert status["raw_status"] is record
