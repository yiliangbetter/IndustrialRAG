"""Regression tests for type-aware multimodal batch orchestration.

`_process_multimodal_content_batch_type_aware` fans out description generation
then stores, extracts, and merges. One bad item or a missing processor must not
drop siblings, and a total description failure must not write empty KG chunks.
"""

from unittest.mock import AsyncMock

import pytest

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


class FakeDocStatusStorage:
    def __init__(self, records=None, *, get_error=None):
        self.records = dict(records or {})
        self.get_error = get_error

    async def get_by_id(self, key):
        if self.get_error is not None:
            raise self.get_error
        return self.records.get(key)


class FakeProcessor:
    def __init__(self, *, description="caption", fail=False):
        self.description = description
        self.fail = fail
        self.calls = []

    async def generate_description_only(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("caption failed")
        return self.description, {
            "entity_name": f"{kwargs['content_type'].title()} 1 ({kwargs['content_type']})",
            "entity_type": kwargs["content_type"],
            "summary": self.description,
        }


def _dummy(storage=None, processors=None):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.modal_processors = processors or {}
    dummy.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "doc_status": storage or FakeDocStatusStorage(),
            "max_parallel_insert": 2,
        },
    )()
    dummy.stage_calls = []

    def convert(data_list, file_path, doc_id):
        dummy.stage_calls.append(("convert", list(data_list), file_path, doc_id))
        return {
            f"chunk-{item['index']}": {"content": item["description"]}
            for item in data_list
        }

    dummy._convert_to_lightrag_chunks_type_aware = convert
    dummy._store_chunks_to_lightrag_storage_type_aware = AsyncMock(
        side_effect=lambda chunks: dummy.stage_calls.append(("store_chunks", chunks))
    )
    dummy._store_multimodal_main_entities = AsyncMock(
        side_effect=lambda *args, **kwargs: dummy.stage_calls.append(
            ("store_entities", args, kwargs)
        )
    )
    dummy._batch_extract_entities_lightrag_style_type_aware = AsyncMock(
        side_effect=lambda chunks: dummy.stage_calls.append(("extract", chunks))
        or [("nodes", "edges")]
    )
    dummy._batch_add_belongs_to_relations_type_aware = AsyncMock(
        side_effect=lambda results, data: dummy.stage_calls.append(
            ("belongs_to", results, data)
        )
        or [("enhanced",)]
    )
    dummy._batch_merge_lightrag_style_type_aware = AsyncMock(
        side_effect=lambda results, file_path, doc_id: dummy.stage_calls.append(
            ("merge", file_path, doc_id)
        )
    )
    dummy._update_doc_status_with_chunks_type_aware = AsyncMock(
        side_effect=lambda doc_id, chunk_ids: dummy.stage_calls.append(
            ("update_status", doc_id, list(chunk_ids))
        )
    )
    return dummy


def _stage_names(dummy):
    return [call[0] for call in dummy.stage_calls]


@pytest.mark.asyncio
async def test_empty_items_return_without_downstream_stages():
    dummy = _dummy()
    await dummy._process_multimodal_content_batch_type_aware([], "doc.pdf", "doc-1")
    assert dummy.stage_calls == []


@pytest.mark.asyncio
async def test_missing_processor_and_failed_item_do_not_drop_sibling():
    image_proc = FakeProcessor(description="pump photo")
    table_proc = FakeProcessor(fail=True)
    dummy = _dummy(
        processors={"image": image_proc, "table": table_proc}
        # no generic → unknown types are skipped
    )

    await dummy._process_multimodal_content_batch_type_aware(
        [
            {"type": "table", "table_body": "a|b", "page_idx": 0},
            {"type": "image", "img_path": "/abs/fig.png", "page_idx": 1},
            {"type": "chart", "content": "unsupported"},
        ],
        "doc.pdf",
        "doc-1",
    )

    assert len(image_proc.calls) == 1
    assert image_proc.calls[0]["content_type"] == "image"
    assert len(table_proc.calls) == 1
    convert = dummy.stage_calls[0]
    assert convert[0] == "convert"
    data_list = convert[1]
    assert len(data_list) == 1
    assert data_list[0]["content_type"] == "image"
    assert data_list[0]["index"] == 1
    assert _stage_names(dummy) == [
        "convert",
        "store_chunks",
        "store_entities",
        "extract",
        "belongs_to",
        "merge",
        "update_status",
    ]
    assert dummy.stage_calls[-1][1] == "doc-1"
    assert dummy.stage_calls[-1][2] == ["chunk-1"]


@pytest.mark.asyncio
async def test_all_description_failures_skip_storage():
    dummy = _dummy(processors={"image": FakeProcessor(fail=True)})
    await dummy._process_multimodal_content_batch_type_aware(
        [{"type": "image", "img_path": "/abs/fig.png"}],
        "doc.pdf",
        "doc-1",
    )
    assert dummy.stage_calls == []


@pytest.mark.asyncio
async def test_existing_chunk_count_offsets_order_index():
    dummy = _dummy(
        storage=FakeDocStatusStorage({"doc-1": {"chunks_count": 5}}),
        processors={"image": FakeProcessor()},
    )
    await dummy._process_multimodal_content_batch_type_aware(
        [{"type": "image", "img_path": "/abs/fig.png", "page_idx": 2}],
        "doc.pdf",
        "doc-1",
    )
    item = dummy.stage_calls[0][1][0]
    assert item["chunk_order_index"] == 5
    assert item["item_info"]["page_idx"] == 2


@pytest.mark.asyncio
async def test_status_lookup_error_defaults_order_index_to_zero():
    dummy = _dummy(
        storage=FakeDocStatusStorage(get_error=RuntimeError("status down")),
        processors={"table": FakeProcessor()},
    )
    await dummy._process_multimodal_content_batch_type_aware(
        [{"type": "table", "table_body": "a|b"}],
        "doc.pdf",
        "doc-1",
    )
    assert dummy.stage_calls[0][1][0]["chunk_order_index"] == 0
