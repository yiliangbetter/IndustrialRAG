"""Type-aware multimodal batch must keep going when one item cannot be described.

``_process_multimodal_content_batch_type_aware`` is the path that writes image,
table, and equation chunks for a document. A missing processor or a caption
failure has to drop only that item. The surviving item keeps its original index
in ``chunk_order_index`` (existing chunk count plus that index), and a total
failure must not write chunks or run extraction.

This is the inner per-item gate. It is distinct from the outer batch-to-individual
fallback and from individual chunk merge.
"""

from types import SimpleNamespace

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


class StatusStorage:
    def __init__(self, record=None, error=None):
        self.record = record
        self.error = error
        self.calls = []

    async def get_by_id(self, doc_id):
        self.calls.append(doc_id)
        if self.error is not None:
            raise self.error
        return self.record


class DescriptionProcessor:
    def __init__(self, description, entity_name, error=None):
        self.description = description
        self.entity_name = entity_name
        self.error = error
        self.calls = []

    async def generate_description_only(
        self, modal_content, content_type, item_info, entity_name
    ):
        self.calls.append(
            {
                "modal_content": modal_content,
                "content_type": content_type,
                "item_info": item_info,
                "entity_name": entity_name,
            }
        )
        if self.error is not None:
            raise self.error
        return self.description, {
            "entity_name": self.entity_name,
            "entity_type": content_type,
        }


def _processor(status, processors):
    processor = ProcessorMixin()
    processor.logger = FakeLogger()
    processor.lightrag = SimpleNamespace(
        doc_status=status,
        max_parallel_insert=1,
    )
    processor.modal_processors = processors
    calls = {
        "convert": [],
        "store": [],
        "entities": [],
        "extract": [],
        "update": [],
    }

    def convert(data_list, file_path, doc_id):
        calls["convert"].append((data_list, file_path, doc_id))
        return {"chunk-bearing": {"content": data_list[0]["description"]}}

    async def store(chunks):
        calls["store"].append(chunks)

    async def entities(data_list, chunks, file_path, doc_id):
        calls["entities"].append((data_list, chunks, file_path, doc_id))

    async def extract(chunks):
        calls["extract"].append(chunks)
        return []

    async def belongs(chunk_results, data_list):
        return chunk_results

    async def merge(enhanced, file_path, doc_id=None):
        return None

    async def update(doc_id, chunk_ids):
        calls["update"].append((doc_id, chunk_ids))

    processor._convert_to_lightrag_chunks_type_aware = convert
    processor._store_chunks_to_lightrag_storage_type_aware = store
    processor._store_multimodal_main_entities = entities
    processor._batch_extract_entities_lightrag_style_type_aware = extract
    processor._batch_add_belongs_to_relations_type_aware = belongs
    processor._batch_merge_lightrag_style_type_aware = merge
    processor._update_doc_status_with_chunks_type_aware = update
    return processor, calls


@pytest.mark.asyncio
async def test_failed_and_missing_items_do_not_drop_the_later_table():
    image = DescriptionProcessor(
        "unused",
        "Scan",
        error=RuntimeError("caption model down"),
    )
    table = DescriptionProcessor("bearing load 12 kN", "BearingLoad")
    status = StatusStorage({"chunks_count": 4, "status": "processed"})
    processor, calls = _processor(
        status,
        {"image": image, "table": table},
    )
    items = [
        {"type": "image", "img_path": "/manuals/scan.png", "page_idx": 3},
        {"type": "equation", "text": "F=ma", "page_idx": 4},
        {"type": "table", "table_body": "bearing load 12 kN", "page_idx": 5},
    ]

    result = await processor._process_multimodal_content_batch_type_aware(
        items,
        "manuals/bearing.pdf",
        "doc-bearing",
    )

    assert result is None
    assert status.calls == ["doc-bearing"]
    assert len(image.calls) == 1
    assert image.calls[0]["content_type"] == "image"
    assert image.calls[0]["entity_name"] is None
    assert len(table.calls) == 1
    assert table.calls[0]["item_info"] == {
        "page_idx": 5,
        "index": 2,
        "type": "table",
    }
    assert table.calls[0]["modal_content"]["table_body"] == "bearing load 12 kN"

    assert len(calls["convert"]) == 1
    data_list, file_path, doc_id = calls["convert"][0]
    assert file_path == "manuals/bearing.pdf"
    assert doc_id == "doc-bearing"
    assert len(data_list) == 1
    survivor = data_list[0]
    assert survivor["content_type"] == "table"
    assert survivor["description"] == "bearing load 12 kN"
    assert survivor["chunk_order_index"] == 6
    assert survivor["original_item"]["table_body"] == "bearing load 12 kN"
    assert survivor["file_path"] == "manuals/bearing.pdf"
    assert calls["store"] == [{"chunk-bearing": {"content": "bearing load 12 kN"}}]
    assert calls["extract"] == [{"chunk-bearing": {"content": "bearing load 12 kN"}}]
    assert calls["update"] == [("doc-bearing", ["chunk-bearing"])]
    assert len(calls["entities"]) == 1
    assert calls["entities"][0][2:] == ("manuals/bearing.pdf", "doc-bearing")


@pytest.mark.asyncio
async def test_status_read_failure_still_stores_the_table_at_its_index():
    table = DescriptionProcessor("bearing load 12 kN", "BearingLoad")
    status = StatusStorage(error=ConnectionError("status down"))
    processor, calls = _processor(status, {"table": table})

    await processor._process_multimodal_content_batch_type_aware(
        [{"type": "table", "table_body": "bearing load 12 kN", "page_idx": 1}],
        "manuals/bearing.pdf",
        "doc-bearing",
    )

    assert status.calls == ["doc-bearing"]
    data_list, _, _ = calls["convert"][0]
    assert data_list[0]["chunk_order_index"] == 0
    assert data_list[0]["description"] == "bearing load 12 kN"
    assert calls["store"] == [{"chunk-bearing": {"content": "bearing load 12 kN"}}]


@pytest.mark.asyncio
async def test_no_described_items_skips_chunk_storage_and_extraction():
    image = DescriptionProcessor(
        "unused",
        "Scan",
        error=RuntimeError("caption model down"),
    )
    status = StatusStorage({"chunks_count": 4})
    processor, calls = _processor(status, {"image": image})

    result = await processor._process_multimodal_content_batch_type_aware(
        [
            {"type": "image", "img_path": "/manuals/scan.png", "page_idx": 0},
            {"type": "equation", "latex": "F=ma", "page_idx": 1},
        ],
        "manuals/bearing.pdf",
        "doc-bearing",
    )

    assert result is None
    assert len(image.calls) == 1
    assert calls["convert"] == []
    assert calls["store"] == []
    assert calls["entities"] == []
    assert calls["extract"] == []
    assert calls["update"] == []
