"""Regression tests for type-aware multimodal batch orchestration."""

import pytest

from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []
        self.debugs = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, msg, *args, **kwargs):
        self.errors.append(str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg))


class FakeDocStatusStorage:
    def __init__(self, records=None):
        self.records = records or {}

    async def get_by_id(self, key):
        return self.records.get(key)


class FakeModalProcessor:
    def __init__(self, *, fail=False, entity_name="Entity"):
        self.fail = fail
        self.entity_name = entity_name
        self.calls = []

    async def generate_description_only(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("description boom")
        content_type = kwargs["content_type"]
        return (
            f"{content_type} description",
            {"entity_name": self.entity_name, "entity_type": content_type},
        )


def _make_processor(*, records=None, modal_processors=None, max_parallel_insert=2):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type("Config", (), {"use_full_path": False})()
    processor.modal_processors = modal_processors or {}
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "doc_status": FakeDocStatusStorage(records),
            "max_parallel_insert": max_parallel_insert,
            "tokenizer": type(
                "Tok", (), {"encode": staticmethod(lambda text: [1, 2])}
            )(),
        },
    )()
    return processor


@pytest.mark.asyncio
async def test_batch_orchestration_noop_on_empty_items():
    processor = _make_processor()
    stage_calls = []

    def boom(*args, **kwargs):
        stage_calls.append("convert")
        raise AssertionError("should not convert empty input")

    processor._convert_to_lightrag_chunks_type_aware = boom

    await processor._process_multimodal_content_batch_type_aware([], "doc.pdf", "doc-1")

    assert stage_calls == []
    assert any(
        "No multimodal content to process" in msg for msg in processor.logger.debugs
    )


@pytest.mark.asyncio
async def test_batch_orchestration_selects_processors_and_runs_stages_in_order():
    table_proc = FakeModalProcessor(entity_name="Spec Table (table)")
    image_proc = FakeModalProcessor(entity_name="Figure 1 (image)")
    processor = _make_processor(
        records={"doc-1": {"chunks_count": 5, "chunks_list": ["c0"]}},
        modal_processors={"table": table_proc, "image": image_proc},
    )

    stage_order = []
    converted = {"chunk-mm": {"content": "x"}}

    def fake_convert(data_list, file_path, doc_id):
        stage_order.append("convert")
        processor.convert_args = (data_list, file_path, doc_id)
        return converted

    async def fake_store(chunks):
        stage_order.append("store")
        processor.store_args = chunks

    async def fake_store_entities(data_list, chunks, file_path, doc_id):
        stage_order.append("entities")
        processor.entity_args = (data_list, chunks, file_path, doc_id)

    async def fake_extract(chunks):
        stage_order.append("extract")
        processor.extract_args = chunks
        return [("nodes", "edges")]

    async def fake_belongs(chunk_results, data_list):
        stage_order.append("belongs")
        processor.belongs_args = (chunk_results, data_list)
        return [("enhanced", "edges")]

    async def fake_merge(enhanced, file_path, doc_id=None):
        stage_order.append("merge")
        processor.merge_args = (enhanced, file_path, doc_id)

    async def fake_update(doc_id, chunk_ids):
        stage_order.append("update")
        processor.update_args = (doc_id, chunk_ids)

    processor._convert_to_lightrag_chunks_type_aware = fake_convert
    processor._store_chunks_to_lightrag_storage_type_aware = fake_store
    processor._store_multimodal_main_entities = fake_store_entities
    processor._batch_extract_entities_lightrag_style_type_aware = fake_extract
    processor._batch_add_belongs_to_relations_type_aware = fake_belongs
    processor._batch_merge_lightrag_style_type_aware = fake_merge
    processor._update_doc_status_with_chunks_type_aware = fake_update

    items = [
        {"type": "table", "table_body": "| a |", "page_idx": 1},
        {"type": "image", "img_path": "/tmp/a.png", "page_idx": 2},
    ]
    await processor._process_multimodal_content_batch_type_aware(
        items, "/docs/manual.pdf", "doc-1"
    )

    assert stage_order == [
        "convert",
        "store",
        "entities",
        "extract",
        "belongs",
        "merge",
        "update",
    ]

    assert len(table_proc.calls) == 1
    assert table_proc.calls[0]["content_type"] == "table"
    assert table_proc.calls[0]["item_info"] == {
        "page_idx": 1,
        "index": 0,
        "type": "table",
    }
    assert len(image_proc.calls) == 1
    assert image_proc.calls[0]["content_type"] == "image"
    assert image_proc.calls[0]["item_info"]["index"] == 1

    data_list, file_path, doc_id = processor.convert_args
    assert file_path == "/docs/manual.pdf"
    assert doc_id == "doc-1"
    assert len(data_list) == 2
    assert {d["content_type"] for d in data_list} == {"table", "image"}
    # Existing chunks_count offsets multimodal chunk order indexes.
    by_type = {d["content_type"]: d for d in data_list}
    assert by_type["table"]["chunk_order_index"] == 5
    assert by_type["image"]["chunk_order_index"] == 6
    assert by_type["table"]["file_path"] == "/docs/manual.pdf"

    assert processor.store_args is converted
    assert processor.entity_args[1] is converted
    assert processor.extract_args is converted
    assert processor.belongs_args[0] == [("nodes", "edges")]
    assert processor.merge_args == ([("enhanced", "edges")], "/docs/manual.pdf", "doc-1")
    assert processor.update_args == ("doc-1", ["chunk-mm"])


@pytest.mark.asyncio
async def test_batch_orchestration_filters_unknown_and_failed_items():
    good = FakeModalProcessor(entity_name="Good Table (table)")
    bad = FakeModalProcessor(fail=True, entity_name="Bad Image (image)")
    processor = _make_processor(
        records={},
        modal_processors={"table": good, "image": bad},
    )

    convert_calls = []

    def fake_convert(data_list, file_path, doc_id):
        convert_calls.append(data_list)
        return {"chunk-ok": {"content": "ok"}}

    async def fake_extract(chunks):
        return []

    async def fake_belongs(chunk_results, data_list):
        return []

    async def fake_merge(enhanced, file_path, doc_id=None):
        return None

    async def fake_update(doc_id, chunk_ids):
        return None

    async def fake_store(chunks):
        return None

    async def fake_entities(*args, **kwargs):
        return None

    processor._convert_to_lightrag_chunks_type_aware = fake_convert
    processor._store_chunks_to_lightrag_storage_type_aware = fake_store
    processor._store_multimodal_main_entities = fake_entities
    processor._batch_extract_entities_lightrag_style_type_aware = fake_extract
    processor._batch_add_belongs_to_relations_type_aware = fake_belongs
    processor._batch_merge_lightrag_style_type_aware = fake_merge
    processor._update_doc_status_with_chunks_type_aware = fake_update

    await processor._process_multimodal_content_batch_type_aware(
        [
            {"type": "unknown", "payload": "x"},
            {"type": "image", "img_path": "/tmp/bad.png"},
            {"type": "table", "table_body": "| ok |"},
        ],
        "manual.pdf",
        "doc-1",
    )

    assert len(good.calls) == 1
    assert len(bad.calls) == 1
    assert len(convert_calls) == 1
    assert len(convert_calls[0]) == 1
    assert convert_calls[0][0]["content_type"] == "table"
    assert any(
        "No processor found for type: unknown" in w for w in processor.logger.warnings
    )
    assert any(
        "Error generating description for image item" in e
        for e in processor.logger.errors
    )


@pytest.mark.asyncio
async def test_batch_orchestration_aborts_when_all_descriptions_fail():
    bad = FakeModalProcessor(fail=True)
    processor = _make_processor(modal_processors={"table": bad})
    stage_calls = []

    def fake_convert(*args, **kwargs):
        stage_calls.append("convert")
        return {}

    processor._convert_to_lightrag_chunks_type_aware = fake_convert

    await processor._process_multimodal_content_batch_type_aware(
        [{"type": "table", "table_body": "| x |"}],
        "manual.pdf",
        "doc-1",
    )

    assert stage_calls == []
    assert any(
        "No valid multimodal descriptions generated" in w
        for w in processor.logger.warnings
    )
