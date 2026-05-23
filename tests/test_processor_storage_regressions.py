from types import SimpleNamespace

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


class FakeStorage:
    def __init__(self, initial=None):
        self.records = dict(initial or {})
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class FakeGraph:
    def __init__(self):
        self.nodes = {}

    async def upsert_node(self, name, data):
        self.nodes[name] = data


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeLightRAG:
    def __init__(self, full_entities_initial=None):
        self.text_chunks = FakeStorage()
        self.chunks_vdb = FakeStorage()
        self.doc_status = FakeStorage()
        self.entities_vdb = FakeStorage()
        self.full_entities = FakeStorage(full_entities_initial)
        self.chunk_entity_relation_graph = FakeGraph()
        self.tokenizer = FakeTokenizer()
        self.insert_done_calls = 0

    async def _insert_done(self):
        self.insert_done_calls += 1


class DummyProcessor(ProcessorMixin):
    pass


def make_processor(*, allow_embedding_only_ingestion=False, full_entities_initial=None):
    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(
        allow_embedding_only_ingestion=allow_embedding_only_ingestion,
        content_format="mineru",
        display_content_stats=False,
        use_full_path=False,
    )
    processor.lightrag = FakeLightRAG(full_entities_initial)

    async def fake_ensure_lightrag_initialized():
        return {"success": True}

    processor._ensure_lightrag_initialized = fake_ensure_lightrag_initialized
    return processor


@pytest.mark.asyncio
async def test_embedding_only_content_list_recovers_mineru_v2_text_and_status():
    processor = make_processor(allow_embedding_only_ingestion=True)
    content_list = [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Pump report"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Pressure rose above limit"}
                    ]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {
                            "prefix": "1.",
                            "item_content": [
                                {"type": "text", "content": "Inspect valve"}
                            ],
                        }
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "<table><tr><td>psi</td></tr></table>"},
            },
            {
                "type": "image",
                "content": {"image_caption": ["Pump schematic"]},
            },
        ]
    ]

    await processor.insert_content_list(
        content_list,
        file_path="/tmp/reports/pump_content_list_v2.json",
    )

    assert processor.lightrag.insert_done_calls == 1
    assert processor.lightrag.text_chunks.records == processor.lightrag.chunks_vdb.records

    chunk_records = sorted(
        processor.lightrag.text_chunks.records.values(),
        key=lambda record: record["chunk_order_index"],
    )
    assert [record["content"] for record in chunk_records] == [
        "Pump report",
        "Pressure rose above limit",
        "1. Inspect valve",
        "<table><tr><td>psi</td></tr></table>",
        "Pump schematic",
    ]
    assert {record["file_path"] for record in chunk_records} == {
        "pump_content_list_v2.json"
    }

    [doc_id] = processor.lightrag.doc_status.records.keys()
    doc_status = processor.lightrag.doc_status.records[doc_id]
    assert doc_status["status"] == DocStatus.PROCESSED
    assert doc_status["multimodal_processed"] is True
    assert doc_status["chunks_count"] == 5
    assert doc_status["chunks_list"] == list(
        processor.lightrag.text_chunks.records.keys()
    )
    assert processor.lightrag.doc_status.index_done_calls == 1


@pytest.mark.asyncio
async def test_store_multimodal_main_entities_writes_all_required_storages():
    processor = make_processor()
    multimodal_data = [
        {
            "entity_info": {
                "entity_name": "Pump Diagram (image)",
                "entity_type": "image",
                "summary": "A labeled pump diagram",
            },
            "description": "A labeled pump diagram",
            "content_type": "image",
            "original_item": {
                "type": "image",
                "img_path": "/tmp/pump.png",
                "image_caption": ["Pump Diagram"],
                "image_footnote": [],
            },
        }
    ]

    await processor._store_multimodal_main_entities(
        multimodal_data,
        lightrag_chunks={},
        file_path="/tmp/reports/pump.pdf",
        doc_id="doc-pump",
    )

    graph_node = processor.lightrag.chunk_entity_relation_graph.nodes[
        "Pump Diagram (image)"
    ]
    assert graph_node["entity_type"] == "image"
    assert graph_node["description"] == "A labeled pump diagram"
    assert graph_node["file_path"] == "pump.pdf"

    [entity] = processor.lightrag.entities_vdb.records.values()
    assert entity["entity_name"] == "Pump Diagram (image)"
    assert entity["file_path"] == "pump.pdf"
    assert entity["source_id"].startswith("chunk-")
    assert processor.lightrag.entities_vdb.index_done_calls == 1

    assert processor.lightrag.full_entities.records["doc-pump"]["entity_names"] == [
        "Pump Diagram (image)"
    ]
    assert processor.lightrag.full_entities.records["doc-pump"]["count"] == 1
    assert processor.lightrag.full_entities.index_done_calls == 1


@pytest.mark.asyncio
async def test_store_multimodal_entities_preserves_existing_full_entity_metadata():
    processor = make_processor(
        full_entities_initial={
            "doc-pump": {
                "entity_names": ["Existing Entity", "Pump Diagram (image)"],
                "count": 2,
                "update_time": 10,
                "source": "text_pipeline",
                "doc_status": "indexed",
            }
        }
    )
    entities_to_store = {
        "ent-existing": {"entity_name": "Pump Diagram (image)"},
        "ent-new": {"entity_name": "Valve Table (table)"},
    }

    await processor._store_multimodal_entities_to_full_entities(
        entities_to_store,
        doc_id="doc-pump",
    )

    merged = processor.lightrag.full_entities.records["doc-pump"]
    assert merged["entity_names"] == [
        "Existing Entity",
        "Pump Diagram (image)",
        "Valve Table (table)",
    ]
    assert merged["count"] == 3
    assert merged["source"] == "text_pipeline"
    assert merged["doc_status"] == "indexed"
    assert merged["update_time"] != 10
    assert processor.lightrag.full_entities.index_done_calls == 1
