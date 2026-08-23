"""Per-item isolation in individual multimodal fallback processing.

One bad figure or a missing processor must not drop the remaining items
or skip the final multimodal_processed mark.
"""

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


class FakeDocStatus:
    def __init__(self, records):
        self.records = records
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class _GoodProcessor:
    async def process_multimodal_content(self, **kwargs):
        return (
            "caption",
            {"entity_name": "GoodEntity", "chunk_id": "chunk-good"},
            [("nodes", {})],
        )


class _BadProcessor:
    async def process_multimodal_content(self, **kwargs):
        raise RuntimeError("caption failed")


def _processor(modal_processors, doc_id="doc-1"):
    proc = ProcessorMixin.__new__(ProcessorMixin)
    proc.logger = FakeLogger()
    proc.config = type("Config", (), {"use_full_path": False})()
    proc.modal_processors = modal_processors
    proc.lightrag = type(
        "FakeLightRAG",
        (),
        {
            "doc_status": FakeDocStatus(
                {
                    doc_id: {
                        "status": "processed",
                        "chunks_list": ["chunk-text"],
                        "chunks_count": 1,
                    }
                }
            )
        },
    )()
    return proc


@pytest.mark.asyncio
async def test_missing_processor_still_marks_complete():
    proc = _processor({})

    await proc._process_multimodal_content_individual(
        [{"type": "chart", "title": "x"}],
        "manual.pdf",
        "doc-1",
    )

    record = proc.lightrag.doc_status.records["doc-1"]
    assert record["multimodal_processed"] is True
    assert record["chunks_list"] == ["chunk-text"]
    assert proc.lightrag.doc_status.index_done_calls >= 1


@pytest.mark.asyncio
async def test_one_item_failure_does_not_drop_successful_sibling(monkeypatch):
    proc = _processor({"image": _GoodProcessor(), "table": _BadProcessor()})
    merge_calls = []

    async def fake_merge(**kwargs):
        merge_calls.append(kwargs)

    async def fake_namespace(_name):
        return {}

    def fake_lock():
        return object()

    async def fake_insert_done():
        return None

    monkeypatch.setattr("lightrag.operate.merge_nodes_and_edges", fake_merge)
    monkeypatch.setattr("lightrag.kg.shared_storage.get_namespace_data", fake_namespace)
    monkeypatch.setattr(
        "lightrag.kg.shared_storage.get_pipeline_status_lock", fake_lock
    )
    proc.lightrag._insert_done = fake_insert_done
    proc.lightrag.chunk_entity_relation_graph = object()
    proc.lightrag.entities_vdb = object()
    proc.lightrag.relationships_vdb = object()
    proc.lightrag.full_entities = object()
    proc.lightrag.full_relations = object()
    proc.lightrag.llm_response_cache = object()
    proc.lightrag.entity_chunks = object()
    proc.lightrag.relation_chunks = object()

    await proc._process_multimodal_content_individual(
        [
            {"type": "table", "table_body": "bad"},
            {"type": "image", "img_path": "ok.png"},
        ],
        "manual.pdf",
        "doc-1",
    )

    record = proc.lightrag.doc_status.records["doc-1"]
    assert record["chunks_list"] == ["chunk-text", "chunk-good"]
    assert record["chunks_count"] == 2
    assert record["multimodal_processed"] is True
    assert len(merge_calls) == 1
    assert merge_calls[0]["chunk_results"] == [("nodes", {})]
    assert merge_calls[0]["doc_id"] == "doc-1"
