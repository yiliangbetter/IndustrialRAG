"""Regression tests for multimodal → LightRAG chunk metadata wiring.

Template wording is covered separately. These tests lock the retrieval
metadata (`is_multimodal`, entity name, page, order, file_path, tokens,
stable chunk ids) that a regression would silently corrupt.
"""

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


class FakeTokenizer:
    def encode(self, text):
        return list(text)


def _processor(use_full_path=False, template_return=None):
    proc = ProcessorMixin()
    proc.logger = FakeLogger()
    proc.config = type("Config", (), {"use_full_path": use_full_path})()
    proc.lightrag = type("FakeLightRAG", (), {"tokenizer": FakeTokenizer()})()
    if template_return is not None:
        proc._apply_chunk_template = lambda *args, **kwargs: template_return
    return proc


def _item(
    description="desc",
    entity_name="Pump (image)",
    content_type="image",
    chunk_order_index=2,
    page_idx=4,
    original_item=None,
):
    return {
        "description": description,
        "entity_info": {
            "entity_name": entity_name,
            "entity_type": content_type,
            "summary": description,
        },
        "chunk_order_index": chunk_order_index,
        "content_type": content_type,
        "original_item": original_item or {"img_path": "/docs/fig.png"},
        "item_info": {"page_idx": page_idx, "index": 0},
    }


class TestConvertToLightragChunksTypeAware:
    def test_single_item_sets_multimodal_metadata(self):
        proc = _processor(template_return="FORMATTED-CHUNK")
        chunks = proc._convert_to_lightrag_chunks_type_aware(
            [_item()],
            file_path="/data/manuals/pump.pdf",
            doc_id="doc-pump",
        )
        assert len(chunks) == 1
        chunk = next(iter(chunks.values()))
        assert chunk["content"] == "FORMATTED-CHUNK"
        assert chunk["tokens"] == len("FORMATTED-CHUNK")
        assert chunk["full_doc_id"] == "doc-pump"
        assert chunk["chunk_order_index"] == 2
        assert chunk["file_path"] == "pump.pdf"
        assert chunk["is_multimodal"] is True
        assert chunk["modal_entity_name"] == "Pump (image)"
        assert chunk["original_type"] == "image"
        assert chunk["page_idx"] == 4
        assert chunk["llm_cache_list"] == []

    def test_use_full_path_preserves_file_reference(self):
        proc = _processor(use_full_path=True, template_return="CHUNK")
        chunks = proc._convert_to_lightrag_chunks_type_aware(
            [_item()],
            file_path="/data/manuals/pump.pdf",
            doc_id="doc-pump",
        )
        chunk = next(iter(chunks.values()))
        assert chunk["file_path"] == "/data/manuals/pump.pdf"

    def test_missing_page_idx_defaults_to_zero(self):
        proc = _processor(template_return="CHUNK")
        item = _item()
        item["item_info"] = {"index": 3}
        chunks = proc._convert_to_lightrag_chunks_type_aware(
            [item], file_path="pump.pdf", doc_id="doc-1"
        )
        chunk = next(iter(chunks.values()))
        assert chunk["page_idx"] == 0

    def test_two_items_get_distinct_ids_and_preserve_order_indexes(self):
        proc = _processor()
        proc._apply_chunk_template = (
            lambda content_type, original_item, description: description
        )
        chunks = proc._convert_to_lightrag_chunks_type_aware(
            [
                _item(description="first-body", entity_name="A", chunk_order_index=0),
                _item(description="second-body", entity_name="B", chunk_order_index=5),
            ],
            file_path="pump.pdf",
            doc_id="doc-1",
        )
        assert len(chunks) == 2
        orders = sorted(chunk["chunk_order_index"] for chunk in chunks.values())
        assert orders == [0, 5]
        by_order = {chunk["chunk_order_index"]: chunk for chunk in chunks.values()}
        assert by_order[0]["modal_entity_name"] == "A"
        assert by_order[5]["modal_entity_name"] == "B"

    def test_identical_formatted_content_is_stable_chunk_id(self):
        proc = _processor(template_return="SAME-CONTENT")
        first = proc._convert_to_lightrag_chunks_type_aware(
            [_item()], file_path="a.pdf", doc_id="doc-a"
        )
        second = proc._convert_to_lightrag_chunks_type_aware(
            [_item()], file_path="b.pdf", doc_id="doc-b"
        )
        assert set(first) == set(second)
        assert next(iter(first)).startswith("chunk-")
