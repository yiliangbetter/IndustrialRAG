"""Regression tests for multimodal belongs_to relation wiring.

Extracted entities must attach to the modal parent (figure/table/equation).
A missed or self-looping edge breaks KG navigation for every multimodal
chunk produced by the type-aware batch path.
"""

import pytest
from lightrag.utils import compute_mdhash_id

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


def _processor():
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type("Config", (), {"use_full_path": False})()
    dummy.lightrag = type(
        "FakeLightRAG",
        (),
        {"tokenizer": type("Tok", (), {"encode": staticmethod(lambda text: text.split())})()},
    )()
    return dummy


def _table_item(description="Pump curve data"):
    return {
        "description": description,
        "content_type": "table",
        "original_item": {
            "img_path": "",
            "table_caption": ["Pump curve"],
            "table_body": "rpm|flow\n1800|12",
            "table_footnote": [],
        },
        "entity_info": {
            "entity_name": "Pump curve (table)",
            "entity_type": "table",
            "summary": description,
        },
        "file_path": "spec.pdf",
    }


@pytest.mark.asyncio
async def test_belongs_to_links_extracted_entities_to_modal_parent():
    dummy = _processor()
    item = _table_item()
    formatted = dummy._apply_chunk_template(
        item["content_type"], item["original_item"], item["description"]
    )
    chunk_id = compute_mdhash_id(formatted, prefix="chunk-")
    modal = item["entity_info"]["entity_name"]

    chunk_results = [
        (
            {
                modal: [{"source_id": chunk_id}],
                "Impeller": [{"source_id": chunk_id}],
                "RPM": [{"source_id": chunk_id}],
            },
            {},
        )
    ]

    enhanced = await dummy._batch_add_belongs_to_relations_type_aware(
        chunk_results, [item]
    )
    _, maybe_edges = enhanced[0]

    assert (modal, modal) not in maybe_edges
    impeller = maybe_edges[("Impeller", modal)][0]
    rpm = maybe_edges[("RPM", modal)][0]
    assert impeller["src_id"] == "Impeller"
    assert impeller["tgt_id"] == modal
    assert impeller["source_id"] == chunk_id
    assert impeller["file_path"] == "spec.pdf"
    assert impeller["weight"] == 10.0
    assert "belongs_to" in impeller["keywords"]
    assert rpm["src_id"] == "RPM"


@pytest.mark.asyncio
async def test_belongs_to_appends_without_clobbering_existing_edge():
    dummy = _processor()
    item = _table_item()
    formatted = dummy._apply_chunk_template(
        item["content_type"], item["original_item"], item["description"]
    )
    chunk_id = compute_mdhash_id(formatted, prefix="chunk-")
    modal = item["entity_info"]["entity_name"]
    existing = {"description": "pre-existing edge"}
    chunk_results = [
        (
            {
                modal: [{"source_id": chunk_id}],
                "Impeller": [{"source_id": chunk_id}],
            },
            {("Impeller", modal): [existing]},
        )
    ]

    enhanced = await dummy._batch_add_belongs_to_relations_type_aware(
        chunk_results, [item]
    )
    _, maybe_edges = enhanced[0]
    edges = maybe_edges[("Impeller", modal)]
    assert edges[0] is existing
    assert edges[1]["description"].startswith("Entity Impeller belongs to")


@pytest.mark.asyncio
async def test_belongs_to_skips_unmapped_or_empty_source_chunks():
    dummy = _processor()
    item = _table_item()
    chunk_results = [
        ({"Orphan": [{"source_id": "chunk-not-from-template"}]}, {}),
        ({"Empty": []}, {}),
    ]

    enhanced = await dummy._batch_add_belongs_to_relations_type_aware(
        chunk_results, [item]
    )
    assert enhanced[0][1] == {}
    assert enhanced[1][1] == {}
