"""Regression tests for multimodal belongs_to relation wiring."""

from lightrag.utils import compute_mdhash_id

from raganything.processor import ProcessorMixin
from raganything.prompt import PROMPTS


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    return processor


def _table_chunk_id(description: str, original_item: dict) -> str:
    formatted = PROMPTS["table_chunk"].format(
        table_img_path=original_item.get("img_path", ""),
        table_caption=", ".join(original_item.get("table_caption", [])) or "None",
        table_body=original_item.get("table_body", ""),
        table_footnote=", ".join(original_item.get("table_footnote", [])) or "None",
        enhanced_caption=description,
    )
    return compute_mdhash_id(formatted, prefix="chunk-")


def test_batch_add_belongs_to_links_extracted_entities_to_modal_entity():
    processor = _make_processor()
    original_item = {
        "type": "table",
        "table_body": "| a | b |",
        "table_caption": ["Spec table"],
        "table_footnote": [],
        "img_path": "",
    }
    description = "A specification table"
    modal_name = "Spec table (table)"
    chunk_id = _table_chunk_id(description, original_item)

    multimodal_data_list = [
        {
            "description": description,
            "content_type": "table",
            "original_item": original_item,
            "entity_info": {"entity_name": modal_name, "entity_type": "table"},
            "file_path": "manual.pdf",
        }
    ]
    chunk_results = [
        (
            {
                "Pressure": [{"source_id": chunk_id}],
                modal_name: [{"source_id": chunk_id}],
            },
            {},
        )
    ]

    enhanced = processor._batch_add_belongs_to_relations_type_aware(
        chunk_results, multimodal_data_list
    )

    assert len(enhanced) == 1
    maybe_nodes, maybe_edges = enhanced[0]
    assert "Pressure" in maybe_nodes
    edge_key = ("Pressure", modal_name)
    assert edge_key in maybe_edges
    relation = maybe_edges[edge_key][0]
    assert relation["src_id"] == "Pressure"
    assert relation["tgt_id"] == modal_name
    assert relation["source_id"] == chunk_id
    assert relation["weight"] == 10.0
    assert "belongs_to" in relation["keywords"]
    # Avoid self-relation for the modal entity itself.
    assert (modal_name, modal_name) not in maybe_edges


def test_batch_add_belongs_to_skips_chunks_without_modal_mapping():
    processor = _make_processor()
    chunk_results = [
        (
            {"Orphan": [{"source_id": "chunk-unrelated"}]},
            {("Orphan", "Other"): [{"src_id": "Orphan", "tgt_id": "Other"}]},
        )
    ]

    enhanced = processor._batch_add_belongs_to_relations_type_aware(
        chunk_results,
        [
            {
                "description": "desc",
                "content_type": "table",
                "original_item": {"table_body": "x", "table_caption": []},
                "entity_info": {"entity_name": "Table (table)"},
                "file_path": "doc.pdf",
            }
        ],
    )

    maybe_nodes, maybe_edges = enhanced[0]
    assert maybe_nodes == {"Orphan": [{"source_id": "chunk-unrelated"}]}
    # Existing edges preserved; no belongs_to edges added for unmapped chunk.
    assert list(maybe_edges.keys()) == [("Orphan", "Other")]
