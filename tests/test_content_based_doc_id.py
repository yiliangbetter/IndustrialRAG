"""Regression tests for content-based document ID generation.

Doc IDs drive storage keys and dedup. Unstable hashing or empty signatures from
MinerU v2 nested wrappers can silently collide or overwrite unrelated documents.
"""

import pytest

pytest.importorskip("lightrag")

import raganything.processor as processor_module


class DummyProcessor(processor_module.ProcessorMixin):
    pass


def test_content_based_doc_id_is_deterministic_for_same_blocks():
    processor = DummyProcessor()
    content = [
        {"type": "text", "text": "Wear eye protection."},
        {"type": "table", "table_body": "<table><tr><td>OK</td></tr></table>"},
        {"type": "image", "img_path": "/data/figures/shutoff.png"},
    ]

    first = processor._generate_content_based_doc_id(content)
    second = processor._generate_content_based_doc_id(list(content))

    assert first == second
    assert first.startswith("doc-")


def test_content_based_doc_id_changes_when_text_changes():
    processor = DummyProcessor()
    base = [{"type": "text", "text": "Procedure A"}]
    other = [{"type": "text", "text": "Procedure B"}]

    assert processor._generate_content_based_doc_id(
        base
    ) != processor._generate_content_based_doc_id(other)


def test_content_based_doc_id_skips_nested_list_wrappers_without_dicts():
    """Un-normalized MinerU [[...]] wrappers yield no hashable dict items.

    Callers must normalize before hashing; this locks the current contract so a
    silent change to hashing nested lists does not go unnoticed.
    """
    processor = DummyProcessor()
    nested_only = [[{"type": "text", "text": "hidden inside wrapper"}]]
    empty = []

    nested_id = processor._generate_content_based_doc_id(nested_only)
    empty_id = processor._generate_content_based_doc_id(empty)

    assert nested_id == empty_id
    assert nested_id.startswith("doc-")


def test_normalize_then_hash_recovers_nested_mineru_v2_identity():
    processor = DummyProcessor()
    nested = [
        [
            {"type": "text", "text": "Safety Protocol"},
            {"type": "paragraph", "content": {"paragraph_content": []}},
        ]
    ]
    flat = processor._normalize_nested_content_list(nested)

    assert processor._generate_content_based_doc_id(flat) != processor._generate_content_based_doc_id(
        []
    )
    assert processor._generate_content_based_doc_id(flat) == processor._generate_content_based_doc_id(
        flat
    )
