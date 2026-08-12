"""Unit tests for MinerU v2 nested content-list flattening.

Open #68 covers normalize-then-hash for doc IDs. These cases lock filtering and
shape contracts used by ``insert_content_list`` before separation/hashing.
"""

from __future__ import annotations

from raganything.processor import ProcessorMixin


class DummyProcessor(ProcessorMixin):
    pass


def test_normalize_flattens_single_nested_wrapper():
    processor = DummyProcessor()
    nested = [
        [
            {"type": "text", "text": "alpha"},
            {"type": "image", "img_path": "/tmp/a.png"},
        ]
    ]

    assert processor._normalize_nested_content_list(nested) == [
        {"type": "text", "text": "alpha"},
        {"type": "image", "img_path": "/tmp/a.png"},
    ]


def test_normalize_keeps_top_level_dicts_and_flattens_mixed_wrappers():
    processor = DummyProcessor()
    mixed = [
        {"type": "text", "text": "outer"},
        [{"type": "table", "table_body": "<table/>"}],
        {"type": "equation", "text": "E=mc^2"},
    ]

    assert processor._normalize_nested_content_list(mixed) == [
        {"type": "text", "text": "outer"},
        {"type": "table", "table_body": "<table/>"},
        {"type": "equation", "text": "E=mc^2"},
    ]


def test_normalize_drops_non_dict_nested_and_top_level_junk():
    """Non-dict noise must not become blocks (would break separate_content)."""
    processor = DummyProcessor()
    noisy = [
        "ignore-string",
        42,
        None,
        [{"type": "text", "text": "keep"}, "skip", {"type": "image", "img_path": "x"}],
        [{"not": "a problem"}, ["still", "ignored"]],
    ]

    assert processor._normalize_nested_content_list(noisy) == [
        {"type": "text", "text": "keep"},
        {"type": "image", "img_path": "x"},
        {"not": "a problem"},
    ]


def test_normalize_empty_inputs():
    processor = DummyProcessor()
    assert processor._normalize_nested_content_list([]) == []
    assert processor._normalize_nested_content_list([[], []]) == []
