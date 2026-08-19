"""Isolated flatten of MinerU *_content_list_v2.json wrappers.

insert_content_list hashes and separates blocks after this step. Nested
list wrappers or non-dict junk must not drop real blocks or invent empty
doc_ids.
"""

from raganything.processor import ProcessorMixin


class DummyProcessor(ProcessorMixin):
    pass


def test_flattens_top_level_list_wrappers_and_keeps_dicts():
    processor = DummyProcessor()
    flat = processor._normalize_nested_content_list(
        [
            [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
            {"type": "text", "text": "C"},
        ]
    )
    assert [item["text"] for item in flat] == ["A", "B", "C"]


def test_drops_non_dict_items_inside_nested_and_top_level():
    processor = DummyProcessor()
    flat = processor._normalize_nested_content_list(
        [
            [{"type": "text", "text": "keep"}, "skip-string", None, 12],
            "also-skip",
            None,
            {"type": "image", "img_path": "/abs/a.png"},
            [],
        ]
    )
    assert flat == [
        {"type": "text", "text": "keep"},
        {"type": "image", "img_path": "/abs/a.png"},
    ]


def test_empty_input_returns_empty_list():
    processor = DummyProcessor()
    assert processor._normalize_nested_content_list([]) == []
