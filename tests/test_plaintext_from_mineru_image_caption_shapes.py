"""MinerU v2 image captions must stay searchable regardless of JSON shape.

`_plaintext_from_mineru_blocks` is the skip-multimodal harvest used by
`insert_content_list`. List captions are already exercised via embedding-only
ingest. A string (or mixed/empty) caption must not drop figure text or crash.
"""

from raganything.processor import ProcessorMixin


def _harvest(items):
    processor = ProcessorMixin.__new__(ProcessorMixin)
    return processor._plaintext_from_mineru_blocks(items)


def test_list_captions_are_joined():
    text = _harvest(
        [{"type": "image", "content": {"image_caption": ["Pump diagram", "Fig. 2"]}}]
    )
    assert text == "Pump diagram Fig. 2"


def test_string_caption_is_used_as_is():
    text = _harvest(
        [{"type": "image", "content": {"image_caption": "Cooling loop overview"}}]
    )
    assert text == "Cooling loop overview"


def test_empty_and_missing_captions_yield_no_plaintext():
    text = _harvest(
        [
            {"type": "image", "content": {"image_caption": []}},
            {"type": "image", "content": {}},
            {"type": "image", "content": {"image_caption": ""}},
            {"type": "image", "content": {"image_caption": None}},
        ]
    )
    assert text == ""


def test_falsy_list_entries_are_dropped():
    text = _harvest(
        [{"type": "image", "content": {"image_caption": ["", None, "Keep"]}}]
    )
    assert text == "Keep"
