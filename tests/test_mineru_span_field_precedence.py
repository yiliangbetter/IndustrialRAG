"""MinerU span precedence and list/caption plaintext edges.

``_mineru_span_text`` prefers a text span's string ``content``, then a
sibling ``text`` field, then a nested ``content`` tree. MinerU v2 list items
store their body in that span tree, and image captions drop empty entries.
String-only harvest does not cover these shapes. Mixing the branches up
drops or duplicates manual text before it is embedded.
"""

from raganything.processor import ProcessorMixin


class _Processor(ProcessorMixin):
    pass


def _proc():
    return _Processor()


def test_string_span_content_wins_over_a_sibling_text_field():
    proc = _proc()
    node = {"type": "text", "content": "  torque  ", "text": "ignored sibling"}
    assert proc._mineru_span_text(node) == "torque"

    # A blank string is still the span body, so the sibling must not leak in.
    blank = {"type": "text", "content": "   ", "text": "should not leak"}
    assert proc._mineru_span_text(blank) == ""


def test_non_string_content_uses_text_field_then_nested_tree():
    proc = _proc()
    with_text = {
        "type": "text",
        "content": ["nested", "ignored"],
        "text": "  explicit  ",
    }
    assert proc._mineru_span_text(with_text) == "explicit"

    nested = {
        "type": "text",
        "content": ["  left ", {"text": "right"}, ""],
    }
    assert proc._mineru_span_text(nested) == "left right"

    wrapper = {
        "type": "span",
        "text": "  visible  ",
        "content": {"type": "text", "content": "hidden"},
    }
    assert proc._mineru_span_text(wrapper) == "visible"

    # A numeric ``text`` field is not prose; the nested string content is.
    numeric_text = {"text": 12, "content": "kept"}
    assert proc._mineru_span_text(numeric_text) == "kept"


def test_list_item_bodies_can_be_span_trees():
    items = [
        {
            "prefix": "1.",
            "item_content": {"type": "text", "content": "  Check torque  "},
        },
        {
            "prefix": "2.",
            "item_content": ["Hold ", {"text": "the panel"}],
        },
        {"prefix": None, "item_content": {"type": "text", "content": "No marker"}},
        {"prefix": "3.", "item_content": {"type": "text", "content": "   "}},
        "skip",
        {"prefix": "4.", "item_content": None},
    ]
    assert _proc()._mineru_list_items_text(items) == (
        "1. Check torque\n2. Hold the panel\nNo marker"
    )


def test_plaintext_keeps_span_lists_and_drops_blank_captions_and_tables():
    items = [
        {
            "type": "paragraph",
            "content": {
                "paragraph_content": [
                    {"type": "text", "content": "Install"},
                    {"text": "the seal."},
                ]
            },
        },
        {"type": "paragraph", "content": ["not a v2 wrapper"]},
        {
            "type": "list",
            "content": {
                "list_items": [
                    {
                        "prefix": "-",
                        "item_content": {"type": "text", "content": "220V"},
                    }
                ]
            },
        },
        {"type": "table", "content": {"html": "  \n"}},
        {"type": "table", "content": {}},
        {"type": "image", "content": {"image_caption": "   "}},
        {
            "type": "image",
            "content": {"image_caption": ["Fig. 1", None, "", "panel"]},
        },
    ]
    assert _proc()._plaintext_from_mineru_blocks(items) == (
        "Install the seal.\n\n- 220V\n\nFig. 1 panel"
    )
