"""Unit tests for MinerU v2 plaintext harvest helpers on ProcessorMixin.

These helpers recover searchable text when content-list blocks are not
``type=text`` (paragraph/title/list/table/image). A regression here silently
drops industrial-manual content from the knowledge graph.
"""

from raganything.processor import ProcessorMixin


class DummyProcessor(ProcessorMixin):
    pass


def _proc():
    return DummyProcessor()


class TestMineruSpanText:
    def test_none_and_non_container_return_empty(self):
        proc = _proc()
        assert proc._mineru_span_text(None) == ""
        assert proc._mineru_span_text(42) == ""

    def test_string_is_stripped(self):
        assert _proc()._mineru_span_text("  hello  ") == "hello"

    def test_nested_list_joins_non_empty_parts(self):
        node = ["  alpha  ", "", ["beta"], None]
        assert _proc()._mineru_span_text(node) == "alpha beta"

    def test_text_span_uses_content(self):
        node = {"type": "text", "content": "  span body  "}
        assert _proc()._mineru_span_text(node) == "span body"

    def test_plain_text_field(self):
        assert _proc()._mineru_span_text({"text": " title "}) == "title"

    def test_nested_content_tree(self):
        node = {
            "content": [
                {"type": "text", "content": "left"},
                {"content": {"text": "right"}},
            ]
        }
        assert _proc()._mineru_span_text(node) == "left right"


class TestMineruListItemsText:
    def test_non_list_returns_empty(self):
        assert _proc()._mineru_list_items_text(None) == ""
        assert _proc()._mineru_list_items_text({"item": "x"}) == ""

    def test_prefix_and_item_content(self):
        items = [
            {"prefix": "1.", "item_content": "First"},
            {"prefix": "", "item_content": "Second"},
            "skip-me",
            {"prefix": "-", "item_content": "   "},
        ]
        text = _proc()._mineru_list_items_text(items)
        assert text == "1. First\nSecond"


class TestPlaintextFromMineruBlocks:
    def test_text_blocks_and_non_dicts_are_handled(self):
        items = [
            "ignore",
            {"type": "text", "text": "  hello  "},
            {"type": "text", "text": "   "},
            {"type": "paragraph"},
        ]
        assert _proc()._plaintext_from_mineru_blocks(items) == "hello"

    def test_paragraph_title_list_table_image(self):
        items = [
            {
                "type": "title",
                "content": {"title_content": {"type": "text", "content": "Spec"}},
            },
            {
                "type": "paragraph",
                "content": {"paragraph_content": "Install the unit."},
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {"prefix": "-", "item_content": "Check voltage"},
                    ]
                },
            },
            {
                "type": "table",
                "content": {"html": "  <table><td>220V</td></table>  "},
            },
            {
                "type": "image",
                "content": {"image_caption": ["Fig. 1", "wiring"]},
            },
        ]
        text = _proc()._plaintext_from_mineru_blocks(items)
        assert text == (
            "Spec\n\n"
            "Install the unit.\n\n"
            "- Check voltage\n\n"
            "<table><td>220V</td></table>\n\n"
            "Fig. 1 wiring"
        )

    def test_image_caption_string_and_unknown_type_ignored(self):
        items = [
            {"type": "image", "content": {"image_caption": "single caption"}},
            {"type": "code", "content": {"code_body": "print(1)"}},
        ]
        assert _proc()._plaintext_from_mineru_blocks(items) == "single caption"
