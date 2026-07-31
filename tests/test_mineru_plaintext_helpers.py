"""Unit tests for MinerU v2 plaintext recovery helpers on ProcessorMixin.

Distinct from open PRs #58/#60/#67 (end-to-end process_document / insert_content_list
paths). These lock the helper contracts for nested spans, list prefixes, and
wrapper flattening used by every MinerU v2 ingest path.
"""

from types import SimpleNamespace

from raganything.processor import ProcessorMixin


class DummyProcessor(ProcessorMixin):
    """Minimal stand-in; helpers under test are pure methods."""

    def __init__(self):
        self.config = SimpleNamespace(use_full_path=False, parser="mineru", parse_method="auto")
        self.logger = SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        )


class TestNormalizeNestedContentList:
    def setup_method(self):
        self.p = DummyProcessor()

    def test_flattens_top_level_list_wrappers(self):
        nested = [
            [{"type": "paragraph", "content": {"paragraph_content": "A"}}],
            {"type": "text", "text": "B"},
            [[{"type": "text", "text": "nested-too-deep"}]],  # inner list of list → skipped
        ]
        out = self.p._normalize_nested_content_list(nested)
        types = [x.get("type") for x in out]
        assert "paragraph" in types
        assert "text" in types
        # only dict children of top-level lists are kept
        assert all(isinstance(x, dict) for x in out)
        assert not any(x.get("text") == "nested-too-deep" for x in out)

    def test_empty_and_non_dict_ignored(self):
        assert self.p._normalize_nested_content_list([]) == []
        assert self.p._normalize_nested_content_list(["x", 1, None]) == []


class TestMineruSpanAndListText:
    def setup_method(self):
        self.p = DummyProcessor()

    def test_span_text_from_string_and_nested(self):
        assert self.p._mineru_span_text("  hello  ") == "hello"
        assert self.p._mineru_span_text(None) == ""
        nested = {
            "content": [
                {"type": "text", "content": "foo"},
                {"type": "text", "content": "bar"},
            ]
        }
        assert self.p._mineru_span_text(nested) == "foo bar"

    def test_list_items_with_prefix(self):
        items = [
            {"prefix": "1.", "item_content": "First"},
            {"prefix": "", "item_content": [{"type": "text", "content": "Second"}]},
            {"prefix": "-", "item_content": None},  # skipped — empty body
        ]
        text = self.p._mineru_list_items_text(items)
        assert "1. First" in text
        assert "Second" in text
        assert text.count("\n") == 1

    def test_list_items_non_list_returns_empty(self):
        assert self.p._mineru_list_items_text("not-a-list") == ""


class TestPlaintextFromMineruBlocks:
    def setup_method(self):
        self.p = DummyProcessor()

    def test_recovers_paragraph_title_list_table_image(self):
        blocks = [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "Overview"}]
                },
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [{"type": "text", "content": "Body text."}]
                },
            },
            {
                "type": "list",
                "content": {
                    "list_items": [
                        {"prefix": "-", "item_content": "Item A"},
                    ]
                },
            },
            {"type": "table", "content": {"html": "<table><tr><td>1</td></tr></table>"}},
            {
                "type": "image",
                "content": {"image_caption": ["Fig. 1", "Detail"]},
            },
            {"type": "text", "text": "Legacy flat text"},
            {"type": "unknown", "content": {"x": 1}},  # ignored
        ]
        text = self.p._plaintext_from_mineru_blocks(blocks)
        assert "Overview" in text
        assert "Body text." in text
        assert "- Item A" in text
        assert "<table>" in text
        assert "Fig. 1" in text and "Detail" in text
        assert "Legacy flat text" in text
        # blocks joined with blank lines
        assert "\n\n" in text

    def test_skips_empty_and_non_dict(self):
        blocks = [
            None,
            "x",
            {"type": "paragraph", "content": {"paragraph_content": "   "}},
            {"type": "text", "text": "   "},
        ]
        # filter None/"x" via isinstance; whitespace-only omitted
        assert self.p._plaintext_from_mineru_blocks(blocks) == ""
