"""Dict-shaped context sources must still yield surrounding text for captions.

MinerU-adjacent callers sometimes pass a page map (`{"p1": "...", "p2": "..."}`)
or a `{text: ...}` blob instead of a `content` key. If those fallbacks regress,
image/table captions are generated without nearby procedure text.
"""

from raganything.modalprocessors import ContextExtractor


def test_dict_source_uses_text_key_when_content_missing():
    extractor = ContextExtractor()
    context = extractor.extract_context(
        {"text": "from-text-key", "other": 1},
        {},
    )
    assert context == "from-text-key"


def test_dict_source_joins_string_values_when_no_content_or_text_key():
    extractor = ContextExtractor()
    context = extractor.extract_context(
        {"page_1": "intro paragraph", "page_2": "isolation steps", "page_idx": 3},
        {},
    )
    assert "intro paragraph" in context
    assert "isolation steps" in context
    # Non-string values are not coerced into the context window.
    assert "3" not in context
