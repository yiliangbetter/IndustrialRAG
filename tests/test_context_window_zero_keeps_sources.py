"""context_window=0 disables neighbor pages, not the item's own source.

The early return in ContextExtractor.extract_context only skips work when the
source is empty *and* the window is zero. A plain string or a dict ``content``
field is the caption context itself, so a zero window must still return it.
An unrecognized context_mode stays on the page window instead of raising or
switching to chunk indexing.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def test_zero_window_keeps_text_and_dict_sources():
    extractor = ContextExtractor(ContextConfig(context_window=0))

    assert extractor.extract_context("isolation steps", {}) == "isolation steps"
    assert (
        extractor.extract_context({"content": "torque spec 40 Nm"}, {})
        == "torque spec 40 Nm"
    )


def test_zero_window_empty_sources_return_empty_string():
    extractor = ContextExtractor(ContextConfig(context_window=0))

    assert extractor.extract_context(None, {}) == ""
    assert extractor.extract_context("", {}) == ""
    assert extractor.extract_context([], {}) == ""
    assert extractor.extract_context({}, {}) == ""


def test_unknown_context_mode_uses_page_window():
    extractor = ContextExtractor(
        ContextConfig(
            context_mode="bogus",
            context_window=1,
            include_headers=False,
            filter_content_types=["text"],
        )
    )
    pages = [
        {"type": "text", "text": "intro", "page_idx": 0},
        {"type": "text", "text": "body", "page_idx": 1},
        {"type": "text", "text": "later", "page_idx": 2},
    ]

    context = extractor.extract_context(pages, {"page_idx": 0})

    assert context.split("\n") == ["intro", "[Page 1] body"]
