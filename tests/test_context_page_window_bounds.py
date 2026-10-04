"""Page-context windows must stop at the neighboring page.

ContextExtractor feeds nearby text into image and table descriptions.
The page window is half-open: current_page ± window, excluding the next
page beyond that. A distant page (already covered elsewhere) can stay
out while an off-by-one still pulls in the adjacent procedure.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def _pages():
    return [
        {"type": "text", "text": "page zero", "page_idx": 0},
        {"type": "text", "text": "page one", "page_idx": 1},
        {"type": "text", "text": "page two", "page_idx": 2},
        {"type": "text", "text": "page three", "page_idx": 3},
        {"type": "text", "text": "page four", "page_idx": 4},
        {"type": "text", "text": "unpaged warning"},
    ]


def test_page_window_includes_neighbors_and_drops_the_next_page():
    extractor = ContextExtractor(
        ContextConfig(context_window=1, context_mode="page", include_headers=False)
    )

    context = extractor.extract_context(
        _pages(), {"page_idx": 2}, content_format="minerU"
    )

    assert context.split("\n") == [
        "[Page 1] page one",
        "page two",
        "[Page 3] page three",
    ]


def test_zero_page_window_keeps_only_the_current_page():
    extractor = ContextExtractor(
        ContextConfig(context_window=0, context_mode="page", include_headers=False)
    )

    context = extractor.extract_context(
        _pages(), {"page_idx": 2}, content_format="minerU"
    )

    assert context == "page two"
