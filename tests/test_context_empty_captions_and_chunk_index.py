"""Empty captions and chunk-index defaults must not attach the wrong context.

Page context is already covered for non-empty captions and explicit chunk
indexes. An explicit empty canonical caption has to suppress a leftover
legacy alias, and a missing or out-of-range chunk index has to stay inside
the window instead of pulling in the current item or the rest of the document.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def test_empty_canonical_captions_do_not_revive_legacy_aliases():
    extractor = ContextExtractor(
        ContextConfig(
            context_window=1,
            context_mode="page",
            include_headers=False,
            filter_content_types=["text", "image", "table"],
        )
    )
    context = extractor.extract_context(
        [
            {"type": "text", "text": "   ", "page_idx": 1},
            {"type": "text", "text": "Isolation steps", "page_idx": 1},
            {
                "type": "image",
                "image_caption": [],
                "img_caption": ["stale nameplate"],
                "page_idx": 1,
            },
            {"type": "table", "table_caption": [], "page_idx": 1},
        ],
        {"page_idx": 1},
    )

    assert context == "Isolation steps"


def test_omitted_chunk_index_excludes_the_first_item_and_blank_neighbors():
    chunks = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "   "},
        {"type": "text", "text": "second"},
        {"type": "text", "text": "third"},
    ]
    extractor = ContextExtractor(
        ContextConfig(context_mode="chunk", context_window=2, include_headers=False)
    )

    assert extractor.extract_context(chunks, {}) == "second"


def test_chunk_window_zero_and_out_of_range_index_yield_no_context():
    chunks = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    zero_window = ContextExtractor(
        ContextConfig(context_mode="chunk", context_window=0)
    )
    wide = ContextExtractor(ContextConfig(context_mode="chunk", context_window=2))

    assert zero_window.extract_context(chunks, {"index": 0}) == ""
    assert wide.extract_context(chunks, {"index": 50}) == ""


def test_text_chunk_window_skips_blanks_and_stringifies_neighbors():
    extractor = ContextExtractor(ContextConfig(context_window=2))

    context = extractor.extract_context(
        ["current", "   ", 12, "too-far"],
        {},
        content_format="text_chunks",
    )
    zero_window = ContextExtractor(ContextConfig(context_window=0))

    assert context == "12"
    assert (
        zero_window.extract_context(
            ["only"],
            {"index": 0},
            content_format="text_chunks",
        )
        == ""
    )
