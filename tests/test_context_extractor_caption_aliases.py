"""ContextExtractor caption flags and MinerU 1.x caption field names.

Image and table descriptions are built from this surrounding text. Dropping
the legacy ``img_caption`` key, or ignoring ``include_captions``, changes
what the caption model sees for the same manual.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def test_legacy_img_caption_is_used_when_image_caption_is_absent():
    extractor = ContextExtractor(
        ContextConfig(
            context_window=0,
            context_mode="page",
            filter_content_types=["text", "image"],
        )
    )
    context = extractor.extract_context(
        [{"type": "image", "img_caption": ["Nameplate"], "page_idx": 0}],
        {"page_idx": 0},
        content_format="minerU",
    )
    assert context == "[Image: Nameplate]"


def test_canonical_image_caption_wins_over_legacy_img_caption():
    extractor = ContextExtractor(
        ContextConfig(
            context_window=0,
            context_mode="page",
            filter_content_types=["image"],
        )
    )
    context = extractor.extract_context(
        [
            {
                "type": "image",
                "image_caption": ["Canonical"],
                "img_caption": ["Legacy"],
                "page_idx": 2,
            }
        ],
        {"page_idx": 2},
    )
    assert context == "[Image: Canonical]"
    assert "Legacy" not in context


def test_include_captions_false_keeps_text_and_drops_captions():
    extractor = ContextExtractor(
        ContextConfig(
            context_window=1,
            context_mode="page",
            include_captions=False,
            filter_content_types=["text", "image", "table"],
        )
    )
    context = extractor.extract_context(
        [
            {"type": "text", "text": "Isolation steps", "page_idx": 1},
            {"type": "image", "image_caption": ["Pump diagram"], "page_idx": 1},
            {"type": "table", "table_caption": ["Torque table"], "page_idx": 1},
            {"type": "text", "text": "   ", "page_idx": 1},
        ],
        {"page_idx": 1},
    )
    assert context == "Isolation steps"


def test_missing_page_idx_stays_on_page_zero():
    extractor = ContextExtractor(ContextConfig(context_window=0, context_mode="page"))
    pages = [
        {"type": "text", "text": "Unlabeled block"},
        {"type": "text", "text": "Later page", "page_idx": 4},
    ]

    on_first = extractor.extract_context(pages, {"page_idx": 0})
    on_later = extractor.extract_context(pages, {"page_idx": 4})

    assert on_first == "Unlabeled block"
    assert on_later == "Later page"
