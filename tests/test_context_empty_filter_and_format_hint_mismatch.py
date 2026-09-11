"""ContextExtractor empty filters and format-hint mismatches.

Empty filter_content_types is not treated as 'all types' (None still defaults to
text). A minerU format hint on a dict source must still extract nearby procedure
text via auto-detect. Distinct from #122 (page/chunk windows, captions, explicit
text_chunks/text/dict dispatch) and #155 (dict key preference without a hint).
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def _pages():
    return [
        {"type": "text", "text": "Intro", "page_idx": 0},
        {"type": "text", "text": "Current page body", "page_idx": 1},
        {"type": "image", "image_caption": ["Pump diagram"], "page_idx": 1},
    ]


class TestEmptyFilterContentTypes:
    def test_none_filter_defaults_to_text_only(self):
        cfg = ContextConfig()
        assert cfg.filter_content_types == ["text"]

        extractor = ContextExtractor(
            ContextConfig(context_window=1, context_mode="page")
        )
        context = extractor.extract_context(
            _pages(), {"page_idx": 1}, content_format="minerU"
        )
        assert "Current page body" in context
        assert "Pump diagram" not in context

    def test_empty_filter_list_drops_all_context(self):
        extractor = ContextExtractor(
            ContextConfig(
                context_window=2,
                context_mode="page",
                filter_content_types=[],
            )
        )
        context = extractor.extract_context(
            _pages(), {"page_idx": 1}, content_format="minerU"
        )
        assert context == ""


class TestFormatHintMismatch:
    def test_mineru_hint_with_dict_source_still_extracts_content(self):
        extractor = ContextExtractor()
        context = extractor.extract_context(
            {"content": "nearby-procedure", "text": "ignored"},
            {"page_idx": 0},
            content_format="minerU",
        )
        assert context == "nearby-procedure"

    def test_text_hint_with_list_source_uses_content_list_path(self):
        extractor = ContextExtractor(
            ContextConfig(context_window=1, context_mode="page")
        )
        context = extractor.extract_context(
            _pages(), {"page_idx": 1}, content_format="text"
        )
        assert "Current page body" in context
        assert "[Page 0] Intro" in context
