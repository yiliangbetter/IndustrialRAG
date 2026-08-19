"""ContextExtractor page/chunk windows, captions, and token truncation.

Multimodal processors rely on this helper to attach surrounding text. A
regression here silently changes image/table descriptions or drops nearby
headers, which is a large blast radius for retrieval quality.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def _pages():
    return [
        {"type": "text", "text": "Intro", "page_idx": 0, "text_level": 1},
        {"type": "text", "text": "Current page body", "page_idx": 1},
        {"type": "image", "image_caption": ["Pump diagram"], "page_idx": 1},
        {"type": "table", "table_caption": ["Torque table"], "page_idx": 2},
        {"type": "text", "text": "Far away", "page_idx": 5},
    ]


class TestPageContext:
    def test_window_includes_adjacent_pages_with_markers(self):
        extractor = ContextExtractor(
            ContextConfig(context_window=1, context_mode="page")
        )
        context = extractor.extract_context(
            _pages(),
            {"page_idx": 1},
            content_format="minerU",
        )

        assert "Current page body" in context
        assert "[Page 0] # Intro" in context
        assert "[Page 2]" not in context  # table captions are not type=text by default
        assert "Far away" not in context

    def test_headers_can_be_disabled(self):
        extractor = ContextExtractor(
            ContextConfig(context_window=1, context_mode="page", include_headers=False)
        )
        context = extractor.extract_context(_pages(), {"page_idx": 1})
        assert "# Intro" not in context
        assert "Intro" in context

    def test_captions_included_when_filter_allows(self):
        extractor = ContextExtractor(
            ContextConfig(
                context_window=1,
                context_mode="page",
                filter_content_types=["text", "image", "table"],
            )
        )
        context = extractor.extract_context(_pages(), {"page_idx": 1})
        assert "[Image: Pump diagram]" in context
        assert "[Table: Torque table]" in context
        assert "[Page 2] [Table: Torque table]" in context

    def test_unknown_mode_falls_back_to_page(self):
        extractor = ContextExtractor(
            ContextConfig(context_mode="token", context_window=0)
        )
        context = extractor.extract_context(_pages(), {"page_idx": 1})
        assert "Current page body" in context
        assert "Far away" not in context


class TestChunkContext:
    def test_window_excludes_current_index(self):
        chunks = [
            {"type": "text", "text": "before"},
            {"type": "text", "text": "CURRENT"},
            {"type": "text", "text": "after"},
            {"type": "text", "text": "too far"},
        ]
        extractor = ContextExtractor(
            ContextConfig(context_mode="chunk", context_window=1)
        )
        context = extractor.extract_context(chunks, {"index": 1})
        assert "before" in context
        assert "after" in context
        assert "CURRENT" not in context
        assert "too far" not in context


class TestFormatDispatch:
    def test_text_chunks_format(self):
        extractor = ContextExtractor(ContextConfig(context_window=1))
        context = extractor.extract_context(
            ["alpha", "beta", "gamma"],
            {"index": 1},
            content_format="text_chunks",
        )
        assert "alpha" in context
        assert "gamma" in context
        assert "beta" not in context

    def test_plain_text_is_truncated(self):
        extractor = ContextExtractor(ContextConfig(max_context_tokens=8))
        context = extractor.extract_context(
            "abcdefghijklmnop",
            {},
            content_format="text",
        )
        assert context == "abcdefgh..."

    def test_dict_source_prefers_content_key(self):
        extractor = ContextExtractor()
        context = extractor.extract_context(
            {"content": "from-content", "text": "ignored"}, {}
        )
        assert context == "from-content"

    def test_unsupported_type_returns_empty(self):
        extractor = ContextExtractor()
        assert extractor.extract_context(12345, {}) == ""

    def test_empty_source_and_zero_window_returns_empty(self):
        extractor = ContextExtractor(ContextConfig(context_window=0))
        assert extractor.extract_context([], {"page_idx": 0}) == ""

    def test_extraction_errors_fail_closed(self):
        extractor = ContextExtractor()

        class Boom(list):
            def __iter__(self):
                raise RuntimeError("corrupt content list")

        assert extractor.extract_context(Boom(), {"page_idx": 0}) == ""


class FakeTokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


class TestTruncation:
    def test_character_fallback_uses_sentence_boundary(self):
        extractor = ContextExtractor(ContextConfig(max_context_tokens=22))
        # Period must sit past 80% of the truncated window to be used.
        text = "Keep this sentence. trailing words"
        assert extractor._truncate_context(text) == "Keep this sentence."

    def test_character_fallback_uses_newline_boundary(self):
        extractor = ContextExtractor(ContextConfig(max_context_tokens=22))
        text = "first line is here\nextra trailing words"
        assert extractor._truncate_context(text) == "first line is here"

    def test_tokenizer_truncates_and_adds_ellipsis(self):
        extractor = ContextExtractor(
            ContextConfig(max_context_tokens=6), tokenizer=FakeTokenizer()
        )
        assert extractor._truncate_context("abcdefghij") == "abcdef..."

    def test_tokenizer_keeps_short_context(self):
        extractor = ContextExtractor(
            ContextConfig(max_context_tokens=50), tokenizer=FakeTokenizer()
        )
        assert extractor._truncate_context("keep me") == "keep me"
