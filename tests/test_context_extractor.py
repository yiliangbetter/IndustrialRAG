"""Regression tests for ContextExtractor page/chunk windows and truncation.

ContextExtractor feeds surrounding text into every multimodal caption/entity
call. Windowing, caption aliases, filters, and token truncation bugs silently
degrade KG quality across the whole ingest path. Not covered by open PRs.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def _mineru_pages():
    return [
        {"type": "text", "text": "Intro page 0", "page_idx": 0, "text_level": 0},
        {
            "type": "text",
            "text": "Section Title",
            "page_idx": 1,
            "text_level": 1,
        },
        {
            "type": "image",
            "page_idx": 1,
            "img_caption": ["Legacy caption"],
            "image_caption": ["Canonical caption"],
        },
        {
            "type": "table",
            "page_idx": 2,
            "table_caption": ["Spec table"],
        },
        {"type": "text", "text": "Closing page 2", "page_idx": 2},
    ]


class TestContextExtractorPageMode:
    def test_page_window_includes_neighbors_and_excludes_far_pages(self):
        extractor = ContextExtractor(
            ContextConfig(context_window=1, context_mode="page")
        )
        ctx = extractor.extract_context(
            _mineru_pages(),
            {"page_idx": 1, "index": 1},
            content_format="minerU",
        )
        assert "Intro page 0" in ctx
        assert "Section Title" in ctx
        assert "[Page 0]" in ctx
        # page 2 is within window=1 of page 1
        assert "Closing page 2" in ctx or "Spec table" in ctx

    def test_headers_prefixed_when_include_headers(self):
        extractor = ContextExtractor(
            ContextConfig(
                context_window=0,
                context_mode="page",
                include_headers=True,
                filter_content_types=["text"],
            )
        )
        ctx = extractor.extract_context(
            _mineru_pages(),
            {"page_idx": 1},
            content_format="minerU",
        )
        assert "# Section Title" in ctx

    def test_img_caption_alias_when_image_caption_absent(self):
        content = [
            {
                "type": "image",
                "page_idx": 0,
                "img_caption": ["Alias only"],
            }
        ]
        extractor = ContextExtractor(
            ContextConfig(
                context_window=0,
                include_captions=True,
                filter_content_types=["image"],
            )
        )
        ctx = extractor.extract_context(content, {"page_idx": 0}, content_format="minerU")
        assert "[Image: Alias only]" in ctx

    def test_filter_excludes_non_matching_types(self):
        extractor = ContextExtractor(
            ContextConfig(
                context_window=5,
                filter_content_types=["text"],
                include_captions=True,
            )
        )
        ctx = extractor.extract_context(
            _mineru_pages(),
            {"page_idx": 1},
            content_format="minerU",
        )
        assert "Image:" not in ctx
        assert "Table:" not in ctx
        assert "Intro page 0" in ctx


class TestContextExtractorChunkMode:
    def test_chunk_window_skips_current_index(self):
        chunks = [
            {"type": "text", "text": "A"},
            {"type": "text", "text": "B-current"},
            {"type": "text", "text": "C"},
        ]
        extractor = ContextExtractor(
            ContextConfig(context_window=1, context_mode="chunk")
        )
        ctx = extractor.extract_context(
            chunks, {"index": 1}, content_format="minerU"
        )
        assert "A" in ctx
        assert "C" in ctx
        assert "B-current" not in ctx


class TestContextExtractorTextChunksAndDict:
    def test_text_chunks_format_excludes_current(self):
        extractor = ContextExtractor(ContextConfig(context_window=1))
        ctx = extractor.extract_context(
            ["one", "two", "three"],
            {"index": 1},
            content_format="text_chunks",
        )
        assert "one" in ctx
        assert "three" in ctx
        assert "two" not in ctx

    def test_dict_source_uses_content_key(self):
        extractor = ContextExtractor(ContextConfig(max_context_tokens=2000))
        ctx = extractor.extract_context(
            {"content": "dict body"},
            {},
            content_format="auto",
        )
        assert ctx == "dict body"


class TestContextExtractorTruncation:
    def test_char_truncation_without_tokenizer(self):
        extractor = ContextExtractor(ContextConfig(max_context_tokens=20))
        long_text = "Sentence one. " + ("word " * 50)
        ctx = extractor._truncate_context(long_text)
        assert len(ctx) <= 20 + 3  # may append "..."
        assert ctx  # non-empty

    def test_tokenizer_truncation_respects_max_tokens(self):
        class FakeTok:
            def encode(self, text):
                return list(text)

            def decode(self, tokens):
                return "".join(tokens)

        extractor = ContextExtractor(
            ContextConfig(max_context_tokens=40), tokenizer=FakeTok()
        )
        # Period early enough (>80% of truncated window) → cut at sentence end
        text = ("word " * 6) + "Done. " + ("tail " * 20)
        ctx = extractor._truncate_context(text)
        assert "Done." in ctx
        assert "tail" not in ctx
        assert len(FakeTok().encode(ctx)) <= 40

    def test_extract_context_swallows_errors(self):
        class BoomExtractor(ContextExtractor):
            def _extract_from_content_list(self, *args, **kwargs):
                raise RuntimeError("boom")

        extractor = BoomExtractor(ContextConfig())
        assert extractor.extract_context([{"type": "text"}], {"page_idx": 0}) == ""
