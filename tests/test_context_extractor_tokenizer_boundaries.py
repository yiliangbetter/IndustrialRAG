"""Tokenizer truncation boundaries and MinerU caption aliases.

#122 covers character-based period/newline cuts and tokenizer ellipsis.
The tokenizer path still has the same 80% sentence/newline rule: a
regression there silently over-truncates VLM context or drops nearby
headers. img_caption is the MinerU field; image_caption is the Docling
one. include_captions=False must not leak captions when filters include
image/table types.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


class FakeTokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


class TestTokenizerBoundaries:
    def test_tokenizer_uses_sentence_boundary(self):
        extractor = ContextExtractor(
            ContextConfig(max_context_tokens=22), tokenizer=FakeTokenizer()
        )
        text = "Keep this sentence. trailing words"
        assert extractor._truncate_context(text) == "Keep this sentence."

    def test_tokenizer_uses_newline_boundary(self):
        extractor = ContextExtractor(
            ContextConfig(max_context_tokens=22), tokenizer=FakeTokenizer()
        )
        text = "first line is here\nextra trailing words"
        assert extractor._truncate_context(text) == "first line is here"

    def test_tokenizer_prefers_period_when_both_qualify(self):
        extractor = ContextExtractor(
            ContextConfig(max_context_tokens=22), tokenizer=FakeTokenizer()
        )
        # Period and newline both sit past 80% of the 22-char window.
        text = "Keep this sentence.\ntrailing words"
        assert extractor._truncate_context(text) == "Keep this sentence."


class TestCaptionAliases:
    def test_img_caption_alias_is_included(self):
        extractor = ContextExtractor(
            ContextConfig(
                context_window=0,
                context_mode="page",
                filter_content_types=["image"],
            )
        )
        context = extractor.extract_context(
            [{"type": "image", "img_caption": ["Legacy caption"], "page_idx": 0}],
            {"page_idx": 0},
        )
        assert "[Image: Legacy caption]" in context

    def test_include_captions_false_skips_image_and_table(self):
        extractor = ContextExtractor(
            ContextConfig(
                context_window=0,
                filter_content_types=["text", "image", "table"],
                include_captions=False,
            )
        )
        context = extractor.extract_context(
            [
                {"type": "text", "text": "body", "page_idx": 0},
                {"type": "image", "image_caption": ["hidden image"], "page_idx": 0},
                {"type": "table", "table_caption": ["hidden table"], "page_idx": 0},
            ],
            {"page_idx": 0},
        )
        assert "body" in context
        assert "hidden image" not in context
        assert "hidden table" not in context
