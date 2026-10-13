"""Tokenizer context cuts must honor the late sentence and line boundaries.

Character-only truncation is already proposed. Production captions use a
tokenizer, and the cut is strict: a period or newline is kept only when it
sits past 80% of the decoded window, and a period wins over a newline.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


class CharTokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def _extractor(max_tokens):
    return ContextExtractor(
        ContextConfig(max_context_tokens=max_tokens),
        tokenizer=CharTokenizer(),
    )


def test_tokenizer_cuts_at_a_late_period():
    extractor = _extractor(10)
    assert extractor._truncate_context("abcdefghi.EXTRA") == "abcdefghi."


def test_tokenizer_period_wins_over_an_earlier_newline():
    extractor = _extractor(10)
    assert extractor._truncate_context("ab\ncdefgh.EXTRA") == "ab\ncdefgh."


def test_tokenizer_cuts_at_a_late_newline_when_the_period_is_early():
    extractor = _extractor(10)
    assert extractor._truncate_context("a.bcdefgh\nEXTRA") == "a.bcdefgh"


def test_tokenizer_boundary_at_80_percent_keeps_the_ellipsis():
    extractor = _extractor(10)
    # Index 8 of a 10-character window is exactly 80%, which must not win.
    assert extractor._truncate_context("abcdefgh.XEXTRA") == "abcdefgh.X..."
    assert extractor._truncate_context("abcdefgh\nXEXTRA") == "abcdefgh\nX..."
