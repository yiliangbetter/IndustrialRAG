"""Character-only context cuts must keep the strict 80% boundary.

Production captions prefer a tokenizer. When none is attached,
``ContextExtractor._truncate_context`` falls back to characters. A context
that is exactly ``max_context_tokens`` long must stay whole. A period or
newline is kept only when it sits past 80% of that window, and a late period
wins over a late newline.

Distinct from the tokenizer equality checks and from the character tests that
only place the period past 80%.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def test_character_window_keeps_exact_length_and_marks_one_extra_char():
    extractor = ContextExtractor(ContextConfig(max_context_tokens=8))

    assert extractor._truncate_context("abcdefgh") == "abcdefgh"
    assert extractor._truncate_context("abcdefghi") == "abcdefgh..."


def test_character_boundary_at_80_percent_keeps_the_ellipsis():
    extractor = ContextExtractor(ContextConfig(max_context_tokens=10))
    # Index 8 of a 10-character window is exactly 80%, which must not win.
    assert extractor._truncate_context("abcdefgh.XEXTRA") == "abcdefgh.X..."
    assert extractor._truncate_context("abcdefgh\nXEXTRA") == "abcdefgh\nX..."


def test_character_cuts_at_a_period_past_80_percent():
    extractor = ContextExtractor(ContextConfig(max_context_tokens=10))

    assert extractor._truncate_context("abcdefghi.EXTRA") == "abcdefghi."


def test_character_late_period_wins_over_a_late_newline():
    extractor = ContextExtractor(ContextConfig(max_context_tokens=20))
    text = "abcdefghijklmnopq\n.ZEXTRA"

    assert extractor._truncate_context(text) == "abcdefghijklmnopq\n."
