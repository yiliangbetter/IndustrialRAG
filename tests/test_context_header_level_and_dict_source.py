"""Context headers and dictionary sources must keep the surrounding text honest.

A text level of 0 is body text and must not gain a markdown heading. An
explicit empty ``content`` field must not fall through to a sibling ``text``
field. When a dictionary has neither field, only string values are joined.
"""

from raganything.modalprocessors import ContextConfig, ContextExtractor


def test_text_level_zero_is_body_and_level_two_is_a_heading():
    pages = [
        {"type": "text", "text": "Body", "page_idx": 0, "text_level": 0},
        {"type": "text", "text": "Untitled", "page_idx": 0},
        {"type": "text", "text": "Section", "page_idx": 0, "text_level": 2},
    ]
    extractor = ContextExtractor(
        ContextConfig(context_window=0, context_mode="page", include_headers=True)
    )

    context = extractor.extract_context(pages, {"page_idx": 0}, content_format="minerU")

    assert context.split("\n") == ["Body", "Untitled", "## Section"]


def test_empty_content_key_does_not_fall_through_to_text():
    extractor = ContextExtractor()

    assert extractor.extract_context({"content": "", "text": "sibling"}, {}) == ""
    assert extractor.extract_context({"text": "from-text"}, {}) == "from-text"


def test_dict_source_without_named_fields_keeps_strings_only():
    extractor = ContextExtractor()

    context = extractor.extract_context(
        {"title": "Pump", "page": 3, "ok": True, "note": "check torque"},
        {},
    )

    assert context == "Pump\ncheck torque"
