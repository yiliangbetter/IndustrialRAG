"""Tests for BaseModalProcessor progressive JSON quote/escape fixing (strategy 3).

LLM captions often embed LaTeX-like escapes (e.g. \\alpha) that break json.loads.
Strategy 3 recovers those payloads before the regex last-resort path.
"""

import json

from raganything.modalprocessors import BaseModalProcessor


class ConcreteProcessor(BaseModalProcessor):
    """Minimal subclass that skips LightRAG-dependent __init__."""

    def __init__(self):
        pass

    async def process_multimodal_content(self, *args, **kwargs):
        pass


INVALID_LATEX_JSON = (
    '{"detailed_description": "Greek letter \\alpha in diagram", '
    '"entity_info": {'
    '"entity_name": "symbol_\\alpha", '
    '"entity_type": "equation", '
    '"summary": "contains \\beta"}}'
)


class TestProgressiveQuoteFix:
    def setup_method(self):
        self.proc = ConcreteProcessor()

    def test_fixes_unescaped_latex_letter_escapes(self):
        # Raw string is invalid JSON because \a / \b are not valid escapes.
        assert self.proc._try_parse_json(INVALID_LATEX_JSON) is None

        fixed = self.proc._progressive_quote_fix(INVALID_LATEX_JSON)
        parsed = json.loads(fixed)

        assert parsed["entity_info"]["entity_name"] == "symbol_\\alpha"
        assert "\\alpha" in parsed["detailed_description"]
        assert parsed["entity_info"]["summary"] == "contains \\beta"

    def test_fix_json_escapes_delegates_to_progressive_fix(self):
        via_legacy = self.proc._fix_json_escapes(INVALID_LATEX_JSON)
        via_progressive = self.proc._progressive_quote_fix(INVALID_LATEX_JSON)
        assert via_legacy == via_progressive
        assert json.loads(via_legacy)["entity_info"]["entity_type"] == "equation"

    def test_robust_parse_uses_strategy_three_when_direct_and_cleanup_fail(self):
        # Strategies 1–2 fail on invalid escapes; strategy 3 must recover entity_name.
        assert self.proc._try_parse_json(INVALID_LATEX_JSON) is None
        cleaned = self.proc._basic_json_cleanup(INVALID_LATEX_JSON)
        assert self.proc._try_parse_json(cleaned) is None

        result = self.proc._robust_json_parse(INVALID_LATEX_JSON)
        assert result["entity_info"]["entity_name"] == "symbol_\\alpha"
        assert result["entity_info"]["entity_type"] == "equation"
        assert "Greek letter" in result["detailed_description"]
