"""Regression tests for BaseModalProcessor robust JSON parsing strategies.

LLM caption responses are often wrapped in fences, smart quotes, or think tags.
_robust_json_parse / regex fallback decide entity_name/type stored in the KG —
silent mis-parses corrupt multimodal graph nodes. Distinct from
test_strip_thinking_tags.py (which only covers tag stripping on parse failure).
"""

from raganything.modalprocessors import BaseModalProcessor


class ConcreteProcessor(BaseModalProcessor):
    def __init__(self):
        # Skip LightRAG-dependent __init__
        pass

    async def process_multimodal_content(self, *args, **kwargs):
        pass

    async def generate_description_only(self, *args, **kwargs):
        pass


class TestRobustJsonParse:
    def setup_method(self):
        self.proc = ConcreteProcessor()

    def test_direct_json_object(self):
        raw = '{"detailed_description": "d", "entity_info": {"entity_name": "Pump", "entity_type": "equipment", "summary": "s"}}'
        result = self.proc._robust_json_parse(raw)
        assert result["entity_info"]["entity_name"] == "Pump"
        assert result["detailed_description"] == "d"

    def test_json_in_fenced_code_block(self):
        raw = (
            "Here is the result:\n"
            "```json\n"
            '{"detailed_description": "fig", "entity_info": '
            '{"entity_name": "Valve", "entity_type": "part", "summary": "ok"}}\n'
            "```\n"
        )
        result = self.proc._robust_json_parse(raw)
        assert result["entity_info"]["entity_name"] == "Valve"

    def test_strips_think_tags_before_parsing(self):
        raw = (
            "<think>I should output JSON</think>\n"
            '{"detailed_description": "clean", "entity_info": '
            '{"entity_name": "Motor", "entity_type": "device", "summary": "m"}}'
        )
        result = self.proc._robust_json_parse(raw)
        assert result["entity_info"]["entity_name"] == "Motor"
        assert result["detailed_description"] == "clean"

    def test_basic_cleanup_strips_trailing_commas(self):
        cleaned = self.proc._basic_json_cleanup(
            '{"a": 1, "b": [2, 3,],}'
        )
        assert self.proc._try_parse_json(cleaned) == {"a": 1, "b": [2, 3]}

    def test_trailing_comma_cleanup(self):
        raw = '{"detailed_description": "x", "entity_info": {"entity_name": "Y", "entity_type": "t", "summary": "s",},}'
        result = self.proc._robust_json_parse(raw)
        assert result["entity_info"]["entity_name"] == "Y"

    def test_regex_fallback_extracts_fields(self):
        # Intentionally broken JSON that still has quoted fields
        raw = (
            'almost json {"detailed_description": "fallback desc", '
            '"entity_name": "Sensor", "entity_type": "instrument", '
            '"summary": "short"} trailing junk {'
        )
        # Force strategies 1-3 to fail by making braces unbalanced for full parse
        # but keep field patterns for regex strategy
        result = self.proc._extract_fields_with_regex(raw)
        assert result["detailed_description"] == "fallback desc"
        assert result["entity_info"]["entity_name"] == "Sensor"
        assert result["entity_info"]["entity_type"] == "instrument"
        assert result["entity_info"]["summary"] == "short"

    def test_regex_fallback_defaults_when_fields_missing(self):
        result = self.proc._extract_fields_with_regex("no json here")
        assert result["detailed_description"] == ""
        assert result["entity_info"]["entity_name"] == "unknown_entity"
        assert result["entity_info"]["entity_type"] == "unknown"

    def test_extract_candidates_from_balanced_braces(self):
        raw = 'prefix {"a": 1} middle {"b": 2} suffix'
        candidates = self.proc._extract_all_json_candidates(raw)
        assert '{"a": 1}' in candidates
        assert '{"b": 2}' in candidates

    def test_try_parse_json_empty_returns_none(self):
        assert self.proc._try_parse_json("") is None
        assert self.proc._try_parse_json("   ") is None
        assert self.proc._try_parse_json("{bad") is None
