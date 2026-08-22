"""Regression tests for BaseModalProcessor JSON recovery.

Every modal processor (image/table/equation/generic) turns LLM output into
knowledge-graph entities through `_robust_json_parse`. A regression here
silently drops fields, stores regex fallbacks, or pollutes entity names.
"""

from raganything.modalprocessors import BaseModalProcessor, ImageModalProcessor


class _BareProcessor(BaseModalProcessor):
    """Skip LightRAG construction; only the JSON helpers are under test."""

    def __init__(self):
        pass

    async def generate_description_only(self, *args, **kwargs):
        raise NotImplementedError


def _valid_payload(
    description="A detailed visual description",
    name="pump_diagram",
    entity_type="image",
    summary="Centrifugal pump cutaway",
):
    return (
        "{"
        f'"detailed_description": "{description}", '
        f'"entity_info": {{'
        f'"entity_name": "{name}", '
        f'"entity_type": "{entity_type}", '
        f'"summary": "{summary}"'
        "}}"
        "}"
    )


class TestRobustJsonParse:
    def setup_method(self):
        self.proc = _BareProcessor()

    def test_parses_json_inside_fenced_code_block(self):
        response = f"Here is the analysis:\n```json\n{_valid_payload()}\n```\n"
        result = self.proc._robust_json_parse(response)
        assert result["detailed_description"] == "A detailed visual description"
        assert result["entity_info"]["entity_name"] == "pump_diagram"

    def test_strips_think_tags_before_candidate_extraction(self):
        response = (
            "<think>I will emit JSON after reasoning.</think>\n" + _valid_payload()
        )
        result = self.proc._robust_json_parse(response)
        assert result["entity_info"]["entity_type"] == "image"
        assert "think" not in result["detailed_description"]

    def test_trailing_comma_recovered_by_basic_cleanup(self):
        messy = (
            '{"detailed_description": "Table of ratings", '
            '"entity_info": {"entity_name": "ratings", '
            '"entity_type": "table", "summary": "Ratings table"},}'
        )
        result = self.proc._robust_json_parse(messy)
        assert result["detailed_description"] == "Table of ratings"
        assert result["entity_info"]["entity_name"] == "ratings"

    def test_unescaped_latex_backslash_recovered_by_quote_fix(self):
        # Invalid JSON escape `\a` must be repaired before json.loads succeeds.
        messy = (
            '{"detailed_description": "formula \\alpha = 1", '
            '"entity_info": {"entity_name": "alpha_eq", '
            '"entity_type": "equation", "summary": "alpha"}}'
        )
        result = self.proc._robust_json_parse(messy)
        assert "alpha" in result["detailed_description"].lower()
        assert result["entity_info"]["entity_name"] == "alpha_eq"

    def test_first_valid_candidate_wins_when_multiple_objects_present(self):
        response = (
            _valid_payload(description="first", name="first_entity")
            + "\n"
            + _valid_payload(description="second", name="second_entity")
        )
        result = self.proc._robust_json_parse(response)
        assert result["detailed_description"] == "first"
        assert result["entity_info"]["entity_name"] == "first_entity"

    def test_regex_fallback_when_json_is_unrecoverable(self):
        response = (
            "not json at all "
            '"detailed_description": "partial caption" '
            '"entity_name": "fallback_name" '
            '"entity_type": "image" '
            '"summary": "partial summary"'
        )
        result = self.proc._robust_json_parse(response)
        assert result["detailed_description"] == "partial caption"
        assert result["entity_info"]["entity_name"] == "fallback_name"
        assert result["entity_info"]["entity_type"] == "image"
        assert result["entity_info"]["summary"] == "partial summary"

    def test_regex_fallback_defaults_for_empty_response(self):
        result = self.proc._robust_json_parse("")
        assert result["detailed_description"] == ""
        assert result["entity_info"]["entity_name"] == "unknown_entity"
        assert result["entity_info"]["entity_type"] == "unknown"
        assert result["entity_info"]["summary"] == ""


class TestImageParseResponseSuccess:
    def setup_method(self):
        self.proc = _BareProcessor()
        self.proc.__class__ = ImageModalProcessor

    def test_success_appends_entity_type_suffix(self):
        caption, entity = self.proc._parse_response(_valid_payload())
        assert caption == "A detailed visual description"
        assert entity["entity_name"] == "pump_diagram (image)"
        assert entity["entity_type"] == "image"
        assert entity["summary"] == "Centrifugal pump cutaway"

    def test_predefined_entity_name_overrides_suffixed_name(self):
        caption, entity = self.proc._parse_response(
            _valid_payload(), entity_name="FIG-12"
        )
        assert caption == "A detailed visual description"
        assert entity["entity_name"] == "FIG-12"

    def test_missing_entity_info_falls_back_instead_of_raising(self):
        raw = '{"detailed_description": "only a caption"}'
        caption, entity = self.proc._parse_response(raw)
        # Fallback uses the stripped raw response, not a partial JSON field.
        assert caption == raw
        assert entity["entity_type"] == "image"
        assert entity["entity_name"].startswith("image_")
        assert entity["summary"] == raw
