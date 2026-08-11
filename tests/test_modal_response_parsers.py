"""Typed modal response parsers: success suffixing and missing-field fallback.

Existing main coverage strips think-tags on forced parse failure
(test_strip_thinking_tags). Open PR #79 covers BaseModalProcessor JSON
strategies. These tests lock the typed/generic parser contracts that decide
entity_name stored in the KG after a successful JSON parse, and the fallback
when required fields are missing.
"""

import json

from raganything.modalprocessors import (
    EquationModalProcessor,
    GenericModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)


class _BareModal:
    def __init__(self):
        pass


def _as_processor(cls):
    proc = _BareModal()
    proc.__class__ = cls
    return proc


def _ok_payload(name: str, entity_type: str, summary: str = "summary") -> str:
    return json.dumps(
        {
            "detailed_description": f"Description of {name}",
            "entity_info": {
                "entity_name": name,
                "entity_type": entity_type,
                "summary": summary,
            },
        }
    )


class TestImageParseResponse:
    def test_success_appends_entity_type_suffix(self):
        proc = _as_processor(ImageModalProcessor)
        caption, entity = proc._parse_response(_ok_payload("Pump", "image", "cutaway"))

        assert caption == "Description of Pump"
        assert entity["entity_name"] == "Pump (image)"
        assert entity["entity_type"] == "image"
        assert entity["summary"] == "cutaway"

    def test_caller_entity_name_overrides_suffix(self):
        proc = _as_processor(ImageModalProcessor)
        caption, entity = proc._parse_response(
            _ok_payload("Pump", "image"),
            entity_name="Canonical Pump",
        )

        assert caption == "Description of Pump"
        assert entity["entity_name"] == "Canonical Pump"

    def test_missing_description_falls_back(self):
        proc = _as_processor(ImageModalProcessor)
        raw = json.dumps(
            {
                "detailed_description": "",
                "entity_info": {
                    "entity_name": "X",
                    "entity_type": "image",
                    "summary": "s",
                },
            }
        )
        caption, entity = proc._parse_response(raw, entity_name="Forced")

        assert caption == raw
        assert entity["entity_name"] == "Forced"
        assert entity["entity_type"] == "image"
        # Fallback summaries truncate long raw responses at 100 chars.
        assert entity["summary"] == raw[:100] + "..."
        assert len(raw) > 100

    def test_incomplete_entity_info_falls_back_with_hash_name(self):
        proc = _as_processor(ImageModalProcessor)
        raw = json.dumps(
            {
                "detailed_description": "partial",
                "entity_info": {"entity_name": "OnlyName"},
            }
        )
        caption, entity = proc._parse_response(raw)

        assert caption == raw
        assert entity["entity_name"].startswith("image_")
        assert entity["entity_type"] == "image"


class TestTableParseResponse:
    def test_success_and_override(self):
        proc = _as_processor(TableModalProcessor)
        caption, entity = proc._parse_table_response(
            _ok_payload("Ratings", "table"),
            entity_name="Spec Table",
        )

        assert caption == "Description of Ratings"
        assert entity["entity_name"] == "Spec Table"
        assert entity["entity_type"] == "table"

    def test_missing_entity_info_falls_back(self):
        proc = _as_processor(TableModalProcessor)
        raw = json.dumps({"detailed_description": "table text"})
        caption, entity = proc._parse_table_response(raw)

        assert caption == raw
        assert entity["entity_name"].startswith("table_")
        assert entity["entity_type"] == "table"


class TestEquationParseResponse:
    def test_success_appends_suffix(self):
        proc = _as_processor(EquationModalProcessor)
        caption, entity = proc._parse_equation_response(
            _ok_payload("Ohm", "equation", "V=IR")
        )

        assert caption == "Description of Ohm"
        assert entity["entity_name"] == "Ohm (equation)"
        assert entity["summary"] == "V=IR"

    def test_incomplete_entity_info_falls_back(self):
        proc = _as_processor(EquationModalProcessor)
        raw = json.dumps(
            {
                "detailed_description": "eq",
                "entity_info": {"entity_name": "E", "entity_type": "equation"},
            }
        )
        caption, entity = proc._parse_equation_response(raw, entity_name="CallerEq")

        assert caption == raw
        assert entity["entity_name"] == "CallerEq"
        assert entity["entity_type"] == "equation"


class TestGenericParseResponse:
    def test_success_appends_suffix(self):
        proc = _as_processor(GenericModalProcessor)
        caption, entity = proc._parse_generic_response(
            _ok_payload("Clip", "audio"),
            content_type="audio",
        )

        assert caption == "Description of Clip"
        assert entity["entity_name"] == "Clip (audio)"
        assert entity["entity_type"] == "audio"

    def test_missing_fields_use_content_type_in_fallback(self):
        proc = _as_processor(GenericModalProcessor)
        raw = json.dumps({"detailed_description": "", "entity_info": {}})
        caption, entity = proc._parse_generic_response(
            raw,
            content_type="chart",
        )

        assert caption == raw
        assert entity["entity_name"].startswith("chart_")
        assert entity["entity_type"] == "chart"
