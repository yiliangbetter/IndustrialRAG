"""Regression tests for modal-response entity identity.

Parsed captions become knowledge-graph entities. These tests lock the type
suffix, caller-name override, and the refusal to store a partial entity when
the model response is missing a description or a required entity field.
"""

import pytest
from lightrag.utils import compute_mdhash_id

from raganything.modalprocessors import (
    EquationModalProcessor,
    GenericModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)

COMPLETE = (
    '{"detailed_description": "A bolt.", '
    '"entity_info": {"entity_name": "Bolt", "entity_type": "part", "summary": "Fastener"}}'
)
MISSING_SUMMARY = (
    '{"detailed_description": "A bolt.", '
    '"entity_info": {"entity_name": "Bolt", "entity_type": "part"}}'
)
EMPTY_DESCRIPTION = (
    '{"detailed_description": "", '
    '"entity_info": {"entity_name": "Bolt", "entity_type": "part", "summary": "Fastener"}}'
)

PARSERS = [
    (ImageModalProcessor, "image"),
    (TableModalProcessor, "table"),
    (EquationModalProcessor, "equation"),
    (GenericModalProcessor, "chart"),
]


def _processor(cls):
    return cls.__new__(cls)


def _parse(proc, response, entity_name=None, content_type="chart"):
    if isinstance(proc, ImageModalProcessor):
        return proc._parse_response(response, entity_name)
    if isinstance(proc, TableModalProcessor):
        return proc._parse_table_response(response, entity_name)
    if isinstance(proc, EquationModalProcessor):
        return proc._parse_equation_response(response, entity_name)
    return proc._parse_generic_response(response, entity_name, content_type)


@pytest.mark.parametrize(("cls", "fallback_type"), PARSERS)
def test_complete_response_appends_model_type_suffix(cls, fallback_type):
    caption, entity = _parse(_processor(cls), COMPLETE)

    assert caption == "A bolt."
    assert entity["entity_name"] == "Bolt (part)"
    assert entity["entity_type"] == "part"
    assert entity["summary"] == "Fastener"
    assert fallback_type not in entity["entity_name"]


@pytest.mark.parametrize(("cls", "fallback_type"), PARSERS)
def test_caller_name_replaces_suffix_and_blank_name_does_not(cls, fallback_type):
    proc = _processor(cls)

    _, named = _parse(proc, COMPLETE, entity_name="CatalogBolt")
    _, blank = _parse(proc, COMPLETE, entity_name="")

    assert named["entity_name"] == "CatalogBolt"
    assert named["entity_type"] == "part"
    assert fallback_type not in named["entity_name"]
    assert blank["entity_name"] == "Bolt (part)"


@pytest.mark.parametrize(("cls", "fallback_type"), PARSERS)
def test_missing_summary_falls_back_instead_of_partial_entity(cls, fallback_type):
    caption, entity = _parse(_processor(cls), MISSING_SUMMARY, entity_name="KeepMe")
    _, hashed = _parse(_processor(cls), MISSING_SUMMARY)

    assert caption == MISSING_SUMMARY
    assert entity["entity_name"] == "KeepMe"
    assert entity["entity_type"] == fallback_type
    assert (
        hashed["entity_name"] == f"{fallback_type}_{compute_mdhash_id(MISSING_SUMMARY)}"
    )
    assert hashed["entity_type"] == fallback_type
    assert hashed["entity_name"] != "Bolt (part)"


@pytest.mark.parametrize(("cls", "fallback_type"), PARSERS)
def test_empty_description_falls_back(cls, fallback_type):
    caption, entity = _parse(_processor(cls), EMPTY_DESCRIPTION)

    assert caption == EMPTY_DESCRIPTION
    assert entity["entity_type"] == fallback_type
    assert entity["entity_name"] == (
        f"{fallback_type}_{compute_mdhash_id(EMPTY_DESCRIPTION)}"
    )
    assert entity["summary"] != "Fastener"


@pytest.mark.parametrize(("cls", "fallback_type"), PARSERS)
def test_fallback_summary_cuts_only_after_100_characters(cls, fallback_type):
    exact = "A" * 100
    over = "B" * 101
    proc = _processor(cls)

    exact_caption, exact_entity = _parse(proc, exact)
    over_caption, over_entity = _parse(proc, over)

    assert exact_caption == exact
    assert exact_entity["summary"] == exact
    assert exact_entity["entity_name"] == f"{fallback_type}_{compute_mdhash_id(exact)}"
    assert over_caption == over
    assert over_entity["summary"] == ("B" * 100) + "..."
    assert over_entity["entity_name"] == f"{fallback_type}_{compute_mdhash_id(over)}"


@pytest.mark.parametrize(("cls", "fallback_type"), PARSERS)
def test_thinking_block_is_excluded_from_fallback_entity_id(cls, fallback_type):
    answer = "C" * 100
    raw = f"<think>secret reasoning</think>{answer}"

    caption, entity = _parse(_processor(cls), raw)

    assert caption == answer
    assert entity["summary"] == answer
    assert "<think>" not in caption
    assert "<think>" not in entity["summary"]
    assert entity["entity_name"] == f"{fallback_type}_{compute_mdhash_id(answer)}"
    assert entity["entity_type"] == fallback_type
