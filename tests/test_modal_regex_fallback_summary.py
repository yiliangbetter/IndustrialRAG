"""Regex fallback must keep a bounded entity summary when JSON cannot be repaired.

Every modal processor stores `_robust_json_parse` output. When the model emits
field text that is not valid JSON, the last resort copies `summary` from the
first 100 characters of `detailed_description`. Truncating an explicit summary,
or keeping an unbounded one, changes knowledge-graph entity text.
"""

from raganything.modalprocessors import BaseModalProcessor


def _parse(response: str) -> dict:
    processor = BaseModalProcessor.__new__(BaseModalProcessor)
    return processor._robust_json_parse(response)


def test_missing_summary_uses_first_100_characters_of_description():
    description = "D" * 101
    response = (
        "not json\n"
        f'"detailed_description": "{description}"\n'
        '"entity_name": "gearbox"\n'
        '"entity_type": "image"'
    )

    result = _parse(response)

    assert result["detailed_description"] == description
    assert result["entity_info"]["summary"] == description[:100]
    assert len(result["entity_info"]["summary"]) == 100
    assert result["entity_info"]["entity_name"] == "gearbox"
    assert result["entity_info"]["entity_type"] == "image"


def test_description_at_100_characters_is_copied_in_full_when_summary_missing():
    description = "S" * 100
    response = f'"detailed_description": "{description}"\n"entity_name": "seal"'

    result = _parse(response)

    assert result["entity_info"]["summary"] == description
    assert result["entity_info"]["entity_type"] == "unknown"


def test_explicit_summary_longer_than_100_characters_is_not_truncated():
    summary = "K" * 150
    response = (
        '"detailed_description": "short caption"\n'
        '"entity_name": "torque_table"\n'
        '"entity_type": "table"\n'
        f'"summary": "{summary}"'
    )

    result = _parse(response)

    assert result["entity_info"]["summary"] == summary
    assert result["detailed_description"] == "short caption"
