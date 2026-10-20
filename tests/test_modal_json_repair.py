"""Modal caption JSON repair used by every content processor.

Model output often has trailing commas or LaTeX backslashes. Those responses
must become the parsed object, not the regex fallback, and a later valid
object must still be reachable when an earlier one does not parse.
"""

from raganything.modalprocessors import BaseModalProcessor


class _Processor(BaseModalProcessor):
    def __init__(self):
        pass


def _parse(response: str) -> dict:
    return _Processor()._robust_json_parse(response)


def test_trailing_commas_in_object_and_array_parse():
    raw = '{"detailed_description": "two parts", "parts": ["a", "b",],}'
    assert _parse(raw) == {
        "detailed_description": "two parts",
        "parts": ["a", "b"],
    }


def test_comma_inside_a_string_is_preserved():
    raw = '{"detailed_description": "ends with comma,"}'
    assert _parse(raw) == {"detailed_description": "ends with comma,"}


def test_latex_backslash_escapes_parse_as_literal_commands():
    raw = '{"detailed_description": "torque is \\alpha times \\beta"}'
    assert _parse(raw) == {"detailed_description": "torque is \\alpha times \\beta"}


def test_fenced_json_wins_over_a_later_object():
    raw = (
        '```json\n{"detailed_description": "from fence"}\n```\n'
        '{"detailed_description": "from prose"}'
    )
    assert _parse(raw) == {"detailed_description": "from fence"}


def test_later_valid_object_survives_an_earlier_trailing_comma():
    raw = '{"bad": true,}\n{"detailed_description": "kept", "entity_name": "motor"}'
    assert _parse(raw) == {
        "detailed_description": "kept",
        "entity_name": "motor",
    }
