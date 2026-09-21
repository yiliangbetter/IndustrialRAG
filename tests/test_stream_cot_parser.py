"""Response parsing tests for hidden reasoning wrappers."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stream_cot_parser import StreamCotParser, parse_complete_cot


def test_parse_complete_cot_recognizes_thinking_tag() -> None:
    thinking, answer = parse_complete_cot(
        "<thinking>internal reasoning</thinking>user-facing answer"
    )

    assert thinking == "internal reasoning"
    assert answer == "user-facing answer"


def test_stream_parser_handles_thinking_tag_across_chunks() -> None:
    parser = StreamCotParser()
    events: list[tuple[str, str]] = []

    for chunk in ("<thi", "nking>secret", "</think", "ing>answer"):
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())

    thinking = "".join(text for kind, text in events if kind == "thinking")
    answer = "".join(text for kind, text in events if kind == "answer")
    assert thinking == "secret"
    assert answer == "answer"


def test_stream_parser_matches_reasoning_tags_case_insensitively() -> None:
    thinking, answer = parse_complete_cot("<THINKING>secret</THINKING>answer")

    assert thinking == "secret"
    assert answer == "answer"


def test_plain_answer_is_unchanged() -> None:
    assert parse_complete_cot("plain answer") == ("", "plain answer")
