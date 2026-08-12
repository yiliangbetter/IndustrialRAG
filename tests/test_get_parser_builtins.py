"""Builtin ``get_parser`` selection and name normalization.

PaddleOCR + custom registry coverage already exists; this locks mineru/docling
factory routing and empty/whitespace defaults used by RAGAnything init.
"""

from __future__ import annotations

import pytest

from raganything.parser import (
    DoclingParser,
    MineruParser,
    PaddleOCRParser,
    get_parser,
)


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("mineru", MineruParser),
        ("MinerU", MineruParser),
        (" docling ", DoclingParser),
        ("paddleocr", PaddleOCRParser),
    ],
)
def test_get_parser_returns_builtin_instances(name, expected_cls):
    parser = get_parser(name)
    assert isinstance(parser, expected_cls)


@pytest.mark.parametrize("name", [None, ""])
def test_get_parser_defaults_blank_names_to_mineru(name):
    parser = get_parser(name)
    assert isinstance(parser, MineruParser)


def test_get_parser_whitespace_only_name_is_rejected():
    """Whitespace is truthy before strip, so it does not fall back to mineru."""
    with pytest.raises(ValueError, match="Unsupported parser type"):
        get_parser("   ")


def test_get_parser_rejects_unknown_with_supported_list():
    with pytest.raises(ValueError, match="Unsupported parser type") as exc_info:
        get_parser("not-a-real-parser")

    message = str(exc_info.value)
    assert "mineru" in message
    assert "docling" in message
    assert "paddleocr" in message
