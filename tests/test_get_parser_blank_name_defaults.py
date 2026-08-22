"""Regression tests for get_parser name normalization and defaults.

Blank or missing parser names must resolve to MinerU instead of crashing
ingest, while whitespace-only names that survive the `or` default must
still fail closed.
"""

import pytest

from raganything.parser import (
    DoclingParser,
    MineruParser,
    PaddleOCRParser,
    get_parser,
)


class TestGetParserBlankNameDefaults:
    def test_none_defaults_to_mineru(self):
        parser = get_parser(None)
        assert isinstance(parser, MineruParser)

    def test_empty_string_defaults_to_mineru(self):
        parser = get_parser("")
        assert isinstance(parser, MineruParser)

    def test_padded_builtin_name_is_normalized(self):
        parser = get_parser("  MinerU  ")
        assert isinstance(parser, MineruParser)

    def test_padded_docling_and_paddleocr_are_normalized(self):
        assert isinstance(get_parser(" Docling "), DoclingParser)
        assert isinstance(get_parser("PADDLEOCR"), PaddleOCRParser)

    def test_whitespace_only_name_is_rejected(self):
        # `"   " or "mineru"` is `"   "`, then strip() yields "" and no match.
        with pytest.raises(ValueError, match="Unsupported parser type"):
            get_parser("   ")
