"""Builtin parser names stay case-insensitive and cannot be shadowed.

``PARSER=MinerU`` and ``PARSER="  Docling  "`` must still select the built-in
classes. Registering those same spellings must not hide the built-ins.
Distinct from the lowercase registry checks in ``tests/test_custom_parser.py``
and from the open claim that only ``None`` and a blank name fall back to MinerU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PARSER_PATH = Path(__file__).resolve().parent.parent / "raganything" / "parser.py"


def _load_parser_module():
    name = "_raganything_parser_builtin_case_767b"
    spec = importlib.util.spec_from_file_location(name, PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def parser_mod():
    return _load_parser_module()


@pytest.fixture(autouse=True)
def _clean_registry(parser_mod):
    parser_mod._CUSTOM_PARSERS.clear()
    yield
    parser_mod._CUSTOM_PARSERS.clear()


class _MarkerParser:
    """Replaced per test with a real Parser subclass once the module is loaded."""


def _marker_parser(parser_mod):
    class MarkerParser(parser_mod.Parser):
        def check_installation(self) -> bool:
            return True

        def parse_document(self, file_path, output_dir="./output", method="auto", **kw):
            return [{"type": "text", "text": "marker", "page_idx": 0}]

    return MarkerParser


def test_get_parser_folds_builtin_case_and_padding(parser_mod):
    assert isinstance(parser_mod.get_parser("MinerU"), parser_mod.MineruParser)
    assert isinstance(parser_mod.get_parser("  Docling  "), parser_mod.DoclingParser)
    assert isinstance(parser_mod.get_parser("PaddleOCR"), parser_mod.PaddleOCRParser)
    assert isinstance(parser_mod.get_parser(" mineru "), parser_mod.MineruParser)


def test_register_parser_rejects_cased_builtin_names(parser_mod):
    marker = _marker_parser(parser_mod)
    for name in ("MinerU", "  Docling  ", "PaddleOCR"):
        with pytest.raises(ValueError, match="Cannot override built-in"):
            parser_mod.register_parser(name, marker)

    assert parser_mod._CUSTOM_PARSERS == {}
    assert isinstance(parser_mod.get_parser("mineru"), parser_mod.MineruParser)
    assert isinstance(parser_mod.get_parser("docling"), parser_mod.DoclingParser)
    assert "MinerU" not in parser_mod.list_parsers()
    assert parser_mod.list_parsers()["mineru"] == "MineruParser"


def test_get_parser_folds_custom_lookup_case(parser_mod):
    marker = _marker_parser(parser_mod)
    parser_mod.register_parser("marker", marker)

    resolved = parser_mod.get_parser("  MARKER  ")
    assert isinstance(resolved, marker)
    assert parser_mod.list_parsers()["marker"] == "MarkerParser"
