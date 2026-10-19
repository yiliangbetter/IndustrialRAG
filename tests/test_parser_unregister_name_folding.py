"""Custom parser removal must use the same folded name as registration.

Operators register ``marker`` and later remove ``MARKER`` or ``  marker  ``.
A case-sensitive unregister leaves that parser installed, so the next ingest
still runs third-party code. Unregistering ``MinerU`` must not delete the
built-in. Re-registering one custom name must keep a single entry in the
unsupported-parser error, or operators cannot tell which parsers are live.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PARSER_PATH = Path(__file__).resolve().parent.parent / "raganything" / "parser.py"


def _load_parser_module():
    name = "_raganything_parser_unregister_0894"
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


def _parser_class(parser_mod, class_name: str, text: str):
    def parse_document(self, file_path, output_dir="./output", method="auto", **kw):
        return [{"type": "text", "text": text, "page_idx": 0}]

    return type(
        class_name,
        (parser_mod.Parser,),
        {
            "check_installation": lambda self: True,
            "parse_document": parse_document,
        },
    )


def test_unregister_folds_case_and_leaves_other_custom_parsers(parser_mod):
    marker = _parser_class(parser_mod, "MarkerParser", "marker")
    surya = _parser_class(parser_mod, "SuryaParser", "surya")
    parser_mod.register_parser("marker", marker)
    parser_mod.register_parser("surya", surya)

    parser_mod.unregister_parser("  MARKER  ")

    with pytest.raises(ValueError, match="Unsupported parser type: marker"):
        parser_mod.get_parser("marker")
    with pytest.raises(KeyError, match="marker"):
        parser_mod.unregister_parser("marker")

    assert isinstance(parser_mod.get_parser("  Surya  "), surya)
    assert "marker" not in parser_mod.list_parsers()
    assert parser_mod.list_parsers()["surya"] == "SuryaParser"
    assert parser_mod.get_supported_parsers().count("marker") == 0
    assert parser_mod.get_supported_parsers().count("surya") == 1
    assert isinstance(parser_mod.get_parser("mineru"), parser_mod.MineruParser)


def test_unregister_builtin_name_does_not_remove_the_builtin(parser_mod):
    marker = _parser_class(parser_mod, "MarkerParser", "marker")
    parser_mod.register_parser("marker", marker)

    for name in ("MinerU", "  Docling  ", "PaddleOCR"):
        with pytest.raises(KeyError, match="No custom parser"):
            parser_mod.unregister_parser(name)

    assert isinstance(parser_mod.get_parser("mineru"), parser_mod.MineruParser)
    assert isinstance(parser_mod.get_parser("docling"), parser_mod.DoclingParser)
    assert isinstance(parser_mod.get_parser("paddleocr"), parser_mod.PaddleOCRParser)
    assert isinstance(parser_mod.get_parser("marker"), marker)
    assert set(parser_mod.list_parsers()) == {
        "mineru",
        "docling",
        "paddleocr",
        "marker",
    }


def test_reregister_lists_each_custom_parser_once(parser_mod):
    first = _parser_class(parser_mod, "MarkerParser", "first")
    second = _parser_class(parser_mod, "MarkerParserV2", "second")
    surya = _parser_class(parser_mod, "SuryaParser", "surya")
    parser_mod.register_parser("marker", first)
    parser_mod.register_parser("  Marker  ", second)
    parser_mod.register_parser("surya", surya)

    resolved = parser_mod.get_parser("marker")
    supported = parser_mod.get_supported_parsers()

    assert isinstance(resolved, second)
    assert supported.count("marker") == 1
    assert supported == ("mineru", "docling", "paddleocr", "marker", "surya")
    assert parser_mod.list_parsers()["marker"] == "MarkerParserV2"

    with pytest.raises(ValueError, match="Unsupported parser type: missing") as exc:
        parser_mod.get_parser("missing")

    message = str(exc.value)
    assert message.endswith(
        "Supported parsers: mineru, docling, paddleocr, marker, surya"
    )
    assert message.count("marker") == 1
