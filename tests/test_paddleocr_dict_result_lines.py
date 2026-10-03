"""PaddleOCR dict and result-object text extraction.

Newer PaddleOCR builds return a page object (or a dict) with ``rec_texts``
plus geometry, instead of the classic ``[box, (text, score)]`` tuples covered
by ``tests/testpaddleocr_parser.py``. Walking every dict value after reading
``rec_texts`` / ``text`` / ``texts`` would insert each line twice. Dropping
``to_dict()`` would insert nothing. A single result object whose ``to_dict``
fails must not discard the rest of the page.
"""

import importlib.util
from pathlib import Path


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_paddle_dict_a8ee", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser_module = _load_parser_module()
PaddleOCRParser = parser_module.PaddleOCRParser


class _PageResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _BrokenResult:
    def to_dict(self):
        raise RuntimeError("corrupt paddle result")


def _paddle_page(label, extra_text):
    return {
        "rec_texts": [f"  {label}  ", {"text": "inner"}],
        "text": extra_text,
        "texts": ["Footer"],
        "rec_scores": [0.98],
        "dt_polys": [[[0, 0], [10, 10]]],
    }


def test_parse_image_keeps_dict_lines_once(tmp_path, monkeypatch):
    image = tmp_path / "scan.png"
    image.write_bytes(b"not-a-real-png")
    parser = PaddleOCRParser()

    class DictOCR:
        def ocr(self, input_data, cls=True):
            assert input_data == str(image)
            assert cls is True
            return [_PageResult(_paddle_page("Valve label", "Header"))]

    monkeypatch.setattr(parser, "_get_ocr", lambda lang=None: DictOCR())

    assert parser.parse_image(image, page_idx=3) == [
        {"type": "text", "text": "Valve label", "page_idx": 3},
        {"type": "text", "text": "inner", "page_idx": 3},
        {"type": "text", "text": "Header", "page_idx": 3},
        {"type": "text", "text": "Footer", "page_idx": 3},
    ]


def test_blank_and_non_text_rec_items_are_dropped():
    parser = PaddleOCRParser()
    lines = parser._extract_text_lines(
        {"rec_texts": ["  ", "", "Keep", 1, None, {"text": "  "}]}
    )
    assert lines == ["Keep"]


def test_text_score_pair_does_not_stringify_the_score():
    parser = PaddleOCRParser()
    assert parser._extract_text_lines(["  score-line  ", 0.87]) == ["score-line"]


def test_broken_to_dict_does_not_drop_sibling_pages():
    parser = PaddleOCRParser()
    lines = parser._extract_text_lines(
        [
            _BrokenResult(),
            _PageResult({"rec_texts": ["kept"], "rec_scores": [0.5]}),
        ]
    )
    assert lines == ["kept"]


def test_to_dict_failure_on_the_only_result_returns_no_lines():
    parser = PaddleOCRParser()
    assert parser._extract_text_lines(_BrokenResult()) == []
