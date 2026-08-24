"""PaddleOCR result adapters: harvest text from v2/v3/PaddleX shapes without dropping lines."""

import pytest

from raganything.parser import PaddleOCRParser


def test_plain_string_is_stripped():
    parser = PaddleOCRParser()
    assert parser._extract_text_lines("  hello  ") == ["hello"]
    assert parser._extract_text_lines("   ") == []


def test_rec_texts_and_texts_keys_are_harvested_without_double_count():
    parser = PaddleOCRParser()
    result = {
        "rec_texts": ["A", "  B  ", ""],
        "texts": ["C"],
        "ignored": None,
    }
    assert parser._extract_text_lines(result) == ["A", "B", "C"]


def test_to_dict_objects_are_unwrapped():
    parser = PaddleOCRParser()

    class Node:
        def to_dict(self):
            return {"text": "from-dict"}

    assert parser._extract_text_lines(Node()) == ["from-dict"]


def test_classic_ocr_tuple_shape():
    parser = PaddleOCRParser()
    classic = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("First line", 0.99)],
            [[[0, 2], [1, 2], [1, 3], [0, 3]], ("Second line", 0.95)],
        ]
    ]
    assert parser._extract_text_lines(classic) == ["First line", "Second line"]


def test_ocr_cls_kwarg_typeerror_falls_back():
    parser = PaddleOCRParser()
    calls = []

    class FakeOCR:
        def ocr(self, input_data):
            calls.append(input_data)
            return "fallback-line"

    parser._get_ocr = lambda lang=None: FakeOCR()
    assert parser._ocr_input("page-bytes") == ["fallback-line"]
    assert calls == ["page-bytes"]


def test_predict_api_is_used_when_ocr_is_absent():
    parser = PaddleOCRParser()

    class FakeOCR:
        def predict(self, input_data):
            return {"rec_texts": ["via-predict"]}

    parser._get_ocr = lambda lang=None: FakeOCR()
    assert parser._ocr_input("img") == ["via-predict"]


def test_unsupported_ocr_api_raises():
    parser = PaddleOCRParser()

    class FakeOCR:
        pass

    parser._get_ocr = lambda lang=None: FakeOCR()
    with pytest.raises(RuntimeError, match="ocr` or `predict"):
        parser._ocr_input("img")
