"""PaddleOCR language codes are stripped before the recognizer is built or reused."""

import importlib.util
import sys
from pathlib import Path

_MODULE_NAME = "_raganything_parser_paddleocr_lang_1030"
_PARSER_PATH = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PARSER_PATH)
_parser_module = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = _parser_module
_spec.loader.exec_module(_parser_module)

PaddleOCRParser = _parser_module.PaddleOCRParser


class _RecordingPaddleOCR:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def ocr(self, input_data, cls=True):
        return "bolt torque"


def _parser_with_recorder(default_lang="en"):
    parser = PaddleOCRParser(default_lang=default_lang)
    constructed = []

    class RecordingPaddleOCR(_RecordingPaddleOCR):
        def __init__(self, **kwargs):
            constructed.append(dict(kwargs))
            super().__init__(**kwargs)

    parser._require_paddleocr = lambda: RecordingPaddleOCR
    return parser, constructed


def test_parse_image_reuses_one_model_for_padded_language(tmp_path):
    parser, constructed = _parser_with_recorder()
    image = tmp_path / "scan.png"
    image.write_bytes(b"png")

    first = parser.parse_image(image, lang="  ch  ")
    second = parser.parse_image(image, lang="ch")
    third = parser.parse_image(image, lang=" ch ")

    assert (
        first
        == second
        == third
        == [{"type": "text", "text": "bolt torque", "page_idx": 0}]
    )
    assert constructed == [{"lang": "ch", "show_log": False}]
    assert list(parser._ocr_instances) == ["ch"]


def test_blank_language_uses_default_model_once():
    parser, constructed = _parser_with_recorder(default_lang="en")

    from_none = parser._get_ocr(None)
    from_empty = parser._get_ocr("")
    from_spaces = parser._get_ocr("   ")
    from_padded_default = parser._get_ocr(" en ")

    assert from_empty is from_none
    assert from_spaces is from_none
    assert from_padded_default is from_none
    assert constructed == [{"lang": "en", "show_log": False}]

    chinese = parser._get_ocr("ch")
    assert chinese is not from_none
    assert [item["lang"] for item in constructed] == ["en", "ch"]
    assert set(parser._ocr_instances) == {"en", "ch"}
