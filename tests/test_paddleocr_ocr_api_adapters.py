"""Regression tests for PaddleOCR API adapters.

Covers init-candidate fallbacks, ocr/predict routing, and temp-file cleanup for
rendered PDF pages — paths that silently break OCR ingest across Paddle versions.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from raganything.parser import PaddleOCRParser


class TestGetOcr:
    def test_tries_init_candidates_and_caches_by_lang(self, monkeypatch):
        parser = PaddleOCRParser(default_lang="en")
        calls = []

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if "show_log" in kwargs:
                    raise TypeError("show_log unsupported")
                if kwargs.get("lang") == "ch":
                    self.marker = "ch"
                    return
                raise TypeError("lang unsupported")

        monkeypatch.setattr(parser, "_require_paddleocr", lambda: FakePaddleOCR)

        first = parser._get_ocr(lang="ch")
        second = parser._get_ocr(lang="ch")

        assert first is second
        assert first.marker == "ch"
        assert calls == [
            {"lang": "ch", "show_log": False},
            {"lang": "ch"},
        ]

    def test_falls_back_to_empty_kwargs_and_raises_when_all_fail(self, monkeypatch):
        parser = PaddleOCRParser(default_lang="en")

        class AlwaysFail:
            def __init__(self, **kwargs):
                raise ValueError(f"bad kwargs: {kwargs}")

        monkeypatch.setattr(parser, "_require_paddleocr", lambda: AlwaysFail)

        with pytest.raises(RuntimeError, match="Unable to initialize PaddleOCR"):
            parser._get_ocr(lang="en")

    def test_blank_lang_uses_default(self, monkeypatch):
        parser = PaddleOCRParser(default_lang="en")
        seen = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                seen["kwargs"] = kwargs
                self.ok = True

        monkeypatch.setattr(parser, "_require_paddleocr", lambda: FakePaddleOCR)
        ocr = parser._get_ocr(lang="   ")
        assert ocr.ok is True
        assert "en" in parser._ocr_instances
        assert seen["kwargs"]["lang"] == "en"


class TestOcrInput:
    def test_ocr_cls_typeerror_falls_back_without_cls(self, monkeypatch):
        parser = PaddleOCRParser()
        calls = []

        class FakeOcr:
            def ocr(self, input_data, **kwargs):
                calls.append((input_data, kwargs))
                if "cls" in kwargs:
                    raise TypeError("unexpected keyword argument 'cls'")
                return [[["box", ("hello", 0.9)]]]

        monkeypatch.setattr(parser, "_get_ocr", lambda lang=None: FakeOcr())
        lines = parser._ocr_input("img.png", lang="en", cls_enabled=True)

        assert lines == ["hello"]
        assert calls == [
            ("img.png", {"cls": True}),
            ("img.png", {}),
        ]

    def test_uses_predict_when_ocr_method_absent(self, monkeypatch):
        parser = PaddleOCRParser()

        class PredictOnly:
            def predict(self, input_data):
                return [{"text": "from-predict"}]

        monkeypatch.setattr(parser, "_get_ocr", lambda lang=None: PredictOnly())
        assert parser._ocr_input("x.png") == ["from-predict"]

    def test_raises_when_neither_ocr_nor_predict(self, monkeypatch):
        parser = PaddleOCRParser()
        monkeypatch.setattr(parser, "_get_ocr", lambda lang=None: object())

        with pytest.raises(RuntimeError, match="Unsupported PaddleOCR API"):
            parser._ocr_input("x.png")


class TestOcrRenderedPage:
    def test_saves_pil_to_temp_and_deletes_even_on_ocr_failure(self, monkeypatch, tmp_path):
        parser = PaddleOCRParser()
        saved_paths = []

        class FakeImage:
            def save(self, path):
                path = Path(path)
                path.write_bytes(b"png")
                saved_paths.append(path)

        def boom(input_data, lang=None, cls_enabled=True):
            assert Path(input_data).exists()
            raise RuntimeError("ocr failed")

        monkeypatch.setattr(parser, "_ocr_input", boom)

        with pytest.raises(RuntimeError, match="ocr failed"):
            parser._ocr_rendered_page(FakeImage(), lang="en")

        assert len(saved_paths) == 1
        assert not saved_paths[0].exists()

    def test_non_pil_page_forwards_to_ocr_input(self, monkeypatch):
        parser = PaddleOCRParser()
        seen = {}

        def fake_ocr(input_data, lang=None, cls_enabled=True):
            seen["args"] = (input_data, lang, cls_enabled)
            return ["ok"]

        monkeypatch.setattr(parser, "_ocr_input", fake_ocr)
        page = SimpleNamespace(array=[1, 2, 3])  # no .save
        assert parser._ocr_rendered_page(page, lang="ch", cls_enabled=False) == ["ok"]
        assert seen["args"] == (page, "ch", False)
