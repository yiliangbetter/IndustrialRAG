"""Regression tests for ProcessorMixin HTML → Docling parse_html routing.

On main, parse_document bundled .html/.htm/.xhtml with Office formats and always
called parse_office_doc. DoclingParser.parse_office_doc rejects non-office
suffixes, so Docling HTML ingest failed closed despite parse_html existing.
"""

import pytest


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_dummy(processor_module, tmp_path, parser_name="docling"):
    class DummyProcessor(processor_module.ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": parser_name,
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
            "use_full_path": False,
        },
    )()
    dummy.logger = FakeLogger()
    dummy.parse_cache = None
    return DummyProcessor, dummy


def _stub_cache_and_doc_id(monkeypatch, DummyProcessor, doc_id="doc-html"):
    async def fake_store_cached_result(*args, **kwargs):
        return None

    monkeypatch.setattr(
        DummyProcessor,
        "_store_cached_result",
        fake_store_cached_result,
        raising=False,
    )
    monkeypatch.setattr(
        DummyProcessor,
        "_generate_content_based_doc_id",
        lambda self, content_list: doc_id,
        raising=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [".html", ".htm", ".xhtml"])
async def test_html_uses_parse_html_when_available(monkeypatch, tmp_path, suffix):
    import raganything.processor as processor_module

    calls = {"parse_html": 0, "parse_office_doc": 0}
    captured = {}

    class FakeDoclingParser:
        def parse_html(self, **kwargs):
            calls["parse_html"] += 1
            captured.update(kwargs)
            return [{"type": "text", "text": "html via parse_html", "page_idx": 0}]

        def parse_office_doc(self, **kwargs):
            calls["parse_office_doc"] += 1
            raise AssertionError("HTML must not call parse_office_doc for Docling")

    monkeypatch.setattr(
        processor_module,
        "get_parser",
        lambda parser_name: FakeDoclingParser(),
    )

    DummyProcessor, dummy = _make_dummy(processor_module, tmp_path, "docling")
    _stub_cache_and_doc_id(monkeypatch, DummyProcessor)

    html_file = tmp_path / f"sample{suffix}"
    html_file.write_text("<html><body><p>hello</p></body></html>", encoding="utf-8")

    content_list, doc_id = await dummy.parse_document(
        str(html_file), parse_method="auto", method="auto", lang="en"
    )

    assert calls["parse_html"] == 1
    assert calls["parse_office_doc"] == 0
    assert doc_id == "doc-html"
    assert content_list == [
        {"type": "text", "text": "html via parse_html", "page_idx": 0}
    ]
    assert captured["html_path"] == html_file
    assert captured["output_dir"] == str(tmp_path / "output")
    assert captured.get("lang") == "en"
    # parse_html does not accept MinerU-style method; strip it from kwargs.
    assert "method" not in captured


@pytest.mark.asyncio
async def test_office_still_uses_parse_office_doc_when_parse_html_exists(
    monkeypatch, tmp_path
):
    import raganything.processor as processor_module

    calls = {"parse_html": 0, "parse_office_doc": 0}
    captured = {}

    class FakeDoclingParser:
        def parse_html(self, **kwargs):
            calls["parse_html"] += 1
            raise AssertionError("Office docs must not call parse_html")

        def parse_office_doc(self, **kwargs):
            calls["parse_office_doc"] += 1
            captured.update(kwargs)
            return [{"type": "text", "text": "office via parse_office_doc", "page_idx": 0}]

    monkeypatch.setattr(
        processor_module,
        "get_parser",
        lambda parser_name: FakeDoclingParser(),
    )

    DummyProcessor, dummy = _make_dummy(processor_module, tmp_path, "docling")
    _stub_cache_and_doc_id(monkeypatch, DummyProcessor, doc_id="doc-office")

    docx = tmp_path / "sample.docx"
    docx.write_bytes(b"PK\x03\x04")

    content_list, doc_id = await dummy.parse_document(str(docx), parse_method="auto")

    assert calls["parse_office_doc"] == 1
    assert calls["parse_html"] == 0
    assert doc_id == "doc-office"
    assert content_list == [
        {"type": "text", "text": "office via parse_office_doc", "page_idx": 0}
    ]
    assert captured["doc_path"] == docx
    assert captured.get("method") == "auto"


@pytest.mark.asyncio
async def test_html_falls_back_to_parse_office_doc_without_parse_html(
    monkeypatch, tmp_path
):
    """MinerU/PaddleOCR lack parse_html; keep LibreOffice office-conversion path."""
    import raganything.processor as processor_module

    calls = {"parse_office_doc": 0}
    captured = {}

    class FakeMineruParser:
        def parse_office_doc(self, **kwargs):
            calls["parse_office_doc"] += 1
            captured.update(kwargs)
            return [{"type": "text", "text": "html via office fallback", "page_idx": 0}]

    monkeypatch.setattr(
        processor_module,
        "get_parser",
        lambda parser_name: FakeMineruParser(),
    )

    DummyProcessor, dummy = _make_dummy(processor_module, tmp_path, "mineru")
    _stub_cache_and_doc_id(monkeypatch, DummyProcessor, doc_id="doc-html-fallback")

    html_file = tmp_path / "sample.html"
    html_file.write_text("<html><body>fallback</body></html>", encoding="utf-8")

    content_list, doc_id = await dummy.parse_document(
        str(html_file), parse_method="ocr", method="should-be-stripped-from-kwargs"
    )

    assert calls["parse_office_doc"] == 1
    assert doc_id == "doc-html-fallback"
    assert content_list == [
        {"type": "text", "text": "html via office fallback", "page_idx": 0}
    ]
    assert captured["doc_path"] == html_file
    assert captured.get("method") == "ocr"
