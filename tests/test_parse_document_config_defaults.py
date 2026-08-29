"""parse_document must apply config defaults when optional args are omitted.

Callers (process_document_complete, folder batch) pass None to inherit
parser_output_dir / parse_method. Silently using hardcoded auto/./output
would mix OCR and text extracts under the same cache key.
"""

import pytest

from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class RecordingParser:
    def __init__(self):
        self.pdf_calls = []

    def parse_pdf(self, pdf_path, output_dir="./output", method="auto", **kwargs):
        self.pdf_calls.append(
            {
                "pdf_path": pdf_path,
                "output_dir": output_dir,
                "method": method,
                "kwargs": kwargs,
            }
        )
        return [{"type": "text", "text": "parsed body"}]


def _processor(tmp_path, parse_method="ocr", output_dir=None):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": "mineru",
            "parse_method": parse_method,
            "parser_output_dir": str(output_dir or (tmp_path / "parsed")),
            "display_content_stats": False,
        },
    )()
    dummy.doc_parser = RecordingParser()
    return dummy


@pytest.mark.asyncio
async def test_none_parse_method_and_output_dir_use_config(tmp_path):
    output_dir = tmp_path / "mineru_out"
    dummy = _processor(tmp_path, parse_method="ocr", output_dir=output_dir)
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    content_list, doc_id = await dummy.parse_document(str(pdf))

    assert content_list == [{"type": "text", "text": "parsed body"}]
    assert doc_id.startswith("doc-")
    assert len(dummy.doc_parser.pdf_calls) == 1
    call = dummy.doc_parser.pdf_calls[0]
    assert call["method"] == "ocr"
    assert call["output_dir"] == str(output_dir)
    assert call["kwargs"] == {}


@pytest.mark.asyncio
async def test_explicit_args_override_config_and_forward_lang(tmp_path):
    dummy = _processor(tmp_path, parse_method="auto")
    pdf = tmp_path / "spec.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    override_dir = str(tmp_path / "override")

    await dummy.parse_document(
        str(pdf),
        output_dir=override_dir,
        parse_method="txt",
        lang="ch",
        start_page=2,
    )

    call = dummy.doc_parser.pdf_calls[0]
    assert call["method"] == "txt"
    assert call["output_dir"] == override_dir
    assert call["kwargs"]["lang"] == "ch"
    assert call["kwargs"]["start_page"] == 2
