"""BatchParser must honor parsers registered via register_parser.

Plant ingest often uses a Bring-Your-Own parser. If BatchParser only constructs
built-ins, registered parsers are silently rejected and batch jobs never reach
the custom parse_document implementation. Distinct from tests/test_custom_parser.py
(registry/get_parser/CLI) and from #128/#144 BatchMixin wrappers around builtins.
"""

import pytest

from raganything.batch_parser import BatchParser
from raganything.parser import (
    Parser,
    _CUSTOM_PARSERS,
    register_parser,
)


class RecordingParser(Parser):
    """Minimal custom parser that records parse_document calls."""

    calls = []

    def check_installation(self) -> bool:
        return True

    def parse_document(self, file_path, output_dir="./output", method="auto", **kw):
        self.calls.append(
            {
                "file_path": str(file_path),
                "output_dir": str(output_dir),
                "method": method,
                "kwargs": dict(kw),
            }
        )
        return [{"type": "text", "text": "custom-parsed", "page_idx": 0}]


@pytest.fixture(autouse=True)
def _clean_registry():
    _CUSTOM_PARSERS.clear()
    RecordingParser.calls = []
    yield
    _CUSTOM_PARSERS.clear()
    RecordingParser.calls = []


class TestBatchParserCustomParser:
    def test_constructs_registered_parser_instance(self):
        register_parser("marker", RecordingParser)
        bp = BatchParser(
            parser_type="marker",
            skip_installation_check=True,
            show_progress=False,
        )
        assert isinstance(bp.parser, RecordingParser)

    def test_process_single_file_forwards_method_and_kwargs(self, tmp_path):
        register_parser("marker", RecordingParser)
        bp = BatchParser(
            parser_type="marker",
            skip_installation_check=True,
            show_progress=False,
        )
        pdf = tmp_path / "manual.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        out = tmp_path / "out"

        ok, path, err = bp.process_single_file(
            str(pdf), str(out), parse_method="ocr", lang="ch"
        )

        assert ok is True
        assert err is None
        assert path == str(pdf)
        assert len(RecordingParser.calls) == 1
        call = RecordingParser.calls[0]
        assert call["method"] == "ocr"
        assert call["kwargs"]["lang"] == "ch"
        assert call["file_path"] == str(pdf)
        assert "manual" in call["output_dir"]

    def test_process_batch_uses_custom_parser(self, tmp_path):
        register_parser("marker", RecordingParser)
        bp = BatchParser(
            parser_type="marker",
            skip_installation_check=True,
            show_progress=False,
            max_workers=1,
        )
        pdf = tmp_path / "plant.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        out = tmp_path / "out"

        result = bp.process_batch([str(pdf)], str(out), parse_method="txt", lang="en")

        assert result.successful_files == [str(pdf)]
        assert result.failed_files == []
        assert len(RecordingParser.calls) == 1
        assert RecordingParser.calls[0]["method"] == "txt"
        assert RecordingParser.calls[0]["kwargs"]["lang"] == "en"
