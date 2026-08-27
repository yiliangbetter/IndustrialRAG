"""Regression tests for ProcessorMixin._get_file_reference.

Citation and LightRAG file_path identity depend on use_full_path. A basename
vs full-path swap looks like a successful ingest with broken citations.
"""

from raganything.processor import ProcessorMixin


def _processor(use_full_path: bool) -> ProcessorMixin:
    processor = ProcessorMixin.__new__(ProcessorMixin)
    processor.config = type("Config", (), {"use_full_path": use_full_path})()
    return processor


class TestGetFileReference:
    def test_basename_when_use_full_path_false(self):
        processor = _processor(False)
        assert processor._get_file_reference("/data/manuals/pump.pdf") == "pump.pdf"
        assert processor._get_file_reference("relative/dir/spec.docx") == "spec.docx"

    def test_preserves_path_string_when_use_full_path_true(self):
        processor = _processor(True)
        assert (
            processor._get_file_reference("/data/manuals/pump.pdf")
            == "/data/manuals/pump.pdf"
        )
        # Current contract: relative paths are not resolved.
        assert (
            processor._get_file_reference("relative/dir/spec.docx")
            == "relative/dir/spec.docx"
        )
