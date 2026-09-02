"""BatchParser.process_batch must forward extra parser kwargs to each file.

The sync ThreadPoolExecutor path accepts kwargs (unlike process_batch_async
extra-kwargs TypeError). Dropping lang/device here OCR's every plant PDF with
the parser default and mixes languages in one index.
"""

from raganything.batch_parser import BatchParser


class StubParser:
    OFFICE_FORMATS = set()
    IMAGE_FORMATS = set()
    TEXT_FORMATS = set()

    def __init__(self):
        self.calls = []

    def parse_document(self, file_path, output_dir, method, **kwargs):
        self.calls.append(
            {
                "file_path": file_path,
                "output_dir": output_dir,
                "method": method,
                "kwargs": kwargs,
            }
        )
        return [{"type": "text", "text": "ok"}]

    def check_installation(self):
        return True


def test_process_batch_forwards_lang_and_device_to_each_file(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    bp = BatchParser(
        parser_type="mineru",
        skip_installation_check=True,
        show_progress=False,
        max_workers=2,
    )
    stub = StubParser()
    bp.parser = stub

    result = bp.process_batch(
        [str(tmp_path)],
        output_dir=str(tmp_path / "out"),
        parse_method="ocr",
        recursive=False,
        lang="ch",
        device="cpu",
    )

    assert result.total_files == 2
    assert result.failed_files == []
    assert len(stub.calls) == 2
    names = {call["file_path"].rsplit("/", 1)[-1] for call in stub.calls}
    assert names == {"a.pdf", "b.pdf"}
    for call in stub.calls:
        assert call["method"] == "ocr"
        assert call["kwargs"]["lang"] == "ch"
        assert call["kwargs"]["device"] == "cpu"


def test_process_batch_without_extra_kwargs_still_parses(tmp_path):
    (tmp_path / "only.pdf").write_bytes(b"%PDF")
    bp = BatchParser(
        parser_type="mineru",
        skip_installation_check=True,
        show_progress=False,
    )
    stub = StubParser()
    bp.parser = stub

    result = bp.process_batch(
        [str(tmp_path)],
        output_dir=str(tmp_path / "out"),
        recursive=False,
    )

    assert result.successful_files
    assert stub.calls[0]["kwargs"] == {}
