"""Regression tests for BatchParser process_batch / process_single_file.

Concurrent folder ingest must record per-file success and failure independently.
Dry-run must list files without calling the parser.
"""

import pytest

from raganything.batch_parser import BatchParser


class StubParser:
    OFFICE_FORMATS = set()
    IMAGE_FORMATS = set()
    TEXT_FORMATS = set()

    def __init__(self, fail_names=None):
        self.fail_names = set(fail_names or [])
        self.calls = []

    def parse_document(self, file_path, output_dir, method, **kwargs):
        self.calls.append((file_path, output_dir, method, kwargs))
        name = file_path.rsplit("/", 1)[-1]
        if name in self.fail_names:
            raise RuntimeError(f"parse failed for {name}")
        return [{"type": "text", "text": name}]

    def check_installation(self):
        return True


def _parser(tmp_path, **kwargs) -> BatchParser:
    bp = BatchParser(
        parser_type="mineru",
        skip_installation_check=True,
        show_progress=False,
        max_workers=2,
        **kwargs,
    )
    return bp


class TestProcessSingleFile:
    def test_success_creates_output_dir(self, tmp_path):
        src = tmp_path / "doc.pdf"
        src.write_bytes(b"%PDF")
        out = tmp_path / "out"
        bp = _parser(tmp_path)
        stub = StubParser()
        bp.parser = stub

        success, path, error = bp.process_single_file(
            str(src), str(out), parse_method="ocr"
        )

        assert success is True
        assert path == str(src)
        assert error is None
        assert (out / "doc").is_dir()
        assert stub.calls[0][2] == "ocr"

    def test_parse_error_returns_failure_tuple(self, tmp_path):
        src = tmp_path / "bad.pdf"
        src.write_bytes(b"%PDF")
        bp = _parser(tmp_path)
        bp.parser = StubParser(fail_names={"bad.pdf"})

        success, path, error = bp.process_single_file(str(src), str(tmp_path / "out"))

        assert success is False
        assert path == str(src)
        assert "parse failed for bad.pdf" in error


class TestProcessBatch:
    def test_dry_run_lists_pdfs_without_parsing(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / "b.pdf").write_bytes(b"%PDF")
        (tmp_path / "skip.txt").write_text("nope")
        bp = _parser(tmp_path)
        stub = StubParser()
        bp.parser = stub

        result = bp.process_batch(
            [str(tmp_path)],
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            recursive=False,
        )

        assert result.dry_run is True
        names = {p.split("/")[-1] for p in result.successful_files}
        assert names == {"a.pdf", "b.pdf"}
        assert result.failed_files == []
        assert result.total_files == 2
        assert stub.calls == []

    def test_mixed_success_and_failure(self, tmp_path):
        (tmp_path / "ok.pdf").write_bytes(b"%PDF")
        (tmp_path / "bad.pdf").write_bytes(b"%PDF")
        bp = _parser(tmp_path)
        bp.parser = StubParser(fail_names={"bad.pdf"})

        result = bp.process_batch(
            [str(tmp_path)],
            output_dir=str(tmp_path / "out"),
            parse_method="auto",
            recursive=False,
        )

        success_names = {p.split("/")[-1] for p in result.successful_files}
        fail_names = {p.split("/")[-1] for p in result.failed_files}
        assert success_names == {"ok.pdf"}
        assert fail_names == {"bad.pdf"}
        assert result.total_files == 2
        assert any("parse failed for bad.pdf" in msg for msg in result.errors.values())

    def test_no_supported_files_returns_empty_result(self, tmp_path):
        (tmp_path / "notes.csv").write_text("a,b\n")
        bp = _parser(tmp_path)
        bp.parser = StubParser()

        result = bp.process_batch(
            [str(tmp_path)],
            output_dir=str(tmp_path / "out"),
            recursive=False,
        )

        assert result.total_files == 0
        assert result.successful_files == []
        assert result.failed_files == []

    @pytest.mark.asyncio
    async def test_process_batch_async_matches_sync_success(self, tmp_path):
        (tmp_path / "only.pdf").write_bytes(b"%PDF")
        bp = _parser(tmp_path)
        bp.parser = StubParser()

        result = await bp.process_batch_async(
            [str(tmp_path)],
            output_dir=str(tmp_path / "out"),
            recursive=False,
        )

        assert [p.split("/")[-1] for p in result.successful_files] == ["only.pdf"]
        assert result.failed_files == []
