"""Regression tests for BatchParser concurrency and per-file error isolation.

Distinct from open PR #66 (output-dir isolation / empty-content) and #69
(empty content_list false success). Focus: one file failure must not abort
siblings; dry-run / recursive discovery; async wrapper.
"""

from pathlib import Path

import pytest

from raganything.batch_parser import BatchParser, BatchProcessingResult


@pytest.fixture
def batch_parser():
    return BatchParser(
        parser_type="mineru",
        max_workers=2,
        show_progress=False,
        skip_installation_check=True,
        timeout_per_file=30,
    )


class TestFilterSupportedRecursive:
    def test_recursive_finds_nested_supported_files(self, batch_parser, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (tmp_path / "root.pdf").write_bytes(b"%PDF")
        (nested / "deep.png").write_bytes(b"\x89PNG")
        (nested / "skip.py").write_text("x = 1")

        found = batch_parser.filter_supported_files([str(tmp_path)], recursive=True)
        names = {Path(p).name for p in found}
        assert names == {"root.pdf", "deep.png"}

    def test_non_recursive_skips_nested(self, batch_parser, tmp_path):
        nested = tmp_path / "sub"
        nested.mkdir()
        (tmp_path / "root.pdf").write_bytes(b"%PDF")
        (nested / "deep.pdf").write_bytes(b"%PDF")

        found = batch_parser.filter_supported_files([str(tmp_path)], recursive=False)
        names = {Path(p).name for p in found}
        assert names == {"root.pdf"}


class TestProcessBatchDryRunAndEmpty:
    def test_dry_run_lists_files_without_calling_parser(self, batch_parser, tmp_path, monkeypatch):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF")
        out = tmp_path / "out"

        def boom(*args, **kwargs):
            raise AssertionError("parser must not run during dry_run")

        monkeypatch.setattr(batch_parser.parser, "parse_document", boom)

        result = batch_parser.process_batch(
            [str(pdf)], str(out), dry_run=True
        )
        assert isinstance(result, BatchProcessingResult)
        assert result.dry_run is True
        assert result.total_files == 1
        assert result.failed_files == []
        assert str(pdf) in result.successful_files

    def test_no_supported_files_returns_empty_result(self, batch_parser, tmp_path):
        junk = tmp_path / "x.py"
        junk.write_text("pass")
        result = batch_parser.process_batch([str(junk)], str(tmp_path / "out"))
        assert result.total_files == 0
        assert result.successful_files == []
        assert result.failed_files == []


class TestProcessBatchErrorIsolation:
    def test_one_failure_does_not_abort_siblings(self, batch_parser, tmp_path, monkeypatch):
        good = tmp_path / "good.pdf"
        bad = tmp_path / "bad.pdf"
        good.write_bytes(b"%PDF")
        bad.write_bytes(b"%PDF")
        out = tmp_path / "out"

        def parse_document(file_path, output_dir, method="auto", **kwargs):
            if Path(file_path).name == "bad.pdf":
                raise RuntimeError("simulated parse failure")
            return [{"type": "text", "text": "ok"}]

        monkeypatch.setattr(batch_parser.parser, "parse_document", parse_document)

        result = batch_parser.process_batch(
            [str(good), str(bad)], str(out), recursive=False
        )
        assert result.total_files == 2
        assert str(good) in result.successful_files
        assert str(bad) in result.failed_files
        assert "simulated parse failure" in result.errors[str(bad)]
        assert len(result.successful_files) == 1
        assert len(result.failed_files) == 1

    def test_all_success_with_parallel_workers(self, batch_parser, tmp_path, monkeypatch):
        files = []
        for i in range(4):
            p = tmp_path / f"doc{i}.pdf"
            p.write_bytes(b"%PDF")
            files.append(str(p))

        monkeypatch.setattr(
            batch_parser.parser,
            "parse_document",
            lambda *a, **k: [{"type": "text", "text": "x"}],
        )

        result = batch_parser.process_batch(files, str(tmp_path / "out"))
        assert result.failed_files == []
        assert set(result.successful_files) == set(files)
        assert result.success_rate == 100.0


class TestProcessBatchAsync:
    @pytest.mark.asyncio
    async def test_async_wrapper_matches_sync_isolation(
        self, batch_parser, tmp_path, monkeypatch
    ):
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF")
        b.write_bytes(b"%PDF")

        def parse_document(file_path, output_dir, method="auto", **kwargs):
            if Path(file_path).name == "b.pdf":
                raise ValueError("b failed")
            return [{"type": "text", "text": "ok"}]

        monkeypatch.setattr(batch_parser.parser, "parse_document", parse_document)

        result = await batch_parser.process_batch_async(
            [str(a), str(b)], str(tmp_path / "out")
        )
        assert str(a) in result.successful_files
        assert str(b) in result.failed_files
