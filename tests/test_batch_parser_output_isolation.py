"""Critical BatchParser correctness: path collision, async kwargs, timeout harvest."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from raganything.batch_parser import BatchParser


def _make_parser(tmp_path: Path, parse_side_effect=None) -> BatchParser:
    bp = BatchParser(parser_type="mineru", skip_installation_check=True, max_workers=2)
    mock_parser = MagicMock()
    if parse_side_effect is None:

        def _parse(file_path, output_dir, method="auto", **kwargs):
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "marker.txt").write_text(Path(file_path).resolve().as_posix(), encoding="utf-8")
            return [{"type": "text", "text": Path(file_path).name}]

        mock_parser.parse_document.side_effect = _parse
    else:
        mock_parser.parse_document.side_effect = parse_side_effect
    mock_parser.OFFICE_FORMATS = set()
    mock_parser.IMAGE_FORMATS = set()
    mock_parser.TEXT_FORMATS = set()
    bp.parser = mock_parser
    return bp


def test_same_stem_files_use_distinct_output_dirs(tmp_path: Path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    file_a = dir_a / "manual.pdf"
    file_b = dir_b / "manual.pdf"
    file_a.write_bytes(b"%PDF-a")
    file_b.write_bytes(b"%PDF-b")

    out = tmp_path / "out"
    bp = _make_parser(tmp_path)
    result = bp.process_batch(
        [str(file_a), str(file_b)],
        str(out),
        parse_method="auto",
        recursive=False,
    )

    assert result.failed_files == []
    assert sorted(Path(p).name for p in result.successful_files) == [
        "manual.pdf",
        "manual.pdf",
    ]
    child_dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert len(child_dirs) == 2
    assert child_dirs[0] != child_dirs[1]
    assert all(name.startswith("manual_") for name in child_dirs)

    markers = {p.read_text(encoding="utf-8") for p in out.glob("*/marker.txt")}
    assert markers == {str(file_a.resolve()), str(file_b.resolve())}


@pytest.mark.asyncio
async def test_process_batch_async_forwards_parser_kwargs(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    out = tmp_path / "out"
    bp = _make_parser(tmp_path)

    result = await bp.process_batch_async(
        [str(pdf)],
        str(out),
        parse_method="auto",
        recursive=False,
        lang="ch",
        formula=True,
    )

    assert result.failed_files == []
    assert len(result.successful_files) == 1
    kwargs = bp.parser.parse_document.call_args.kwargs
    assert kwargs.get("lang") == "ch"
    assert kwargs.get("formula") is True


def test_timeout_harvests_already_completed_futures(tmp_path: Path):
    """Wall-clock as_completed timeout must not drop finished work."""
    files = []
    for i in range(3):
        p = tmp_path / f"doc{i}.pdf"
        p.write_bytes(b"%PDF")
        files.append(p)

    def slow_parse(file_path, output_dir, method="auto", **kwargs):
        name = Path(file_path).name
        # First two finish quickly; the third sleeps past the batch budget.
        if name == "doc2.pdf":
            time.sleep(2.0)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return [{"type": "text", "text": name}]

    bp = _make_parser(tmp_path, parse_side_effect=slow_parse)
    bp.max_workers = 2
    bp.timeout_per_file = 0.3  # batch_timeout = 0.3 * ceil(3/2) = 0.6s

    result = bp.process_batch(
        [str(p) for p in files],
        str(tmp_path / "out"),
        parse_method="auto",
        recursive=False,
    )

    # At least the quickly completed files must appear somewhere (success).
    accounted = set(result.successful_files) | set(result.failed_files)
    assert len(accounted) == 3
    assert len(result.successful_files) >= 1
    assert any(Path(p).name == "doc0.pdf" for p in result.successful_files) or any(
        Path(p).name == "doc1.pdf" for p in result.successful_files
    )
