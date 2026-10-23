"""Batch dry-run must list supported files and must not parse or create output."""

import pytest

from raganything.batch_parser import BatchParser


class _RaisingParser:
    """Parser stand-in that fails if a dry run actually parses."""

    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {
        ".png",
        ".jpeg",
        ".jpg",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".webp",
    }
    TEXT_FORMATS = {".txt", ".md"}

    def __init__(self):
        self.calls = []

    def check_installation(self) -> bool:
        return True

    def parse_document(self, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("dry-run must not parse")


def _batch(monkeypatch):
    parser = _RaisingParser()
    monkeypatch.setattr(
        "raganything.batch_parser.get_parser",
        lambda parser_type: parser,
    )
    batch = BatchParser(
        parser_type="mineru",
        show_progress=False,
        skip_installation_check=True,
        max_workers=2,
    )
    return batch, parser


def test_dry_run_lists_supported_files_without_parsing(monkeypatch, tmp_path):
    batch, parser = _batch(monkeypatch)
    manual = tmp_path / "manual.pdf"
    manual.write_bytes(b"%PDF-1.4")
    notes = tmp_path / "notes.log"
    notes.write_text("not a manual", encoding="utf-8")
    missing = tmp_path / "gone.pdf"
    output_dir = tmp_path / "out"

    result = batch.process_batch(
        [str(manual), str(notes), str(missing)],
        output_dir=str(output_dir),
        parse_method="ocr",
        recursive=False,
        dry_run=True,
    )

    assert result.successful_files == [str(manual)]
    assert result.failed_files == []
    assert result.errors == {}
    assert result.total_files == 1
    assert result.processing_time == 0.0
    assert result.dry_run is True
    assert result.output_dir == str(output_dir)
    assert result.success_rate == 100.0
    assert "Dry run: True" in result.summary()
    assert parser.calls == []
    assert not output_dir.exists()


def test_empty_selection_preserves_dry_run_flag(monkeypatch, tmp_path):
    batch, parser = _batch(monkeypatch)
    notes = tmp_path / "notes.log"
    notes.write_text("nope", encoding="utf-8")
    output_dir = tmp_path / "out"

    dry = batch.process_batch(
        [str(notes)],
        output_dir=str(output_dir),
        dry_run=True,
    )
    real = batch.process_batch(
        [str(notes)],
        output_dir=str(output_dir),
        dry_run=False,
    )

    assert dry.total_files == 0
    assert dry.successful_files == []
    assert dry.failed_files == []
    assert dry.dry_run is True
    assert dry.processing_time == 0.0
    assert real.total_files == 0
    assert real.dry_run is False
    assert real.processing_time == 0.0
    assert parser.calls == []
    assert not output_dir.exists()


@pytest.mark.asyncio
async def test_async_dry_run_forwards_flag(monkeypatch, tmp_path):
    batch, parser = _batch(monkeypatch)
    manual = tmp_path / "spec.pdf"
    manual.write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "async-out"

    result = await batch.process_batch_async(
        [str(manual)],
        output_dir=str(output_dir),
        parse_method="auto",
        recursive=True,
        dry_run=True,
    )

    assert result.successful_files == [str(manual)]
    assert result.failed_files == []
    assert result.errors == {}
    assert result.dry_run is True
    assert result.processing_time == 0.0
    assert parser.calls == []
    assert not output_dir.exists()
