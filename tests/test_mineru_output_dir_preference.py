"""MinerU output-dir selection must prefer the requested parse method.

Pipeline writes {stem}/{method}/ (auto/, ocr/, txt/). A later OCR re-ingest
into the same unique output dir leaves the older auto/ tree in place. Scanning
subdirectories in filesystem order would return the stale parse.
"""

import json
from pathlib import Path

from raganything.parser import MineruParser


def _write_content_list(subdir: Path, stem: str, text: str) -> None:
    subdir.mkdir(parents=True)
    (subdir / f"{stem}_content_list.json").write_text(
        json.dumps([{"type": "text", "text": text, "page_idx": 0}]),
        encoding="utf-8",
    )


def test_read_output_files_prefers_requested_method_over_older_sibling(tmp_path):
    stem = "manual"
    stem_dir = tmp_path / stem
    _write_content_list(stem_dir / "auto", stem, "stale auto parse")
    _write_content_list(stem_dir / "ocr", stem, "ocr recovered text")

    content_list, _ = MineruParser._read_output_files(
        tmp_path, stem, method="ocr"
    )

    assert len(content_list) == 1
    assert content_list[0]["text"] == "ocr recovered text"


def test_read_output_files_still_uses_matching_auto_dir(tmp_path):
    stem = "manual"
    stem_dir = tmp_path / stem
    _write_content_list(stem_dir / "auto", stem, "auto parse")
    _write_content_list(stem_dir / "ocr", stem, "ocr parse")

    content_list, _ = MineruParser._read_output_files(
        tmp_path, stem, method="auto"
    )

    assert content_list[0]["text"] == "auto parse"


def test_read_output_files_scans_when_method_dir_missing(tmp_path):
    stem = "manual"
    _write_content_list(tmp_path / stem / "hybrid_auto", stem, "hybrid parse")

    content_list, _ = MineruParser._read_output_files(
        tmp_path, stem, method="auto"
    )

    assert content_list[0]["text"] == "hybrid parse"
