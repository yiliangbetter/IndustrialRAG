"""Regression: MinerU must not return stale method subdirectory content."""

import json
import time
from pathlib import Path

from raganything.parser import MineruParser


def _write_method_output(stem_dir: Path, method: str, text: str) -> None:
    sub = stem_dir / method
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "paper_content_list.json").write_text(
        json.dumps([{"type": "text", "text": text, "page_idx": 0}]),
        encoding="utf-8",
    )
    (sub / "paper.md").write_text(text, encoding="utf-8")


def test_read_output_files_prefers_requested_method_over_stale_sibling(tmp_path):
    stem_dir = tmp_path / "paper"
    _write_method_output(stem_dir, "auto", "STALE AUTO CONTENT")
    time.sleep(0.02)
    _write_method_output(stem_dir, "ocr", "FRESH OCR CONTENT")

    content_list, _ = MineruParser._read_output_files(
        tmp_path, "paper", method="ocr"
    )

    assert len(content_list) == 1
    assert content_list[0]["text"] == "FRESH OCR CONTENT"


def test_read_output_files_prefers_requested_even_if_older_mtime(tmp_path):
    stem_dir = tmp_path / "paper"
    _write_method_output(stem_dir, "ocr", "REQUESTED OCR")
    time.sleep(0.02)
    # Newer sibling must not win when the requested method dir exists.
    _write_method_output(stem_dir, "auto", "NEWER AUTO")

    content_list, _ = MineruParser._read_output_files(
        tmp_path, "paper", method="ocr"
    )

    assert content_list[0]["text"] == "REQUESTED OCR"


def test_read_output_files_uses_newest_when_requested_missing(tmp_path):
    stem_dir = tmp_path / "paper"
    _write_method_output(stem_dir, "auto", "OLDER")
    time.sleep(0.02)
    _write_method_output(stem_dir, "vlm", "NEWEST")

    content_list, _ = MineruParser._read_output_files(
        tmp_path, "paper", method="hybrid_auto"
    )

    assert content_list[0]["text"] == "NEWEST"
