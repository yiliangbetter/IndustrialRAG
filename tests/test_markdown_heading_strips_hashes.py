"""Markdown-to-PDF must keep heading text and drop the hash markers.

``.md`` files are converted to PDF before MinerU parsing. Heading lines are
stripped and styled separately from body lines. If the hashes stay in the
paragraph, or a hash-only line is emitted as body text, the indexed manual
heading is wrong.
"""

import pytest

from raganything.parser import Parser


def _spy_paragraphs(monkeypatch):
    from reportlab.platypus import Paragraph as RealParagraph

    seen = []

    class SpyParagraph(RealParagraph):
        def __init__(self, text, style, *args, **kwargs):
            seen.append(
                (
                    text,
                    getattr(style, "name", ""),
                    getattr(style, "fontSize", None),
                )
            )
            super().__init__(text, style, *args, **kwargs)

    monkeypatch.setattr("reportlab.platypus.Paragraph", SpyParagraph)
    return seen


def test_markdown_headings_drop_hashes_and_keep_body(tmp_path, monkeypatch):
    pytest.importorskip("reportlab")
    seen = _spy_paragraphs(monkeypatch)
    src = tmp_path / "manual.md"
    src.write_text(
        "# Torque spec\n"
        "\n"
        "Body line.\n"
        "## Detail\n"
        "###\n"
        "#### Deep\n"
        "  # Indented\n"
        "Use item #3\n",
        encoding="utf-8",
    )

    pdf_path = Parser.convert_text_to_pdf(src, output_dir=str(tmp_path / "out"))

    assert pdf_path.exists()
    assert pdf_path.stat().st_size >= 100
    assert [item[0] for item in seen] == [
        "Torque spec",
        "Body line.",
        "Detail",
        "Deep",
        "Indented",
        "Use item #3",
    ]
    assert seen[0][1] == "Heading1"
    assert seen[0][2] == 15
    assert seen[1][1] == "Normal"
    assert seen[2][1] == "Heading2"
    assert seen[2][2] == 14
    assert seen[3][1] == "Heading4"
    assert seen[3][2] == 12
    assert seen[4][1] == "Heading1"
    assert all(not text.startswith("#") for text, _, _ in seen)
