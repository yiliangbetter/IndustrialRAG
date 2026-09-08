"""MinerU parse_document must not treat parse method as image OCR language.

parse_image has no method argument and always runs MinerU with method=ocr.
If parse_document forwarded method as the next positional argument, it would
bind to lang and OCR plant photos with the wrong language (or crash). Open
PR #128 routes images to parse_image but does not pass method= on that path.
"""

from pathlib import Path

from raganything.parser import MineruParser


def test_parse_document_image_keeps_lang_and_drops_method(tmp_path, monkeypatch):
    parser = MineruParser()
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG\r\n")
    called = {}

    def fake_parse_image(file_path, output_dir, lang=None, **kwargs):
        called["path"] = Path(file_path)
        called["output_dir"] = output_dir
        called["lang"] = lang
        called["kwargs"] = kwargs
        return [{"type": "text", "text": "from-image"}]

    monkeypatch.setattr(parser, "parse_image", fake_parse_image)

    result = parser.parse_document(
        image,
        method="txt",
        output_dir=str(tmp_path / "out"),
        lang="ch",
        formula=False,
    )

    assert result == [{"type": "text", "text": "from-image"}]
    assert called["path"] == image
    assert called["lang"] == "ch"
    assert called["output_dir"] == str(tmp_path / "out")
    assert "method" not in called["kwargs"]
    assert called["kwargs"]["formula"] is False
