"""Docling JSON-reference conversion into MinerU-shaped content blocks."""

import base64
from pathlib import Path

from raganything.parser import DoclingParser

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_formula_text_becomes_equation_block():
    parser = DoclingParser()
    result = parser.read_from_block(
        {"orig": "E = mc^2", "label": "formula"},
        "texts",
        Path("/tmp"),
        cnt=10,
        num="0",
    )
    assert result["type"] == "equation"
    assert result["text"] == "E = mc^2"
    assert result["page_idx"] == 1


def test_non_formula_text_preserves_orig():
    parser = DoclingParser()
    result = parser.read_from_block(
        {"orig": "Hello plant", "label": "paragraph"},
        "texts",
        Path("/tmp"),
        cnt=0,
        num="0",
    )
    assert result == {"type": "text", "text": "Hello plant", "page_idx": 0}


def test_picture_with_data_uri_writes_image(tmp_path):
    parser = DoclingParser()
    png_b64 = base64.b64encode(TINY_PNG).decode()
    result = parser.read_from_block(
        {
            "image": {"uri": f"data:image/png;base64,{png_b64}"},
            "caption": "Fig. 1",
            "footnote": "note",
        },
        "pictures",
        tmp_path,
        cnt=0,
        num="3",
    )
    assert result["type"] == "image"
    image_path = Path(result["img_path"])
    assert image_path.exists()
    assert image_path.name == "image_3.png"
    assert image_path.read_bytes() == TINY_PNG
    assert result["image_caption"] == "Fig. 1"
    assert result["image_footnote"] == "note"


def test_picture_without_data_uri_prefix_still_decodes(tmp_path):
    parser = DoclingParser()
    png_b64 = base64.b64encode(TINY_PNG).decode()
    result = parser.read_from_block(
        {"image": {"uri": png_b64}, "caption": "", "footnote": ""},
        "pictures",
        tmp_path,
        cnt=0,
        num="0",
    )
    assert result["type"] == "image"
    assert Path(result["img_path"]).read_bytes() == TINY_PNG


def test_corrupt_picture_falls_back_to_text(tmp_path):
    parser = DoclingParser()
    result = parser.read_from_block(
        {
            "image": {"uri": "data:image/png;base64,@@@not-valid-base64@@@"},
            "caption": "Broken figure",
        },
        "pictures",
        tmp_path,
        cnt=20,
        num="1",
    )
    assert result["type"] == "text"
    assert "Image processing failed" in result["text"]
    assert "Broken figure" in result["text"]
    assert result["page_idx"] == 2


def test_table_body_and_captions_are_preserved():
    parser = DoclingParser()
    result = parser.read_from_block(
        {
            "caption": "Table 1",
            "footnote": "units in mm",
            "data": [["A", "B"], ["1", "2"]],
        },
        "tables",
        Path("/tmp"),
        cnt=0,
        num="0",
    )
    assert result["type"] == "table"
    assert result["table_caption"] == "Table 1"
    assert result["table_footnote"] == "units in mm"
    assert result["table_body"] == [["A", "B"], ["1", "2"]]


def test_recursive_resolves_json_refs_and_skips_bad_ones(tmp_path):
    parser = DoclingParser()
    docling = {
        "body": {
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "not-a-ref"},
                {"$ref": "#/texts/99"},
                {"$ref": "#/texts/1"},
            ]
        },
        "texts": [
            {"orig": "Intro", "label": "paragraph"},
            {"orig": "x^2 + y^2", "label": "formula"},
        ],
    }

    blocks = parser.read_from_block_recursive(
        docling["body"], "body", tmp_path, 0, "0", docling
    )

    assert [b["type"] for b in blocks] == ["text", "equation"]
    assert blocks[0]["text"] == "Intro"
    assert blocks[1]["text"] == "x^2 + y^2"
