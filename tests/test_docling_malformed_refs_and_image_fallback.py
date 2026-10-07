"""Docling walks skip bad content pointers and survive corrupt pictures.

A single unresolvable ``$ref`` must not abort the walk, or
``_read_output_files`` drops the whole document. A picture whose payload
cannot be decoded becomes a text placeholder so later children are still
harvested. Exporters that omit the ``data:...;base64,`` prefix are decoded
as raw base64.
"""

import base64
import importlib.util
from pathlib import Path


def _load_docling_parser():
    """Load parser.py without importing the raganything package."""
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_docling_bad_refs", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DoclingParser


DoclingParser = _load_docling_parser()

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_unresolvable_refs_are_skipped_and_siblings_kept(tmp_path):
    parser = DoclingParser()
    docling = {
        "body": {
            "children": [
                {"$ref": "#/texts"},
                {"$ref": "texts/0"},
                {"$ref": "#/texts/not-an-index"},
                {"$ref": "#/missing/0"},
                {"$ref": "#/texts/9"},
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
            ]
        },
        "texts": [
            {"orig": "kept paragraph", "label": "paragraph"},
            {"orig": "E = mc^2", "label": "formula"},
        ],
    }

    blocks = parser.read_from_block_recursive(
        docling["body"], "body", tmp_path, 0, "0", docling
    )

    assert [block["type"] for block in blocks] == ["text", "equation"]
    assert blocks[0]["text"] == "kept paragraph"
    assert blocks[1]["text"] == "E = mc^2"
    assert blocks[1]["text_format"] == "unknown"


def test_picture_uri_without_data_prefix_is_decoded(tmp_path):
    parser = DoclingParser()
    raw_b64 = base64.b64encode(TINY_PNG).decode()
    assert "," not in raw_b64
    docling = {
        "pictures": [
            {
                "image": {"uri": raw_b64},
                "caption": "Nameplate",
                "footnote": "",
            }
        ]
    }

    blocks = parser.read_from_block_recursive(
        docling["pictures"][0], "pictures", tmp_path, 0, "2", docling
    )

    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    image_path = Path(blocks[0]["img_path"])
    assert image_path.name == "image_2.png"
    assert image_path.read_bytes() == TINY_PNG
    assert blocks[0]["image_caption"] == "Nameplate"


def test_corrupt_picture_becomes_text_and_keeps_child(tmp_path):
    parser = DoclingParser()
    docling = {
        "pictures": [
            {
                "image": {"uri": "data:image/png;base64,abc"},
                "caption": "Fig. 2",
                "children": [{"$ref": "#/texts/0"}],
            }
        ],
        "texts": [{"orig": "still harvested", "label": "paragraph"}],
    }

    blocks = parser.read_from_block_recursive(
        docling["pictures"][0], "pictures", tmp_path, 0, "1", docling
    )

    assert [block["type"] for block in blocks] == ["text", "text"]
    assert blocks[0]["text"] == "[Image processing failed: Fig. 2]"
    assert blocks[1]["text"] == "still harvested"
