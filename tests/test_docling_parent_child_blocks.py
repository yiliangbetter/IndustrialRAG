"""Docling containers: groups are not tables; pictures and tables keep children.

Docling section and list nodes are ``groups`` with ``$ref`` children. Unknown
block types fall through to a table, so emitting a group itself would index
every section as an empty table. A picture or table that has children must
keep the parent block and still harvest the child text.
"""

import base64
import importlib.util
from pathlib import Path


def _load_docling_parser():
    """Load parser.py without importing the raganything package."""
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_docling_groups", module_path
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


def test_nested_groups_are_not_emitted_as_tables(tmp_path):
    parser = DoclingParser()
    docling = {
        "groups": [
            {
                "label": "section",
                "children": [
                    {"$ref": "#/groups/1"},
                    {"$ref": "#/texts/0"},
                ],
            },
            {
                "label": "list",
                "children": [{"$ref": "#/texts/1"}],
            },
        ],
        "texts": [
            {"orig": "Nameplate data", "label": "paragraph"},
            {"orig": "F = ma", "label": "formula"},
        ],
    }

    blocks = parser.read_from_block_recursive(
        docling["groups"][0], "groups", tmp_path, 0, "0", docling
    )

    assert [block["type"] for block in blocks] == ["equation", "text"]
    assert blocks[0]["text"] == "F = ma"
    assert blocks[1]["text"] == "Nameplate data"
    assert not any(block["type"] == "table" for block in blocks)


def test_picture_with_children_keeps_image_and_child_text(tmp_path):
    parser = DoclingParser()
    png_b64 = base64.b64encode(TINY_PNG).decode()
    docling = {
        "pictures": [
            {
                "image": {"uri": f"data:image/png;base64,{png_b64}"},
                "caption": "Fig. 1",
                "footnote": "nameplate",
                "children": [{"$ref": "#/texts/0"}],
            }
        ],
        "texts": [{"orig": "Pressure vessel MAWP 1.6 MPa", "label": "caption"}],
    }

    blocks = parser.read_from_block_recursive(
        docling["pictures"][0], "pictures", tmp_path, 0, "4", docling
    )

    assert [block["type"] for block in blocks] == ["image", "text"]
    image_path = Path(blocks[0]["img_path"])
    assert image_path.name == "image_4.png"
    assert image_path.read_bytes() == TINY_PNG
    assert blocks[0]["image_caption"] == "Fig. 1"
    assert blocks[0]["image_footnote"] == "nameplate"
    assert blocks[1]["text"] == "Pressure vessel MAWP 1.6 MPa"


def test_table_with_children_keeps_table_body_and_child_text(tmp_path):
    parser = DoclingParser()
    docling = {
        "tables": [
            {
                "caption": "Bolt torque",
                "footnote": "N·m",
                "data": [["M12", "12"]],
                "children": [{"$ref": "#/texts/0"}],
            }
        ],
        "texts": [{"orig": "See note A", "label": "paragraph"}],
    }

    blocks = parser.read_from_block_recursive(
        docling["tables"][0], "tables", tmp_path, 0, "0", docling
    )

    assert [block["type"] for block in blocks] == ["table", "text"]
    assert blocks[0]["table_caption"] == "Bolt torque"
    assert blocks[0]["table_footnote"] == "N·m"
    assert blocks[0]["table_body"] == [["M12", "12"]]
    assert blocks[1]["text"] == "See note A"
