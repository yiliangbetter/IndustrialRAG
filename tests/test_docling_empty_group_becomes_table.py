"""Docling groups with no children are indexed as tables.

``read_from_block_recursive`` treats a missing or empty ``children`` list as
a leaf. Group nodes are not texts or pictures, so that leaf is stored as a
table. A text leaf with the same shape stays text.
"""

import importlib.util
from pathlib import Path


def _load_docling_parser():
    """Load parser.py without importing the raganything package."""
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_docling_empty_groups", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DoclingParser


DoclingParser = _load_docling_parser()


def test_group_with_no_children_is_indexed_as_a_table(tmp_path):
    parser = DoclingParser()
    blocks_by_caption = {
        "Empty section": {"label": "section", "caption": "Empty section"},
        "Empty list": {
            "label": "list",
            "caption": "Empty list",
            "children": [],
        },
    }

    for caption, block in blocks_by_caption.items():
        blocks = parser.read_from_block_recursive(
            block, "groups", tmp_path, 0, "0", {"groups": [block]}
        )
        assert blocks == [
            {
                "type": "table",
                "img_path": "",
                "table_caption": caption,
                "table_footnote": "",
                "table_body": [],
                "page_idx": 0,
            }
        ]


def test_text_leaf_without_children_stays_text(tmp_path):
    parser = DoclingParser()
    block = {"orig": "Torque 12 N·m", "label": "paragraph"}

    blocks = parser.read_from_block_recursive(
        block, "texts", tmp_path, 0, "0", {"texts": [block]}
    )

    assert blocks == [
        {
            "type": "text",
            "text": "Torque 12 N·m",
            "page_idx": 0,
        }
    ]
