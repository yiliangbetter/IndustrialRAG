"""Regression tests for Docling → MinerU multimodal field normalization.

DoclingDocument JSON uses ``captions`` / ``footnotes`` RefItems and TableData
dicts. Emitting raw ``caption`` strings or TableData objects caused silent
caption loss and unusable table bodies; string captions also got character-
joined by downstream chunk builders.
"""

from __future__ import annotations

import base64

import pytest

from raganything.parser import DoclingParser
from raganything.processor import ProcessorMixin
from raganything.prompt import PROMPTS
from raganything.utils import join_text_field


TINY_PNG_B64 = base64.b64encode(
    (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
).decode("ascii")


@pytest.fixture
def docling_parser():
    return DoclingParser()


class TestJoinTextField:
    def test_string_not_character_split(self):
        assert join_text_field("Figure 1: Cooling loop") == "Figure 1: Cooling loop"

    def test_list_joined(self):
        assert join_text_field(["Fig 1", "alt"]) == "Fig 1, alt"

    def test_empty_values(self):
        assert join_text_field("") == ""
        assert join_text_field([]) == ""
        assert join_text_field(None) == ""


class TestDoclingCaptionAndTableNormalization:
    def test_picture_resolves_caption_refs(self, docling_parser, tmp_path):
        docling_content = {
            "texts": [
                {"text": "Fig. 1 Cooling loop overview", "label": "caption"},
                {"text": "Source: plant manual", "label": "footnote"},
            ]
        }
        block = {
            "image": {"uri": f"data:image/png;base64,{TINY_PNG_B64}"},
            "captions": [{"$ref": "#/texts/0"}],
            "footnotes": [{"$ref": "#/texts/1"}],
        }

        result = docling_parser.read_from_block(
            block, "pictures", tmp_path, cnt=1, num="0", docling_content=docling_content
        )

        assert result["type"] == "image"
        assert result["image_caption"] == ["Fig. 1 Cooling loop overview"]
        assert result["image_footnote"] == ["Source: plant manual"]

    def test_picture_legacy_string_caption_becomes_list(self, docling_parser, tmp_path):
        block = {
            "image": {"uri": f"data:image/png;base64,{TINY_PNG_B64}"},
            "caption": "Legacy string caption",
            "footnote": "Legacy footnote",
        }

        result = docling_parser.read_from_block(
            block, "pictures", tmp_path, cnt=1, num="1", docling_content={}
        )

        assert result["image_caption"] == ["Legacy string caption"]
        assert result["image_footnote"] == ["Legacy footnote"]

    def test_table_resolves_refs_and_formats_table_data(self, docling_parser, tmp_path):
        docling_content = {
            "texts": [{"text": "Scores", "label": "caption"}],
        }
        block = {
            "captions": [{"$ref": "#/texts/0"}],
            "footnotes": [],
            "data": {
                "num_rows": 2,
                "num_cols": 2,
                "table_cells": [
                    {
                        "text": "Metric",
                        "start_row_offset_idx": 0,
                        "end_row_offset_idx": 1,
                        "start_col_offset_idx": 0,
                        "end_col_offset_idx": 1,
                    },
                    {
                        "text": "Value",
                        "start_row_offset_idx": 0,
                        "end_row_offset_idx": 1,
                        "start_col_offset_idx": 1,
                        "end_col_offset_idx": 2,
                    },
                    {
                        "text": "Pressure",
                        "start_row_offset_idx": 1,
                        "end_row_offset_idx": 2,
                        "start_col_offset_idx": 0,
                        "end_col_offset_idx": 1,
                    },
                    {
                        "text": "42",
                        "start_row_offset_idx": 1,
                        "end_row_offset_idx": 2,
                        "start_col_offset_idx": 1,
                        "end_col_offset_idx": 2,
                    },
                ],
            },
        }

        result = docling_parser.read_from_block(
            block, "tables", tmp_path, cnt=2, num="0", docling_content=docling_content
        )

        assert result["type"] == "table"
        assert result["table_caption"] == ["Scores"]
        assert isinstance(result["table_body"], str)
        assert "Metric" in result["table_body"]
        assert "Pressure" in result["table_body"]
        assert "42" in result["table_body"]
        # Must not leave the raw TableData dict in the body
        assert "table_cells" not in result["table_body"]

    def test_recursive_passes_docling_content_for_captions(
        self, docling_parser, tmp_path
    ):
        docling_content = {
            "body": {"children": [{"$ref": "#/pictures/0"}]},
            "pictures": [
                {
                    "image": {"uri": f"data:image/png;base64,{TINY_PNG_B64}"},
                    "captions": [{"$ref": "#/texts/0"}],
                    "footnotes": [],
                }
            ],
            "texts": [{"text": "Resolved via recursive walk", "label": "caption"}],
        }

        results = docling_parser.read_from_block_recursive(
            docling_content["body"],
            "body",
            tmp_path,
            0,
            "0",
            docling_content,
        )

        assert len(results) == 1
        assert results[0]["image_caption"] == ["Resolved via recursive walk"]


class TestChunkTemplateDoesNotCorruptStringCaptions:
    def test_image_chunk_keeps_string_caption_intact(self):
        mixin = ProcessorMixin.__new__(ProcessorMixin)
        chunk = mixin._apply_chunk_template(
            "image",
            {
                "img_path": "/tmp/fig.png",
                "image_caption": "Figure 1: Cooling loop overview",
                "image_footnote": "Plant manual p.12",
            },
            "enhanced",
        )

        assert "Figure 1: Cooling loop overview" in chunk
        assert "F, i, g, u, r, e" not in chunk
        assert "Plant manual p.12" in chunk
        # Template still renders
        assert "{captions}" not in chunk
        assert PROMPTS["image_chunk"]
