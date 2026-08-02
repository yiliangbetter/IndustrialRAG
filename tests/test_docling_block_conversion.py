"""Regression tests for Docling JSON → MinerU block conversion.

Covers schema bridging in DoclingParser.read_from_block /
read_from_block_recursive / _read_output_files without invoking the Docling CLI.
"""

import base64
import json

from raganything.parser import DoclingParser


def _tiny_png_data_uri() -> str:
    # 1x1 PNG
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class TestReadFromBlock:
    def test_formula_label_maps_to_equation(self, tmp_path):
        parser = DoclingParser()
        block = {"label": "formula", "orig": "E=mc^2"}
        result = parser.read_from_block(block, "texts", tmp_path, cnt=10, num="0")
        assert result["type"] == "equation"
        assert result["text"] == "E=mc^2"
        assert result["page_idx"] == 1

    def test_plain_text_label_maps_to_text(self, tmp_path):
        parser = DoclingParser()
        block = {"label": "paragraph", "orig": "Hello world"}
        result = parser.read_from_block(block, "texts", tmp_path, cnt=0, num="0")
        assert result == {
            "type": "text",
            "text": "Hello world",
            "page_idx": 0,
        }

    def test_picture_writes_image_and_preserves_caption(self, tmp_path):
        parser = DoclingParser()
        block = {
            "image": {"uri": _tiny_png_data_uri()},
            "caption": "Fig 1",
            "footnote": "note",
        }
        result = parser.read_from_block(block, "pictures", tmp_path, cnt=0, num="3")
        assert result["type"] == "image"
        assert result["image_caption"] == "Fig 1"
        assert result["image_footnote"] == "note"
        assert result["img_path"].endswith("image_3.png")
        assert (tmp_path / "images" / "image_3.png").exists()
        assert (tmp_path / "images" / "image_3.png").stat().st_size > 0

    def test_picture_without_prefix_still_decodes(self, tmp_path):
        parser = DoclingParser()
        raw_b64 = _tiny_png_data_uri().split(",", 1)[1]
        block = {"image": {"uri": raw_b64}, "caption": "raw"}
        result = parser.read_from_block(block, "pictures", tmp_path, cnt=0, num="1")
        assert result["type"] == "image"
        assert (tmp_path / "images" / "image_1.png").exists()

    def test_picture_failure_falls_back_to_text(self, tmp_path):
        parser = DoclingParser()
        # Missing image URI forces the soft-fallback text block.
        block = {"caption": "broken"}
        result = parser.read_from_block(block, "pictures", tmp_path, cnt=0, num="9")
        assert result["type"] == "text"
        assert "Image processing failed" in result["text"]
        assert "broken" in result["text"]

    def test_table_block_maps_fields(self, tmp_path):
        parser = DoclingParser()
        block = {
            "caption": "Scores",
            "footnote": "fn",
            "data": [["a", "b"], ["1", "2"]],
        }
        result = parser.read_from_block(block, "tables", tmp_path, cnt=20, num="0")
        assert result["type"] == "table"
        assert result["table_caption"] == "Scores"
        assert result["table_footnote"] == "fn"
        assert result["table_body"] == [["a", "b"], ["1", "2"]]
        assert result["page_idx"] == 2


class TestReadFromBlockRecursive:
    def test_resolves_refs_and_skips_broken_ones(self, tmp_path):
        parser = DoclingParser()
        docling_content = {
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "bad-ref"},
                    {"$ref": "#/texts/99"},
                    {"$ref": "#/pictures/0"},
                ]
            },
            "texts": [
                {"label": "paragraph", "orig": "Intro"},
            ],
            "pictures": [
                {
                    "image": {"uri": _tiny_png_data_uri()},
                    "caption": "pic",
                    "footnote": "",
                }
            ],
        }
        result = parser.read_from_block_recursive(
            docling_content["body"],
            "body",
            tmp_path,
            cnt=0,
            num="0",
            docling_content=docling_content,
        )
        types = [item["type"] for item in result]
        assert "text" in types
        assert "image" in types
        assert all(item.get("text") != "" for item in result if item["type"] == "text")
        # Broken refs are skipped, not raised.
        assert len(result) == 2

    def test_group_container_skips_parent_emits_children(self, tmp_path):
        parser = DoclingParser()
        docling_content = {
            "groups": [
                {
                    "label": "section",
                    "orig": "Section title",
                    "children": [{"$ref": "#/texts/0"}],
                }
            ],
            "texts": [{"label": "paragraph", "orig": "Child text"}],
        }
        group = docling_content["groups"][0]
        result = parser.read_from_block_recursive(
            group,
            "groups",
            tmp_path,
            cnt=0,
            num="0",
            docling_content=docling_content,
        )
        # type in {"groups","body"}: container itself is not emitted; children are.
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Child text"


class TestReadOutputFiles:
    def test_converts_json_fixture_and_reads_markdown(self, tmp_path):
        parser = DoclingParser()
        file_stem = "sample"
        subdir = tmp_path / file_stem / "docling"
        subdir.mkdir(parents=True)
        (subdir / f"{file_stem}.md").write_text("# Title\n\nBody", encoding="utf-8")
        docling_json = {
            "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}]},
            "texts": [
                {"label": "paragraph", "orig": "Alpha"},
                {"label": "formula", "orig": "x^2"},
            ],
        }
        (subdir / f"{file_stem}.json").write_text(
            json.dumps(docling_json), encoding="utf-8"
        )

        content_list, md = parser._read_output_files(tmp_path, file_stem)
        assert md.startswith("# Title")
        assert [c["type"] for c in content_list] == ["text", "equation"]
        assert content_list[0]["text"] == "Alpha"
        assert content_list[1]["text"] == "x^2"

    def test_missing_json_returns_empty_list(self, tmp_path):
        parser = DoclingParser()
        file_stem = "empty"
        subdir = tmp_path / file_stem / "docling"
        subdir.mkdir(parents=True)
        (subdir / f"{file_stem}.md").write_text("only md", encoding="utf-8")
        content_list, md = parser._read_output_files(tmp_path, file_stem)
        assert content_list == []
        assert md == "only md"
