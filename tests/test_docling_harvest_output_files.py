"""Docling harvest path: JSON/MD layout and fail-open on unreadable artifacts.

parse_html / parse_pdf mock `_read_output_files`. This file covers the real
layout (`<output>/<stem>/docling/<stem>.json`) that BatchParser and ingest
rely on. A wrong subdirectory silently returns an empty content list.
"""

import json

from raganything.parser import DoclingParser


def _write_docling_artifacts(output_dir, stem, *, body=None, md=""):
    subdir = output_dir / stem / "docling"
    subdir.mkdir(parents=True, exist_ok=True)
    if body is not None:
        (subdir / f"{stem}.json").write_text(json.dumps(body), encoding="utf-8")
    if md:
        (subdir / f"{stem}.md").write_text(md, encoding="utf-8")
    return subdir


def test_reads_json_and_md_from_stem_docling_subdir(tmp_path):
    stem = "manual"
    _write_docling_artifacts(
        tmp_path,
        stem,
        body={
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                ]
            },
            "texts": [
                {"orig": "Plant overview", "label": "paragraph"},
                {"orig": "E = mc^2", "label": "formula"},
            ],
        },
        md="# Manual\n",
    )

    content_list, md = DoclingParser()._read_output_files(tmp_path, stem)

    assert md == "# Manual\n"
    assert [item["type"] for item in content_list] == ["text", "equation"]
    assert content_list[0]["text"] == "Plant overview"
    assert content_list[1]["text"] == "E = mc^2"


def test_missing_artifacts_return_empty_without_raising(tmp_path):
    content_list, md = DoclingParser()._read_output_files(tmp_path, "absent")

    assert content_list == []
    assert md == ""


def test_md_only_returns_markdown_and_empty_blocks(tmp_path):
    _write_docling_artifacts(tmp_path, "notes", md="just markdown")

    content_list, md = DoclingParser()._read_output_files(tmp_path, "notes")

    assert content_list == []
    assert md == "just markdown"


def test_corrupt_json_returns_empty_list_without_raising(tmp_path):
    stem = "broken"
    subdir = tmp_path / stem / "docling"
    subdir.mkdir(parents=True)
    (subdir / f"{stem}.json").write_text("{not-json", encoding="utf-8")
    (subdir / f"{stem}.md").write_text("still readable", encoding="utf-8")

    content_list, md = DoclingParser()._read_output_files(tmp_path, stem)

    assert content_list == []
    assert md == "still readable"


def test_sibling_json_outside_docling_subdir_is_ignored(tmp_path):
    stem = "report"
    (tmp_path / f"{stem}.json").write_text(
        json.dumps(
            {
                "body": {"children": [{"$ref": "#/texts/0"}]},
                "texts": [{"orig": "should be ignored", "label": "paragraph"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{stem}.md").write_text("ignored md", encoding="utf-8")

    content_list, md = DoclingParser()._read_output_files(tmp_path, stem)

    assert content_list == []
    assert md == ""
