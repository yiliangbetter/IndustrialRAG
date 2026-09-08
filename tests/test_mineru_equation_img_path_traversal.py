"""MinerU harvest must sandbox equation image paths.

Open PR #136 clears traversal in img_path / table_img_path and absolutizes
a *safe* equation_img_path. Equation figures still need the same fail-closed
clearing: a relative path that escapes the MinerU output dir must not be
promoted to an absolute path outside the harvest tree.
"""

import json

from raganything.parser import MineruParser


def test_path_traversal_in_equation_img_path_is_cleared(tmp_path):
    stem = "formula"
    nested = tmp_path / stem / "ocr"
    nested.mkdir(parents=True)
    safe = nested / "eq.png"
    safe.write_bytes(b"png")
    (nested / f"{stem}_content_list.json").write_text(
        json.dumps(
            [
                {
                    "type": "equation",
                    "equation_img_path": "../../secret.png",
                    "text": "E=mc^2",
                },
                {
                    "type": "equation",
                    "equation_img_path": "eq.png",
                    "text": "F=ma",
                },
            ]
        ),
        encoding="utf-8",
    )

    content_list, _ = MineruParser._read_output_files(tmp_path, stem, method="ocr")

    assert content_list[0]["equation_img_path"] == ""
    assert content_list[0]["text"] == "E=mc^2"
    assert content_list[1]["equation_img_path"] == str(safe.resolve())
    assert content_list[1]["text"] == "F=ma"
