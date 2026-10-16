"""Stored multimodal chunks must not revive an empty canonical caption.

``_apply_chunk_template`` prefers ``image_caption`` / ``image_footnote`` and
falls back to the legacy keys only when those fields are absent. An explicit
empty list is a cleared caption: joining it, or substituting ``img_caption``,
would persist a stale nameplate into the knowledge-graph chunk.
"""

import logging

from raganything.processor import ProcessorMixin


class _Host:
    logger = logging.getLogger("chunk-template-empty-caption")


def test_empty_image_caption_fields_do_not_use_legacy_aliases():
    text = ProcessorMixin._apply_chunk_template(
        _Host(),
        "image",
        {
            "img_path": "figures/nameplate.png",
            "image_caption": [],
            "img_caption": ["stale nameplate"],
            "image_footnote": [],
            "img_footnote": ["stale note"],
        },
        "visual description",
    )

    assert "Image Path: figures/nameplate.png" in text
    assert "Captions: None" in text
    assert "Footnotes: None" in text
    assert "Visual Analysis: visual description" in text
    assert "stale nameplate" not in text
    assert "stale note" not in text


def test_empty_table_caption_and_footnote_use_none_placeholder():
    text = ProcessorMixin._apply_chunk_template(
        _Host(),
        "table",
        {
            "img_path": "tables/torque.png",
            "table_caption": [],
            "table_body": "<table><tr><td>20 Nm</td></tr></table>",
            "table_footnote": [],
        },
        "torque table",
    )

    assert "Image Path: tables/torque.png" in text
    assert "Caption: None" in text
    assert "Structure: <table><tr><td>20 Nm</td></tr></table>" in text
    assert "Footnotes: None" in text
    assert "Analysis: torque table" in text
