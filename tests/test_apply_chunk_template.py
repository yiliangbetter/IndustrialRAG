"""Multimodal chunk templates used for LightRAG retrieval text.

If templates drop table bodies, image paths, or equations, retrieval and
citations lose the structured payload even when ingest appears successful.
"""

from raganything.processor import ProcessorMixin
from raganything.prompt import PROMPTS


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _processor():
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type("Config", (), {"use_full_path": False})()
    return dummy


def test_image_chunk_joins_captions_and_footnotes():
    dummy = _processor()
    text = dummy._apply_chunk_template(
        "image",
        {
            "img_path": "/abs/fig-2.png",
            "image_caption": ["Front view", "Pump housing"],
            "image_footnote": ["See §3.1"],
        },
        "housing bolts visible",
    )
    assert "Image Path: /abs/fig-2.png" in text
    assert "Captions: Front view, Pump housing" in text
    assert "Footnotes: See §3.1" in text
    assert "Visual Analysis: housing bolts visible" in text


def test_image_chunk_falls_back_to_img_caption_keys_and_none_placeholders():
    dummy = _processor()
    text = dummy._apply_chunk_template(
        "image",
        {"img_path": "fig.png", "img_caption": ["alt caption"]},
        "desc",
    )
    assert "Captions: alt caption" in text
    assert "Footnotes: None" in text


def test_table_chunk_preserves_body_caption_and_footnote():
    dummy = _processor()
    body = "| Torque | Value |\n| --- | --- |\n| M12 | 80Nm |"
    text = dummy._apply_chunk_template(
        "table",
        {
            "img_path": "table.png",
            "table_caption": ["Bolt torque"],
            "table_body": body,
            "table_footnote": ["Nm at 20C"],
        },
        "M12 requires 80Nm",
    )
    assert text.startswith("Table Analysis:")
    assert "Image Path: table.png" in text
    assert "Caption: Bolt torque" in text
    assert body in text
    assert "Footnotes: Nm at 20C" in text
    assert "Analysis: M12 requires 80Nm" in text


def test_equation_chunk_uses_text_and_format():
    dummy = _processor()
    text = dummy._apply_chunk_template(
        "equation",
        {"text": "E=mc^2", "text_format": "latex"},
        "mass-energy equivalence",
    )
    assert "Equation: E=mc^2" in text
    assert "Format: latex" in text
    assert "Mathematical Analysis: mass-energy equivalence" in text


def test_generic_chunk_stringifies_content_and_titles_type():
    dummy = _processor()
    text = dummy._apply_chunk_template(
        "chart",
        {"content": "series A vs B"},
        "upward trend",
    )
    expected = PROMPTS["generic_chunk"].format(
        content_type="Chart",
        content="series A vs B",
        enhanced_caption="upward trend",
    )
    assert text == expected


def test_template_error_falls_back_to_description():
    dummy = _processor()
    # Non-iterable captions make ", ".join(...) raise; template must not abort ingest.
    text = dummy._apply_chunk_template(
        "image",
        {"img_path": "x.png", "image_caption": 123},
        "fallback description",
    )
    assert text == "fallback description"
