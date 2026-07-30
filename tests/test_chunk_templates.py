"""Regression tests for multimodal chunk templating and file references.

Chunk text is what gets embedded and retrieved. Wrong templates, caption-alias
misses, or basename/full-path citation flips change retrieval identity and
citations across every multimodal ingest path.
"""

from types import SimpleNamespace

from raganything.processor import ProcessorMixin


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def info(self, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg))

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeTokenizer:
    def encode(self, text):
        return list(range(max(1, len(text) // 4)))


def _make_processor(use_full_path=False):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = SimpleNamespace(use_full_path=use_full_path)
    processor.lightrag = SimpleNamespace(tokenizer=FakeTokenizer())
    return processor


def test_get_file_reference_basename_vs_full_path():
    basename_proc = _make_processor(use_full_path=False)
    full_proc = _make_processor(use_full_path=True)
    path = "/data/docs/manual.pdf"

    assert basename_proc._get_file_reference(path) == "manual.pdf"
    assert full_proc._get_file_reference(path) == path


def test_apply_chunk_template_image_uses_caption_aliases():
    processor = _make_processor()
    content = processor._apply_chunk_template(
        "image",
        {
            "img_path": "/abs/fig.png",
            "img_caption": ["legacy caption"],
            "img_footnote": ["legacy note"],
        },
        "vision says gears",
    )

    assert "Image Path: /abs/fig.png" in content
    assert "legacy caption" in content
    assert "legacy note" in content
    assert "vision says gears" in content


def test_apply_chunk_template_image_prefers_canonical_caption_keys():
    processor = _make_processor()
    content = processor._apply_chunk_template(
        "image",
        {
            "img_path": "/abs/fig.png",
            "image_caption": ["canonical"],
            "img_caption": ["legacy"],
            "image_footnote": ["canon-note"],
            "img_footnote": ["legacy-note"],
        },
        "desc",
    )

    assert "canonical" in content
    assert "canon-note" in content
    assert "legacy" not in content
    assert "legacy-note" not in content


def test_apply_chunk_template_table_equation_and_generic():
    processor = _make_processor()

    table = processor._apply_chunk_template(
        "table",
        {
            "img_path": "/t.png",
            "table_caption": ["Specs"],
            "table_body": "| A | B |\n| 1 | 2 |",
            "table_footnote": ["units mm"],
        },
        "table insight",
    )
    assert "Specs" in table
    assert "| A | B |" in table
    assert "units mm" in table
    assert "table insight" in table

    equation = processor._apply_chunk_template(
        "equation",
        {"text": "E=mc^2", "text_format": "latex"},
        "mass-energy",
    )
    assert "E=mc^2" in equation
    assert "latex" in equation
    assert "mass-energy" in equation

    generic = processor._apply_chunk_template(
        "audio",
        {"content": "beep sequence"},
        "audio insight",
    )
    assert "Audio Content Analysis" in generic
    assert "beep sequence" in generic
    assert "audio insight" in generic


def test_apply_chunk_template_falls_back_to_description_on_format_error(monkeypatch):
    processor = _make_processor()

    class BrokenPrompts(dict):
        def __getitem__(self, key):
            template = object()

            class Broken:
                def format(self, **kwargs):
                    raise KeyError("missing field")

            return Broken()

    import raganything.prompt as prompt_module

    monkeypatch.setattr(prompt_module, "PROMPTS", BrokenPrompts())

    result = processor._apply_chunk_template(
        "image",
        {"img_path": "/x.png"},
        "fallback description only",
    )

    assert result == "fallback description only"
    assert any("Error applying chunk template" in w for w in processor.logger.warnings)


def test_convert_to_lightrag_chunks_uses_template_and_file_ref():
    processor = _make_processor(use_full_path=False)
    multimodal_data = [
        {
            "description": "gear diagram",
            "entity_info": {"entity_name": "Figure1"},
            "chunk_order_index": 3,
            "content_type": "image",
            "original_item": {
                "img_path": "/docs/fig.png",
                "image_caption": ["front view"],
                "image_footnote": [],
            },
            "item_info": {"page_idx": 2},
        }
    ]

    chunks = processor._convert_to_lightrag_chunks_type_aware(
        multimodal_data,
        file_path="/workspace/uploads/manual.pdf",
        doc_id="doc-123",
    )

    assert len(chunks) == 1
    chunk = next(iter(chunks.values()))
    assert chunk["full_doc_id"] == "doc-123"
    assert chunk["file_path"] == "manual.pdf"
    assert chunk["chunk_order_index"] == 3
    assert chunk["is_multimodal"] is True
    assert chunk["modal_entity_name"] == "Figure1"
    assert chunk["original_type"] == "image"
    assert chunk["page_idx"] == 2
    assert "front view" in chunk["content"]
    assert "gear diagram" in chunk["content"]
    assert chunk["tokens"] >= 1
