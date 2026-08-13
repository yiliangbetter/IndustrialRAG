"""Regression tests for content-based doc_id identity (D19).

insert_content_list and parse_document hash the content list when callers omit
doc_id (batch graph ingest does this). LightRAG then keys storage on that id,
and _process_multimodal_content returns immediately if multimodal_processed is
already set. Colliding hashes therefore drop the second document silently.
"""

import pytest

from raganything.base import DocStatus
from raganything.processor import ProcessorMixin


class _Proc(ProcessorMixin):
    pass


def _doc_id(content_list):
    return _Proc()._generate_content_based_doc_id(content_list)


def test_identical_content_lists_share_doc_id():
    content = [
        {"type": "text", "text": "The transfer function is:"},
        {
            "type": "equation",
            "text": "first-order plant",
            "latex": "G(s) = 1/(s+1)",
        },
    ]
    assert _doc_id(content) == _doc_id(content)


def test_equations_with_same_text_different_latex_do_not_collide():
    shared_text = {"type": "text", "text": "The transfer function is:"}
    plant_a = [
        shared_text,
        {
            "type": "equation",
            "text": "first-order plant",
            "latex": "G(s) = 1/(s+1)",
        },
    ]
    plant_b = [
        shared_text,
        {
            "type": "equation",
            "text": "first-order plant",
            "latex": "G(s) = 1/(s+2)",
        },
    ]
    assert _doc_id(plant_a) != _doc_id(plant_b)


def test_equation_latex_only_is_part_of_identity():
    eq_a = [{"type": "equation", "latex": "E = mc^2"}]
    eq_b = [{"type": "equation", "latex": "E = hf"}]
    assert _doc_id(eq_a) != _doc_id(eq_b)


def test_tables_with_same_body_different_captions_do_not_collide():
    body = "| Param | Value |\n| --- | --- |\n| Voltage | 24V |"
    model_a = [
        {"type": "text", "text": "Electrical specifications"},
        {
            "type": "table",
            "table_body": body,
            "table_caption": ["Table 1: Model A power board"],
        },
    ]
    model_b = [
        {"type": "text", "text": "Electrical specifications"},
        {
            "type": "table",
            "table_body": body,
            "table_caption": ["Table 1: Model B power board"],
        },
    ]
    assert _doc_id(model_a) != _doc_id(model_b)


def test_images_with_same_path_different_captions_do_not_collide():
    path = "/data/parsed/images/image_0.png"
    fig_a = [
        {
            "type": "image",
            "img_path": path,
            "image_caption": ["Figure 1: Pump assembly A"],
        }
    ]
    fig_b = [
        {
            "type": "image",
            "img_path": path,
            "image_caption": ["Figure 1: Pump assembly B"],
        }
    ]
    assert _doc_id(fig_a) != _doc_id(fig_b)


def test_image_footnotes_are_part_of_identity():
    path = "/data/parsed/images/schematic.png"
    note_a = [
        {
            "type": "image",
            "img_path": path,
            "image_caption": ["Schematic"],
            "image_footnote": ["Applies to 380V cabinets"],
        }
    ]
    note_b = [
        {
            "type": "image",
            "img_path": path,
            "image_caption": ["Schematic"],
            "image_footnote": ["Applies to 220V cabinets"],
        }
    ]
    assert _doc_id(note_a) != _doc_id(note_b)


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeDocStatusStorage:
    def __init__(self):
        self.records = {}

    async def get_by_id(self, key):
        return self.records.get(key)


@pytest.mark.asyncio
async def test_reused_doc_id_drops_second_documents_equations():
    """If two manuals share a doc_id, the second never reaches processors."""
    processor = _Proc()
    processor.logger = FakeLogger()
    processor.lightrag = type(
        "FakeLightRAG",
        (),
        {"doc_status": FakeDocStatusStorage()},
    )()
    processor.callback_manager = None

    async def fake_ensure():
        return {"success": True}

    processed = []

    async def fake_batch(*, multimodal_items, file_path, doc_id):
        processed.append((file_path, multimodal_items))

    async def fake_mark(doc_id):
        processor.lightrag.doc_status.records[doc_id] = {
            "status": DocStatus.PROCESSED,
            "multimodal_processed": True,
        }

    processor._ensure_lightrag_initialized = fake_ensure
    processor._process_multimodal_content_batch_type_aware = fake_batch
    processor._mark_multimodal_processing_complete = fake_mark

    eq_a = {"type": "equation", "text": "first-order plant", "latex": "G(s) = 1/(s+1)"}
    eq_b = {"type": "equation", "text": "first-order plant", "latex": "G(s) = 1/(s+2)"}
    collided_id = "doc-collided"

    await processor._process_multimodal_content([eq_a], "manual_a.json", collided_id)
    await processor._process_multimodal_content([eq_b], "manual_b.json", collided_id)

    assert processed == [("manual_a.json", [eq_a])]
