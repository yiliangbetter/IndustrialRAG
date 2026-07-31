"""Regression: skip-multimodal / embedding-only must not drop table/equation text."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from raganything.processor import ProcessorMixin
from raganything.utils import separate_content


class _Harness(ProcessorMixin):
    def __init__(self, *, embedding_only: bool = False):
        self.logger = MagicMock()
        self.config = SimpleNamespace(
            display_content_stats=False,
            content_format="minerU",
            allow_embedding_only_ingestion=embedding_only,
        )
        self.lightrag = MagicMock()
        self.callback_manager = None
        self._ensure_lightrag_initialized = AsyncMock(
            return_value={"success": True}
        )
        self._mark_multimodal_processing_complete = AsyncMock()
        self._process_multimodal_content = AsyncMock()
        self._insert_text_content_embedding_only = AsyncMock()
        self._get_file_reference = lambda p: p
        self._generate_content_based_doc_id = lambda _items: "doc-test"


LEGACY_MIXED: List[Dict[str, Any]] = [
    {"type": "text", "text": "Intro paragraph about the machine.", "page_idx": 0},
    {
        "type": "table",
        "table_body": "| A | B |\n|---|---|\n| 1 | 2 |",
        "table_caption": ["Specs"],
        "page_idx": 1,
    },
    {
        "type": "equation",
        "latex": "E=mc^2",
        "text": "mass-energy equivalence",
        "page_idx": 2,
    },
    {
        "type": "image",
        "img_path": "/tmp/x.png",
        "image_caption": ["Figure 1 overview"],
        "page_idx": 3,
    },
]


def test_separate_content_leaves_tables_out_of_text_stream():
    text, multimodal = separate_content(LEGACY_MIXED)
    assert text == "Intro paragraph about the machine."
    assert {m["type"] for m in multimodal} == {"table", "equation", "image"}


def test_plaintext_harvest_includes_legacy_table_equation_caption():
    harvested = _Harness()._plaintext_from_mineru_blocks(LEGACY_MIXED)
    assert "Intro paragraph about the machine." in harvested
    assert "| A | B |" in harvested
    assert "Specs" in harvested
    assert "E=mc^2" in harvested or "mass-energy equivalence" in harvested
    assert "Figure 1 overview" in harvested


def test_plaintext_harvest_includes_v2_table_and_equation():
    v2 = [
        {
            "type": "paragraph",
            "content": {"paragraph_content": [{"text": "Intro v2"}]},
        },
        {"type": "table", "content": {"html": "<table><tr><td>X</td></tr></table>"}},
        {
            "type": "equation_interline",
            "content": {"math_content": [{"text": "a^2+b^2=c^2"}]},
        },
    ]
    harvested = _Harness()._plaintext_from_mineru_blocks(v2)
    assert "Intro v2" in harvested
    assert "<table>" in harvested
    assert "a^2+b^2=c^2" in harvested


@pytest.mark.asyncio
async def test_skip_multimodal_merges_table_text_into_lightrag_insert(monkeypatch):
    harness = _Harness()
    inserted: Dict[str, Any] = {}

    async def _fake_insert(lightrag, input, file_paths=None, **kwargs):
        inserted["input"] = input

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", _fake_insert
    )

    await harness.insert_content_list(
        LEGACY_MIXED,
        file_path="manual.pdf",
        skip_multimodal_processing=True,
    )

    assert "Intro paragraph about the machine." in inserted["input"]
    assert "| A | B |" in inserted["input"]
    assert "E=mc^2" in inserted["input"] or "mass-energy" in inserted["input"]
    harness._mark_multimodal_processing_complete.assert_awaited()
    harness._process_multimodal_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_only_merges_table_text(monkeypatch):
    harness = _Harness(embedding_only=True)

    await harness.insert_content_list(LEGACY_MIXED, file_path="manual.pdf")

    harness._insert_text_content_embedding_only.assert_awaited()
    args = harness._insert_text_content_embedding_only.await_args
    text_content = args.kwargs.get("text_content") or args.args[0]
    assert "Intro paragraph about the machine." in text_content
    assert "| A | B |" in text_content
    harness._process_multimodal_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_multimodal_path_does_not_premerge_table_into_text(monkeypatch):
    """When multimodal processors run, keep table bodies out of the text insert."""
    harness = _Harness()
    inserted: Dict[str, Any] = {}

    async def _fake_insert(lightrag, input, file_paths=None, **kwargs):
        inserted["input"] = input

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", _fake_insert
    )

    await harness.insert_content_list(
        LEGACY_MIXED,
        file_path="manual.pdf",
        skip_multimodal_processing=False,
    )

    assert inserted["input"] == "Intro paragraph about the machine."
    assert "| A | B |" not in inserted["input"]
    harness._process_multimodal_content.assert_awaited()
