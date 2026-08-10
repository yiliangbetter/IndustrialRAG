"""Regression: MinerU 3 list/code/chart must not be dropped on skip-multimodal.

D17 — open PR #81 harvests table/equation/image captions when multimodal is skipped,
but does not cover MinerU 3 legacy ``list`` / ``code`` / ``chart`` blocks (top-level
``list_items`` / ``code_body`` / ``chart_caption``). Production graph-batch scripts
default to ``skip_multimodal_processing=True``.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from raganything.processor import ProcessorMixin
from raganything.utils import separate_content


LEGACY_LIST_CODE_CHART: List[Dict[str, Any]] = [
    {"type": "text", "text": "Intro about the algorithm.", "page_idx": 0},
    {
        "type": "list",
        "sub_type": "text",
        "list_items": ["First requirement", "Second requirement"],
        "page_idx": 1,
    },
    {
        "type": "code",
        "sub_type": "algorithm",
        "code_caption": ["Algorithm 1 Search"],
        "code_body": "1: function SEARCH(x)\n2: return x",
        "page_idx": 2,
    },
    {
        "type": "chart",
        "img_path": "images/chart.png",
        "chart_caption": ["Figure 2 trend"],
        "content": "| Year | Value |\n| 2020 | 10 |",
        "page_idx": 3,
    },
]


class _Harness(ProcessorMixin):
    def __init__(self):
        self.logger = type(
            "L",
            (),
            {
                "info": lambda *a, **k: None,
                "debug": lambda *a, **k: None,
                "warning": lambda *a, **k: None,
            },
        )()
        self.config = type(
            "C",
            (),
            {
                "allow_embedding_only_ingestion": False,
                "content_format": "minerU",
                "use_full_path": False,
                "display_content_stats": False,
            },
        )()
        self.lightrag = object()
        self.callback_manager = None
        self._ensure_lightrag_initialized = AsyncMock(
            return_value={"success": True}
        )
        self._mark_multimodal_processing_complete = AsyncMock()
        self._process_multimodal_content = AsyncMock()
        self._insert_text_content_embedding_only = AsyncMock()

    def _get_file_reference(self, file_path: str) -> str:
        return file_path


def test_separate_content_routes_list_code_chart_to_multimodal():
    text, multimodal = separate_content(LEGACY_LIST_CODE_CHART)
    assert text == "Intro about the algorithm."
    assert {m["type"] for m in multimodal} == {"list", "code", "chart"}


def test_plaintext_harvest_includes_legacy_list_code_chart():
    harvested = _Harness()._plaintext_from_mineru_blocks(LEGACY_LIST_CODE_CHART)
    assert "Intro about the algorithm." in harvested
    assert "First requirement" in harvested
    assert "Second requirement" in harvested
    assert "Algorithm 1 Search" in harvested
    assert "function SEARCH" in harvested
    assert "Figure 2 trend" in harvested
    assert "Year" in harvested and "2020" in harvested


def test_string_list_items_and_v2_nested_list_both_work():
    harness = _Harness()
    legacy = harness._mineru_list_items_text(["alpha", "beta"])
    assert "alpha" in legacy and "beta" in legacy

    nested = harness._mineru_list_items_text(
        [
            {
                "prefix": "-",
                "item_content": [{"type": "text", "content": "nested item"}],
            }
        ]
    )
    assert "nested item" in nested


@pytest.mark.asyncio
async def test_skip_multimodal_inserts_list_code_chart(monkeypatch):
    harness = _Harness()
    inserted: Dict[str, Any] = {}

    async def fake_insert(lightrag, input=None, **kwargs):
        inserted["input"] = input

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", fake_insert
    )

    await harness.insert_content_list(
        LEGACY_LIST_CODE_CHART,
        file_path="manual.pdf",
        skip_multimodal_processing=True,
    )

    assert "Intro about the algorithm." in inserted["input"]
    assert "First requirement" in inserted["input"]
    assert "function SEARCH" in inserted["input"]
    assert "Figure 2 trend" in inserted["input"]
    harness._mark_multimodal_processing_complete.assert_awaited()
    harness._process_multimodal_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_multimodal_path_does_not_premerge_list_into_text(monkeypatch):
    harness = _Harness()
    inserted: Dict[str, Any] = {}

    async def fake_insert(lightrag, input=None, **kwargs):
        inserted["input"] = input

    monkeypatch.setattr(
        "raganything.processor.insert_text_content", fake_insert
    )

    await harness.insert_content_list(
        LEGACY_LIST_CODE_CHART,
        file_path="manual.pdf",
        skip_multimodal_processing=False,
    )

    assert inserted["input"] == "Intro about the algorithm."
    assert "First requirement" not in inserted["input"]
    assert "function SEARCH" not in inserted["input"]
    harness._process_multimodal_content.assert_awaited()


@pytest.mark.asyncio
async def test_embedding_only_merges_list_code_chart(monkeypatch):
    harness = _Harness()
    harness.config.allow_embedding_only_ingestion = True

    await harness.insert_content_list(
        LEGACY_LIST_CODE_CHART, file_path="manual.pdf"
    )

    harness._insert_text_content_embedding_only.assert_awaited()
    args = harness._insert_text_content_embedding_only.await_args
    text_content = args.kwargs.get("text_content") or args.args[0]
    assert "Intro about the algorithm." in text_content
    assert "First requirement" in text_content
    assert "function SEARCH" in text_content
    harness._process_multimodal_content.assert_not_awaited()
