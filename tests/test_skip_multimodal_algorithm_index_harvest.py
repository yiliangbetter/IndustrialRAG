"""Regression: skip-multimodal must harvest MinerU v2 algorithm/index/equations.

D20 — official MinerU 3 ``*_content_list_v2.json`` emits ``type=algorithm``,
``type=index``, and ``type=equation_interline``. ``separate_content`` only keeps
``type=text``. Default graph ingest (``skip_multimodal_processing=True``) then
marks multimodal complete, so those blocks never enter LightRAG whenever any
paragraph/title/list text made the harvest stream non-empty.

Production scripts ``batch_ingest_content_lists_with_graph.py`` and
``batch_ingest_content_lists_local_hf.py`` ingest ``*_content_list_v2.json``.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from raganything.processor import ProcessorMixin
from raganything.utils import separate_content


V2_PROSE_AND_ALGORITHM: List[Dict[str, Any]] = [
    {
        "type": "paragraph",
        "content": {
            "paragraph_content": [
                {"type": "text", "content": "PID control strategy follows."}
            ]
        },
        "bbox": [10, 10, 90, 20],
    },
    {
        "type": "algorithm",
        "content": {
            "algorithm_caption": [{"type": "text", "content": "Algorithm 1 PID"}],
            "algorithm_content": [
                {"type": "text", "content": "1: e = sp - pv"},
                {"type": "text", "content": "2: u = Kp*e + Ki*integral"},
            ],
            "algorithm_footnote": [{"type": "text", "content": "discrete form"}],
        },
        "bbox": [10, 30, 90, 60],
    },
    {
        "type": "index",
        "content": {
            "list_items": [
                {
                    "prefix": "",
                    "item_content": [
                        {"type": "text", "content": "Pump P-100 .............. 12"}
                    ],
                },
                {
                    "item_content": [
                        {"type": "text", "content": "Valve V-201 .............. 18"}
                    ],
                },
            ]
        },
        "bbox": [10, 70, 90, 90],
    },
    {
        "type": "equation_interline",
        "content": {
            "math_content": "G(s)=K_p(1+1/(T_i s))",
            "math_type": "latex",
        },
        "bbox": [10, 92, 90, 98],
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
        self._ensure_lightrag_initialized = AsyncMock(return_value={"success": True})
        self._mark_multimodal_processing_complete = AsyncMock()
        self._process_multimodal_content = AsyncMock()
        self._insert_text_content_embedding_only = AsyncMock()


def test_separate_content_routes_v2_algorithm_index_to_multimodal():
    text, multimodal = separate_content(V2_PROSE_AND_ALGORITHM)
    assert text == ""
    assert {m["type"] for m in multimodal} == {
        "paragraph",
        "algorithm",
        "index",
        "equation_interline",
    }


def test_plaintext_harvest_includes_algorithm_index_equation():
    harvested = _Harness()._plaintext_from_mineru_blocks(V2_PROSE_AND_ALGORITHM)
    assert "PID control strategy follows." in harvested
    assert "Algorithm 1 PID" in harvested
    assert "e = sp - pv" in harvested
    assert "Kp*e + Ki*integral" in harvested
    assert "discrete form" in harvested
    assert "Pump P-100" in harvested
    assert "Valve V-201" in harvested
    assert "G(s)=K_p(1+1/(T_i s))" in harvested


def test_plaintext_harvest_algorithm_string_payload():
    harvested = _Harness()._plaintext_from_mineru_blocks(
        [
            {
                "type": "algorithm",
                "content": {
                    "algorithm_caption": "Algorithm 2 Search",
                    "algorithm_content": "1: function SEARCH(x)\n2: return x",
                },
            }
        ]
    )
    assert "Algorithm 2 Search" in harvested
    assert "function SEARCH" in harvested


@pytest.mark.asyncio
async def test_skip_multimodal_inserts_algorithm_index_equation(monkeypatch):
    harness = _Harness()
    inserted: Dict[str, Any] = {}

    async def fake_insert(lightrag, input=None, **kwargs):
        inserted["input"] = input

    monkeypatch.setattr("raganything.processor.insert_text_content", fake_insert)

    await harness.insert_content_list(
        V2_PROSE_AND_ALGORITHM,
        file_path="pid_manual.pdf",
        skip_multimodal_processing=True,
    )

    assert "PID control strategy follows." in inserted["input"]
    assert "Algorithm 1 PID" in inserted["input"]
    assert "e = sp - pv" in inserted["input"]
    assert "Pump P-100" in inserted["input"]
    assert "G(s)=K_p(1+1/(T_i s))" in inserted["input"]
    harness._mark_multimodal_processing_complete.assert_awaited()
    harness._process_multimodal_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_multimodal_path_still_processes_algorithm_items(monkeypatch):
    harness = _Harness()
    inserted: Dict[str, Any] = {}

    async def fake_insert(lightrag, input=None, **kwargs):
        inserted["input"] = input

    monkeypatch.setattr("raganything.processor.insert_text_content", fake_insert)

    await harness.insert_content_list(
        V2_PROSE_AND_ALGORITHM,
        file_path="pid_manual.pdf",
        skip_multimodal_processing=False,
    )

    # Pure v2 lists have no type=text, so harvest already fills the text stream.
    # Multimodal processors must still run on the original algorithm/index items.
    assert "PID control strategy follows." in inserted["input"]
    harness._process_multimodal_content.assert_awaited()
    mm_items = harness._process_multimodal_content.await_args.args[0]
    assert {item["type"] for item in mm_items} == {
        "paragraph",
        "algorithm",
        "index",
        "equation_interline",
    }
