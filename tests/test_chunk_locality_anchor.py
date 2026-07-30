"""Phase 1–2: chunk_locality anchor + unified figure targets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from iqr_anchor import (  # noqa: E402
    _best_anchor_chunk,
    _chunk_anchor_structural_adjustment,
    _figure_doc_for_anchor_neighbor,
)
from iqr_figure_target import (  # noqa: E402
    _chunk_is_table_heavy,
    _refs_from_unified_figure_targets,
    _resolve_doc_with_order_index,
    _should_use_unified_figure_targets,
    extract_figure_targets,
)
from iqr_placement import build_inline_placements  # noqa: E402


def test_table_heavy_detects_appendix_table():
    blob = "附表2 设备保养规程表\n\n[Table]\n<tr><td>电控板</td></tr>" + ("x" * 200)
    assert _chunk_is_table_heavy(blob)


def test_structural_penalty_prefers_section_heading_over_table():
    table = "附表2\n\n[Table]\n" + "<tr><td>电控板元件检测</td></tr>" * 5
    heading = "3.2 电控板\n\n电控板检查：每半年保养一次"
    assert _chunk_anchor_structural_adjustment(
        table
    ) < _chunk_anchor_structural_adjustment(heading)


def test_best_anchor_chunk_prefers_section_heading():
    table_doc = {
        "id": "chunk-table",
        "chunk_order_index": 10,
        "content": "附表2 设备保养规程表\n\n[Table]\n"
        + "<tr><td>电控板元件检测</td><td>每半年</td></tr>" * 8,
    }
    heading_doc = {
        "id": "chunk-heading",
        "chunk_order_index": 11,
        "content": "3.2 电控板\n\n电控板检查：每半年保养一次",
    }
    fig_doc = {
        "id": "chunk-fig",
        "chunk_order_index": 12,
        "content": "步骤\n\n[图片]\n图片路径：/tmp/x.jpg\n页码：17",
    }
    manual = [table_doc, heading_doc, fig_doc]
    bullet = "**自动封边机**：电控板检查的保养周期为**每半年一次**。"
    anchor = _best_anchor_chunk(
        manual,
        bullet,
        cite_pool=[table_doc],
        query="这四种封边机的电控板的保养周期分别是多久",
    )
    assert anchor is not None
    assert anchor["id"] == "chunk-heading"


def test_figure_neighbor_uses_resolved_order_index():
    anchor = {
        "id": "chunk-anchor",
        "chunk_order_index": 4,
        "content": "3.2 电控板\n\n电控板检查：每半年保养一次",
    }
    fig = {
        "id": "chunk-fig",
        "chunk_order_index": 5,
        "content": "检查控制箱\n\n[图片]\n图片路径：/tmp/a.jpg\n页码：17",
    }
    manual = [anchor, fig]
    got, source = _figure_doc_for_anchor_neighbor(anchor, manual, window=4)
    assert source == "neighbor"
    assert got is not None
    assert got["id"] == "chunk-fig"


def test_refs_from_chunk_locality_listing_four_machines(monkeypatch):
    """Smoke: four machine bullets → four figures when KB has heading+neighbor split."""

    def fake_load(manual_hint: str) -> list[dict[str, Any]]:
        base = _normalize_key(manual_hint)
        heading = {
            "id": f"chunk-h-{base}",
            "chunk_order_index": 20,
            "file_path": f"{manual_hint}.pdf",
            "content": "3.2 电控板\n\n电控板检查：每半年保养一次",
        }
        fig = {
            "id": f"chunk-f-{base}",
            "chunk_order_index": 21,
            "file_path": f"{manual_hint}.pdf",
            "content": f"检查控制箱\n\n[图片]\n图片路径：/tmp/{base}.jpg\n页码：17",
        }
        return [heading, fig]

    def _normalize_key(h: str) -> str:
        return h.replace(" ", "")[:8]

    monkeypatch.setattr("iqr_anchor._load_manual_chunks_for_locality", fake_load)
    monkeypatch.setattr(
        "iqr_figure_target._load_manual_chunks_for_locality", fake_load
    )

    answer = """
* **高速智能封边机**：电控板**每季度一次** [1]。
* **高速自动封边机**：电控板**每半年一次** [2]。
* **自动封边机**：电控板**每半年一次** [3]。
* **双端封边机**：电控板**每半年一次** [4]。
"""
    refs, meta = _refs_from_unified_figure_targets(
        "四种封边机电控板保养周期",
        answer,
        retrieved_docs=[],
        cite_pool=[],
    )
    assert meta["picked"] == 4
    assert meta["mode"] == "unified_figure_targets"
    assert len(refs) == 4
    for row in meta["targets"]:
        if row.get("status") == "ok":
            assert row.get("anchor_idx") is not None


def test_build_inline_placements_keeps_all_machine_bullet_images():
    answer = """
* **高速智能封边机**：电控板检查的保养周期为**每季度一次**。
* **自动封边机**：电控板检查的保养周期为**每半年一次**。
* **高速自动封边机**：电控板检查的保养周期为**每半年一次**。
* **双端封边机**：电控板元件检测的保养周期为**每半年一次**。
"""
    images = [
        {"caption": "周期A", "source_key": "高速智能封边机维护保养手册"},
        {"caption": "周期B", "source_key": "自动封边机维护保养手册8-25"},
        {"caption": "周期C", "source_key": "高速自动封边机维护保养手册8-24"},
        {"caption": "周期D", "source_key": "双端封边机维护保养手册 新版8-24最终版"},
    ]
    placements = build_inline_placements(answer, images, query="四种封边机电控板周期")
    assert len(placements) >= 2
    assert len(images) == 4


def test_semantic_single_placement_does_not_drop_multi_machine_images():
    """Regression: one semantic match must not shrink a 4-image chunk_locality set."""
    answer = """
* **高速智能封边机**：电控板**每季度一次**。
* **自动封边机**：电控板**每半年一次**。
"""
    images = [
        {"caption": "保养周期：每季度一次", "source_key": "高速智能封边机维护保养手册"},
        {"caption": "fig2", "source_key": "自动封边机维护保养手册8-25"},
        {"caption": "fig3", "source_key": "高速自动封边机维护保养手册8-24"},
        {"caption": "fig4", "source_key": "双端封边机维护保养手册 新版8-24最终版"},
    ]
    build_inline_placements(
        answer,
        images,
        query="这四种封边机的电控板的保养周期分别是多久",
    )
    assert len(images) == 4


def test_best_anchor_chunk_rejects_cross_manual_cite_pool():
    from iqr_figure_target import _doc_matches_manual_hint
    from iqr_store import _load_manual_chunks_for_locality

    manual = _load_manual_chunks_for_locality("双端封边机")
    row = _resolve_doc_with_order_index(
        {"id": "chunk-67666d23e448adaa5734cddab58a02aa"}
    )
    if not row.get("file_path"):
        pytest.skip("KV store not available")
    line = "* **双端封边机**：电控板元件检测**每半年一次**。"
    wrong = _best_anchor_chunk(
        manual,
        line,
        cite_pool=[row],
        query="四种封边机电控板保养周期",
        manual_hint="双端封边机",
    )
    right = _best_anchor_chunk(
        manual,
        line,
        cite_pool=[],
        query="四种封边机电控板保养周期",
        manual_hint="双端封边机",
    )
    assert wrong is not None and right is not None
    assert wrong.get("id") != "chunk-67666d23e448adaa5734cddab58a02aa"
    assert _doc_matches_manual_hint(right, "双端封边机")


def test_resolve_doc_fills_order_index_from_kv(monkeypatch):
    store = {
        "chunk-abc": {
            "chunk_order_index": 42,
            "content": "3.2 电控板",
            "file_path": "manual.pdf",
        }
    }
    monkeypatch.setattr("iqr_store._kv_text_chunks_store", lambda: store)
    out = _resolve_doc_with_order_index({"id": "chunk-abc", "content": "3.2 电控板"})
    assert out.get("chunk_order_index") == 42


def test_unified_gate_multi_machine_bullet():
    query = "这四种封边机的电控板的保养周期分别是多久"
    answer = """
* **高速智能封边机**：电控板**每季度一次**。
* **自动封边机**：电控板**每半年一次**。
"""
    assert _should_use_unified_figure_targets(query, answer)


def test_unified_gate_component_listing():
    query = "所有机型中，各自哪些部件需要清除残胶"
    answer = """
**高速智能封边机**
1. **压带轮**残胶清理
2. **仿形靠模**残胶清理

**自动封边机**
- **激光发生器出光口**残胶须彻底清理
"""
    targets = extract_figure_targets(query, answer)
    assert len(targets) >= 2
    assert targets[0].kind == "machine_component_pair"
    assert _should_use_unified_figure_targets(query, answer)


def test_single_target_uses_unified_pipeline():
    """Single-target answers now go through the unified pipeline directly."""
    query = "高速智能封边机预铣刀保养步骤是什么"
    answer = (
        "预铣刀每季度检查一次。\n\n### References\n- [1] 高速智能封边机维护保养手册.pdf"
    )
    assert _should_use_unified_figure_targets(query, answer)
