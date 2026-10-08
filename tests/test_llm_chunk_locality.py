"""Query-side ±order_index neighbor expansion for LLM chunks (Q7-style siblings)."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from image_query_refs import (  # noqa: E402
    merge_order_neighbors_into_llm_chunks,
    supplement_unique_chunks_with_order_neighbors,
)


def _doc(
    chunk_id: str,
    idx: int,
    content: str,
    manual: str = "高速智能封边机维护保养手册.pdf",
):
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "chunk_order_index": idx,
        "file_path": manual,
        "content": content,
    }


def test_pre_rerank_adds_missing_neighbor(monkeypatch):
    anchor = _doc(
        "chunk-157e3049d904eeccd71f3e73feea68c9",
        49,
        "3.1.3 进料部分保养\n\n保养周期：每半年保养一次",
    )
    steps = _doc(
        "chunk-d82fdc07b77593f3a47b6437e647a148",
        50,
        "保养步骤：百分表吸在机架上，表针跳动＜0.15mm。",
    )

    def _fake_load(_manual: str):
        return [anchor, steps]

    monkeypatch.setattr("iqr_anchor._load_manual_chunks_for_locality", _fake_load)
    monkeypatch.setattr(
        "iqr_figure_target._load_manual_chunks_for_locality", _fake_load
    )
    monkeypatch.setenv("RAG_QUERY_CHUNK_LOCALITY", "1")

    expanded, meta = supplement_unique_chunks_with_order_neighbors(
        "对高速智能封边机的进料部分保养时，需要使用什么表？",
        [anchor],
    )
    ids = {d["chunk_id"] for d in expanded}
    assert meta["added"] == 1
    assert "chunk-d82fdc07b77593f3a47b6437e647a148" in ids


def test_post_merge_inserts_neighbor_after_anchor(monkeypatch):
    anchor = _doc(
        "chunk-157e3049d904eeccd71f3e73feea68c9",
        49,
        "3.1.3 进料部分保养\n\n保养周期：每半年保养一次",
    )
    steps = _doc(
        "chunk-d82fdc07b77593f3a47b6437e647a148",
        50,
        "保养步骤：百分表吸在机架上，表针跳动＜0.15mm。",
    )

    def _fake_load(_manual: str):
        return [anchor, steps]

    monkeypatch.setattr("iqr_anchor._load_manual_chunks_for_locality", _fake_load)
    monkeypatch.setattr(
        "iqr_figure_target._load_manual_chunks_for_locality", _fake_load
    )
    monkeypatch.setenv("RAG_QUERY_CHUNK_LOCALITY", "1")

    merged = merge_order_neighbors_into_llm_chunks(
        "对高速智能封边机的进料部分保养时，需要使用什么表？",
        [{**anchor, "id": "DC1"}],
        global_config={},
        query_param=None,
    )
    assert len(merged) == 2
    assert merged[0]["id"] == "DC1"
    assert merged[1]["id"] == "DC2"
    assert "百分表" in merged[1]["content"]


def test_skips_listing_queries(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_CHUNK_LOCALITY", "1")
    anchor = _doc("chunk-a", 1, "3.2 电控板")

    def _fail_load(_manual: str):
        raise AssertionError("should not load manual chunks for listing query")

    monkeypatch.setattr("iqr_anchor._load_manual_chunks_for_locality", _fail_load)
    monkeypatch.setattr(
        "iqr_figure_target._load_manual_chunks_for_locality", _fail_load
    )
    out, meta = supplement_unique_chunks_with_order_neighbors(
        "高速智能封边机的保养中，哪些部件需要清理残胶",
        [anchor],
    )
    assert out == [anchor]
    assert meta.get("reason") == "listing_or_catalog"
