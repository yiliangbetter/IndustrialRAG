"""Chunk-locality image selection for multi-manual listings (Q15-style)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from iqr_figure_target import (  # noqa: E402
    _figure_doc_for_anchor_neighbor,
    _heading_before_image_block,
    _refs_from_unified_figure_targets,
    _should_use_unified_figure_targets,
)
from iqr_protocol import extract_image_refs_from_context  # noqa: E402
from iqr_store import _dedupe_doc_list_by_chunk_identity  # noqa: E402


def _img_block(path: str, *, page: int = 20, context: str = "bullet text") -> str:
    return (
        f"{context}\n\n"
        f"[图片]\n"
        f"图片路径：images/{path}.jpg\n"
        f"页码：{page}\n"
        f"关联正文：{context}"
    )


def test_dedupe_keeps_same_text_different_manuals():
    text = "3.2 电控板\n\n电控板检查：每半年保养一次"
    docs = [
        {"id": "a", "file_path": "高速自动封边机.pdf", "content": text},
        {"id": "b", "file_path": "自动封边机.pdf", "content": text},
        {"id": "c", "file_path": "双端封边机.pdf", "content": text},
    ]
    out = _dedupe_doc_list_by_chunk_identity(docs)
    assert len(out) == 3


def test_uses_unified_figure_targets_for_q15_style():
    q = "这四种封边机的电控板的保养周期分别是多久"
    ans = (
        "* **高速智能封边机**：每季度一次。\n"
        "* **双端封边机**：每半年一次。\n"
        "### References\n- [1] a.pdf\n- [2] b.pdf"
    )
    assert _should_use_unified_figure_targets(q, ans)


def test_neighbor_picks_adjacent_figure_not_far_mega():
    anchor = {
        "id": "chunk-anchor",
        "file_path": "双端封边机维护保养手册.pdf",
        "chunk_order_index": 192,
        "content": "3.2 电控板\n\n电控板检查：每半年保养一次",
    }
    sibling = {
        "id": "chunk-sib",
        "file_path": "双端封边机维护保养手册.pdf",
        "chunk_order_index": 193,
        "content": _img_block("fdf72fc8p20", page=20, context="检查控制箱内部各元器件"),
    }
    far_pur = {
        "id": "chunk-pur",
        "file_path": "双端封边机维护保养手册.pdf",
        "chunk_order_index": 236,
        "content": _img_block("f79286453p25", page=25, context="清空胶锅，PUR 清胶剂"),
    }
    manual_chunks = [anchor, sibling, far_pur]
    picked, source = _figure_doc_for_anchor_neighbor(anchor, manual_chunks, window=8)
    assert picked is not None
    assert picked["id"] == "chunk-sib"
    assert source == "neighbor"
    assert "fdf72fc8p20" in picked["content"]


def test_locality_refs_skip_far_pur_when_anchor_text_only(monkeypatch):
    anchor = {
        "id": "chunk-anchor",
        "file_path": "双端封边机维护保养手册 新版8-24最终版.pdf",
        "chunk_order_index": 192,
        "content": "3.2 电控板\n\n电控板检查：每半年保养一次",
    }
    sibling = {
        "id": "chunk-sib",
        "file_path": "双端封边机维护保养手册 新版8-24最终版.pdf",
        "chunk_order_index": 193,
        "content": _img_block("fdf72fc8p20", page=20, context="保养内容： 检查电控板设备"),
    }
    pur = {
        "id": "chunk-pur",
        "file_path": "双端封边机维护保养手册 新版8-24最终版.pdf",
        "chunk_order_index": 236,
        "content": _img_block("f79286453p25", page=25, context="清空胶锅，PUR 清胶剂"),
    }
    smart = {
        "id": "chunk-smart",
        "file_path": "高速智能封边机维护保养手册.pdf",
        "chunk_order_index": 50,
        "content": (
            "3.14.5 电控板检查\n\n保养内容： 检查电控板设备\n\n"
            + _img_block("e06e7d6bsmart", page=34, context="保养内容： 检查电控板设备")
        ),
    }

    import iqr_anchor
    import iqr_figure_target

    def fake_load(hint: str) -> list[dict]:
        if "双端" in hint:
            return [anchor, sibling, pur]
        if "高速智能" in hint:
            return [smart]
        return []

    monkeypatch.setattr(iqr_anchor, "_load_manual_chunks_for_locality", fake_load)
    monkeypatch.setattr(
        iqr_figure_target, "_load_manual_chunks_for_locality", fake_load
    )

    answer = (
        "* **高速智能封边机**：电控板检查的保养周期为**每季度一次**。\n"
        "* **双端封边机**：电气系统中的电控板元件检测保养周期为**每半年一次**。\n"
        "### References\n"
        "- [1] 高速智能封边机维护保养手册.pdf\n"
        "- [2] 双端封边机维护保养手册 新版8-24最终版.pdf"
    )
    refs, meta = _refs_from_unified_figure_targets(
        "这四种封边机的电控板的保养周期分别是多久",
        answer,
        retrieved_docs=[smart, anchor],
        cite_pool=[smart, anchor],
    )
    paths = {Path(r["path"]).name for r in refs}
    assert any("fdf72fc8p20" in p for p in paths)
    assert not any("f79286453p25" in p for p in paths)
    assert any("e06e7d6bsmart" in p for p in paths)
    assert meta["picked"] == 2


def test_heading_before_image_block_stays_in_same_chunk():
    chunk_a = "3.14.5 电控板检查\n\n保养内容： 检查电控板设备"
    chunk_b = "清空胶锅 PUR\n\n" + _img_block("purimg", page=25, context="清空胶锅 PUR")
    merged = chunk_a + "\n\n" + chunk_b
    pos = merged.find("purimg")
    heading = _heading_before_image_block(merged, pos)
    assert heading != "3.14.5 电控板检查"
    assert heading == ""


def test_extract_no_cross_chunk_caption_on_pur():
    chunk = "清空胶锅 PUR\n\n" + _img_block("puronly", page=25, context="清空胶锅 PUR")
    refs = extract_image_refs_from_context(chunk)
    assert len(refs) == 1
    assert refs[0].get("caption") in ("", None) or "PUR" in str(refs[0].get("context"))
