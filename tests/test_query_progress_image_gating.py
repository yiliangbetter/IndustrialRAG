from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import image_query_refs
import iqr_figure_target
import iqr_placement
import query_progress_hooks as hooks


def _set_image_context(monkeypatch, *, query: str) -> None:
    monkeypatch.setattr(hooks, "_docs_for_image_finalize", lambda _snap=None: [])
    hooks.set_query_media_roots([ROOT])
    hooks.set_query_text_for_images(query)


def test_finalize_suppresses_images_when_answer_abstains(monkeypatch):
    _set_image_context(monkeypatch, query="电控板如何保养？")

    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("refusal must stop before image resolution")

    monkeypatch.setattr(image_query_refs, "resolve_query_images", unexpected_resolve)
    result = hooks.finalize_inline_images(
        answer_text="根据当前知识库，没有足够的信息回答这个问题。"
    )

    assert result["images"] == []
    assert result["debug"]["gate"] == {
        "ok": False,
        "reason": "answer_abstained",
    }


def test_finalize_uses_selected_manual_refs_before_empty_short_circuit(
    monkeypatch,
):
    query = "两种封边机的电控板保养周期分别是多久？"
    answer = (
        "* **自动封边机**：电控板每半年检查一次。\n"
        "* **双端封边机**：电控板每半年检查一次。"
    )
    fallback_refs = [
        {
            "path": "/tmp/auto.jpg",
            "locality_machine": "自动封边机",
            "target_id": "target-1",
        },
        {
            "path": "/tmp/double.jpg",
            "locality_machine": "双端封边机",
            "target_id": "target-2",
        },
    ]
    _set_image_context(monkeypatch, query=query)
    monkeypatch.setattr(
        image_query_refs,
        "filter_docs_cited_by_answer",
        lambda *args, **kwargs: ([], {"mode": "answer_no_chunk_match"}),
    )
    monkeypatch.setattr(
        iqr_figure_target,
        "_manual_figure_refs_for_structured_targets",
        lambda *args, **kwargs: (
            fallback_refs,
            {
                "mode": "structured_manual_fallback",
                "picked": 2,
                "selection": {
                    "mode": "unified_figure_targets",
                    "picked": 2,
                    "targets": [],
                },
            },
        ),
    )
    monkeypatch.setattr(
        iqr_placement,
        "api_images_from_figure_refs",
        lambda refs, roots, **kwargs: [
            {
                "url": f"/media/{Path(ref['path']).name}",
                "caption": ref["locality_machine"],
                "target_id": ref["target_id"],
            }
            for ref in refs
        ],
    )

    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("successful structured fallback must not select twice")

    monkeypatch.setattr(image_query_refs, "resolve_query_images", unexpected_resolve)

    result = hooks.finalize_inline_images(answer_text=answer)

    assert [image["target_id"] for image in result["images"]] == [
        "target-1",
        "target-2",
    ]
    assert result["debug"]["gate"]["reason"] == "structured_manual_fallback"
    fallback_meta = result["debug"]["answer_citation"]["structured_manual_fallback"]
    assert fallback_meta["picked"] == 2


def test_public_ref_converter_keeps_seven_target_set(monkeypatch):
    refs = [{"path": f"/tmp/figure-{index}.jpg"} for index in range(7)]
    captured = []

    def fake_convert(selected, _roots):
        captured.extend(selected)
        return selected

    monkeypatch.setattr(iqr_placement, "_api_images_from_figure_refs", fake_convert)
    monkeypatch.setattr(iqr_placement, "_multi_figure_image_limit", lambda: 24)

    converted = iqr_placement.api_images_from_figure_refs(refs, [ROOT], limit=4)

    assert converted == refs
    assert captured == refs


def test_finalize_rejects_partially_resolved_structured_fallback(monkeypatch):
    query = "两种封边机的电控板保养周期分别是多久？"
    answer = (
        "* **自动封边机**：电控板每半年检查一次。\n"
        "* **双端封边机**：电控板每半年检查一次。"
    )
    fallback_refs = [
        {"path": "/tmp/auto.jpg", "target_id": "target-1"},
        {"path": "/tmp/missing.jpg", "target_id": "target-2"},
    ]
    _set_image_context(monkeypatch, query=query)
    monkeypatch.setattr(
        image_query_refs,
        "filter_docs_cited_by_answer",
        lambda *args, **kwargs: ([], {"mode": "answer_no_chunk_match"}),
    )
    monkeypatch.setattr(
        iqr_figure_target,
        "_manual_figure_refs_for_structured_targets",
        lambda *args, **kwargs: (
            fallback_refs,
            {
                "mode": "structured_manual_fallback",
                "picked": 2,
                "selection": {"picked": 2, "targets": []},
            },
        ),
    )
    monkeypatch.setattr(
        iqr_placement,
        "api_images_from_figure_refs",
        lambda *args, **kwargs: [
            {
                "url": "/media/auto.jpg",
                "caption": "自动封边机",
                "target_id": "target-1",
            }
        ],
    )

    result = hooks.finalize_inline_images(answer_text=answer)

    assert result["images"] == []
    assert result["debug"]["gate"]["reason"] == "no_answer_cited_chunks"
    fallback_meta = result["debug"]["answer_citation"]["structured_manual_fallback"]
    assert fallback_meta["reason"] == "incomplete_media_resolution"
    assert fallback_meta["resolved_count"] == 1
    assert fallback_meta["resolved_target_count"] == 1
