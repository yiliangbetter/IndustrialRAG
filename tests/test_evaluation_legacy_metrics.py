"""Focused parity tests for dependency-free legacy evaluation scoring."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from raganything.evaluation_legacy import load_canonical_fixture
from raganything.evaluation_legacy_metrics import (
    LegacyScorerError,
    score_legacy_case,
)
from raganything.evaluation_schema import EvaluationCase, EvaluationTraceResult


def _case(dataset_id: str, case_id: str) -> EvaluationCase:
    dataset = load_canonical_fixture(dataset_id)
    return next(case for case in dataset.cases if case.case_id == case_id)


def test_shili_preserves_text_and_forbidden_image_grading() -> None:
    case = _case("shili17", "shili17:q01")

    passed = score_legacy_case(
        case,
        {"answer": "适用于 NB9-Smart 和 NB10-Smart。", "images": [], "placements": []},
    )
    wrong_answer = score_legacy_case(
        case,
        {"answer": "这是一个完全无关但非空的回答。", "images": [], "placements": []},
    )
    unexpected_image = score_legacy_case(
        case,
        {
            "answer": "适用于 NB9-Smart 和 NB10-Smart。",
            "images": [{"caption": "不应返回"}],
            "placements": [],
        },
    )

    assert passed.metrics == {
        "legacy_text_passed": True,
        "legacy_image_passed": True,
        "legacy_hard_gate_passed": True,
    }
    assert wrong_answer.metrics == {
        "legacy_text_passed": False,
        "legacy_image_passed": True,
        "legacy_hard_gate_passed": False,
    }
    assert wrong_answer.diagnostics["text"]["missing"] == [
        "missing:NB9-Smart",
        "missing:NB10-Smart",
    ]
    assert unexpected_image.metrics["legacy_hard_gate_passed"] is False
    assert unexpected_image.diagnostics["images"]["notes"] == ["unexpected_images:1"]


def test_shili_machine_cycles_remain_scoped_to_each_bullet() -> None:
    case = _case("shili17", "shili17:q15")
    answer = (
        "- 高速智能封边机：季度检查。\n"
        "- 自动封边机：每半年检查。\n"
        "- 双端封边机：每半年检查。\n"
        "- 高速自动封边机：每半年检查。"
    )
    images = [{"caption": "电控", "manual_id": f"manual-{index}"} for index in range(4)]
    trace = {
        "answer": answer,
        "images": images,
        "placements": [{"image_index": index} for index in range(4)],
    }

    score = score_legacy_case(case, trace)

    assert score.metrics["legacy_text_passed"] is True
    assert score.metrics["legacy_image_passed"] is True


def test_shili_gold_image_pairs_require_distinct_placed_provenance() -> None:
    case = _case("shili17", "shili17:q16")
    images = [
        {
            "manual_id": machine,
            "target_component": "气源三联件中的油杯",
            "caption": "解析器原始相邻文本",
        }
        for machine in ("自动封边机", "双端封边机", "高速自动封边机")
    ]

    passed = score_legacy_case(
        case,
        {
            "answer": "answer",
            "images": images,
            "placements": [{"image_index": index} for index in range(3)],
        },
    )
    failed = score_legacy_case(
        case,
        {
            "answer": "answer",
            "images": images,
            "placements": [
                {"image_index": 0},
                {"image_index": 1},
                {"image_index": 1},
            ],
        },
    )

    assert passed.metrics["legacy_image_passed"] is True
    assert failed.metrics["legacy_image_passed"] is False
    assert "placement_count:2<3" in failed.diagnostics["images"]["notes"]


def test_nanxing_preserves_markdown_regex_and_expected_source_checks() -> None:
    case = _case("nanxing-latency", "nanxing:machining-table")
    answer = "每年检查，低于 **9mm** 时更换，用5号内六角和M8螺钉。"
    trace = {
        "answer": answer,
        "context_documents": [
            {"file_path": ("D:\\manuals\\加工中心产品维护保养说明-20240510.pdf")}
        ],
    }

    passed = score_legacy_case(case, trace)
    failed = score_legacy_case(case, {**trace, "context_documents": []})

    assert passed.metrics == {
        "legacy_answer_passed": True,
        "legacy_source_passed": True,
        "legacy_hard_gate_passed": True,
    }
    assert failed.metrics["legacy_answer_passed"] is True
    assert failed.metrics["legacy_source_passed"] is False
    assert failed.diagnostics["sources"]["missing"] == [
        "加工中心产品维护保养说明-20240510.pdf"
    ]


def test_nanxing_requires_every_expected_source_from_sep_documents() -> None:
    case = _case("nanxing-latency", "nanxing:cross-electric-board")
    answer = "NB9-Smart 每季度检查变频器和电弧异味；NB8PCHGM 每半年检查变压器异常。"
    expected = case.metadata["expected_source_files"]
    trace = {
        "answer": answer,
        "context_documents": [{"file_path": "<SEP>".join(expected)}],
    }

    score = score_legacy_case(case, trace)

    assert score.metrics["legacy_answer_passed"] is True
    assert score.metrics["legacy_source_passed"] is True
    assert score.diagnostics["sources"]["observed"] == sorted(expected)


def test_voice_preserves_strict_response_mode_and_spacing() -> None:
    case = _case("voice29", "voice29:vs02")

    passed = score_legacy_case(
        case,
        {"answer": "建议清洗机头部分， 更换易损件，并检查胶轴。"},
    )
    failed = score_legacy_case(case, {"answer": "请检查胶轴。"})

    assert passed.metrics["legacy_hard_gate_passed"] is True
    assert passed.diagnostics["response"]["mode"] == "script"
    assert failed.metrics["legacy_response_passed"] is False
    assert failed.diagnostics["response"] == {
        "passed": False,
        "missing": ["缺：建议清洗机头部分，更换易损件"],
        "mode": "strict",
    }


def test_voice_accept_fallback_behavior_is_preserved() -> None:
    base = _case("voice29", "voice29:vs01")
    metadata = json.loads(json.dumps(base.metadata, ensure_ascii=False))
    metadata["legacy"]["grader_payloads"]["response"]["accept_fallback"] = True
    case = replace(base, metadata=metadata)

    score = score_legacy_case(case, {"answer": "请联系人工客服处理。"})

    assert score.metrics["legacy_response_passed"] is True
    assert score.diagnostics["response"]["mode"] == "fallback"


def test_metrics_fit_trace_schema_and_diagnostics_are_strict_json() -> None:
    case = _case("voice29", "voice29:vs01")
    score = score_legacy_case(case, {"answer": "检查各部位，进行调试。"})

    trace = EvaluationTraceResult(
        case_id=case.case_id,
        status="completed",
        observed_action="answer",
        answer="检查各部位，进行调试。",
        metrics=score.metrics,
    )

    assert trace.metrics["legacy_hard_gate_passed"] is True
    assert (
        json.loads(json.dumps(score.diagnostics, allow_nan=False))[
            "legacy_hard_gate_passed"
        ]
        is True
    )


def test_unknown_legacy_suite_is_rejected() -> None:
    base = _case("voice29", "voice29:vs01")
    metadata = json.loads(json.dumps(base.metadata, ensure_ascii=False))
    metadata["legacy"]["suite"] = "unknown-suite"

    with pytest.raises(LegacyScorerError, match="unsupported"):
        score_legacy_case(replace(base, metadata=metadata), {"answer": "answer"})
