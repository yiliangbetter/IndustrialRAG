"""Parity tests for additive migration of the existing evaluation fixtures."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from raganything.evaluation_legacy import (
    canonical_fixture_path,
    load_canonical_fixture,
    nanxing_latency_legacy_cases,
    ragas_shili17_cases,
    shili17_legacy_cases,
    shili17_legacy_ref,
    voice29_legacy_cases,
)
from raganything.evaluation_schema import ExpectedAction

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _shili_ground_truth() -> dict[int, str]:
    """Read the former Ragas source without importing the Web-path runner."""

    path = ROOT / "scripts" / "build_ragas_ground_truth_shili17.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "GROUND_TRUTH"
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict)
            return value
    raise AssertionError("GROUND_TRUTH assignment not found")


def _stable_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_shili17_canonical_fixture_preserves_every_legacy_case() -> None:
    legacy = _read_json(ROOT / "tests" / "fixtures" / "test_cases_shili17.json")
    dataset = load_canonical_fixture("shili17")

    assert dataset.dataset_version == "1.0.0"
    assert [case.case_id for case in dataset.cases] == [
        f"shili17:q{case_id:02d}" for case_id in range(1, 18)
    ]
    assert all(case.expected_action is ExpectedAction.ANSWER for case in dataset.cases)
    assert shili17_legacy_cases(dataset) == legacy["cases"]
    assert dataset.metadata["legacy_fixture_metadata"] == legacy["_meta"]

    expected_ref: dict[int, dict] = {}
    for row in legacy["cases"]:
        spec = {"query": row["query"]}
        spec.update(row["text"])
        spec.update(row["image"])
        expected_ref[int(row["id"])] = spec
    assert shili17_legacy_ref(dataset) == expected_ref


def test_shili17_canonical_reference_answers_preserve_ragas_builder() -> None:
    dataset = load_canonical_fixture("shili17")
    ground_truth = _shili_ground_truth()

    assert ragas_shili17_cases(dataset) == [
        {
            "id": case_id,
            "question": shili17_legacy_ref(dataset)[case_id]["query"],
            "ground_truth": ground_truth[case_id],
        }
        for case_id in range(1, 18)
    ]


def test_nanxing_canonical_fixture_preserves_every_legacy_case() -> None:
    legacy = _read_json(ROOT / "tests" / "fixtures" / "nanxing_latency_cases.json")
    dataset = load_canonical_fixture("nanxing_latency")

    assert dataset.dataset_id == "nanxing-latency"
    assert [case.case_id for case in dataset.cases] == [
        "nanxing:pc-011",
        "nanxing:machining-table",
        "nanxing:six-drill-magazine",
        "nanxing:cross-electric-board",
    ]
    assert nanxing_latency_legacy_cases(dataset) == legacy["cases"]


def test_voice29_is_fully_recoverable_without_git_or_source_xlsx() -> None:
    dataset = load_canonical_fixture("voice")
    rows = voice29_legacy_cases(dataset)

    assert len(rows) == 29
    assert [case.case_id for case in dataset.cases] == [
        f"voice29:vs{case_id:02d}" for case_id in range(1, 30)
    ]
    assert sum(len(row["utterances"]) for row in rows) == 59
    assert _stable_hash(rows) == (
        "ab9e2601af8ecc86f9f8e6838ba8dbb07464482e6d881e67bddcee52db03d81a"
    )
    assert dataset.metadata["recovered_from_git"]["sha256"] == (
        "8f697f55a0ff49e45c7b7c6bae18d41754a77b924ee0ad4284ac66fd327dded7"
    )

    green8_ids = set(dataset.metadata["green8_voice_ids"])
    assert green8_ids == {2, 4, 6, 7, 11, 14, 26, 28}
    assert [row["standard_question"] for row in rows if row["id"] in green8_ids] == [
        "漏胶",
        "仿形效果不好",
        "气压报警？",
        "靠板上限？",
        "三相电异常",
        "变频器异常报警",
        "未检测到工作（板材）",
        "未检测到工件",
    ]


def test_legacy_adapters_return_detached_payloads() -> None:
    first = shili17_legacy_cases()
    first[0]["text"]["text_all"].append("mutation")

    second = shili17_legacy_cases()
    assert "mutation" not in second[0]["text"]["text_all"]


def test_canonical_fixture_aliases_resolve_to_tracked_json() -> None:
    assert canonical_fixture_path("nanxing") == canonical_fixture_path(
        "nanxing-latency"
    )
    assert canonical_fixture_path("voice").name == "voice29.json"
    assert canonical_fixture_path("shili17").is_file()
