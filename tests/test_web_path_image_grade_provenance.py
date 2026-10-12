"""Image benchmark grading must use gold targets and structured provenance."""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
spec = importlib.util.spec_from_file_location(
    "run_web_path_q1_17_image_grade", _SCRIPTS / "run_web_path_q1_17.py"
)
suite = importlib.util.module_from_spec(spec)
sys.modules["run_web_path_q1_17_image_grade"] = suite
assert spec.loader is not None
spec.loader.exec_module(suite)


def _image(machine: str, component: str) -> dict:
    return {
        "manual_id": machine,
        "source_key": "txt",
        "target_component": component,
        "caption": "解析器原始相邻文本",
    }


def test_pair_grade_uses_explicit_gold_pairs_not_generated_answer_pairs():
    result = {
        "answer": "### 错误机型\n* **错误部件**：无关内容。",
        "images": [_image("自动封边机", "激光器出光口")],
        "placements": [{"image_index": 0}],
    }
    image_spec = {
        "want_images": True,
        "image_answer_pairs": True,
        "image_expected_pairs": [["自动封边机", "激光器出光口"]],
        "image_min_placements": 1,
    }

    ok, notes = suite.grade_images(result, image_spec)

    assert ok, notes


def test_pair_grade_requires_declared_placement_coverage():
    result = {
        "answer": "自动封边机的激光器出光口需要清理。",
        "images": [_image("自动封边机", "激光器出光口")],
        "placements": [],
    }
    image_spec = {
        "want_images": True,
        "image_answer_pairs": True,
        "image_expected_pairs": [["自动封边机", "激光器出光口"]],
        "image_min_placements": 1,
    }

    ok, notes = suite.grade_images(result, image_spec)

    assert not ok
    assert "placement_count:0<1" in notes


def test_caption_topic_mismatch_is_a_failure():
    result = {
        "answer": "answer",
        "images": [{"caption": "每季度一次", "manual_id": "高速智能封边机"}],
        "placements": [{"image_index": 0}],
    }

    ok, notes = suite.grade_images(
        result,
        {
            "want_images": True,
            "caption_min": 1,
            "caption_topic_any": ["电控"],
        },
    )

    assert not ok
    assert "caption_topic:0/1" in notes


def test_source_diversity_prefers_stable_manual_identity():
    result = {
        "answer": "answer",
        "images": [
            {"manual_id": "manual-a", "source_key": "txt"},
            {"manual_id": "manual-b", "source_key": "txt"},
        ],
        "placements": [{"image_index": 0}, {"image_index": 1}],
    }

    ok, notes = suite.grade_images(
        result,
        {"want_images": True, "caption_min": 2, "image_min_sources": 2},
    )

    assert ok, notes


def test_pair_grade_rejects_conflicting_authoritative_provenance():
    image = _image("双端封边机", "激光器出光口")
    image["target_machine"] = "自动封边机"
    image["caption"] = "自动封边机 激光器出光口"
    result = {
        "answer": "自动封边机的激光器出光口需要清理。",
        "images": [image],
        "placements": [{"image_index": 0}],
    }
    image_spec = {
        "want_images": True,
        "image_answer_pairs": True,
        "image_expected_pairs": [["自动封边机", "激光器出光口"]],
        "image_min_placements": 1,
    }

    ok, notes = suite.grade_images(result, image_spec)

    assert not ok
    assert any(note.startswith("pair_missing:") for note in notes)


def test_pair_grade_cannot_reuse_one_placed_image_for_two_gold_pairs():
    image = _image("自动封边机", "激光器出光口")
    image["target_component"] = "激光器出光口 压带轮"
    result = {
        "answer": "answer",
        "images": [image],
        "placements": [{"image_index": 0}, {"image_index": 0}],
    }
    image_spec = {
        "want_images": True,
        "image_answer_pairs": True,
        "image_expected_pairs": [
            ["自动封边机", "激光器出光口"],
            ["自动封边机", "压带轮"],
        ],
        "image_min_placements": 2,
    }

    ok, notes = suite.grade_images(result, image_spec)

    assert not ok
    assert "placement_count:1<2" in notes
    assert any(note.startswith("pair_missing:") for note in notes)


def test_pair_grade_ignores_invalid_placement_indexes():
    result = {
        "answer": "answer",
        "images": [_image("自动封边机", "激光器出光口")],
        "placements": [{"image_index": 5}, {"missing": 0}],
    }
    image_spec = {
        "want_images": True,
        "image_answer_pairs": True,
        "image_expected_pairs": [["自动封边机", "激光器出光口"]],
        "image_min_placements": 1,
    }

    ok, notes = suite.grade_images(result, image_spec)

    assert not ok
    assert "placement_count:0<1" in notes
