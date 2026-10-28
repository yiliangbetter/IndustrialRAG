"""Dependency-free compatibility scoring for the migrated legacy suites.

The canonical evaluation cases retain the old grader inputs under
``metadata.legacy``.  This module evaluates those inputs directly against the
raw product trace produced by ``execute_query_with_trace``.  Keeping the
compatibility scorer here avoids importing benchmark scripts (and their
runtime/model dependencies) into the versioned evaluation runner.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from raganything.evaluation_metrics import DeterministicEvaluationScore
from raganything.evaluation_schema import EvaluationCase

LEGACY_SCORER_VERSION = "rag-legacy-deterministic-scorer/v1"

_FALLBACK_ANY = ("人工客服", "转人工", "专线", "400-888-2033")
_SOURCE_KEYS = (
    "file_path",
    "source",
    "title",
    "document_id",
    "manual_id",
    "source_document_id",
    "source_document",
)
# The legacy Shili image-pair fixture was authored for these four manuals.
# Longest-first matching is important because ``自动封边机`` is a suffix of
# ``高速自动封边机``.
_SHILI_MACHINE_NAMES = tuple(
    sorted(
        ("高速智能封边机", "高速自动封边机", "双端封边机", "自动封边机"),
        key=len,
        reverse=True,
    )
)


class LegacyScorerError(ValueError):
    """Raised when a canonical case lacks a usable legacy grader payload."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LegacyScorerError(f"{path} must be an object")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _legacy_payloads(
    case: EvaluationCase,
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    legacy = _mapping(case.metadata.get("legacy"), f"{case.case_id}: metadata.legacy")
    suite = legacy.get("suite")
    if not isinstance(suite, str) or not suite:
        raise LegacyScorerError(
            f"{case.case_id}: metadata.legacy.suite must be a non-empty string"
        )
    payloads = _mapping(
        legacy.get("grader_payloads"),
        f"{case.case_id}: metadata.legacy.grader_payloads",
    )
    return suite, legacy, payloads


def _answer(raw_trace: Mapping[str, Any]) -> str:
    answer = raw_trace.get("answer")
    return answer if isinstance(answer, str) else "" if answer is None else str(answer)


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _contains(term: str, text: str) -> bool:
    return term in text or _compact_whitespace(term) in _compact_whitespace(text)


def _caption_matches_expected(caption: str, expected: str) -> bool:
    """Return true only when two non-empty caption strings overlap."""

    caption = (caption or "").strip()
    expected = (expected or "").strip()
    return bool(caption and expected) and (expected in caption or caption in expected)


def _answer_bullets(answer: str) -> list[str]:
    bullets: list[str] = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("* ", "- ")):
            bullets.append(line[2:].strip())
    return bullets or [answer or ""]


def _grade_machine_cycle_pairs(answer: str, pairs: Sequence[Any]) -> list[str]:
    misses: list[str] = []
    bullets = _answer_bullets(answer)
    for raw_pair in pairs:
        if not isinstance(raw_pair, Mapping):
            continue
        machine_terms = [str(item) for item in _sequence(raw_pair.get("machine"))]
        cycle_any = [str(item) for item in _sequence(raw_pair.get("cycle_any"))]
        cycle_forbidden = [
            str(item) for item in _sequence(raw_pair.get("cycle_forbidden"))
        ]
        matched = [
            bullet
            for bullet in bullets
            if any(_contains(machine, bullet) for machine in machine_terms)
        ]
        if not matched:
            if any(_contains(machine, answer) for machine in machine_terms):
                matched = [answer]
            else:
                misses.append(f"machine_missing:{machine_terms}")
                continue
        has_cycle = any(
            any(_contains(cycle, line) for cycle in cycle_any) for line in matched
        )
        has_forbidden = any(
            any(_contains(cycle, line) for cycle in cycle_forbidden) for line in matched
        )
        if not has_cycle or has_forbidden:
            misses.append(
                f"machine_cycle:{machine_terms} need {cycle_any}"
                + (f" forbid {cycle_forbidden}" if cycle_forbidden else "")
            )
    return misses


def _grade_shili_text(answer: str, spec: Mapping[str, Any]) -> tuple[bool, list[str]]:
    misses: list[str] = []
    for value in _sequence(spec.get("text_all")):
        term = str(value)
        if not _contains(term, answer):
            misses.append(f"missing:{term}")

    text_any = [str(value) for value in _sequence(spec.get("text_any"))]
    text_need2 = list(_sequence(spec.get("text_need2")))
    if (
        text_any
        and not any(_contains(term, answer) for term in text_any)
        and not text_need2
    ):
        misses.append(f"any_of:{text_any}")
    for raw_group in text_need2:
        group = [str(value) for value in _sequence(raw_group)]
        if not any(_contains(term, answer) for term in group):
            misses.append(f"need_one_of:{group}")

    misses.extend(
        _grade_machine_cycle_pairs(answer, _sequence(spec.get("text_machine_cycles")))
    )
    text_count_any = [str(value) for value in _sequence(spec.get("text_count_any"))]
    if text_count_any and not any(_contains(term, answer) for term in text_count_any):
        misses.append(f"count_any:{text_count_any}")
    for value in _sequence(spec.get("text_none")):
        term = str(value)
        if _contains(term, answer):
            misses.append(f"forbidden:{term}")
    if spec.get("text_min_len") and len(answer.strip()) < int(spec["text_min_len"]):
        misses.append(f"too_short:{len(answer.strip())}<{spec['text_min_len']}")
    return not misses, misses


def _listing_target_head(span: str) -> str:
    span = (span or "").strip().rstrip("：:")
    span = re.sub(r"[（(][^）)]*[）)]", "", span).strip()
    for separator in ("及其", "及"):
        if separator in span:
            span = span.split(separator, 1)[0].strip()
    return span


def _normalize_label_key(label: str) -> str:
    return re.sub(r"\s+", "", (label or "").strip())


def _resolve_shili_machine(text: str) -> str:
    compact = _compact_whitespace(text)
    return next(
        (
            machine
            for machine in _SHILI_MACHINE_NAMES
            if _compact_whitespace(machine) in compact
        ),
        "",
    )


def _image_matches_machine(image: Mapping[str, Any], machine_hint: str) -> bool:
    source_values = [
        str(image.get(key) or "").strip()
        for key in (
            "manual_id",
            "source_document_id",
            "source_document",
            "source_key",
        )
        if str(image.get(key) or "").strip()
    ]
    expected = _resolve_shili_machine(machine_hint)
    resolved_sources = {
        resolved
        for value in source_values
        if (resolved := _resolve_shili_machine(value))
    }
    if expected and resolved_sources:
        return resolved_sources == {expected}

    fallback_keys = (
        "locality_machine",
        "target_machine",
        "caption",
        "context",
        "url",
        "path",
    )
    values = source_values or [
        str(image.get(key) or "").strip()
        for key in fallback_keys
        if str(image.get(key) or "").strip()
    ]
    blob = " ".join(values)
    if not blob.strip():
        return False
    if expected:
        return _compact_whitespace(expected) in _compact_whitespace(blob)
    hint = _compact_whitespace(machine_hint)
    candidate = _compact_whitespace(blob)
    return bool(hint) and (
        hint in candidate or (len(hint) >= 8 and hint[:8] in candidate)
    )


def _cjk_bigrams(text: str) -> set[str]:
    bigrams: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text or ""):
        bigrams.update(run[index : index + 2] for index in range(len(run) - 1))
    return bigrams


def _label_matches_listing_target(label: str, target: str) -> bool:
    """Small, pure equivalent of the legacy caption/target matcher.

    Exact containment is the normal structured-provenance path.  The shared
    bigram rule retains the legacy tolerance for short captions whose word
    order differs from the gold component label.
    """

    label = (label or "").strip()
    target = (target or "").strip()
    if not label or not target:
        return False
    label_key = _normalize_label_key(label)
    for candidate in (target, _listing_target_head(target)):
        candidate_key = _normalize_label_key(candidate)
        if not candidate_key:
            continue
        if candidate_key in label_key:
            return True
        if len(label_key) >= 4 and label_key in candidate_key:
            return True
        if len(_cjk_bigrams(candidate) & _cjk_bigrams(label)) >= 2:
            return True
    return False


def _image_matches_component(image: Mapping[str, Any], component: str) -> bool:
    head = _listing_target_head(component)
    caption = str(image.get("caption") or "")
    context = str(image.get("context") or "")
    target = str(image.get("target_component") or image.get("target_anchor") or "")
    blob = f"{caption} {context} {target}".strip()
    if not blob:
        return False
    if _label_matches_listing_target(caption, head) or _label_matches_listing_target(
        context, head
    ):
        return True
    return head in blob or component in blob


def _expected_image_pairs(value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_pair in _sequence(value):
        pair = _sequence(raw_pair)
        if len(pair) != 2:
            continue
        machine, component = (str(item).strip() for item in pair)
        if machine and component:
            pairs.append((machine, component))
    return pairs


def _valid_placement_indexes(placements: Sequence[Any], image_count: int) -> set[int]:
    indexes: set[int] = set()
    for placement in placements:
        if not isinstance(placement, Mapping):
            continue
        index = placement.get("image_index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if 0 <= index < image_count:
            indexes.add(index)
    return indexes


def _grade_shili_image_pairs(
    images: list[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    eligible_indexes: set[int] | None,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if not images:
        return False, ["no_images"]

    waived_keys = {
        (
            _normalize_label_key(machine),
            _normalize_label_key(_listing_target_head(component)),
        )
        for machine, component in _expected_image_pairs(spec.get("image_pair_waive"))
    }
    raw_pairs = _expected_image_pairs(spec.get("image_expected_pairs"))
    # Canonical Shili pair cases always carry explicit gold pairs.  The old
    # fallback parsed generated Markdown using the product's KB vocabulary,
    # which is intentionally not available to this dependency-free scorer.
    if not raw_pairs:
        return False, ["answer_pairs:0<2"]

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for machine, component in raw_pairs:
        component = _listing_target_head(component)
        key = (_normalize_label_key(machine), _normalize_label_key(component))
        if not machine or not component or key in seen:
            continue
        seen.add(key)
        pairs.append((machine, component))

    required_pairs = [
        pair
        for pair in pairs
        if (
            _normalize_label_key(pair[0]),
            _normalize_label_key(pair[1]),
        )
        not in waived_keys
    ]
    waived = [
        f"{machine}/{component}"
        for machine, component in pairs
        if (
            _normalize_label_key(machine),
            _normalize_label_key(component),
        )
        in waived_keys
    ]
    if waived:
        notes.append(f"pair_waived:{waived}")

    allowed = (
        set(range(len(images))) if eligible_indexes is None else set(eligible_indexes)
    )
    candidates = [
        [
            image_index
            for image_index, image in enumerate(images)
            if image_index in allowed
            and _image_matches_machine(image, machine)
            and _image_matches_component(image, component)
        ]
        for machine, component in required_pairs
    ]
    image_to_pair: dict[int, int] = {}

    def assign(pair_index: int, visited_images: set[int]) -> bool:
        for image_index in candidates[pair_index]:
            if image_index in visited_images:
                continue
            visited_images.add(image_index)
            previous_pair = image_to_pair.get(image_index)
            if previous_pair is None or assign(previous_pair, visited_images):
                image_to_pair[image_index] = pair_index
                return True
        return False

    uncovered: list[str] = []
    for pair_index, (machine, component) in enumerate(required_pairs):
        if not assign(pair_index, set()):
            uncovered.append(f"{machine}/{component}")
    machines = {_normalize_label_key(machine) for machine, _ in required_pairs}
    if uncovered:
        notes.append(f"pair_missing:{uncovered}")
    if machines and len(images) < len(machines):
        notes.append(f"image_count:{len(images)}<{len(machines)}")
    return not uncovered and (not machines or len(images) >= len(machines)), notes


def _grade_shili_images(
    raw_trace: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    images = [
        item for item in _sequence(raw_trace.get("images")) if isinstance(item, Mapping)
    ]
    placements = list(_sequence(raw_trace.get("placements")))
    placed_indexes = _valid_placement_indexes(placements, len(images))
    captions = [str(image.get("caption") or "") for image in images]
    want_images = spec.get("want_images")
    notes: list[str] = []

    if want_images is False:
        if images:
            notes.append(f"unexpected_images:{len(images)}")
        return not images, notes
    if want_images is not True:
        return True, notes

    if spec.get("image_answer_pairs"):
        minimum_placements = int(spec.get("image_min_placements") or 0)
        passed, pair_notes = _grade_shili_image_pairs(
            images,
            spec,
            eligible_indexes=placed_indexes if minimum_placements else None,
        )
        notes.extend(pair_notes)
        if minimum_placements and len(placed_indexes) < minimum_placements:
            notes.append(f"placement_count:{len(placed_indexes)}<{minimum_placements}")
            passed = False
        return passed, notes

    if not placements:
        return False, ["no_placements"]

    expected_captions = [str(value) for value in _sequence(spec.get("caption_any"))]
    if expected_captions:
        empty_caption_indexes = [
            index for index, caption in enumerate(captions) if not caption.strip()
        ]
        if empty_caption_indexes:
            notes.append(f"empty_captions:{empty_caption_indexes}")
        matched = [
            expected
            for expected in expected_captions
            if any(_caption_matches_expected(caption, expected) for caption in captions)
        ]
        minimum_matches = int(spec.get("caption_min_match") or 1)
        if len(matched) < minimum_matches:
            notes.append(f"caption_match:{len(matched)}/{len(expected_captions)}")
        wrong = [
            caption
            for caption in captions
            if caption
            and not any(
                _caption_matches_expected(caption, expected)
                for expected in expected_captions
            )
        ]
        if wrong:
            notes.append(f"extra_or_wrong_caps:{wrong}")
        captions_passed = (
            bool(captions)
            and not empty_caption_indexes
            and all(
                any(
                    _caption_matches_expected(caption, expected)
                    for expected in expected_captions
                )
                for caption in captions
            )
        )
        if captions and not captions_passed:
            notes.append("caption_not_in_expected")
        passed = len(matched) >= minimum_matches and captions_passed
        return passed, notes

    minimum_images = int(spec.get("caption_min") or spec.get("image_min") or 1)
    if len(images) < minimum_images:
        return False, [f"image_count:{len(images)}<{minimum_images}"]

    topics = [str(value) for value in _sequence(spec.get("caption_topic_any"))]
    if topics:
        topic_hits = sum(
            1
            for image in images
            if any(
                topic
                in " ".join(
                    str(image.get(key) or "")
                    for key in (
                        "caption",
                        "context",
                        "target_component",
                        "target_anchor",
                    )
                )
                for topic in topics
            )
        )
        if topic_hits < min(minimum_images, len(images)):
            return False, [f"caption_topic:{topic_hits}/{minimum_images}"]

    minimum_placements = int(spec.get("image_min_placements") or 0)
    if minimum_placements and len(placed_indexes) < minimum_placements:
        return False, [f"placement_count:{len(placed_indexes)}<{minimum_placements}"]

    minimum_sources = int(spec.get("image_min_sources") or 0)
    if minimum_sources:
        sources = {
            str(
                image.get("manual_id")
                or image.get("source_document_id")
                or image.get("source_key")
                or ""
            ).strip()
            for image in images
            if str(
                image.get("manual_id")
                or image.get("source_document_id")
                or image.get("source_key")
                or ""
            ).strip()
        }
        if len(sources) < minimum_sources:
            return False, [f"image_sources:{len(sources)}<{minimum_sources}"]
    return True, notes


def _grade_nanxing_answer(
    answer: str, spec: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    compact = "".join(answer.split()).lower()
    pattern_text = re.sub(r"[*_`~]+", "", compact)
    missing: list[str] = []
    for value in _sequence(spec.get("must_include")):
        token = str(value)
        if "".join(token.split()).lower() not in compact:
            missing.append(token)
    for raw_choices in _sequence(spec.get("one_of")):
        choices = [str(value) for value in _sequence(raw_choices)]
        if not any("".join(choice.split()).lower() in compact for choice in choices):
            missing.append("one_of:" + "|".join(choices))
    for value in _sequence(spec.get("must_match")):
        pattern = str(value)
        if re.search(pattern, pattern_text, flags=re.IGNORECASE | re.DOTALL) is None:
            missing.append("pattern:" + pattern)
    return not missing, missing


def _source_parts(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    parts: list[str] = []
    for part in value.split("<SEP>"):
        normalized = part.strip().replace("\\", "/")
        if not normalized:
            continue
        parts.append(PurePosixPath(normalized).name)
    return parts


def _trace_source_files(raw_trace: Mapping[str, Any]) -> list[str]:
    found: set[str] = set()
    for value in _sequence(raw_trace.get("source_files")):
        found.update(_source_parts(value))

    documents = raw_trace.get("context_documents")
    if not isinstance(documents, list):
        documents = []
    if not documents:
        debug = raw_trace.get("debug_state")
        if isinstance(debug, Mapping):
            llm_input = debug.get("llm_input")
            if isinstance(llm_input, Mapping) and isinstance(
                llm_input.get("chunks"), list
            ):
                documents = llm_input["chunks"]
            elif isinstance(debug.get("retrieved_docs"), list):
                documents = debug["retrieved_docs"]
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        for key in _SOURCE_KEYS:
            found.update(_source_parts(document.get(key)))
    return sorted(found)


def _expected_source_files(
    case: EvaluationCase, payloads: Mapping[str, Any]
) -> list[str]:
    configured = [
        str(value).strip()
        for value in _sequence(case.metadata.get("expected_source_files"))
        if str(value).strip()
    ]
    if configured:
        return configured
    source = payloads.get("source")
    if not isinstance(source, str):
        return []
    return [part.strip() for part in source.split("+") if part.strip()]


def _voice_contains_missing(text: str, needles: Sequence[Any]) -> list[str]:
    compact = _compact_whitespace(text)
    missing: list[str] = []
    for raw_needle in needles:
        needle = str(raw_needle or "").strip()
        if not needle:
            continue
        if _compact_whitespace(needle) not in compact and needle not in text:
            missing.append(needle)
    return missing


def _grade_voice_response(
    answer: str,
    grade: Mapping[str, Any],
    *,
    expect: str,
) -> tuple[bool, list[str], str]:
    if grade.get("skip"):
        if not answer.strip():
            return False, ["（空应答）"], "empty"
        return True, [], "skipped"
    answer = answer.strip()
    if not answer:
        return False, ["（空应答）"], "empty"

    fallback_hits = [token for token in _FALLBACK_ANY if token in answer]
    if grade.get("accept_fallback") and fallback_hits:
        return True, [], "fallback"

    text_all = list(_sequence(grade.get("text_all")))
    text_any = [str(value) for value in _sequence(grade.get("text_any"))]
    missing_all = _voice_contains_missing(answer, text_all)
    if missing_all:
        if grade.get("accept_fallback") and fallback_hits:
            return True, [], "fallback"
        return False, [f"缺：{item}" for item in missing_all], "strict"

    if text_any:
        hits = [
            token
            for token in text_any
            if _compact_whitespace(token) in _compact_whitespace(answer)
            or token in answer
        ]
        # The legacy implementation names the mode but gives both modes the
        # same one-hit threshold.
        minimum_hits = 1
        if len(hits) >= minimum_hits:
            return True, [], "script"
        if grade.get("accept_fallback") and fallback_hits:
            return True, [], "fallback"
        return False, [f"未命中要点：{', '.join(text_any[:3])}"], "strict"

    if grade.get("accept_fallback") and fallback_hits:
        return True, [], "fallback"
    return False, ["无评分要点"], "none"


def _score_shili(
    case: EvaluationCase,
    payloads: Mapping[str, Any],
    raw_trace: Mapping[str, Any],
) -> DeterministicEvaluationScore:
    text_spec = _mapping(
        payloads.get("text"),
        f"{case.case_id}: metadata.legacy.grader_payloads.text",
    )
    image_spec = _mapping(
        payloads.get("image"),
        f"{case.case_id}: metadata.legacy.grader_payloads.image",
    )
    text_passed, text_missing = _grade_shili_text(_answer(raw_trace), text_spec)
    image_passed, image_notes = _grade_shili_images(raw_trace, image_spec)
    conditions = {
        "legacy_text_passed": text_passed,
        "legacy_image_passed": image_passed,
    }
    hard_gate_passed = all(conditions.values())
    return DeterministicEvaluationScore(
        metrics={**conditions, "legacy_hard_gate_passed": hard_gate_passed},
        diagnostics={
            "scorer_version": LEGACY_SCORER_VERSION,
            "case_id": case.case_id,
            "suite": "shili17",
            "text": {"passed": text_passed, "missing": text_missing},
            "images": {"passed": image_passed, "notes": image_notes},
            "legacy_hard_gate": {
                "conditions": conditions,
                "failed_conditions": [
                    name for name, passed in conditions.items() if not passed
                ],
                "passed": hard_gate_passed,
            },
            "legacy_hard_gate_passed": hard_gate_passed,
        },
    )


def _score_nanxing(
    case: EvaluationCase,
    payloads: Mapping[str, Any],
    raw_trace: Mapping[str, Any],
) -> DeterministicEvaluationScore:
    answer_spec = _mapping(
        payloads.get("answer"),
        f"{case.case_id}: metadata.legacy.grader_payloads.answer",
    )
    answer_passed, missing = _grade_nanxing_answer(_answer(raw_trace), answer_spec)
    expected_sources = _expected_source_files(case, payloads)
    observed_sources = _trace_source_files(raw_trace)
    observed_set = set(observed_sources)
    missing_sources = [
        source for source in expected_sources if source not in observed_set
    ]
    source_passed = not missing_sources
    conditions = {
        "legacy_answer_passed": answer_passed,
        "legacy_source_passed": source_passed,
    }
    hard_gate_passed = all(conditions.values())
    return DeterministicEvaluationScore(
        metrics={**conditions, "legacy_hard_gate_passed": hard_gate_passed},
        diagnostics={
            "scorer_version": LEGACY_SCORER_VERSION,
            "case_id": case.case_id,
            "suite": "nanxing-latency",
            "answer": {"passed": answer_passed, "missing": missing},
            "sources": {
                "matching_key": "source_filename",
                "expected": expected_sources,
                "observed": observed_sources,
                "missing": missing_sources,
                "passed": source_passed,
            },
            "legacy_hard_gate": {
                "conditions": conditions,
                "failed_conditions": [
                    name for name, passed in conditions.items() if not passed
                ],
                "passed": hard_gate_passed,
            },
            "legacy_hard_gate_passed": hard_gate_passed,
        },
    )


def _score_voice(
    case: EvaluationCase,
    legacy: Mapping[str, Any],
    payloads: Mapping[str, Any],
    raw_trace: Mapping[str, Any],
) -> DeterministicEvaluationScore:
    grade = _mapping(
        payloads.get("response"),
        f"{case.case_id}: metadata.legacy.grader_payloads.response",
    )
    fields = _mapping(
        legacy.get("case_fields"), f"{case.case_id}: metadata.legacy.case_fields"
    )
    passed, missing, mode = _grade_voice_response(
        _answer(raw_trace),
        grade,
        expect=str(fields.get("expect") or "script"),
    )
    conditions = {"legacy_response_passed": passed}
    return DeterministicEvaluationScore(
        metrics={**conditions, "legacy_hard_gate_passed": passed},
        diagnostics={
            "scorer_version": LEGACY_SCORER_VERSION,
            "case_id": case.case_id,
            "suite": "voice29",
            "response": {"passed": passed, "missing": missing, "mode": mode},
            "legacy_hard_gate": {
                "conditions": conditions,
                "failed_conditions": [] if passed else ["legacy_response_passed"],
                "passed": passed,
            },
            "legacy_hard_gate_passed": passed,
        },
    )


def score_legacy_case(
    case: EvaluationCase | Mapping[str, Any],
    raw_trace: Mapping[str, Any],
) -> DeterministicEvaluationScore:
    """Score one canonical case with its preserved legacy grader.

    ``raw_trace`` is the product trace, not an ``EvaluationTraceResult``.  The
    image grader needs placement/provenance fields that are intentionally kept
    in the raw trace, and the Nanxing source check uses its model-bound context
    documents.
    """

    expected = EvaluationCase.from_dict(case, "case")
    if not isinstance(raw_trace, Mapping):
        raise TypeError("raw_trace must be an object")
    suite, legacy, payloads = _legacy_payloads(expected)
    if suite == "shili17":
        return _score_shili(expected, payloads, raw_trace)
    if suite == "nanxing-latency":
        return _score_nanxing(expected, payloads, raw_trace)
    if suite == "voice29":
        return _score_voice(expected, legacy, payloads, raw_trace)
    raise LegacyScorerError(
        f"{expected.case_id}: unsupported metadata.legacy.suite {suite!r}"
    )


__all__ = [
    "LEGACY_SCORER_VERSION",
    "LegacyScorerError",
    "score_legacy_case",
]
