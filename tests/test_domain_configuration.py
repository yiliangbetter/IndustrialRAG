"""Domain policy is shipped as validated data rather than executable constants."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import iqr_domain_schema
import query_doc_steering


def test_shipped_domain_and_steering_configs_are_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_DOMAIN_SCHEMA", raising=False)
    monkeypatch.delenv("RAG_QUERY_STEERING_PROFILES", raising=False)
    monkeypatch.delenv("RAG_QUERY_DOC_FILTER_RULES_JSON", raising=False)

    assert iqr_domain_schema.get_domain_schema().catalog_page_marker
    assert [
        profile["id"] for profile in query_doc_steering.load_steering_profiles()
    ] == [
        "high_speed_smart",
        "high_speed_auto",
        "double_end",
        "auto_edge",
    ]


def test_domain_schema_proxy_reloads_custom_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = json.loads(
        (ROOT / "config" / "domain_schema.json").read_text(encoding="utf-8")
    )
    custom = tmp_path / "schema.json"
    data["catalog_page_marker"] = "CUSTOM MARKER ONE"
    custom.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("RAG_DOMAIN_SCHEMA", str(custom))

    assert iqr_domain_schema.schema.catalog_page_marker == "CUSTOM MARKER ONE"

    data["catalog_page_marker"] = "CUSTOM MARKER TWO"
    custom.write_text(json.dumps(data), encoding="utf-8")
    assert iqr_domain_schema.schema.catalog_page_marker == "CUSTOM MARKER TWO"


def test_invalid_domain_or_profile_config_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    broken_schema = tmp_path / "schema.json"
    broken_schema.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("RAG_DOMAIN_SCHEMA", str(broken_schema))
    with pytest.raises(iqr_domain_schema.DomainSchemaError, match="field 'domain'"):
        iqr_domain_schema.get_domain_schema()

    broken_profiles = tmp_path / "profiles.json"
    broken_profiles.write_text('[{"id": "missing-fields"}]', encoding="utf-8")
    monkeypatch.setenv("RAG_QUERY_STEERING_PROFILES", str(broken_profiles))
    with pytest.raises(query_doc_steering.SteeringProfileError, match="requires"):
        query_doc_steering.load_steering_profiles()


def test_cross_manual_prompt_contains_no_fixture_answers() -> None:
    residual_prompt = query_doc_steering.build_cross_manual_listing_answer_prompt(
        "所有机型中哪些部件需要清理残胶？"
    )
    oil_prompt = query_doc_steering.build_cross_manual_listing_answer_prompt(
        "哪些机型使用1#透平油？"
    )

    for leaked_fact in ("涂胶轴", "涂胶电机", "ISOVG32", "ISO VG-32"):
        assert leaked_fact not in residual_prompt
        assert leaked_fact not in oil_prompt
