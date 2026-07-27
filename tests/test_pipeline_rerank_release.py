"""Tests for the single rerank release strategy (RERANK_RELEASE_AFTER_QUERY)."""

import logging

import pytest

import raganything.pipeline_rerank as pipeline_rerank
from raganything.pipeline_rerank import (
    release_rerank_after_query_if_enabled,
    rerank_release_after_query,
)


class TestRerankReleaseAfterQueryEnv:
    def test_unset_is_disabled(self, monkeypatch):
        monkeypatch.delenv("RERANK_RELEASE_AFTER_QUERY", raising=False)
        assert rerank_release_after_query() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", " 1 "])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("RERANK_RELEASE_AFTER_QUERY", value)
        assert rerank_release_after_query() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "on"])
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("RERANK_RELEASE_AFTER_QUERY", value)
        assert rerank_release_after_query() is False


class TestReleaseRerankAfterQueryIfEnabled:
    @pytest.fixture()
    def release_spy(self, monkeypatch):
        calls: list[bool] = []
        monkeypatch.setattr(
            pipeline_rerank,
            "release_cross_encoder",
            lambda: calls.append(True),
        )
        return calls

    def test_disabled_env_does_not_release(self, monkeypatch, release_spy):
        monkeypatch.delenv("RERANK_RELEASE_AFTER_QUERY", raising=False)
        monkeypatch.setenv("RERANK_BINDING", "hf")
        release_rerank_after_query_if_enabled()
        assert release_spy == []

    @pytest.mark.parametrize(
        "binding", ["hf", "local", "cross_encoder", "sentence_transformers"]
    )
    def test_hf_bindings_release(self, monkeypatch, release_spy, binding):
        monkeypatch.setenv("RERANK_RELEASE_AFTER_QUERY", "1")
        monkeypatch.setenv("RERANK_BINDING", binding)
        release_rerank_after_query_if_enabled()
        assert release_spy == [True]

    @pytest.mark.parametrize("binding", ["", "none", "jina", "cohere", "aliyun"])
    def test_non_hf_bindings_do_not_release(self, monkeypatch, release_spy, binding):
        monkeypatch.setenv("RERANK_RELEASE_AFTER_QUERY", "1")
        monkeypatch.setenv("RERANK_BINDING", binding)
        release_rerank_after_query_if_enabled()
        assert release_spy == []


class TestDeprecatedReleaseEnvs:
    @pytest.mark.parametrize(
        "name", ["RERANK_RELEASE_AFTER_PREDICT", "RERANK_RELEASE_AFTER_GATE"]
    )
    def test_warns_when_deprecated_env_set(self, monkeypatch, caplog, name):
        monkeypatch.setenv(name, "1")
        with caplog.at_level(logging.WARNING, logger="raganything.pipeline_rerank"):
            pipeline_rerank._warn_deprecated_release_envs()
        assert any(name in rec.getMessage() for rec in caplog.records)
        assert any(
            "RERANK_RELEASE_AFTER_QUERY" in rec.getMessage() for rec in caplog.records
        )

    def test_silent_when_deprecated_envs_unset(self, monkeypatch, caplog):
        monkeypatch.delenv("RERANK_RELEASE_AFTER_PREDICT", raising=False)
        monkeypatch.delenv("RERANK_RELEASE_AFTER_GATE", raising=False)
        with caplog.at_level(logging.WARNING, logger="raganything.pipeline_rerank"):
            pipeline_rerank._warn_deprecated_release_envs()
        assert caplog.records == []

    def test_removed_switch_helpers_are_gone(self):
        assert not hasattr(pipeline_rerank, "rerank_release_after_predict")
        assert not hasattr(pipeline_rerank, "rerank_release_after_gate")
        assert "rerank_release_after_predict" not in pipeline_rerank.__all__
        assert "rerank_release_after_gate" not in pipeline_rerank.__all__


class TestHfCrossEncoderRerank:
    @pytest.mark.asyncio
    async def test_empty_documents_short_circuits(self):
        assert await pipeline_rerank.hf_cross_encoder_rerank("q", []) == []
