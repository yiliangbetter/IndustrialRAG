"""Regression tests for RAGAnything context extractor creation.

Open PR #89 covers lifecycle/config/processor gating. This module covers the
fail-closed and tokenizer-wiring contracts of _create_context_extractor /
_create_context_config, which decide whether multimodal processors can be
built with a real LightRAG tokenizer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("lightrag")

from raganything.config import RAGAnythingConfig
from raganything.modalprocessors import ContextConfig, ContextExtractor
from raganything.raganything import RAGAnything


class FakeTokenizer:
    def encode(self, text):
        return list(range(len(text)))


def _make_rag(*, lightrag=None, **config_overrides):
    rag = object.__new__(RAGAnything)
    rag.config = RAGAnythingConfig(**config_overrides)
    rag.lightrag = lightrag
    rag.logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return rag


def test_create_context_extractor_requires_lightrag():
    rag = _make_rag(lightrag=None)

    with pytest.raises(
        ValueError,
        match="LightRAG must be initialized before creating context extractor",
    ):
        rag._create_context_extractor()


def test_create_context_extractor_wires_config_and_tokenizer(monkeypatch):
    tokenizer = FakeTokenizer()
    rag = _make_rag(
        lightrag=SimpleNamespace(tokenizer=tokenizer),
        context_window=3,
        context_mode="chunk",
        max_context_tokens=1500,
        include_headers=False,
        include_captions=True,
        context_filter_content_types=["text", "table"],
    )

    captured = {}

    class CapturingExtractor(ContextExtractor):
        def __init__(self, config=None, tokenizer=None):
            captured["config"] = config
            captured["tokenizer"] = tokenizer
            super().__init__(config=config, tokenizer=tokenizer)

    monkeypatch.setattr(
        "raganything.raganything.ContextExtractor",
        CapturingExtractor,
    )

    extractor = rag._create_context_extractor()

    assert isinstance(extractor, CapturingExtractor)
    assert captured["tokenizer"] is tokenizer
    cfg = captured["config"]
    assert isinstance(cfg, ContextConfig)
    assert cfg.context_window == 3
    assert cfg.context_mode == "chunk"
    assert cfg.max_context_tokens == 1500
    assert cfg.include_headers is False
    assert cfg.include_captions is True
    assert cfg.filter_content_types == ["text", "table"]


def test_create_context_config_maps_rag_config_fields():
    rag = _make_rag(
        context_window=2,
        context_mode="page",
        max_context_tokens=999,
        include_headers=True,
        include_captions=False,
        context_filter_content_types=["text"],
    )

    cfg = rag._create_context_config()

    assert cfg == ContextConfig(
        context_window=2,
        context_mode="page",
        max_context_tokens=999,
        include_headers=True,
        include_captions=False,
        filter_content_types=["text"],
    )
