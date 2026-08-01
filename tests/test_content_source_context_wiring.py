"""Regression tests for multimodal context-source wiring.

set_content_source / _get_context_for_item decide whether modal processors see
surrounding page text. Broken wiring silently drops captions/context from
image/table/equation entities. RAGAnything helpers must fan out to all
processors and refresh extractors when context config changes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("lightrag")

from raganything.config import RAGAnythingConfig
from raganything.modalprocessors import BaseModalProcessor
from raganything.raganything import RAGAnything


class ConcreteProcessor(BaseModalProcessor):
    def __init__(self):
        # Skip LightRAG-dependent __init__
        self.content_source = None
        self.content_format = "auto"
        self.context_extractor = MagicMock()

    async def process_multimodal_content(self, *args, **kwargs):
        pass

    async def generate_description_only(self, *args, **kwargs):
        pass


class TestBaseModalProcessorContentSource:
    def test_set_content_source_stores_source_and_format(self):
        proc = ConcreteProcessor()
        source = [{"type": "text", "text": "page text"}]

        proc.set_content_source(source, content_format="minerU")

        assert proc.content_source is source
        assert proc.content_format == "minerU"

    def test_get_context_returns_empty_without_source(self):
        proc = ConcreteProcessor()

        assert proc._get_context_for_item({"page_idx": 0}) == ""
        proc.context_extractor.extract_context.assert_not_called()

    def test_get_context_uses_extractor(self):
        proc = ConcreteProcessor()
        source = [{"type": "text", "text": "nearby"}]
        proc.set_content_source(source, "minerU")
        proc.context_extractor.extract_context.return_value = "surrounding text"
        item = {"page_idx": 2, "index": 1}

        result = proc._get_context_for_item(item)

        assert result == "surrounding text"
        proc.context_extractor.extract_context.assert_called_once_with(
            source, item, "minerU"
        )

    def test_get_context_swallows_extractor_errors(self):
        proc = ConcreteProcessor()
        proc.set_content_source([{"type": "text", "text": "x"}], "auto")
        proc.context_extractor.extract_context.side_effect = RuntimeError("boom")

        assert proc._get_context_for_item({"page_idx": 0}) == ""


class TestRAGAnythingContextWiring:
    def test_set_content_source_for_context_noop_without_processors(self):
        rag = RAGAnything(config=RAGAnythingConfig(working_dir="/tmp/rag-test-ctx"))
        # modal_processors empty by default before init
        rag.set_content_source_for_context([{"type": "text", "text": "x"}], "minerU")
        assert rag.modal_processors == {}

    def test_set_content_source_for_context_applies_to_all_processors(self):
        rag = RAGAnything(config=RAGAnythingConfig(working_dir="/tmp/rag-test-ctx2"))
        image = ConcreteProcessor()
        table = ConcreteProcessor()
        failing = ConcreteProcessor()
        failing.set_content_source = MagicMock(side_effect=RuntimeError("fail"))
        rag.modal_processors = {
            "image": image,
            "table": table,
            "generic": failing,
        }
        source = [{"type": "text", "text": "doc"}]

        rag.set_content_source_for_context(source, "text_chunks")

        assert image.content_source is source
        assert image.content_format == "text_chunks"
        assert table.content_source is source
        assert table.content_format == "text_chunks"
        failing.set_content_source.assert_called_once_with(source, "text_chunks")

    def test_update_context_config_updates_known_keys_and_propagates_extractor(self):
        rag = RAGAnything(config=RAGAnythingConfig(working_dir="/tmp/rag-test-ctx3"))
        rag.config.context_window = 1
        rag.lightrag = SimpleNamespace(tokenizer=object())
        new_extractor = object()
        rag._create_context_extractor = MagicMock(return_value=new_extractor)
        image = ConcreteProcessor()
        table = ConcreteProcessor()
        rag.modal_processors = {"image": image, "table": table}

        rag.update_context_config(context_window=4, not_a_real_param=99)

        assert rag.config.context_window == 4
        assert not hasattr(rag.config, "not_a_real_param")
        rag._create_context_extractor.assert_called_once()
        assert image.context_extractor is new_extractor
        assert table.context_extractor is new_extractor
