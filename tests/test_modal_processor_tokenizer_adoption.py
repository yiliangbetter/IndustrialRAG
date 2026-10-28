"""Modal processors must trim caption context with the intended tokenizer.

Image and generic processors share ``BaseModalProcessor``. A caller-supplied
tokenizer has to win over the LightRAG tokenizer. An extractor that has no
tokenizer must adopt the LightRAG one, on the same extractor object. Omitting
the extractor must still build one that token-truncates instead of cutting
by character count.
"""

from dataclasses import dataclass

from raganything.modalprocessors import (
    ContextExtractor,
    GenericModalProcessor,
    ImageModalProcessor,
)

_SOURCE = "procedure text beside the figure"


class _MarkTokenizer:
    def __init__(self, label: str) -> None:
        self.label = label

    def encode(self, text: str):
        return list(range(2001))

    def decode(self, tokens) -> str:
        return f"{self.label}-{len(tokens)}"


@dataclass
class _LightRAGStandIn:
    tokenizer: object
    text_chunks: object = None
    chunks_vdb: object = None
    entities_vdb: object = None
    relationships_vdb: object = None
    chunk_entity_relation_graph: object = None
    embedding_func: object = None
    llm_model_func: object = None
    llm_response_cache: object = None


def _context(processor) -> str:
    processor.set_content_source(_SOURCE, "text")
    return processor._get_context_for_item({"page_idx": 0})


def test_image_processor_keeps_caller_tokenizer_for_context_cuts():
    caller = _MarkTokenizer("caller")
    lightrag = _MarkTokenizer("lightrag")
    extractor = ContextExtractor(tokenizer=caller)
    processor = ImageModalProcessor(
        _LightRAGStandIn(tokenizer=lightrag),
        modal_caption_func=lambda *args, **kwargs: None,
        context_extractor=extractor,
    )

    assert processor.context_extractor is extractor
    assert extractor.tokenizer is caller
    assert _context(processor) == "caller-2000..."


def test_missing_extractor_tokenizer_adopts_lightrag_tokenizer():
    lightrag = _MarkTokenizer("lightrag")
    extractor = ContextExtractor()
    assert extractor.tokenizer is None
    processor = ImageModalProcessor(
        _LightRAGStandIn(tokenizer=lightrag),
        modal_caption_func=lambda *args, **kwargs: None,
        context_extractor=extractor,
    )

    assert processor.context_extractor is extractor
    assert extractor.tokenizer is lightrag
    assert _context(processor) == "lightrag-2000..."


def test_omitted_extractor_token_truncates_with_lightrag_tokenizer():
    lightrag = _MarkTokenizer("lightrag")
    processor = ImageModalProcessor(
        _LightRAGStandIn(tokenizer=lightrag),
        modal_caption_func=lambda *args, **kwargs: None,
    )

    assert isinstance(processor.context_extractor, ContextExtractor)
    assert processor.context_extractor.tokenizer is lightrag
    assert _context(processor) == "lightrag-2000..."


def test_generic_processor_keeps_caller_tokenizer():
    caller = _MarkTokenizer("caller")
    extractor = ContextExtractor(tokenizer=caller)
    processor = GenericModalProcessor(
        _LightRAGStandIn(tokenizer=_MarkTokenizer("lightrag")),
        modal_caption_func=lambda *args, **kwargs: None,
        context_extractor=extractor,
    )

    assert processor.context_extractor is extractor
    assert extractor.tokenizer is caller
    assert _context(processor) == "caller-2000..."
