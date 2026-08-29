"""BaseModalProcessor._get_context_for_item must fail open.

Every image/table/equation caption reads surrounding text through this
helper. A missing source, a broken extractor, or an uninitialized
extractor must return empty context rather than abort multimodal ingest.
"""

from raganything.modalprocessors import BaseModalProcessor


class ConcreteProcessor(BaseModalProcessor):
    """Bypass LightRAG-backed __init__; only context fields are needed."""

    def __init__(self, context_extractor=None):
        self.content_source = None
        self.content_format = "auto"
        self.context_extractor = context_extractor

    async def process_multimodal_content(self, *args, **kwargs):
        raise NotImplementedError


class StubExtractor:
    def __init__(self, result="nearby header"):
        self.result = result
        self.calls = []

    def extract_context(self, content_source, item_info, content_format):
        self.calls.append((content_source, item_info, content_format))
        return self.result


class ExplodingExtractor:
    def extract_context(self, content_source, item_info, content_format):
        raise RuntimeError("tokenizer unavailable")


def test_empty_content_source_returns_empty_without_calling_extractor():
    extractor = StubExtractor()
    processor = ConcreteProcessor(context_extractor=extractor)

    assert processor._get_context_for_item({"page_idx": 1}) == ""
    assert extractor.calls == []


def test_extractor_result_is_returned_with_configured_format():
    extractor = StubExtractor(result="[Page 0] Intro")
    processor = ConcreteProcessor(context_extractor=extractor)
    processor.set_content_source([{"type": "text", "text": "Intro"}], "minerU")

    context = processor._get_context_for_item({"page_idx": 1, "index": 0})

    assert context == "[Page 0] Intro"
    assert extractor.calls == [
        (
            [{"type": "text", "text": "Intro"}],
            {"page_idx": 1, "index": 0},
            "minerU",
        )
    ]


def test_extractor_exception_returns_empty_string():
    processor = ConcreteProcessor(context_extractor=ExplodingExtractor())
    processor.content_source = [{"type": "text", "text": "body"}]

    assert processor._get_context_for_item({"page_idx": 2}) == ""


def test_missing_extractor_returns_empty_string():
    processor = ConcreteProcessor(context_extractor=None)
    processor.content_source = [{"type": "text", "text": "body"}]

    assert processor._get_context_for_item({"page_idx": 0}) == ""
