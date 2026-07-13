import pytest

pytest.importorskip("lightrag")

from raganything.query import QueryMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class FakeLightRAG:
    def __init__(self):
        self.calls = []
        self.llm_response_cache = None

    async def aquery(self, query, param, system_prompt=None):
        self.calls.append(
            {
                "query": query,
                "mode": param.mode,
                "system_prompt": system_prompt,
            }
        )
        return "answer"


class FakeCache:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self.requested_keys = []
        self.upserts = []
        self.index_done_calls = 0

    async def get_by_id(self, cache_key):
        self.requested_keys.append(cache_key)
        return None

    async def upsert(self, data):
        self.upserts.append(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


class DummyQuery(QueryMixin):
    def __init__(self):
        self.lightrag = FakeLightRAG()
        self.logger = FakeLogger()
        self.vision_model_func = None
        self.modal_processors = {}

    async def _ensure_lightrag_initialized(self):
        return {"success": True}


@pytest.mark.asyncio
async def test_multimodal_query_without_content_forwards_system_prompt():
    dummy = DummyQuery()

    result = await dummy.aquery_with_multimodal(
        "Summarize the report",
        multimodal_content=None,
        mode="hybrid",
        system_prompt="Use compliance language.",
    )

    assert result == "answer"
    assert dummy.lightrag.calls == [
        {
            "query": "Summarize the report",
            "mode": "hybrid",
            "system_prompt": "Use compliance language.",
        }
    ]


@pytest.mark.asyncio
async def test_multimodal_query_cache_key_and_inner_query_include_system_prompt():
    dummy = DummyQuery()
    cache = FakeCache()
    dummy.lightrag.llm_response_cache = cache
    multimodal_content = [
        {
            "type": "image",
            "img_path": "/tmp/diagram.png",
            "image_caption": ["pipeline"],
        }
    ]
    system_prompt = "Answer as an auditor."

    result = await dummy.aquery_with_multimodal(
        "What changed?",
        multimodal_content=multimodal_content,
        mode="mix",
        system_prompt=system_prompt,
    )

    expected_cache_key = dummy._generate_multimodal_cache_key(
        "What changed?",
        multimodal_content,
        "mix",
        system_prompt=system_prompt,
    )
    assert result == "answer"
    assert cache.requested_keys == [expected_cache_key]
    assert list(cache.upserts[0].keys()) == [expected_cache_key]
    assert cache.index_done_calls == 1
    assert dummy.lightrag.calls[0]["system_prompt"] == system_prompt
    assert dummy.lightrag.calls[0]["query"].startswith("User query: What changed?")
