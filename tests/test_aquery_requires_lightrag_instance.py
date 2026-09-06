"""aquery must fail closed when no LightRAG instance exists.

#125/#130 cover VLM routing and multimodal cache once LightRAG is present.
The public text query still has to refuse a None instance with a clear
ValueError instead of AttributeError, or callers think retrieval ran.
"""

import pytest

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


@pytest.mark.asyncio
async def test_aquery_raises_when_lightrag_is_missing():
    query = QueryMixin()
    query.lightrag = None
    query.logger = FakeLogger()
    query.vision_model_func = None

    with pytest.raises(ValueError, match="No LightRAG instance available"):
        await query.aquery("what is the pump torque?", mode="mix")
