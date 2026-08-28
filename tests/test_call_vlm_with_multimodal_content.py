"""VLM helper must dispatch text vs multimodal payloads and re-raise failures.

aquery_vlm_enhanced builds messages then calls this helper. Sending a list
payload as a string prompt drops images; swallowing errors would return
None and look like a successful empty answer.
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


def _mixin(vision):
    query = QueryMixin()
    query.logger = FakeLogger()
    query.vision_model_func = vision
    return query


@pytest.mark.asyncio
async def test_string_user_content_uses_prompt_and_system_prompt():
    calls = []

    async def vision(prompt, system_prompt=None, messages=None):
        calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "messages": messages}
        )
        return "text-answer"

    query = _mixin(vision)
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Context without images"},
    ]

    result = await query._call_vlm_with_multimodal_content(messages)

    assert result == "text-answer"
    assert calls == [
        {
            "prompt": "Context without images",
            "system_prompt": "Be brief.",
            "messages": None,
        }
    ]


@pytest.mark.asyncio
async def test_list_user_content_passes_full_messages():
    calls = []

    async def vision(prompt, system_prompt=None, messages=None):
        calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "messages": messages}
        )
        return "vision-answer"

    query = _mixin(vision)
    messages = [
        {"role": "system", "content": "Analyze images."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xx"}},
            ],
        },
    ]

    result = await query._call_vlm_with_multimodal_content(messages)

    assert result == "vision-answer"
    assert calls == [{"prompt": "", "system_prompt": None, "messages": messages}]


@pytest.mark.asyncio
async def test_vision_errors_are_reraised():
    async def vision(*args, **kwargs):
        raise RuntimeError("vision endpoint down")

    query = _mixin(vision)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    with pytest.raises(RuntimeError, match="vision endpoint down"):
        await query._call_vlm_with_multimodal_content(messages)
