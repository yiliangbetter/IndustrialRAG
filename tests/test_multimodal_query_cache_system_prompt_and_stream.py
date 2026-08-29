"""Multimodal query cache keys must include system_prompt/stream/max_tokens.

temperature/top_k hashing is covered by an open coverage PR. These remaining
QueryParam fields still change the completion; dropping them serves an answer
generated under a different prompt or token budget.
"""

from raganything.query import QueryMixin


class DummyQuery(QueryMixin):
    pass


def _key(**kwargs):
    query = DummyQuery()
    return query._generate_multimodal_cache_key(
        "what is the torque spec?",
        [{"type": "table", "table_body": "bolt|nm"}],
        "mix",
        **kwargs,
    )


def test_system_prompt_changes_cache_key():
    default = _key()
    with_prompt = _key(system_prompt="Answer only from the manual.")
    other_prompt = _key(system_prompt="Be concise.")

    assert default != with_prompt
    assert with_prompt != other_prompt
    assert default.startswith("multimodal_query:")


def test_stream_and_max_tokens_change_cache_key():
    base = _key(system_prompt="sys", stream=False, max_tokens=512)
    streamed = _key(system_prompt="sys", stream=True, max_tokens=512)
    longer = _key(system_prompt="sys", stream=False, max_tokens=2048)
    response_type = _key(
        system_prompt="sys",
        stream=False,
        max_tokens=512,
        response_type="Multiple Paragraphs",
    )

    assert base != streamed
    assert base != longer
    assert base != response_type


def test_identical_relevant_kwargs_are_stable():
    first = _key(
        system_prompt="sys", stream=False, max_tokens=256, response_type="Bullet"
    )
    second = _key(
        system_prompt="sys", stream=False, max_tokens=256, response_type="Bullet"
    )
    assert first == second
