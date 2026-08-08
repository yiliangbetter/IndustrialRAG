"""Regression: multimodal query cache must isolate QueryParam-affecting kwargs."""

from raganything.query import QueryMixin


class _CacheKeyHost(QueryMixin):
    """Minimal host for exercising QueryMixin._generate_multimodal_cache_key."""

    pass


def _key(**kwargs) -> str:
    host = _CacheKeyHost()
    return host._generate_multimodal_cache_key(
        "What does this show?",
        [{"type": "table", "table_body": "| A | 1 |\n| B | 2 |"}],
        "mix",
        **kwargs,
    )


def test_conversation_history_changes_cache_key():
    base = _key(conversation_history=[])
    follow_up = _key(
        conversation_history=[
            {"role": "user", "content": "Focus on column B"},
            {"role": "assistant", "content": "Understood."},
        ]
    )
    assert base != follow_up


def test_only_need_context_changes_cache_key():
    assert _key(only_need_context=False) != _key(only_need_context=True)


def test_enable_rerank_and_chunk_top_k_change_cache_key():
    base = _key(enable_rerank=True, chunk_top_k=20)
    assert base != _key(enable_rerank=False, chunk_top_k=20)
    assert base != _key(enable_rerank=True, chunk_top_k=5)


def test_identical_kwargs_keep_same_cache_key():
    history = [{"role": "user", "content": "hi"}]
    assert _key(conversation_history=history, only_need_context=True) == _key(
        conversation_history=history, only_need_context=True
    )
