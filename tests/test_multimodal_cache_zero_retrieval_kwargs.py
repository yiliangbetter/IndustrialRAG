"""Zero retrieval knobs must not share a multimodal query cache entry.

``temperature=0`` asks for a greedy completion and ``top_k=0`` changes how
many chunks are retrieved. ``stream=False`` and ``max_tokens=0`` are also
falsy. Dropping them from the cache key serves an answer cached for the
omitted defaults.
"""

from raganything.query import QueryMixin


class _Query(QueryMixin):
    pass


def test_falsy_retrieval_kwargs_do_not_match_the_omitted_key():
    content = [{"type": "image", "img_path": "/manuals/torque.png"}]
    query = _Query()

    omitted = query._generate_multimodal_cache_key("torque spec", content, "mix")
    greedy = query._generate_multimodal_cache_key(
        "torque spec", content, "mix", temperature=0.0
    )
    no_hits = query._generate_multimodal_cache_key(
        "torque spec", content, "mix", top_k=0
    )
    both = query._generate_multimodal_cache_key(
        "torque spec", content, "mix", temperature=0.0, top_k=0
    )
    non_stream = query._generate_multimodal_cache_key(
        "torque spec", content, "mix", stream=False
    )
    zero_tokens = query._generate_multimodal_cache_key(
        "torque spec", content, "mix", max_tokens=0
    )

    keys = {omitted, greedy, no_hits, both, non_stream, zero_tokens}
    assert len(keys) == 6
    assert all(key.startswith("multimodal_query:") for key in keys)
