"""Multimodal query cache keys must ignore query padding and dict key order.

``_generate_multimodal_cache_key`` strips the question before hashing. Padding
must not miss the cache, and a real wording change must not collide with it.
Non-dict multimodal items stay in the key so two different raw values cannot
share an answer. Dict key order is normalized so the same table is one entry.
"""

from raganything.query import QueryMixin


class DummyQuery(QueryMixin):
    pass


def _key(query, content, mode="mix"):
    return DummyQuery()._generate_multimodal_cache_key(query, content, mode)


def test_query_padding_shares_a_key_and_wording_changes_do_not():
    padded = _key("  torque spec  ", [])
    trimmed = _key("torque spec", None)
    blank = _key("   ", [])
    empty = _key("", None)
    internal_space = _key("torque  spec", [])

    assert padded == trimmed
    assert blank == empty
    assert padded != internal_space
    assert padded != blank
    assert padded.startswith("multimodal_query:")


def test_non_dict_multimodal_items_remain_distinct_in_the_key():
    alpha = _key("what is this?", ["alpha"])
    beta = _key("what is this?", ["beta"])
    empty = _key("what is this?", [])

    assert alpha != beta
    assert alpha != empty


def test_multimodal_dict_key_order_does_not_change_the_cache_key():
    first = _key("q", [{"b": 1, "a": "pump"}])
    second = _key("q", [{"a": "pump", "b": 1}])

    assert first == second
