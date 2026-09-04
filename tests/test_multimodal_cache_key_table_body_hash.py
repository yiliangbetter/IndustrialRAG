"""Long table_body values must be hashed in multimodal query cache keys.

table_data hashing is covered by an open coverage PR. table_body uses the
same 200-character threshold; storing it verbatim would bloat keys and
treat equivalent tables as cache misses after trivial whitespace edits
inside a huge markdown table.
"""

from raganything.query import QueryMixin


class DummyQuery(QueryMixin):
    pass


def _key(table_body: str) -> str:
    query = DummyQuery()
    return query._generate_multimodal_cache_key(
        "what is the torque spec?",
        [{"type": "table", "table_body": table_body}],
        "mix",
    )


def test_long_table_body_changes_key_and_is_stable():
    long_a = "A" * 250
    long_b = "B" * 250
    key_a = _key(long_a)
    key_a_again = _key(long_a)
    key_b = _key(long_b)
    short = _key("small")

    assert key_a == key_a_again
    assert key_a != key_b
    assert key_a != short
    assert key_a.startswith("multimodal_query:")


def test_table_body_at_threshold_is_not_hashed_like_long_body():
    """len == 200 stays verbatim; len == 201 is hashed under a different key field."""
    at_limit = "x" * 200
    over_limit = "x" * 201
    assert _key(at_limit) != _key(over_limit)
