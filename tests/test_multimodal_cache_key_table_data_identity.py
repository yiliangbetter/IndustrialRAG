"""Multimodal query cache keys must hash large table_data and ignore None vs [].

table_data (not only table_body) is hashed when longer than 200 chars so cache
identity stays stable without embedding the full table. None multimodal content
must match an empty list, and leading/trailing query whitespace must not fork
the key — otherwise identical questions miss the cache.
"""

from raganything.query import QueryMixin


class DummyQuery(QueryMixin):
    pass


def test_long_table_data_changes_key_and_is_not_verbatim():
    query = DummyQuery()
    long_a = "A" * 250
    long_b = "B" * 250
    key_a = query._generate_multimodal_cache_key(
        "q", [{"type": "table", "table_data": long_a}], "mix"
    )
    key_b = query._generate_multimodal_cache_key(
        "q", [{"type": "table", "table_data": long_b}], "mix"
    )
    short = query._generate_multimodal_cache_key(
        "q", [{"type": "table", "table_data": "small"}], "mix"
    )

    assert key_a != key_b
    assert key_a != short
    assert key_a.startswith("multimodal_query:")


def test_none_multimodal_matches_empty_list():
    query = DummyQuery()
    none_key = query._generate_multimodal_cache_key("q", None, "mix")
    empty_key = query._generate_multimodal_cache_key("q", [], "mix")
    assert none_key == empty_key


def test_query_whitespace_is_stripped_before_hashing():
    query = DummyQuery()
    content = [{"type": "equation", "latex": "E=mc^2"}]
    padded = query._generate_multimodal_cache_key("  rating?  ", content, "mix")
    plain = query._generate_multimodal_cache_key("rating?", content, "mix")
    assert padded == plain


def test_non_dict_items_are_kept_in_normalized_content():
    query = DummyQuery()
    with_note = query._generate_multimodal_cache_key("q", ["not-a-dict"], "mix")
    other_note = query._generate_multimodal_cache_key("q", ["other"], "mix")
    assert with_note != other_note
