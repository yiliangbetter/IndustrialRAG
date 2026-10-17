"""Multimodal cache keys must switch table payloads at 200 characters.

``table_body`` and ``table_data`` longer than 200 characters are replaced with
an MD5 before the cache key is hashed. A body of exactly 200 characters stays
in the key. Answer-shaping kwargs (system prompt, token cap, stream, response
type) are part of that same key, so a different instruction cannot reuse an
answer cached for another one.
"""

import hashlib
import json

from raganything.query import QueryMixin


class _Query(QueryMixin):
    pass


def _cache_key(cache_data: dict) -> str:
    encoded = json.dumps(cache_data, sort_keys=True, ensure_ascii=False).encode()
    return "multimodal_query:" + hashlib.md5(encoded).hexdigest()


def _actual(content, **kwargs) -> str:
    return _Query()._generate_multimodal_cache_key("spec", content, "mix", **kwargs)


def test_table_strings_of_length_200_stay_raw_and_201_are_hashed():
    body_200 = "A" * 200
    body_201 = "B" * 201
    data_200 = "C" * 200
    data_201 = "D" * 201

    assert _actual([{"type": "table", "table_body": body_200}]) == _cache_key(
        {
            "query": "spec",
            "mode": "mix",
            "multimodal_content": [{"type": "table", "table_body": body_200}],
        }
    )
    assert _actual([{"type": "table", "table_body": body_201}]) == _cache_key(
        {
            "query": "spec",
            "mode": "mix",
            "multimodal_content": [
                {
                    "type": "table",
                    "table_body_hash": hashlib.md5(body_201.encode()).hexdigest(),
                }
            ],
        }
    )
    assert _actual([{"type": "table", "table_data": data_200}]) == _cache_key(
        {
            "query": "spec",
            "mode": "mix",
            "multimodal_content": [{"type": "table", "table_data": data_200}],
        }
    )
    assert _actual([{"type": "table", "table_data": data_201}]) == _cache_key(
        {
            "query": "spec",
            "mode": "mix",
            "multimodal_content": [
                {
                    "type": "table",
                    "table_data_hash": hashlib.md5(data_201.encode()).hexdigest(),
                }
            ],
        }
    )


def test_non_string_table_body_is_not_hashed():
    rows = ["row"] * 201
    assert _actual([{"table_body": rows}]) == _cache_key(
        {
            "query": "spec",
            "mode": "mix",
            "multimodal_content": [{"table_body": rows}],
        }
    )


def test_answer_shaping_kwargs_change_the_cache_key():
    content = [{"type": "text", "text": "note"}]
    base = {
        "query": "spec",
        "mode": "mix",
        "multimodal_content": content,
    }

    assert _actual(content, system_prompt="cite the manual") == _cache_key(
        {**base, "system_prompt": "cite the manual"}
    )
    assert _actual(content, max_tokens=128) == _cache_key({**base, "max_tokens": 128})
    assert _actual(content, stream=True) == _cache_key({**base, "stream": True})
    assert _actual(content, response_type="bullet") == _cache_key(
        {**base, "response_type": "bullet"}
    )
    assert _actual(content) == _cache_key(base)
