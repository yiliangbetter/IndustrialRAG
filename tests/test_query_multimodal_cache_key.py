from pathlib import Path

from raganything.query import QueryMixin


class _Dummy(QueryMixin):
    pass


def test_multimodal_cache_key_differs_for_same_basename_different_paths(tmp_path):
    img_a = tmp_path / "a" / "chart.png"
    img_b = tmp_path / "b" / "chart.png"
    img_a.parent.mkdir()
    img_b.parent.mkdir()
    img_a.write_bytes(b"aaa")
    img_b.write_bytes(b"bbb")

    dummy = _Dummy()
    key_a = dummy._generate_multimodal_cache_key(
        "describe this",
        [{"type": "image", "img_path": str(img_a)}],
        mode="mix",
    )
    key_b = dummy._generate_multimodal_cache_key(
        "describe this",
        [{"type": "image", "img_path": str(img_b)}],
        mode="mix",
    )

    assert Path(img_a).name == Path(img_b).name
    assert key_a != key_b


def test_multimodal_cache_key_includes_system_prompt():
    dummy = _Dummy()
    base = {
        "query": "q",
        "multimodal_content": [{"type": "table", "table_data": "a,b\n1,2"}],
        "mode": "mix",
    }
    key1 = dummy._generate_multimodal_cache_key(
        base["query"],
        base["multimodal_content"],
        base["mode"],
        system_prompt="answer briefly",
    )
    key2 = dummy._generate_multimodal_cache_key(
        base["query"],
        base["multimodal_content"],
        base["mode"],
        system_prompt="answer in detail",
    )
    assert key1 != key2
