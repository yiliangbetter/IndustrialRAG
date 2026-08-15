"""Regression: multimodal query cache must not collapse distinct image paths."""

from pathlib import Path

from raganything.query import QueryMixin


class _QueryCacheHarness(QueryMixin):
    """Minimal stand-in so we can call the mixin cache-key helper directly."""

    pass


def _cache_key(query: str, multimodal_content, mode: str = "mix", **kwargs) -> str:
    return QueryMixin._generate_multimodal_cache_key(
        _QueryCacheHarness(),
        query,
        multimodal_content,
        mode,
        **kwargs,
    )


def test_same_basename_different_directories_produce_distinct_cache_keys():
    """MinerU commonly emits image_0.png / chart.png under per-document folders."""
    key_a = _cache_key(
        "Describe this figure",
        [{"type": "image", "img_path": "/uploads/report_a/chart.png"}],
    )
    key_b = _cache_key(
        "Describe this figure",
        [{"type": "image", "img_path": "/uploads/report_b/chart.png"}],
    )
    assert key_a != key_b


def test_image_path_and_file_path_keys_preserve_directory_identity():
    key_a = _cache_key(
        "What is shown?",
        [{"type": "image", "image_path": "/data/doc1/image_0.png"}],
    )
    key_b = _cache_key(
        "What is shown?",
        [{"type": "image", "image_path": "/data/doc2/image_0.png"}],
    )
    assert key_a != key_b

    key_c = _cache_key(
        "Summarize",
        [{"type": "image", "file_path": "/tmp/batch_a/fig.png"}],
    )
    key_d = _cache_key(
        "Summarize",
        [{"type": "image", "file_path": "/tmp/batch_b/fig.png"}],
    )
    assert key_c != key_d


def test_identical_absolute_paths_still_share_cache_key():
    path = "/uploads/report_a/chart.png"
    key_a = _cache_key("Describe this figure", [{"type": "image", "img_path": path}])
    key_b = _cache_key("Describe this figure", [{"type": "image", "img_path": path}])
    assert key_a == key_b


def test_relative_paths_resolve_to_absolute_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "left").mkdir()
    (tmp_path / "right").mkdir()
    left = Path("left") / "chart.png"
    right = Path("right") / "chart.png"
    left.write_bytes(b"a")
    right.write_bytes(b"b")

    key_left = _cache_key("Describe", [{"type": "image", "img_path": str(left)}])
    key_right = _cache_key("Describe", [{"type": "image", "img_path": str(right)}])
    assert key_left != key_right
