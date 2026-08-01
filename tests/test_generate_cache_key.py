"""Regression tests for parse-cache key generation isolation."""

from pathlib import Path

from raganything.processor import ProcessorMixin


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _make_processor(parser="mineru", parse_method="auto"):
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {
            "parser": parser,
            "parse_method": parse_method,
        },
    )()
    return processor


def test_generate_cache_key_stable_for_same_inputs(tmp_path: Path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    key_a = processor._generate_cache_key(file_path, "ocr", lang="en", device="cpu")
    key_b = processor._generate_cache_key(file_path, "ocr", lang="en", device="cpu")

    assert key_a == key_b
    assert len(key_a) == 32


def test_generate_cache_key_changes_when_relevant_kwargs_change(tmp_path: Path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    key_en = processor._generate_cache_key(file_path, "ocr", lang="en")
    key_ch = processor._generate_cache_key(file_path, "ocr", lang="ch")
    key_device = processor._generate_cache_key(
        file_path, "ocr", lang="en", device="cuda"
    )
    key_pages = processor._generate_cache_key(
        file_path, "ocr", lang="en", start_page=0, end_page=2
    )

    assert key_en != key_ch
    assert key_en != key_device
    assert key_en != key_pages


def test_generate_cache_key_ignores_irrelevant_kwargs(tmp_path: Path):
    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    base = processor._generate_cache_key(file_path, "auto", lang="en", formula=True)
    polluted = processor._generate_cache_key(
        file_path,
        "auto",
        lang="en",
        formula=True,
        output_dir="/tmp/out",
        callback=object(),
        verbose=True,
        display_stats=False,
        ignored="noise",
    )

    assert base == polluted


def test_generate_cache_key_uses_config_parse_method_default(tmp_path: Path):
    processor = _make_processor(parse_method="txt")
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    default_key = processor._generate_cache_key(file_path)
    explicit_key = processor._generate_cache_key(file_path, "txt")
    other_key = processor._generate_cache_key(file_path, "ocr")

    assert default_key == explicit_key
    assert default_key != other_key


def test_generate_cache_key_changes_when_mtime_changes(tmp_path: Path):
    import os

    processor = _make_processor()
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4\n")

    before = processor._generate_cache_key(file_path, "auto")
    new_mtime = file_path.stat().st_mtime + 10
    os.utime(file_path, (new_mtime, new_mtime))
    after = processor._generate_cache_key(file_path, "auto")

    assert before != after
