"""Parse-cache key composition for ProcessorMixin._generate_cache_key.

The parse cache is keyed by this hash. Dropping a relevant parser option
serves a stale extract; hashing irrelevant kwargs fragments the cache.
"""

import os

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


def _processor(parser="mineru", parse_method="auto"):
    class DummyProcessor(ProcessorMixin):
        pass

    dummy = DummyProcessor()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {"parser": parser, "parse_method": parse_method},
    )()
    return dummy


def test_same_file_and_config_produce_stable_key(tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    dummy = _processor()

    first = dummy._generate_cache_key(pdf, "ocr", lang="en")
    second = dummy._generate_cache_key(pdf, "ocr", lang="en")

    assert first == second
    assert len(first) == 32


def test_parse_method_and_parser_change_the_key(tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    dummy = _processor(parser="mineru", parse_method="auto")

    auto_key = dummy._generate_cache_key(pdf, "auto")
    ocr_key = dummy._generate_cache_key(pdf, "ocr")
    dummy.config.parser = "paddleocr"
    paddle_key = dummy._generate_cache_key(pdf, "auto")

    assert auto_key != ocr_key
    assert auto_key != paddle_key


def test_none_parse_method_uses_config_default(tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    dummy = _processor(parse_method="txt")

    assert dummy._generate_cache_key(pdf, None) == dummy._generate_cache_key(pdf, "txt")


def test_mtime_change_invalidates_key(tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    dummy = _processor()

    before = dummy._generate_cache_key(pdf, "auto")
    os.utime(pdf, (pdf.stat().st_atime, pdf.stat().st_mtime + 5))
    after = dummy._generate_cache_key(pdf, "auto")

    assert before != after


def test_relevant_kwargs_are_part_of_the_key(tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    dummy = _processor()
    baseline = dummy._generate_cache_key(pdf, "auto")

    relevant = {
        "lang": "ch",
        "device": "cuda",
        "start_page": 2,
        "end_page": 8,
        "formula": False,
        "table": False,
        "backend": "vlm-sglang-engine",
        "source": "pipeline",
    }
    for name, value in relevant.items():
        assert dummy._generate_cache_key(pdf, "auto", **{name: value}) != baseline, name


def test_irrelevant_kwargs_do_not_change_the_key(tmp_path):
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    dummy = _processor()
    baseline = dummy._generate_cache_key(pdf, "auto", lang="en")

    noisy = dummy._generate_cache_key(
        pdf,
        "auto",
        lang="en",
        output_dir="/tmp/out",
        display_stats=True,
        method="ocr",
        callback=object(),
        extra="ignored",
    )

    assert noisy == baseline
