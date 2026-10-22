"""Parse-cache identity for MinerU backend, source, and device.

Formula flags and page bounds already have coverage. ``backend``, ``source``,
and ``device`` also change the extract (pipeline vs VLM, local weights vs a
hub, CPU vs CUDA). An empty string is a real "caller passed this option"
value and must not collapse into an omitted option. If either the cache key
or the stored parse config drops these fields, the next ingest replays the
wrong parse.
"""

import pytest

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


class FakeKVStorage:
    def __init__(self):
        self.records = {}
        self.index_done_calls = 0

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def index_done_callback(self):
        self.index_done_calls += 1


def _processor():
    class DummyProcessor(ProcessorMixin):
        pass

    processor = DummyProcessor()
    processor.logger = FakeLogger()
    processor.config = type(
        "Config",
        (),
        {"parser": "mineru", "parse_method": "auto"},
    )()
    processor.parse_cache = FakeKVStorage()
    return processor


def _pdf(tmp_path):
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    return path


def test_cache_key_separates_backend_source_and_device(tmp_path):
    processor = _processor()
    path = _pdf(tmp_path)

    omitted = processor._generate_cache_key(path, "auto")
    pipeline = processor._generate_cache_key(path, "auto", backend="pipeline")
    vlm = processor._generate_cache_key(path, "auto", backend="vlm-transformers")
    local = processor._generate_cache_key(path, "auto", source="local")
    hub = processor._generate_cache_key(path, "auto", source="modelscope")
    cpu = processor._generate_cache_key(path, "auto", device="cpu")
    cuda = processor._generate_cache_key(path, "auto", device="cuda")
    blank_backend = processor._generate_cache_key(path, "auto", backend="")

    assert len({omitted, pipeline, vlm, local, hub, cpu, cuda, blank_backend}) == 8


@pytest.mark.asyncio
async def test_empty_backend_source_and_device_do_not_match_an_omitted_parse(
    tmp_path,
):
    processor = _processor()
    path = _pdf(tmp_path)
    content = [{"type": "text", "text": "nameplate"}]
    key = processor._generate_cache_key(path, "auto", backend="", source="", device="")

    await processor._store_cached_result(
        key,
        content,
        "doc-blank",
        path,
        backend="",
        source="",
        device="",
    )

    stored = processor.parse_cache.records[key]["parse_config"]
    assert stored == {
        "parser": "mineru",
        "parse_method": "auto",
        "backend": "",
        "source": "",
        "device": "",
    }
    assert processor.parse_cache.index_done_calls == 1

    hit = await processor._get_cached_result(
        key, path, backend="", source="", device=""
    )
    assert hit == (content, "doc-blank")

    # Omitting the options is a different parse than passing empty strings.
    assert await processor._get_cached_result(key, path) is None


@pytest.mark.asyncio
async def test_stored_parse_is_not_reused_for_another_backend_source_or_device(
    tmp_path,
):
    processor = _processor()
    path = _pdf(tmp_path)
    content = [{"type": "text", "text": "torque table"}]
    options = {"backend": "pipeline", "source": "local", "device": "cpu"}
    key = processor._generate_cache_key(path, "auto", **options)

    await processor._store_cached_result(
        key,
        content,
        "doc-pipeline",
        path,
        **options,
    )

    assert await processor._get_cached_result(key, path, **options) == (
        content,
        "doc-pipeline",
    )

    # Same cache row, different runtime target: the stored config must reject it
    # even if a caller reuses the original key.
    assert (
        await processor._get_cached_result(
            key, path, backend="vlm-transformers", source="local", device="cpu"
        )
        is None
    )
    assert (
        await processor._get_cached_result(
            key, path, backend="pipeline", source="modelscope", device="cpu"
        )
        is None
    )
    assert (
        await processor._get_cached_result(
            key, path, backend="pipeline", source="local", device="cuda"
        )
        is None
    )

    vlm_key = processor._generate_cache_key(
        path, "auto", backend="vlm-transformers", source="local", device="cpu"
    )
    hub_key = processor._generate_cache_key(
        path, "auto", backend="pipeline", source="modelscope", device="cpu"
    )
    cuda_key = processor._generate_cache_key(
        path, "auto", backend="pipeline", source="local", device="cuda"
    )
    assert len({key, vlm_key, hub_key, cuda_key}) == 4
