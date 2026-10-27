"""Image parses fall back to MinerU only when the selected parser cannot.

Docling inherits ``Parser.parse_image``, which raises ``NotImplementedError``.
``ProcessorMixin.parse_document`` must send that case to ``MineruParser`` and
cache the result under the configured parser. Any other image-parse failure
must surface, and uppercase or less common image suffixes must stay on the
image route instead of the generic parser.
"""

import pytest

from raganything.callbacks import CallbackManager, ProcessingCallback
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


class RecordingCallback(ProcessingCallback):
    def __init__(self):
        self.events = []

    def on_parse_start(self, file_path, **kwargs):
        self.events.append("start")

    def on_parse_complete(self, file_path, content_blocks=0, doc_id="", **kwargs):
        self.events.append(("complete", content_blocks, doc_id))

    def on_parse_error(self, file_path, error="", **kwargs):
        self.events.append(("error", type(error).__name__))


class FakeParseCache:
    def __init__(self):
        self.records = {}
        self.upserts = []

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.upserts.append(data)
        self.records.update(data)

    async def index_done_callback(self):
        return None


class SelectedParser:
    def __init__(self, image_result=None, image_error=None):
        self.image_result = image_result
        self.image_error = image_error
        self.calls = []

    def parse_image(self, **kwargs):
        self.calls.append(("image", kwargs))
        if self.image_error is not None:
            raise self.image_error
        return self.image_result

    def parse_pdf(self, **kwargs):
        self.calls.append(("pdf", kwargs))
        raise AssertionError("image file must not use the PDF route")

    def parse_office_doc(self, **kwargs):
        self.calls.append(("office", kwargs))
        raise AssertionError("image file must not use the Office route")

    def parse_document(self, **kwargs):
        self.calls.append(("generic", kwargs))
        raise AssertionError("image file must not use the generic route")


def _processor(tmp_path, parser, cache):
    callback = RecordingCallback()
    manager = CallbackManager()
    manager.register(callback)

    dummy = ProcessorMixin()
    dummy.logger = FakeLogger()
    dummy.config = type(
        "Config",
        (),
        {
            "parser": "docling",
            "parser_output_dir": str(tmp_path / "output"),
            "parse_method": "auto",
            "display_content_stats": False,
        },
    )()
    dummy.doc_parser = parser
    dummy.parse_cache = cache
    dummy.callback_manager = manager
    return dummy, callback


def _write_image(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"image-bytes")
    return path


@pytest.mark.asyncio
async def test_not_implemented_image_parse_falls_back_to_mineru_and_caches(
    tmp_path, monkeypatch
):
    source = _write_image(tmp_path, "scan.png")
    output_dir = str(tmp_path / "parsed")
    mineru_blocks = [{"type": "text", "text": "ocr text", "page_idx": 0}]
    created = []

    class FakeMineru:
        def __init__(self):
            created.append(self)
            self.calls = []

        def parse_image(self, **kwargs):
            self.calls.append(kwargs)
            return mineru_blocks

    monkeypatch.setattr("raganything.processor.MineruParser", FakeMineru)

    selected = SelectedParser(
        image_error=NotImplementedError("parse_image must be implemented by subclasses")
    )
    cache = FakeParseCache()
    processor, callback = _processor(tmp_path, selected, cache)

    content_list, doc_id = await processor.parse_document(
        str(source),
        output_dir=output_dir,
        parse_method="ocr",
        lang="de",
        backend="pipeline",
    )

    assert content_list == mineru_blocks
    assert doc_id.startswith("doc-")
    assert processor.doc_parser is selected
    assert len(created) == 1
    assert [name for name, _kwargs in selected.calls] == ["image"]
    selected_kwargs = selected.calls[0][1]
    mineru_kwargs = created[0].calls[0]
    for kwargs in (selected_kwargs, mineru_kwargs):
        assert kwargs["image_path"] == source
        assert kwargs["output_dir"] == output_dir
        assert kwargs["lang"] == "de"
        assert kwargs["backend"] == "pipeline"
        assert "method" not in kwargs

    assert len(cache.upserts) == 1
    stored = next(iter(cache.records.values()))
    assert stored["content_list"] == mineru_blocks
    assert stored["doc_id"] == doc_id
    assert stored["parse_config"] == {
        "parser": "docling",
        "parse_method": "ocr",
        "lang": "de",
        "backend": "pipeline",
    }
    assert callback.events == ["start", ("complete", 1, doc_id)]

    content_again, doc_id_again = await processor.parse_document(
        str(source),
        output_dir=output_dir,
        parse_method="ocr",
        lang="de",
        backend="pipeline",
    )

    assert content_again == mineru_blocks
    assert doc_id_again == doc_id
    assert len(selected.calls) == 1
    assert len(created) == 1
    assert len(created[0].calls) == 1
    assert len(cache.upserts) == 1


@pytest.mark.asyncio
async def test_empty_mineru_image_fallback_is_not_cached(tmp_path, monkeypatch):
    source = _write_image(tmp_path, "blank.png")
    created = []

    class FakeMineru:
        def __init__(self):
            created.append(self)

        def parse_image(self, **kwargs):
            return []

    monkeypatch.setattr("raganything.processor.MineruParser", FakeMineru)

    selected = SelectedParser(image_error=NotImplementedError("no image parser"))
    cache = FakeParseCache()
    processor, callback = _processor(tmp_path, selected, cache)

    with pytest.raises(ValueError, match="No content was extracted"):
        await processor.parse_document(str(source), lang="en")

    assert len(created) == 1
    assert [name for name, _kwargs in selected.calls] == ["image"]
    assert processor.doc_parser is selected
    assert cache.upserts == []
    assert cache.records == {}
    assert callback.events == ["start"]


@pytest.mark.asyncio
async def test_image_parse_runtime_error_does_not_fall_back(tmp_path, monkeypatch):
    source = _write_image(tmp_path, "broken.png")
    constructed = []

    class GuardMineru:
        def __init__(self):
            constructed.append(self)
            raise AssertionError("runtime image errors must not fall back to MinerU")

    monkeypatch.setattr("raganything.processor.MineruParser", GuardMineru)

    selected = SelectedParser(image_error=RuntimeError("decode failed"))
    cache = FakeParseCache()
    processor, callback = _processor(tmp_path, selected, cache)

    with pytest.raises(RuntimeError, match="decode failed"):
        await processor.parse_document(str(source), lang="en")

    assert constructed == []
    assert [name for name, _kwargs in selected.calls] == ["image"]
    assert cache.upserts == []
    assert callback.events == ["start", ("error", "RuntimeError")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename",
    ["Photo.PNG", "Scan.GIF", "Plate.WEBP", "Drawing.TIFF", "Legacy.BMP"],
)
async def test_image_suffixes_use_selected_parser_without_mineru(
    tmp_path, monkeypatch, filename
):
    source = _write_image(tmp_path, filename)
    constructed = []

    class GuardMineru:
        def __init__(self):
            constructed.append(self)
            raise AssertionError("a working image parser must not call MinerU")

    monkeypatch.setattr("raganything.processor.MineruParser", GuardMineru)

    selected = SelectedParser(
        image_result=[{"type": "text", "text": "from-selected", "page_idx": 0}]
    )
    processor, callback = _processor(tmp_path, selected, cache=None)

    content_list, doc_id = await processor.parse_document(str(source), lang="ch")

    assert content_list[0]["text"] == "from-selected"
    assert doc_id.startswith("doc-")
    assert constructed == []
    assert [name for name, _kwargs in selected.calls] == ["image"]
    assert selected.calls[0][1]["image_path"] == source
    assert selected.calls[0][1]["lang"] == "ch"
    assert callback.events[0] == "start"
    assert callback.events[1][0] == "complete"
