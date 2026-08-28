"""BatchMixin file-filter wrappers must honor config recursive defaults.

process_documents_batch construction is covered elsewhere. These wrappers
are what folder ingest uses to drop unsupported files; a recursive=None
regression would scan nested dumps or skip nested manuals depending on
the caller's omitted kwarg.
"""

from raganything.batch import BatchMixin
from raganything.config import RAGAnythingConfig


class FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def test_filter_supported_files_uses_config_recursive_default(monkeypatch):
    captured = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def filter_supported_files(self, file_paths, recursive):
            captured["filter"] = {"file_paths": file_paths, "recursive": recursive}
            return ["kept.pdf"]

    monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

    batch = BatchMixin()
    batch.logger = FakeLogger()
    batch.config = RAGAnythingConfig(
        parser="docling",
        recursive_folder_processing=False,
    )

    result = batch.filter_supported_files(["docs/", "a.csv"])

    assert result == ["kept.pdf"]
    assert captured["init"]["parser_type"] == "docling"
    assert captured["filter"] == {
        "file_paths": ["docs/", "a.csv"],
        "recursive": False,
    }


def test_filter_supported_files_explicit_recursive_overrides_config(monkeypatch):
    captured = {}

    class FakeBatchParser:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def filter_supported_files(self, file_paths, recursive):
            captured["recursive"] = recursive
            return []

    monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

    batch = BatchMixin()
    batch.logger = FakeLogger()
    batch.config = RAGAnythingConfig(parser="mineru", recursive_folder_processing=False)

    batch.filter_supported_files(["docs/"], recursive=True)

    assert captured["recursive"] is True
    assert captured["init"]["parser_type"] == "mineru"


def test_get_supported_file_extensions_delegates_to_parser(monkeypatch):
    class FakeBatchParser:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def get_supported_extensions(self):
            return [".pdf", ".docx"]

    monkeypatch.setattr("raganything.batch.BatchParser", FakeBatchParser)

    batch = BatchMixin()
    batch.logger = FakeLogger()
    batch.config = RAGAnythingConfig(parser="paddleocr")

    assert batch.get_supported_file_extensions() == [".pdf", ".docx"]
