"""Regression tests for RAGAnythingConfig comma-separated list parsing.

These list fields drive batch file discovery and context extraction filters.
Whitespace/empty-token edge cases are easy to break when env defaults change.
"""

from raganything.config import RAGAnythingConfig


class TestSupportedFileExtensionsParsing:
    def test_constructor_kwargs_override(self):
        config = RAGAnythingConfig(
            supported_file_extensions=[".pdf", ".md"],
        )
        assert config.supported_file_extensions == [".pdf", ".md"]

    def test_env_split_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv(
            "SUPPORTED_FILE_EXTENSIONS",
            " .pdf , .DOCX , .txt ",
        )
        config = RAGAnythingConfig()
        assert config.supported_file_extensions == [".pdf", ".DOCX", ".txt"]

    def test_env_split_keeps_empty_tokens_from_consecutive_commas(self, monkeypatch):
        # Document current behavior: strip only; empty tokens are retained.
        monkeypatch.setenv(
            "SUPPORTED_FILE_EXTENSIONS",
            ".pdf,,.md,",
        )
        config = RAGAnythingConfig()
        assert config.supported_file_extensions == [".pdf", "", ".md", ""]


class TestContextFilterContentTypesParsing:
    def test_constructor_kwargs_override(self):
        config = RAGAnythingConfig(
            context_filter_content_types=["text", "table", "image"],
        )
        assert config.context_filter_content_types == ["text", "table", "image"]

    def test_env_split_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv(
            "CONTEXT_FILTER_CONTENT_TYPES",
            " text , table ",
        )
        config = RAGAnythingConfig()
        assert config.context_filter_content_types == ["text", "table"]

    def test_env_split_keeps_empty_tokens_from_consecutive_commas(self, monkeypatch):
        monkeypatch.setenv(
            "CONTEXT_FILTER_CONTENT_TYPES",
            "text,,image,",
        )
        config = RAGAnythingConfig()
        assert config.context_filter_content_types == ["text", "", "image", ""]
