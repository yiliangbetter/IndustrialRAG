"""Download cleanup must keep the original failure when unlink fails.

If a URL download errors after the temp file is created, Parser._download_file
must still raise RuntimeError wrapping the download error — not a secondary
OSError from a failed unlink. Callers (Docling URL ingest) depend on that
wrapper to fail closed without leaking temp-path exceptions.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _load_parser_class():
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location("_raganything_parser", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser


Parser = _load_parser_class()


def test_download_file_unlink_failure_still_raises_download_error():
    parser = Parser()
    leaked: list[Path] = []

    real_mkstemp = __import__("tempfile").mkstemp

    def tracking_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        leaked.append(Path(name))
        return fd, name

    response = MagicMock()
    response.headers.get.return_value = ""
    response.read.side_effect = OSError("connection reset")
    response.close = MagicMock()

    original_unlink = Path.unlink

    def boom_unlink(self, *args, **kwargs):
        if leaked and self == leaked[0]:
            raise PermissionError("unlink denied")
        return original_unlink(self, *args, **kwargs)

    with (
        patch("urllib.request.urlopen", return_value=response),
        patch("tempfile.mkstemp", side_effect=tracking_mkstemp),
        patch.object(Path, "unlink", boom_unlink),
    ):
        with pytest.raises(RuntimeError, match="Failed to download") as exc_info:
            parser._download_file("https://example.com/file.pdf")

    assert "connection reset" in str(exc_info.value)
    assert "unlink denied" not in str(exc_info.value)
    response.close.assert_called_once()
    assert leaked, "temp file should have been created before the download failed"
    for path in leaked:
        if path.exists():
            path.unlink()
