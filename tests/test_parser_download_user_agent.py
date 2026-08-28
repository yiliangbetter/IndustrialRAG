"""Parser URL downloads must send a browser User-Agent.

Some plant-manual hosts return 403 without it. Existing download tests cover
timeouts and temp cleanup, not the request headers.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_parser_class():
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location("_raganything_parser_ua", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser


Parser = _load_parser_class()


def test_download_file_sends_user_agent_header():
    parser = Parser()
    response = MagicMock()
    response.headers.get.return_value = "application/pdf"
    response.read = io.BytesIO(b"%PDF-1.4 fake").read
    response.close = MagicMock()

    with patch("urllib.request.urlopen", return_value=response) as mock_open:
        downloaded = parser._download_file("https://example.com/manual.pdf")

    try:
        req = mock_open.call_args.args[0]
        assert mock_open.call_args.kwargs.get("timeout") == 30
        user_agent = req.headers.get("User-agent") or req.headers.get("User-Agent")
        assert user_agent
        assert "Mozilla" in user_agent
    finally:
        if downloaded.exists():
            downloaded.unlink()
