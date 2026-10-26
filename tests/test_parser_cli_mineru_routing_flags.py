"""Parser CLI must forward MinerU routing flags and reject unknown choices.

``--backend``, ``--device``, ``--source``, and ``--vlm_url`` decide whether
MinerU stays on the local pipeline or calls a remote VLM HTTP server, which
accelerator it uses, and where model weights come from. If those arguments
are dropped, a ``vlm-http-client`` run silently uses the local pipeline.
Unknown backend, source, and method values must fail before any parse starts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location(
        "_raganything_parser_cli_routing_9666", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


parser_mod = _load_parser_module()


def _install_fake_parser(monkeypatch):
    fake = MagicMock()
    fake.parse_document.return_value = [{"type": "text", "text": "ok"}]
    monkeypatch.setattr(parser_mod, "get_parser", MagicMock(return_value=fake))
    return fake


def test_cli_forwards_backend_device_source_and_vlm_url(monkeypatch):
    fake = _install_fake_parser(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "parser",
            "manual.pdf",
            "--backend",
            "vlm-http-client",
            "--device",
            "cuda:0",
            "--source",
            "local",
            "--vlm_url",
            "http://127.0.0.1:30000",
        ],
    )

    assert parser_mod.main() == 0

    parser_mod.get_parser.assert_called_once_with("mineru")
    kwargs = fake.parse_document.call_args.kwargs
    assert kwargs["file_path"] == "manual.pdf"
    assert kwargs["backend"] == "vlm-http-client"
    assert kwargs["device"] == "cuda:0"
    assert kwargs["source"] == "local"
    assert kwargs["vlm_url"] == "http://127.0.0.1:30000"
    assert kwargs["formula"] is True
    assert kwargs["table"] is True


def test_cli_omitted_routing_flags_keep_parser_defaults(monkeypatch):
    fake = _install_fake_parser(monkeypatch)
    monkeypatch.setattr("sys.argv", ["parser", "manual.pdf"])

    assert parser_mod.main() == 0

    kwargs = fake.parse_document.call_args.kwargs
    assert kwargs["backend"] == "pipeline"
    assert kwargs["source"] == "huggingface"
    assert kwargs["device"] is None
    assert kwargs["vlm_url"] is None
    assert kwargs["method"] == "auto"
    assert kwargs["output_dir"] is None
    assert kwargs["lang"] is None
    assert kwargs["formula"] is True
    assert kwargs["table"] is True


@pytest.mark.parametrize(
    "extra",
    [
        ["--backend", "not-a-backend"],
        ["--source", "internet"],
        ["--method", "vision"],
    ],
)
def test_cli_rejects_unknown_routing_choices_without_parsing(monkeypatch, extra):
    fake = _install_fake_parser(monkeypatch)
    monkeypatch.setattr("sys.argv", ["parser", "manual.pdf", *extra])

    with pytest.raises(SystemExit) as caught:
        parser_mod.main()

    assert caught.value.code == 2
    fake.parse_document.assert_not_called()
