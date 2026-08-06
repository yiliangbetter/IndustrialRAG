"""Regression tests for Docling timeout and equation/table field alignment."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import importlib.util
import subprocess

import pytest


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "raganything" / "parser.py"
    spec = importlib.util.spec_from_file_location("_raganything_parser_timeout", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_docling_command_passes_default_timeout():
    mod = _load_parser_module()
    parser = mod.DoclingParser()

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="")
        parser._run_docling_command(
            input_path="/tmp/doc.pdf",
            output_dir="/tmp/out",
            file_stem="doc",
        )

    assert mock_run.called
    _, kwargs = mock_run.call_args
    assert kwargs.get("timeout") == mod.Parser.DEFAULT_DOCLING_TIMEOUT


def test_run_docling_command_timeout_raises_timeout_error():
    mod = _load_parser_module()
    parser = mod.DoclingParser()

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["docling"], timeout=1),
    ):
        with pytest.raises(TimeoutError, match="Docling did not finish"):
            parser._run_docling_command(
                input_path="/tmp/doc.pdf",
                output_dir="/tmp/out",
                file_stem="doc",
                timeout=1,
            )


def test_resolve_equation_fields_prefers_documented_latex():
    from raganything.utils import resolve_equation_fields

    text, fmt = resolve_equation_fields(
        {
            "type": "equation",
            "latex": "E = mc^2",
            "text": "mass-energy equivalence",
        }
    )
    assert text == "E = mc^2"
    assert fmt == "latex"

    # Parser-style payload (formula in text only)
    text2, fmt2 = resolve_equation_fields(
        {"type": "equation", "text": "a^2+b^2=c^2", "text_format": "latex"}
    )
    assert text2 == "a^2+b^2=c^2"
    assert fmt2 == "latex"


@pytest.mark.asyncio
async def test_query_table_accepts_table_body_and_equation_text():
    from raganything.query import QueryMixin

    captured = {}

    class Dummy:
        async def modal_caption_func(self, prompt, system_prompt=None):
            captured["prompt"] = prompt
            return "ok"

    mixin = QueryMixin.__new__(QueryMixin)
    mixin.logger = MagicMock()
    processor = Dummy()

    await QueryMixin._describe_table_for_query(
        mixin,
        processor,
        {"type": "table", "table_body": "| A | B |\n| 1 | 2 |", "table_caption": "T"},
    )
    assert "| A | B |" in captured["prompt"]

    await QueryMixin._describe_equation_for_query(
        mixin,
        processor,
        {"type": "equation", "text": "x^2", "equation_caption": "c"},
    )
    assert "x^2" in captured["prompt"]

    # Documented latex still wins over prose text
    await QueryMixin._describe_equation_for_query(
        mixin,
        processor,
        {
            "type": "equation",
            "latex": "E=mc^2",
            "text": "prose description",
            "equation_caption": "c",
        },
    )
    assert "E=mc^2" in captured["prompt"]
    assert "prose description" not in captured["prompt"]
