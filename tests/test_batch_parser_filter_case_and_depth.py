"""BatchParser must select case-variant suffixes and nested manuals.

``process_documents_batch`` filters through ``BatchParser.filter_supported_files``,
not the folder-glob path. A case-sensitive suffix check drops Windows exports
such as ``Manual.PDF``. A recursive flag that ignores depth either skips plant
subfolders or ingests nested dumps. Two files that share a stem in different
folders must both be selected.
"""

import importlib.util
import sys
import types
from pathlib import Path


def _load_batch_parser():
    """Load batch_parser.py without importing the raganything package.

    Package import pulls LightRAG. The filter only needs parser format sets
    and tqdm.
    """
    package_name = "_raganything_batch_filter_67fd"
    root = Path(__file__).resolve().parents[1] / "raganything"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    def load(subname: str, filename: str):
        full_name = f"{package_name}.{subname}"
        spec = importlib.util.spec_from_file_location(full_name, root / filename)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[full_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    load("parser", "parser.py")
    return load("batch_parser", "batch_parser.py")


_batch_parser = _load_batch_parser()
BatchParser = _batch_parser.BatchParser


def _parser() -> BatchParser:
    return BatchParser(
        parser_type="mineru",
        skip_installation_check=True,
        show_progress=False,
    )


def test_recursive_keeps_uppercase_nested_files_and_both_same_stems(tmp_path):
    (tmp_path / "Manual.PDF").write_bytes(b"%PDF")
    (tmp_path / "notes.md").write_text("top")
    (tmp_path / "ignore.CSV").write_text("a,b")
    (tmp_path / "script.py").write_text("print(1)")

    nested = tmp_path / "plant" / "line-a"
    nested.mkdir(parents=True)
    (nested / "spec.DOCX").write_bytes(b"PK")
    (nested / "readme.TXT").write_text("steps")

    sibling = tmp_path / "other-plant"
    sibling.mkdir()
    (sibling / "Manual.PDF").write_bytes(b"%PDF")

    bp = _parser()
    flat = {
        Path(path).resolve()
        for path in bp.filter_supported_files([str(tmp_path)], recursive=False)
    }
    assert flat == {
        (tmp_path / "Manual.PDF").resolve(),
        (tmp_path / "notes.md").resolve(),
    }

    deep = {
        Path(path).resolve()
        for path in bp.filter_supported_files([str(tmp_path)], recursive=True)
    }
    assert deep == {
        (tmp_path / "Manual.PDF").resolve(),
        (tmp_path / "notes.md").resolve(),
        (nested / "spec.DOCX").resolve(),
        (nested / "readme.TXT").resolve(),
        (sibling / "Manual.PDF").resolve(),
    }


def test_direct_uppercase_image_kept_unsupported_and_missing_dropped(tmp_path):
    image = tmp_path / "figure.PNG"
    image.write_bytes(b"png")
    sheet = tmp_path / "table.csv"
    sheet.write_text("x")
    missing = tmp_path / "gone.pdf"

    selected = _parser().filter_supported_files(
        [str(image), str(sheet), str(missing)],
        recursive=True,
    )

    assert [Path(path).resolve() for path in selected] == [image.resolve()]


def test_dry_run_reports_selected_files_without_parsing(tmp_path):
    (tmp_path / "Manual.PDF").write_bytes(b"%PDF")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "spec.DOCX").write_bytes(b"PK")

    bp = _parser()

    def _explode(*args, **kwargs):
        raise AssertionError("dry run must not parse")

    bp.parser.parse_document = _explode

    result = bp.process_batch(
        [str(tmp_path)],
        output_dir=str(tmp_path / "out"),
        dry_run=True,
        recursive=True,
    )

    assert result.dry_run is True
    assert result.failed_files == []
    assert result.errors == {}
    assert result.total_files == 2
    assert result.processing_time == 0.0
    assert {Path(path).resolve() for path in result.successful_files} == {
        (tmp_path / "Manual.PDF").resolve(),
        (nested / "spec.DOCX").resolve(),
    }
