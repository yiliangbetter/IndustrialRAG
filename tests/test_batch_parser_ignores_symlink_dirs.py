"""BatchParser must not walk a directory symlink outside the requested folder.

``process_documents_batch`` selects files with ``filter_supported_files``,
which uses ``Path.rglob``. A symlinked subdirectory would otherwise copy
manuals from another drop into the same parse batch. An alias of a real
subfolder must not select the nested manual a second time. A non-recursive
scan must stay in the top directory.
"""

from pathlib import Path

from raganything.batch_parser import BatchParser


def _write_tree(folder: Path) -> None:
    nested = folder / "line-a"
    nested.mkdir(parents=True)
    (folder / "local.pdf").write_bytes(b"%PDF-1.4\n")
    (folder / "notes.csv").write_text("a,b\n")
    (nested / "child.pdf").write_bytes(b"%PDF-1.4\n")

    outside = folder.parent / "outside"
    outside.mkdir()
    (outside / "secret.pdf").write_bytes(b"%PDF-1.4\n")
    (folder / "escape").symlink_to(outside, target_is_directory=True)
    (folder / "alias").symlink_to(nested, target_is_directory=True)


def _resolved(paths) -> list[Path]:
    return sorted(Path(path).resolve() for path in paths)


def test_recursive_filter_skips_external_and_alias_symlink_dirs(tmp_path):
    folder = tmp_path / "docs"
    _write_tree(folder)
    parser = BatchParser(parser_type="mineru", skip_installation_check=True)

    found = _resolved(parser.filter_supported_files([str(folder)], recursive=True))

    assert found == sorted(
        [
            (folder / "local.pdf").resolve(),
            (folder / "line-a" / "child.pdf").resolve(),
        ]
    )
    assert all(path.name != "secret.pdf" for path in found)
    assert all(path.name != "notes.csv" for path in found)
    assert all("escape" not in path.parts for path in found)
    assert all("alias" not in path.parts for path in found)


def test_non_recursive_filter_ignores_nested_and_symlink_dirs(tmp_path):
    folder = tmp_path / "docs"
    _write_tree(folder)
    parser = BatchParser(parser_type="mineru", skip_installation_check=True)

    found = _resolved(parser.filter_supported_files([str(folder)], recursive=False))

    assert found == [(folder / "local.pdf").resolve()]
