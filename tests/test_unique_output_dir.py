"""Parser output directories must not collide for same-name files (#51)."""

from raganything.parser import Parser


def test_same_basename_in_different_dirs_gets_distinct_output_dirs(tmp_path):
    first = tmp_path / "plant-a" / "manual.pdf"
    second = tmp_path / "plant-b" / "manual.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    output = tmp_path / "parsed"
    dir_a = Parser._unique_output_dir(output, first)
    dir_b = Parser._unique_output_dir(output, second)

    assert dir_a != dir_b
    assert dir_a.parent == output
    assert dir_b.parent == output
    assert dir_a.name.startswith("manual_")
    assert dir_b.name.startswith("manual_")
    assert len(dir_a.name.rsplit("_", 1)[1]) == 8
    assert len(dir_b.name.rsplit("_", 1)[1]) == 8


def test_unique_output_dir_stable_for_same_absolute_path(tmp_path):
    pdf = tmp_path / "docs" / "manual.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    output = tmp_path / "parsed"

    first = Parser._unique_output_dir(output, pdf)
    second = Parser._unique_output_dir(output, pdf)
    via_str = Parser._unique_output_dir(str(output), str(pdf))

    assert first == second == via_str
    assert first.name.startswith("manual_")
