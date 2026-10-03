"""BatchParser must not drop files that finish after as_completed times out."""

import time

from raganything.batch_parser import BatchParser


def test_process_batch_keeps_files_that_finish_after_wait_timeout(tmp_path, monkeypatch):
    """A total as_completed deadline used to omit later files from both lists.

    process_documents_with_rag_batch ingests only successful_files, and
    export_parse_json_no_llm.py exits 0 when failed_files is empty. Files that
    finished during executor shutdown were in neither list, so they were never
    indexed and never retried.
    """
    files = []
    for index in range(3):
        path = tmp_path / f"manual_{index}.txt"
        path.write_text(f"manual {index}\n", encoding="utf-8")
        files.append(str(path))

    def slow_process(self, file_path, output_dir, parse_method="auto", **kwargs):
        time.sleep(0.4)
        return True, file_path, None

    monkeypatch.setattr(BatchParser, "process_single_file", slow_process)

    parser = BatchParser(
        parser_type="mineru",
        max_workers=1,
        show_progress=False,
        timeout_per_file=0.5,
        skip_installation_check=True,
    )
    result = parser.process_batch(
        file_paths=files,
        output_dir=str(tmp_path / "out"),
        parse_method="auto",
        recursive=False,
    )

    assert set(result.successful_files) == set(files)
    assert result.failed_files == []
    assert result.total_files == len(files)
    assert len(result.successful_files) + len(result.failed_files) == result.total_files


def test_process_batch_records_worker_failure_after_wait_timeout(tmp_path, monkeypatch):
    files = []
    for index in range(3):
        path = tmp_path / f"page_{index}.txt"
        path.write_text("x\n", encoding="utf-8")
        files.append(str(path))
    failing = files[1]

    def mixed_process(self, file_path, output_dir, parse_method="auto", **kwargs):
        time.sleep(0.4)
        if file_path == failing:
            return False, file_path, "parse exploded"
        return True, file_path, None

    monkeypatch.setattr(BatchParser, "process_single_file", mixed_process)

    parser = BatchParser(
        parser_type="mineru",
        max_workers=1,
        show_progress=False,
        timeout_per_file=0.5,
        skip_installation_check=True,
    )
    result = parser.process_batch(
        file_paths=files,
        output_dir=str(tmp_path / "out"),
        parse_method="auto",
        recursive=False,
    )

    assert set(result.successful_files) == {files[0], files[2]}
    assert result.failed_files == [failing]
    assert result.errors[failing] == "parse exploded"
    assert len(result.successful_files) + len(result.failed_files) == result.total_files
