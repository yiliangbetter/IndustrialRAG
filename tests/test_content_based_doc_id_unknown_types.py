"""Unknown MinerU block types must still change the content-based doc_id.

Charts, code, and algorithm blocks fall through to `str(item)`. If that branch
is dropped, two manuals that differ only in a non-text/image/table/equation
block would share a doc_id and silently overwrite each other.
"""

from raganything.processor import ProcessorMixin


def _proc():
    class DummyProcessor(ProcessorMixin):
        pass

    return DummyProcessor()


def test_unknown_block_payload_changes_content_based_doc_id():
    proc = _proc()
    id_a = proc._generate_content_based_doc_id(
        [{"type": "chart", "payload": "pump-curve-a"}]
    )
    id_b = proc._generate_content_based_doc_id(
        [{"type": "chart", "payload": "pump-curve-b"}]
    )

    assert id_a.startswith("doc-")
    assert id_b.startswith("doc-")
    assert id_a != id_b


def test_identical_unknown_blocks_are_stable():
    proc = _proc()
    item = {"type": "code", "content": {"code_body": "print(1)"}}
    first = proc._generate_content_based_doc_id([item])
    second = proc._generate_content_based_doc_id([item])
    assert first == second
