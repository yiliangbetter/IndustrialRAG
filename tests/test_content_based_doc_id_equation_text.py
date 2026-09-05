"""Content-based doc_id must treat equation text as its own identity.

Manuals often store formulas as type=equation with a `text` field. If that
branch hashes like a paragraph (no `equation:` prefix), an equation block and
a text block with the same formula collide and re-ingest / citations cross-wire.
"""

from raganything.processor import ProcessorMixin


class DummyProcessor(ProcessorMixin):
    pass


def _processor():
    return DummyProcessor()


def test_equation_text_does_not_collide_with_same_paragraph_text():
    processor = _processor()
    formula = "E = mc^2"
    equation_id = processor._generate_content_based_doc_id(
        [{"type": "equation", "text": formula}]
    )
    text_id = processor._generate_content_based_doc_id(
        [{"type": "text", "text": formula}]
    )

    assert equation_id.startswith("doc-")
    assert text_id.startswith("doc-")
    assert equation_id != text_id


def test_equation_text_change_changes_doc_id_and_is_stable():
    processor = _processor()
    first = processor._generate_content_based_doc_id(
        [{"type": "equation", "text": "F = ma"}]
    )
    same = processor._generate_content_based_doc_id(
        [{"type": "equation", "text": "F = ma"}]
    )
    other = processor._generate_content_based_doc_id(
        [{"type": "equation", "text": "P = IV"}]
    )

    assert first == same
    assert first != other
