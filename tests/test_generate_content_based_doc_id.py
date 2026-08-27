"""Regression tests for ProcessorMixin._generate_content_based_doc_id.

Re-ingest identity and duplicate detection hash selected fields. Changing
which text/table/image identifiers contribute silently forks the knowledge
graph across otherwise identical documents.
"""

from raganything.processor import ProcessorMixin


def _processor() -> ProcessorMixin:
    return ProcessorMixin.__new__(ProcessorMixin)


class TestGenerateContentBasedDocId:
    def test_prefix_and_stability(self):
        processor = _processor()
        content = [{"type": "text", "text": "  Hello pump  "}]
        first = processor._generate_content_based_doc_id(content)
        second = processor._generate_content_based_doc_id(
            [{"type": "text", "text": "Hello pump"}]
        )
        assert first.startswith("doc-")
        assert first == second

    def test_text_change_changes_id(self):
        processor = _processor()
        left = processor._generate_content_based_doc_id(
            [{"type": "text", "text": "alpha"}]
        )
        right = processor._generate_content_based_doc_id(
            [{"type": "text", "text": "beta"}]
        )
        assert left != right

    def test_image_path_and_table_body_contribute(self):
        processor = _processor()
        image_a = processor._generate_content_based_doc_id(
            [{"type": "image", "img_path": "/out/fig-a.png"}]
        )
        image_b = processor._generate_content_based_doc_id(
            [{"type": "image", "img_path": "/out/fig-b.png"}]
        )
        table_a = processor._generate_content_based_doc_id(
            [{"type": "table", "table_body": "<table>1</table>"}]
        )
        table_b = processor._generate_content_based_doc_id(
            [{"type": "table", "table_body": "<table>2</table>"}]
        )
        assert image_a != image_b
        assert table_a != table_b
        assert image_a != table_a

    def test_non_dict_items_are_ignored(self):
        processor = _processor()
        with_noise = processor._generate_content_based_doc_id(
            ["not-a-block", {"type": "text", "text": "keep"}]
        )
        clean = processor._generate_content_based_doc_id(
            [{"type": "text", "text": "keep"}]
        )
        assert with_noise == clean

    def test_whitespace_only_text_still_contributes(self):
        # item.get("text") is truthy for "   ", then strip() stores "".
        processor = _processor()
        with_blank = processor._generate_content_based_doc_id(
            [{"type": "text", "text": "   "}, {"type": "text", "text": "keep"}]
        )
        without_blank = processor._generate_content_based_doc_id(
            [{"type": "text", "text": "keep"}]
        )
        assert with_blank != without_blank
