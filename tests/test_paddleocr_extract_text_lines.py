"""Regression tests for PaddleOCRParser._extract_text_lines payload shapes."""

from raganything.parser import PaddleOCRParser


class _DictLike:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class TestExtractTextLines:
    def test_plain_string(self):
        parser = PaddleOCRParser()
        assert parser._extract_text_lines("  hello  ") == ["hello"]
        assert parser._extract_text_lines("   ") == []

    def test_rec_texts_list(self):
        parser = PaddleOCRParser()
        result = {"rec_texts": ["Line A", "  Line B  ", ""]}
        assert parser._extract_text_lines(result) == ["Line A", "Line B"]

    def test_text_and_texts_keys(self):
        parser = PaddleOCRParser()
        result = {"text": "solo", "texts": ["a", "b"]}
        assert parser._extract_text_lines(result) == ["solo", "a", "b"]

    def test_to_dict_object(self):
        parser = PaddleOCRParser()
        node = _DictLike({"rec_texts": ["via-dict"]})
        assert parser._extract_text_lines(node) == ["via-dict"]

    def test_classic_box_tuple_shape(self):
        parser = PaddleOCRParser()
        # [[[box], (text, conf)], ...]
        classic = [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("recognized", 0.99)],
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("second", 0.8)],
        ]
        assert parser._extract_text_lines(classic) == ["recognized", "second"]

    def test_confidence_pair_without_box(self):
        parser = PaddleOCRParser()
        assert parser._extract_text_lines([("only-text", 0.5)]) == ["only-text"]

    def test_all_string_list(self):
        parser = PaddleOCRParser()
        assert parser._extract_text_lines(["alpha", "beta"]) == ["alpha", "beta"]

    def test_nested_other_keys_not_revisited_for_handled_fields(self):
        """Handled keys (rec_texts/text/texts) must not be re-walked via siblings."""
        parser = PaddleOCRParser()
        payload = {
            "rec_texts": ["once"],
            "meta": {"ignored": True},
            # Nested structure under a different key should still be visited,
            # but must not re-enter the already-handled top-level keys.
            "pages": [{"note": "extra"}],
        }
        lines = parser._extract_text_lines(payload)
        assert lines.count("once") == 1

    def test_nested_page_results(self):
        parser = PaddleOCRParser()
        payload = [
            {"rec_texts": ["page1-line"]},
            {"texts": ["page2-line"]},
        ]
        assert parser._extract_text_lines(payload) == ["page1-line", "page2-line"]
