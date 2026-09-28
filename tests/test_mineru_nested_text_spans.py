"""Nested MinerU text spans must stay searchable prose.

``_mineru_span_text`` returns immediately only when ``type`` is ``text`` and
``content`` is a string. A text node whose content is a list of spans has to
fall through and join those spans. Stringifying the list would store a Python
repr instead of the manual sentence. A non-text span whose content is a
string (inline equation) has to be kept the same way. Paragraph harvest uses
this helper, so a break here drops text from embedding-only and graph ingest.
"""

from raganything.processor import ProcessorMixin


def test_text_span_list_is_joined_not_stringified():
    proc = ProcessorMixin()
    node = {
        "type": "text",
        "content": [
            {"type": "text", "content": "Voltage "},
            {"type": "inline_equation", "content": r"220\,V"},
            {"type": "text", "content": "   "},
        ],
    }

    text = proc._mineru_span_text(node)

    assert text == r"Voltage 220\,V"
    assert "[" not in text


def test_paragraph_harvest_keeps_inline_equation_text():
    proc = ProcessorMixin()
    items = [
        {
            "type": "paragraph",
            "content": {
                "paragraph_content": {
                    "type": "text",
                    "content": [
                        {"type": "text", "content": "Set "},
                        {"type": "equation", "content": "I=V/R"},
                    ],
                }
            },
        }
    ]

    assert proc._plaintext_from_mineru_blocks(items) == "Set I=V/R"
