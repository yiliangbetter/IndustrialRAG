from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import query_progress_hooks
import rag_web_server


def test_cached_clarify_keywords_reused_for_identical_query():
    query_progress_hooks.set_clarify_context_injection(
        {
            "query": "How often?",
            "raw_data": {
                "metadata": {
                    "keywords": {
                        "high_level": ["maintenance"],
                        "low_level": ["NB9", "inspection"],
                    }
                }
            },
        }
    )
    try:
        assert rag_web_server._cached_clarify_keyword_extras("How often?") == {
            "hl_keywords": ["maintenance"],
            "ll_keywords": ["NB9", "inspection"],
        }
    finally:
        query_progress_hooks.clear_clarify_context_injection()


def test_cached_clarify_keywords_not_reused_for_different_query():
    query_progress_hooks.set_clarify_context_injection(
        {
            "query": "original",
            "raw_data": {
                "metadata": {
                    "keywords": {"high_level": ["wrong"], "low_level": ["wrong"]}
                }
            },
        }
    )
    try:
        assert rag_web_server._cached_clarify_keyword_extras("different") == {}
    finally:
        query_progress_hooks.clear_clarify_context_injection()
