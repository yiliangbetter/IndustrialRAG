"""Local HF embeddings must fail with an install hint when the extra is missing.

Operators set EMBEDDING_BACKEND=hf without `raganything[local-embed]`. A bare
ImportError from sentence-transformers would hide the required extra; the
factory must wrap it with the install command.
"""

import builtins

import pytest

from raganything.local_hf_embedding import make_local_hf_embedding_func


def test_missing_sentence_transformers_raises_install_hint(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(ImportError, match="raganything\\[local-embed\\]"):
        make_local_hf_embedding_func(embedding_dim=1024)
