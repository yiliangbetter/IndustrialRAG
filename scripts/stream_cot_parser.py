"""Split streamed LLM output into chain-of-thought vs user-facing answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["thinking", "answer"]

# (open_tag_name, close_tag_name)
_COT_TAGS: tuple[tuple[str, str], ...] = (
    ("think", "think"),
    ("redacted_reasoning", "redacted_reasoning"),
)


def _open_tag(name: str) -> str:
    return "<" + name + ">"


def _close_tag(name: str) -> str:
    return "</" + name + ">"


def _tag_prefixes() -> list[str]:
    prefixes: list[str] = []
    for open_name, close_name in _COT_TAGS:
        for tag in (_open_tag(open_name), _close_tag(close_name)):
            for i in range(1, len(tag)):
                prefixes.append(tag[:i])
    return sorted(set(prefixes), key=len, reverse=True)


_TAG_PREFIXES = _tag_prefixes()


def _keep_tail_for_partial_tag(buf: str, close_tag: str = "") -> int:
    """How many trailing chars to retain because a tag may be incomplete."""
    if not buf:
        return 0
    lower = buf.lower()
    prefixes = list(_TAG_PREFIXES)
    if close_tag:
        for i in range(1, len(close_tag)):
            prefixes.append(close_tag[:i])
    best = 0
    for prefix in prefixes:
        if lower.endswith(prefix.lower()):
            best = max(best, len(prefix))
    return best


def _find_earliest_open(buf: str) -> tuple[int, str, str] | None:
    best: tuple[int, str, str] | None = None
    lower = buf.lower()
    for open_name, close_name in _COT_TAGS:
        token = _open_tag(open_name)
        idx = lower.find(token.lower())
        if idx < 0:
            continue
        if best is None or idx < best[0]:
            best = (idx, token, close_name)
    return best


@dataclass
class StreamCotParser:
    """Incremental parser for interleaved CoT tags in token stream."""

    _mode: Literal["answer", "think"] = "answer"
    _buf: str = ""
    _close_tag: str = ""

    def feed(self, chunk: str) -> list[tuple[Kind, str]]:
        if not chunk:
            return []
        self._buf += chunk
        out: list[tuple[Kind, str]] = []
        while self._buf:
            if self._mode == "answer":
                found = _find_earliest_open(self._buf)
                if found is None:
                    keep = _keep_tail_for_partial_tag(self._buf)
                    emit = self._buf[: len(self._buf) - keep] if keep else self._buf
                    if emit:
                        out.append(("answer", emit))
                    self._buf = self._buf[len(emit) :]
                    break
                idx, _open_token, close_name = found
                if idx > 0:
                    out.append(("answer", self._buf[:idx]))
                self._buf = self._buf[idx + len(_open_token) :]
                self._mode = "think"
                self._close_tag = _close_tag(close_name)
            else:
                lower = self._buf.lower()
                close_token = self._close_tag.lower()
                cidx = lower.find(close_token)
                if cidx < 0:
                    keep = _keep_tail_for_partial_tag(self._buf, self._close_tag)
                    emit = self._buf[: len(self._buf) - keep] if keep else self._buf
                    if emit:
                        out.append(("thinking", emit))
                    self._buf = self._buf[len(emit) :]
                    break
                if cidx > 0:
                    out.append(("thinking", self._buf[:cidx]))
                self._buf = self._buf[cidx + len(self._close_tag) :]
                self._mode = "answer"
                self._close_tag = ""
        return out

    def flush(self) -> list[tuple[Kind, str]]:
        if not self._buf:
            return []
        kind: Kind = "thinking" if self._mode == "think" else "answer"
        out = [(kind, self._buf)]
        self._buf = ""
        return out


def parse_complete_cot(text: str) -> tuple[str, str]:
    """Parse a full completion into (thinking, answer) markdown strings."""
    if not text:
        return "", ""
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    parser = StreamCotParser()
    for kind, segment in parser.feed(text):
        if kind == "thinking":
            thinking_parts.append(segment)
        else:
            answer_parts.append(segment)
    for kind, segment in parser.flush():
        if kind == "thinking":
            thinking_parts.append(segment)
        else:
            answer_parts.append(segment)
    thinking = "".join(thinking_parts).strip()
    answer = "".join(answer_parts).strip()
    if not thinking and not answer:
        return "", text.strip()
    if not answer and thinking:
        return thinking, ""
    return thinking, answer or text.strip()
