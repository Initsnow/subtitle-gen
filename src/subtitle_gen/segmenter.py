from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .types import SubtitleItem, TimedToken


TRAILING_PUNCTUATION = set(",.;:!?，。；：！？、)]}）】」』")
LEADING_PUNCTUATION = set("([{（【「『")
CJK_PUNCTUATION = set("，。；：！？、")
ATTACHABLE_PUNCTUATION = set(
    ",.;:!?，。；：！？、)]}）】」』\"'”’"
)
_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_CONTENT_RE = re.compile(r"[0-9A-Za-z\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


@dataclass(frozen=True)
class SegmentAtom:
    id: int
    text: str
    start: float
    end: float

    def to_prompt_item(self) -> dict[str, object]:
        return {"id": self.id, "text": self.text}


def atomize_tokens(tokens: Iterable[TimedToken]) -> list[SegmentAtom]:
    return [
        SegmentAtom(id=index, text=token.text, start=token.start, end=token.end)
        for index, token in enumerate(_clean_tokens(tokens), start=1)
    ]


def project_context_punctuation(
    tokens: Iterable[TimedToken],
    context_text: str | None,
) -> list[TimedToken]:
    cleaned = _clean_tokens(tokens)
    if not cleaned or not context_text:
        return cleaned

    spans = _match_context_spans(cleaned, context_text)
    if not any(span is not None for span in spans):
        return cleaned

    projected: list[TimedToken] = []
    for index, token in enumerate(cleaned):
        span = spans[index]
        if span is None:
            projected.append(token)
            continue
        _start, end = span
        next_span = spans[index + 1] if index + 1 < len(spans) else None
        if next_span is None and index + 1 < len(spans):
            projected.append(token)
            continue
        next_start = next_span[0] if next_span is not None else len(context_text)
        between = context_text[end:next_start]
        if _CONTENT_RE.search(between):
            projected.append(token)
            continue
        trailing = "".join(char for char in between if char in ATTACHABLE_PUNCTUATION)
        text = _append_missing_trailing_punctuation(token.text, trailing)
        projected.append(TimedToken(text=text, start=token.start, end=token.end))
    return projected


def segments_from_atom_groups(
    groups: Iterable[list[int]], atoms: list[SegmentAtom]
) -> list[SubtitleItem]:
    by_id = {atom.id: atom for atom in atoms}
    segments: list[SubtitleItem] = []
    for group in groups:
        group_atoms = [by_id[token_id] for token_id in group]
        if not group_atoms:
            continue
        text = join_text_parts([atom.text for atom in group_atoms]).strip()
        if not text:
            continue
        if group_atoms[-1].end <= group_atoms[0].start:
            continue
        segments.append(
            SubtitleItem(
                id=len(segments) + 1,
                start=group_atoms[0].start,
                end=group_atoms[-1].end,
                text=text,
            )
        )
    return segments


def groups_from_text_parts(parts: list[str], atoms: list[SegmentAtom]) -> list[list[int]]:
    groups: list[list[int]] = []
    atom_index = 0

    for part in parts:
        target = normalize_for_mapping(part)
        if not target:
            continue

        start_index = atom_index
        consumed = ""
        while atom_index < len(atoms) and len(consumed) < len(target):
            consumed += normalize_for_mapping(atoms[atom_index].text)
            atom_index += 1

        if consumed != target:
            return []

        group = [atom.id for atom in atoms[start_index:atom_index]]
        if not group:
            return []
        groups.append(group)

    if atom_index != len(atoms):
        return []
    return groups


def normalize_for_delimiter_validation(text: str, remove_delimiters: bool) -> str:
    if remove_delimiters:
        text = text.replace("|", "")
    return "".join(char for char in text if not char.isspace())


def normalize_for_mapping(text: str) -> str:
    return normalize_for_delimiter_validation(text, remove_delimiters=True)


def normalize_for_context_mapping(text: str) -> str:
    return "".join(char.lower() for char in text if _CONTENT_RE.match(char))


def join_text_parts(parts: Iterable[str]) -> str:
    output = ""
    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue
        if not output:
            output = part
            continue
        if _needs_space(output[-1], part[0]):
            output += " " + part
        else:
            output += part
    return output


def _needs_space(left: str, right: str) -> bool:
    if left in LEADING_PUNCTUATION or right in TRAILING_PUNCTUATION:
        return False
    if left in CJK_PUNCTUATION:
        return False
    if left == "'" or right == "'":
        return False
    if left.isdigit() and right.isdigit():
        return False
    if _has_cjk(left) or _has_cjk(right):
        return False
    return True


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _clean_tokens(tokens: Iterable[TimedToken]) -> list[TimedToken]:
    cleaned = [token for token in tokens if token.text.strip() and token.end >= token.start]
    cleaned.sort(key=lambda token: (token.start, token.end))
    return cleaned


def _append_missing_trailing_punctuation(text: str, trailing: str) -> str:
    missing = trailing
    while missing and text.rstrip().endswith(missing[0]):
        missing = missing[1:]
    return f"{text}{missing}" if missing and not text.rstrip().endswith(missing) else text


def _match_context_spans(
    tokens: list[TimedToken],
    context_text: str,
) -> list[tuple[int, int] | None]:
    context_units = [
        (char.lower(), index)
        for index, char in enumerate(context_text)
        if _CONTENT_RE.match(char)
    ]
    if not context_units:
        return [None for _token in tokens]

    cursor = 0
    spans: list[tuple[int, int] | None] = []
    for token in tokens:
        target = normalize_for_context_mapping(token.text)
        if not target:
            spans.append(None)
            continue
        start_cursor = _find_context_units(context_units, target, cursor)
        if start_cursor is None:
            spans.append(None)
            continue
        end_cursor = start_cursor + len(target)
        start = context_units[start_cursor][1]
        end = context_units[end_cursor - 1][1] + 1
        spans.append((start, end))
        cursor = end_cursor
    return spans


def _find_context_units(
    context_units: list[tuple[str, int]],
    target: str,
    start_cursor: int,
) -> int | None:
    max_start = len(context_units) - len(target)
    for cursor in range(start_cursor, max_start + 1):
        if "".join(char for char, _index in context_units[cursor : cursor + len(target)]) == target:
            return cursor
    return None
