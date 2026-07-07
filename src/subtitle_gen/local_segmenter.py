from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config import SegmentConfig
from .segmenter import (
    SegmentAtom,
    atomize_tokens,
    groups_from_text_parts,
    join_text_parts,
    project_context_punctuation,
    segments_from_atom_groups,
)
from .types import SubtitleItem, TimedToken


MAJOR_BOUNDARY_PUNCTUATION = ".!?。！？"
SOFT_BOUNDARY_PUNCTUATION = ",;:，；：、"
TRAILING_QUOTES = "\"'”’)]}）】」』"
BREAK_BEFORE_WORDS = {
    "and",
    "as",
    "because",
    "but",
    "for",
    "if",
    "in",
    "nor",
    "of",
    "on",
    "or",
    "so",
    "that",
    "to",
    "when",
    "while",
    "with",
    "yet",
}
_JA_RE = re.compile(r"[\u3040-\u30ff]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class LocalSegmentation:
    atoms: list[SegmentAtom]
    groups: list[list[int]]
    language: str

    def subtitles(self) -> list[SubtitleItem]:
        return segments_from_atom_groups(self.groups, self.atoms)


@dataclass(frozen=True)
class GroupMetrics:
    text: str
    duration: float
    length: int
    cps: float


class LocalSegmenter:
    def __init__(
        self,
        config: SegmentConfig | None = None,
        *,
        refine_mode: str | None = None,
    ) -> None:
        self.config = config or SegmentConfig()
        self.refine_mode = refine_mode or _refine_mode_for_segment_mode(self.config.mode)

    def segment(
        self,
        tokens: list[TimedToken],
        context_text: str | None = None,
        language: str | None = None,
    ) -> list[SubtitleItem]:
        result = self.segment_groups(tokens, context_text, language)
        return result.subtitles()

    def segment_groups(
        self,
        tokens: list[TimedToken],
        context_text: str | None = None,
        language: str | None = None,
    ) -> LocalSegmentation:
        tokens = project_context_punctuation(tokens, context_text)
        atoms = atomize_tokens(tokens)
        if not atoms:
            return LocalSegmentation(atoms=[], groups=[], language=_normalize_language(language, ""))

        source_text = join_text_parts([atom.text for atom in atoms])
        resolved_language = _normalize_language(language, context_text or source_text)
        groups = _sentence_groups(atoms, self.config)
        if self.refine_mode == "none":
            return LocalSegmentation(atoms=atoms, groups=groups, language=resolved_language)

        refined: list[list[int]] = []
        by_id = _atoms_by_id(atoms)
        for group in groups:
            refined.extend(
                _split_group(
                    group,
                    by_id,
                    self.config,
                    resolved_language,
                    allow_weak_fallback=self.refine_mode == "hard",
                )
            )

        merged = _merge_short_groups(refined, by_id, self.config, resolved_language)
        return LocalSegmentation(atoms=atoms, groups=merged, language=resolved_language)


def is_overlong_group(
    group: list[int],
    atoms: list[SegmentAtom],
    config: SegmentConfig,
    language: str | None = None,
) -> bool:
    text = join_text_parts([atom.text for atom in atoms])
    resolved_language = _normalize_language(language, text)
    return _needs_split(group, _atoms_by_id(atoms), config, resolved_language)


def subtitle_from_group(group: list[int], atoms: list[SegmentAtom]) -> SubtitleItem | None:
    subtitles = segments_from_atom_groups([group], atoms)
    return subtitles[0] if subtitles else None


def timed_tokens_from_group(group: list[int], atoms: list[SegmentAtom]) -> list[TimedToken]:
    by_id = _atoms_by_id(atoms)
    return [
        TimedToken(text=by_id[token_id].text, start=by_id[token_id].start, end=by_id[token_id].end)
        for token_id in group
        if token_id in by_id
    ]


def _sentence_groups(atoms: list[SegmentAtom], config: SegmentConfig) -> list[list[int]]:
    source_text = join_text_parts([atom.text for atom in atoms])
    sentence_parts = _split_sentences(source_text)
    groups = groups_from_text_parts(sentence_parts, atoms) if sentence_parts else []
    if groups:
        return groups
    return _fallback_sentence_groups(atoms, config)


def _split_sentences(text: str) -> list[str]:
    try:
        from blingfire import text_to_sentences
    except ImportError:
        return []

    output = text_to_sentences(text)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _fallback_sentence_groups(atoms: list[SegmentAtom], config: SegmentConfig) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for index, atom in enumerate(atoms):
        current.append(atom.id)
        next_atom = atoms[index + 1] if index + 1 < len(atoms) else None
        if next_atom is None:
            continue
        pause = max(0.0, next_atom.start - atom.end)
        if _is_major_boundary(atom.text) or pause >= config.pause_threshold:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups or [[atom.id for atom in atoms]]


def _split_group(
    group: list[int],
    by_id: dict[int, SegmentAtom],
    config: SegmentConfig,
    language: str,
    *,
    allow_weak_fallback: bool,
) -> list[list[int]]:
    if len(group) <= 1 or not _needs_split(group, by_id, config, language):
        return [group]

    split_index = _best_split_index(
        group,
        by_id,
        config,
        language,
        allow_weak_fallback=allow_weak_fallback,
    )
    if split_index is None:
        return [group]

    left = group[:split_index]
    right = group[split_index:]
    if not left or not right:
        return [group]
    return (
        _split_group(
            left,
            by_id,
            config,
            language,
            allow_weak_fallback=allow_weak_fallback,
        )
        + _split_group(
            right,
            by_id,
            config,
            language,
            allow_weak_fallback=allow_weak_fallback,
        )
    )


def _best_split_index(
    group: list[int],
    by_id: dict[int, SegmentAtom],
    config: SegmentConfig,
    language: str,
    *,
    allow_weak_fallback: bool,
) -> int | None:
    metrics = _group_metrics(group, by_id, language)
    allow_weak = allow_weak_fallback and _is_hard_over_limit(metrics, config, language)
    best_index: int | None = None
    best_score = float("-inf")

    for split_index in range(1, len(group)):
        left = group[:split_index]
        right = group[split_index:]
        left_atoms = _group_atoms(left, by_id)
        right_atoms = _group_atoms(right, by_id)
        if not left_atoms or not right_atoms:
            continue

        boundary_score = _boundary_score(
            left_atoms[-1],
            right_atoms[0],
            config=config,
            language=language,
        )
        if boundary_score <= 0.0 and not allow_weak:
            continue

        left_metrics = _group_metrics(left, by_id, language)
        right_metrics = _group_metrics(right, by_id, language)
        left_share = left_metrics.length / max(metrics.length, 1)
        balance_penalty = abs(left_share - 0.5) * 30.0
        min_duration_penalty = _min_duration_penalty(left_metrics, right_metrics, config)
        overflow_penalty = (
            _overflow_penalty(left_metrics, config, language)
            + _overflow_penalty(right_metrics, config, language)
        )
        weak_bonus = 1.0 if boundary_score <= 0.0 else 0.0
        score = boundary_score + weak_bonus - balance_penalty - min_duration_penalty
        score -= overflow_penalty * 25.0

        if score > best_score:
            best_score = score
            best_index = split_index

    return best_index


def _merge_short_groups(
    groups: list[list[int]],
    by_id: dict[int, SegmentAtom],
    config: SegmentConfig,
    language: str,
) -> list[list[int]]:
    merged: list[list[int]] = []
    index = 0
    while index < len(groups):
        group = groups[index]
        metrics = _group_metrics(group, by_id, language)
        if metrics.duration >= config.min_duration or len(groups) == 1:
            merged.append(group)
            index += 1
            continue

        if index + 1 < len(groups) and _can_merge(group, groups[index + 1], by_id, config, language):
            merged.append(group + groups[index + 1])
            index += 2
            continue

        if merged and _can_merge(merged[-1], group, by_id, config, language):
            merged[-1] = merged[-1] + group
        else:
            merged.append(group)
        index += 1

    return merged


def _can_merge(
    left: list[int],
    right: list[int],
    by_id: dict[int, SegmentAtom],
    config: SegmentConfig,
    language: str,
) -> bool:
    combined = left + right
    metrics = _group_metrics(combined, by_id, language)
    max_chars = _max_total_chars(config, language)
    return (
        metrics.duration <= config.max_duration
        and metrics.length <= max_chars
    )


def _needs_split(
    group: list[int],
    by_id: dict[int, SegmentAtom],
    config: SegmentConfig,
    language: str,
) -> bool:
    metrics = _group_metrics(group, by_id, language)
    return (
        metrics.duration > config.max_duration
        or metrics.length > _max_total_chars(config, language)
    )


def _is_hard_over_limit(metrics: GroupMetrics, config: SegmentConfig, language: str) -> bool:
    return (
        metrics.duration > config.max_duration * 1.10
        or metrics.length > _max_total_chars(config, language)
    )


def _boundary_score(
    left: SegmentAtom,
    right: SegmentAtom,
    *,
    config: SegmentConfig,
    language: str,
) -> float:
    pause = max(0.0, right.start - left.end)
    score = 0.0
    if _is_major_boundary(left.text):
        score = max(score, 100.0)
    elif _is_soft_boundary(left.text):
        score = max(score, 80.0)
    if pause >= config.pause_threshold:
        score = max(score, 70.0 + min(pause, 1.5) * 5.0)
    if language == "en" and _starts_with_break_before_word(right.text):
        score = max(score, 45.0)
    return score


def _min_duration_penalty(
    left: GroupMetrics, right: GroupMetrics, config: SegmentConfig
) -> float:
    penalty = 0.0
    if left.duration < config.min_duration:
        penalty += (config.min_duration - left.duration) / max(config.min_duration, 0.001) * 35.0
    if right.duration < config.min_duration:
        penalty += (config.min_duration - right.duration) / max(config.min_duration, 0.001) * 35.0
    return penalty


def _overflow_penalty(metrics: GroupMetrics, config: SegmentConfig, language: str) -> float:
    penalty = 0.0
    if metrics.duration > config.max_duration:
        penalty += metrics.duration / max(config.max_duration, 0.001) - 1.0
    max_chars = _max_total_chars(config, language)
    if metrics.length > max_chars:
        penalty += metrics.length / max(max_chars, 1) - 1.0
    max_cps = _max_cps(config, language)
    if metrics.cps > max_cps:
        penalty += metrics.cps / max(max_cps, 0.001) - 1.0
    return penalty


def _group_metrics(
    group: list[int],
    by_id: dict[int, SegmentAtom],
    language: str,
) -> GroupMetrics:
    atoms = _group_atoms(group, by_id)
    if not atoms:
        return GroupMetrics(text="", duration=0.0, length=0, cps=0.0)
    text = join_text_parts([atom.text for atom in atoms]).strip()
    duration = max(0.001, atoms[-1].end - atoms[0].start)
    length = _measure_text(text, language)
    return GroupMetrics(text=text, duration=duration, length=length, cps=length / duration)


def _group_atoms(group: Iterable[int], by_id: dict[int, SegmentAtom]) -> list[SegmentAtom]:
    return [by_id[token_id] for token_id in group if token_id in by_id]


def _atoms_by_id(atoms: list[SegmentAtom]) -> dict[int, SegmentAtom]:
    return {atom.id: atom for atom in atoms}


def _is_major_boundary(text: str) -> bool:
    return _strip_trailing_quotes(text).endswith(tuple(MAJOR_BOUNDARY_PUNCTUATION))


def _is_soft_boundary(text: str) -> bool:
    return _strip_trailing_quotes(text).endswith(tuple(SOFT_BOUNDARY_PUNCTUATION))


def _strip_trailing_quotes(text: str) -> str:
    return text.strip().rstrip(TRAILING_QUOTES)


def _starts_with_break_before_word(text: str) -> bool:
    match = _WORD_RE.match(text.strip().lower())
    return bool(match and match.group(0) in BREAK_BEFORE_WORDS)


def _normalize_language(language: str | None, text: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized.startswith(("ja", "japanese", "日本")):
        return "ja"
    if normalized.startswith(("zh", "chinese", "mandarin", "中文", "汉", "漢")):
        return "zh"
    if normalized.startswith(("en", "english")):
        return "en"
    if _JA_RE.search(text):
        return "ja"
    if _CJK_RE.search(text):
        return "zh"
    return "en"


def _refine_mode_for_segment_mode(segment_mode: str) -> str:
    if segment_mode == "local":
        return "hard"
    if segment_mode == "hybrid":
        return "soft"
    return "none"


def _max_chars_per_line(config: SegmentConfig, language: str) -> int:
    if language == "ja":
        return config.max_chars_ja
    if language == "zh":
        return config.max_chars_zh
    return config.max_chars_en


def _max_total_chars(config: SegmentConfig, language: str) -> int:
    return max(1, config.max_lines) * _max_chars_per_line(config, language)


def _max_cps(config: SegmentConfig, language: str) -> float:
    if language == "ja":
        return config.max_cps_ja
    if language == "zh":
        return config.max_cps_zh
    return config.max_cps_en


def _measure_text(text: str, language: str) -> int:
    if language in {"ja", "zh"}:
        return sum(1 for char in text if not char.isspace())
    return len(" ".join(text.split()))
