from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from .config import LLMConfig, SegmentConfig
from .llm import OpenAICompatibleLLM
from .segmenter import (
    SegmentAtom,
    atomize_tokens,
    groups_from_text_parts,
    join_text_parts,
    normalize_for_delimiter_validation,
    normalize_for_mapping,
    project_context_punctuation,
    segments_from_atom_groups,
)
from .types import SubtitleItem, TimedToken


SEGMENT_DELIMITER = "|"

SEGMENT_SYSTEM_PROMPT = """You are a subtitle segmentation engine.
Return only the processed text.
Do not explain.
Do not use JSON.
Your only edit is inserting the delimiter | at natural subtitle boundaries.
Do not change, delete, reorder, or add any other character."""

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SegmentChunkInput:
    tokens: list[TimedToken]
    context_text: str | None = None
    language: str | None = None
    require_split: bool = False


@dataclass(frozen=True)
class DelimitedTextResult:
    groups: list[list[int]]
    error: str | None = None
    boundary_times: list[float] | None = None


class LLMSegmenter:
    def __init__(
        self,
        llm: OpenAICompatibleLLM,
        segment_config: SegmentConfig | None = None,
        llm_config: LLMConfig | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.llm = llm
        self.segment_config = segment_config or SegmentConfig(mode="llm")
        self.llm_config = llm_config or llm.config
        self.progress = progress

    def segment(
        self, tokens: list[TimedToken], context_text: str | None = None
    ) -> list[SubtitleItem]:
        return asyncio.run(self.segment_async(tokens=tokens, context_text=context_text))

    async def segment_async(
        self, tokens: list[TimedToken], context_text: str | None = None
    ) -> list[SubtitleItem]:
        chunks = await self.segment_chunks_async([SegmentChunkInput(tokens, context_text)])
        return chunks[0] if chunks else []

    async def segment_chunks_async(
        self, chunk_inputs: list[SegmentChunkInput]
    ) -> list[list[SubtitleItem]]:
        total_chunks = len(chunk_inputs)
        _report(self.progress, f"LLM segmentation requests: {total_chunks}")
        chunk_atoms = [
            atomize_tokens(project_context_punctuation(chunk.tokens, chunk.context_text))
            for chunk in chunk_inputs
        ]
        semaphore = asyncio.Semaphore(max(1, self.llm_config.concurrency))
        tasks = [
            self._segment_chunk_async(
                chunk_index=chunk_index,
                atoms=atoms,
                context_text=chunk_inputs[chunk_index].context_text,
                require_split=chunk_inputs[chunk_index].require_split,
                semaphore=semaphore,
                total_chunks=total_chunks,
            )
            for chunk_index, atoms in enumerate(chunk_atoms)
        ]
        chunk_results = await asyncio.gather(*tasks)
        sorted_results = sorted(chunk_results, key=lambda item: item[0])

        subtitles_by_chunk: list[list[SubtitleItem]] = []
        for (_index, groups, boundary_times), atoms in zip(
            sorted_results, chunk_atoms, strict=True
        ):
            subtitles = segments_from_atom_groups(groups, atoms)
            _apply_boundary_times(subtitles, boundary_times)
            subtitles_by_chunk.append(subtitles)
        return subtitles_by_chunk

    async def _segment_chunk_async(
        self,
        chunk_index: int,
        atoms: list[SegmentAtom],
        context_text: str | None,
        require_split: bool,
        semaphore: asyncio.Semaphore,
        total_chunks: int,
    ) -> tuple[int, list[list[int]], list[float] | None]:
        if not atoms:
            return chunk_index, [], None
        async with semaphore:
            fallback_reason = "unknown error"
            try:
                prompt = build_delimiter_prompt(
                    atoms,
                    context_text,
                    require_split=require_split,
                )
                content = await self.llm.complete_text_async(SEGMENT_SYSTEM_PROMPT, prompt)
                result = parse_delimited_text(content, atoms, self.segment_config)
                if result.groups:
                    if require_split and len(result.groups) == 1:
                        fallback_reason = "invalid output: no usable split"
                    else:
                        _report(
                            self.progress,
                            f"LLM segmentation {chunk_index + 1}/{total_chunks}: ok",
                        )
                        return chunk_index, result.groups, result.boundary_times
                else:
                    fallback_reason = f"invalid output: {result.error or fallback_reason}"
            except Exception as exc:
                fallback_reason = f"request error: {_format_exception(exc)}"
            _report(
                self.progress,
                (
                    f"LLM segmentation {chunk_index + 1}/{total_chunks}: "
                    f"fallback ({fallback_reason})"
                ),
            )
            return chunk_index, unsegmented_atom_group(atoms), None


def build_delimiter_prompt(
    atoms: list[SegmentAtom],
    context_text: str | None = None,
    *,
    require_split: bool = False,
) -> str:
    source_text = join_text_parts([atom.text for atom in atoms])
    requirements = [
        "Insert | where the text should be split into subtitle lines.",
        "Output ONLY the processed source text.",
        "Do not wrap the answer in quotes or code fences.",
        "Do not change spacing, punctuation, words, or characters.",
        "Prefer complete natural clauses over fixed-length chopping.",
    ]
    if require_split:
        requirements.append("This source is over subtitle limits; insert at least one |.")
    if context_text:
        requirements.append("Use the context only to understand sentence boundaries.")
    lines = ["Requirements:"]
    lines.extend(f"- {requirement}" for requirement in requirements)
    if context_text:
        lines.extend(["", "Context:", context_text])
    lines.extend(["", "Source text:", source_text])
    return "\n".join(lines)


def groups_from_delimited_text(
    raw_output: str, atoms: list[SegmentAtom], config: SegmentConfig
) -> list[list[int]]:
    return parse_delimited_text(raw_output, atoms, config).groups


def parse_delimited_text(
    raw_output: str, atoms: list[SegmentAtom], config: SegmentConfig
) -> DelimitedTextResult:
    source_text = join_text_parts([atom.text for atom in atoms])
    output = strip_text_response(raw_output)

    if normalize_for_delimiter_validation(output, remove_delimiters=True) != (
        normalize_for_delimiter_validation(source_text, remove_delimiters=False)
    ):
        return DelimitedTextResult([], "text changed outside delimiters")

    parts = [part for part in output.split(SEGMENT_DELIMITER) if part.strip()]
    if not parts:
        return DelimitedTextResult([], "empty output")

    groups = groups_from_text_parts(parts, atoms)
    boundary_times: list[float] | None = None
    if not groups:
        groups, boundary_times = _groups_with_proportional_times(parts, atoms)
    if not groups:
        return DelimitedTextResult([], "delimiter positions could not be mapped to tokens")
    return DelimitedTextResult(groups, boundary_times=boundary_times)


def unsegmented_atom_group(atoms: list[SegmentAtom]) -> list[list[int]]:
    if not atoms:
        return []
    return [[atom.id for atom in atoms]]


def strip_text_response(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _groups_with_proportional_times(
    parts: list[str], atoms: list[SegmentAtom]
) -> tuple[list[list[int]], list[float] | None]:
    if len(parts) < 2 or len(atoms) < 2:
        return [], None

    normalized_total = sum(len(normalize_for_mapping(atom.text)) for atom in atoms)
    if normalized_total <= 0:
        return [], None

    split_offsets: list[int] = []
    cursor = 0
    for part in parts[:-1]:
        cursor += len(normalize_for_mapping(part))
        if 0 < cursor < normalized_total:
            split_offsets.append(cursor)
    if not split_offsets:
        return [], None

    boundary_offsets: list[tuple[int, int]] = []
    cursor = 0
    for atom_index, atom in enumerate(atoms[:-1], start=1):
        cursor += len(normalize_for_mapping(atom.text))
        if 0 < cursor < normalized_total:
            boundary_offsets.append((cursor, atom_index))
    if not boundary_offsets:
        return [], None

    offset_to_snapped: dict[int, int] = {}
    for split_offset in split_offsets:
        nearest = min(
            boundary_offsets,
            key=lambda boundary: (abs(boundary[0] - split_offset), boundary[0]),
        )
        offset_to_snapped[split_offset] = nearest[1]

    unique_indices: dict[int, int] = {}
    for split_offset, snapped_idx in offset_to_snapped.items():
        if snapped_idx not in unique_indices:
            unique_indices[snapped_idx] = split_offset

    snapped_indices = sorted(unique_indices)
    if not snapped_indices:
        return [], None

    proportional_times = _compute_boundary_times(list(unique_indices.values()), atoms)
    snapped_times = [proportional_times[unique_indices[idx]] for idx in snapped_indices]

    groups: list[list[int]] = []
    start_index = 0
    for end_index in snapped_indices:
        group = [atom.id for atom in atoms[start_index:end_index]]
        if group:
            groups.append(group)
        start_index = end_index
    tail = [atom.id for atom in atoms[start_index:]]
    if tail:
        groups.append(tail)
    return groups, snapped_times if len(snapped_times) == len(groups) - 1 else None


def _compute_boundary_times(split_offsets: list[int], atoms: list[SegmentAtom]) -> dict[int, float]:
    """Map each delimiter character offset to a proportional time within its token."""
    normalized_texts = [normalize_for_mapping(a.text) for a in atoms]
    atom_lengths = [len(t) for t in normalized_texts]

    result: dict[int, float] = {}
    for offset in split_offsets:
        cursor = 0
        for atom_index, length in enumerate(atom_lengths):
            if cursor + length >= offset:
                ratio = (offset - cursor) / length if length > 0 else 0.5
                atom = atoms[atom_index]
                result[offset] = atom.start + ratio * (atom.end - atom.start)
                break
            cursor += length
    return result


def _apply_boundary_times(
    subtitles: list[SubtitleItem], boundary_times: list[float] | None
) -> None:
    if not boundary_times or len(subtitles) < 2:
        return
    for i in range(min(len(boundary_times), len(subtitles) - 1)):
        time = boundary_times[i]
        subtitles[i] = subtitles[i].with_end(time)
        subtitles[i + 1] = subtitles[i + 1].with_start(time)


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _format_exception(exc: Exception) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
