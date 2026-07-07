from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .config import SegmentConfig
from .llm_segmenter import LLMSegmenter, SegmentChunkInput
from .local_segmenter import (
    LocalSegmenter,
    is_overlong_group,
    subtitle_from_group,
    timed_tokens_from_group,
)
from .segmenter import join_text_parts
from .types import SubtitleItem

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class _OverlongRef:
    chunk_index: int
    group_index: int


class HybridSegmenter:
    def __init__(
        self,
        local_segmenter: LocalSegmenter,
        llm_segmenter: LLMSegmenter,
        config: SegmentConfig,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.local_segmenter = local_segmenter
        self.llm_segmenter = llm_segmenter
        self.config = config
        self.progress = progress

    async def segment_chunks_async(
        self, chunk_inputs: list[SegmentChunkInput]
    ) -> list[list[SubtitleItem]]:
        local_results = [
            self.local_segmenter.segment_groups(
                chunk.tokens,
                context_text=chunk.context_text,
                language=chunk.language,
            )
            for chunk in chunk_inputs
        ]
        soft_segment_count = sum(len(result.groups) for result in local_results)

        refs: list[_OverlongRef] = []
        llm_inputs: list[SegmentChunkInput] = []
        for chunk_index, local_result in enumerate(local_results):
            for group_index, group in enumerate(local_result.groups):
                if not is_overlong_group(
                    group,
                    local_result.atoms,
                    self.config,
                    local_result.language,
                ):
                    continue
                tokens = timed_tokens_from_group(group, local_result.atoms)
                text = join_text_parts([token.text for token in tokens])
                refs.append(_OverlongRef(chunk_index=chunk_index, group_index=group_index))
                llm_inputs.append(
                    SegmentChunkInput(
                        tokens=tokens,
                        context_text=text,
                        language=local_result.language,
                        require_split=True,
                    )
                )
        _report(
            self.progress,
            (
                f"hybrid soft split: {soft_segment_count} segment(s), "
                f"{len(llm_inputs)} overlong segment(s) for LLM"
            ),
        )

        llm_outputs = (
            await self.llm_segmenter.segment_chunks_async(llm_inputs)
            if llm_inputs
            else []
        )
        replacements = {
            (ref.chunk_index, ref.group_index): output
            for ref, output in zip(refs, llm_outputs, strict=True)
            if output
        }
        overlong_keys = {(ref.chunk_index, ref.group_index) for ref in refs}

        output_chunks: list[list[SubtitleItem]] = []
        hard_fallback_count = 0
        for chunk_index, local_result in enumerate(local_results):
            chunk_subtitles: list[SubtitleItem] = []
            for group_index, group in enumerate(local_result.groups):
                key = (chunk_index, group_index)
                replacement = replacements.get(key)
                if replacement is not None and _is_llm_replacement_useful(
                    replacement,
                    group,
                    local_result.atoms,
                ):
                    chunk_subtitles.extend(replacement)
                    continue
                if key in overlong_keys:
                    hard_fallback_count += 1
                    chunk_subtitles.extend(
                        _hard_fallback_subtitles(group, local_result, self.config)
                    )
                    continue
                subtitle = subtitle_from_group(group, local_result.atoms)
                if subtitle is not None:
                    chunk_subtitles.append(subtitle)
            output_chunks.append(chunk_subtitles)
        if hard_fallback_count:
            _report(
                self.progress,
                f"hybrid hard fallback: {hard_fallback_count} segment(s)",
            )
        return output_chunks


def _is_llm_replacement_useful(
    replacement: list[SubtitleItem],
    group: list[int],
    atoms: list,
) -> bool:
    if not replacement:
        return False
    original_text = _group_text(group, atoms)
    if len(replacement) == 1 and _normalize_text(replacement[0].text) == _normalize_text(original_text):
        return False
    return True


def _hard_fallback_subtitles(
    group: list[int],
    local_result,
    config: SegmentConfig,
) -> list[SubtitleItem]:
    tokens = timed_tokens_from_group(group, local_result.atoms)
    context_text = join_text_parts([token.text for token in tokens])
    return LocalSegmenter(config, refine_mode="hard").segment(
        tokens,
        context_text=context_text,
        language=local_result.language,
    )


def _group_text(group: list[int], atoms: list) -> str:
    by_id = {atom.id: atom for atom in atoms}
    return join_text_parts([by_id[token_id].text for token_id in group if token_id in by_id])


def _normalize_text(text: str) -> str:
    return "".join(char for char in text if not char.isspace())


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
