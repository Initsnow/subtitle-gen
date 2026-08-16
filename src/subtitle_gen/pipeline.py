from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .cache import cleanup_cache, media_cache_id
from .config import AppConfig
from .hybrid_segmenter import HybridSegmenter
from .llm import OpenAICompatibleLLM
from .llm_segmenter import LLMSegmenter, SegmentChunkInput
from .local_segmenter import LocalSegmenter
from .media import extract_wav, probe_duration
from .processing import format_duration as _format_duration, process_windows, report as _report
from .segmenter import project_context_punctuation
from .proofreader import SubtitleProofreader
from .translator import SubtitleTranslator
from .types import SubtitleItem, TimedToken, TranscriptChunk
from .vad import plan_audio_windows


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PipelineOptions:
    input_path: Path
    translate: str | None = None
    segment_mode: str | None = None
    overwrite_cache: bool = False
    progress: ProgressCallback | None = None


@dataclass(frozen=True)
class PipelineResult:
    subtitles: list[SubtitleItem]
    transcript_chunks: list[TranscriptChunk]


@dataclass(frozen=True)
class AlignedChunk:
    transcript: TranscriptChunk
    tokens: list[TimedToken]


class SubtitlePipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self, options: PipelineOptions) -> PipelineResult:
        if self.config.cache.enabled:
            return self._run_with_cache_root(
                options,
                Path(self.config.cache_dir),
                cache_enabled=True,
            )

        with tempfile.TemporaryDirectory(prefix="subtitle-gen-") as temp_dir:
            return self._run_with_cache_root(
                options,
                Path(temp_dir),
                cache_enabled=False,
            )

    def _run_with_cache_root(
        self,
        options: PipelineOptions,
        cache_root: Path,
        *,
        cache_enabled: bool,
    ) -> PipelineResult:
        progress = options.progress
        input_path = Path(options.input_path)
        audio_cache = cache_root / "audio"
        segment_mode = options.segment_mode or self.config.segment.mode
        if segment_mode not in {"none", "blingfire", "local", "hybrid", "llm"}:
            raise ValueError(f"Unsupported segment mode: {segment_mode}")

        _report(progress, f"input: {input_path}")
        cache_label = f"enabled at {cache_root}" if cache_enabled else "disabled"
        if options.overwrite_cache:
            cache_label = f"{cache_label}; overwrite requested"
        _report(
            progress,
            f"mode: {segment_mode}; cache: {cache_label}",
        )
        _report(progress, "preparing 16 kHz mono audio")
        wav_path = extract_wav(
            input_path,
            audio_cache,
            overwrite=options.overwrite_cache or not cache_enabled,
        )
        media_id = media_cache_id(wav_path)
        _report(progress, "probing audio duration")
        duration = probe_duration(wav_path)
        _report(progress, "planning speech windows")
        windows = plan_audio_windows(wav_path, duration, self.config.vad)
        _report(
            progress,
            f"audio: {_format_duration(duration)}; chunks: {len(windows)}; media id: {media_id}",
        )
        if cache_enabled and self.config.cache.cleanup_enabled:
            _report(progress, "cleaning stale cache entries")
            cleanup_cache(
                cache_root,
                keep_media_ids={media_id},
                active_windows_by_media={media_id: windows},
                max_media_entries=self.config.cache.max_media_entries,
            )

        processed = process_windows(
            wav_path,
            cache_root,
            self.config,
            windows,
            language=self.config.model.language,
            progress=progress,
            overwrite_cache=options.overwrite_cache,
            with_alignment=segment_mode != "none",
        )

        chunks: list[TranscriptChunk] = []
        aligned_chunks: list[AlignedChunk] = []
        for processed_window in processed.windows:
            transcript_chunk = TranscriptChunk(
                index=processed_window.window.index,
                start=processed_window.window.start,
                end=processed_window.window.end,
                text=processed_window.transcript,
                language=processed_window.language,
            )
            chunks.append(transcript_chunk)

            if processed_window.local_tokens is not None:
                global_tokens = project_context_punctuation(
                    [
                        token.shifted(processed_window.window.start)
                        for token in processed_window.local_tokens
                    ],
                    processed_window.transcript,
                )
                aligned_chunks.append(
                    AlignedChunk(transcript=transcript_chunk, tokens=global_tokens)
                )

        if segment_mode != "none":
            _report(progress, f"segmenting with {segment_mode}")
            segment_inputs = [
                SegmentChunkInput(
                    tokens=aligned_chunk.tokens,
                    context_text=aligned_chunk.transcript.text,
                    language=aligned_chunk.transcript.language,
                )
                for aligned_chunk in aligned_chunks
            ]
            if segment_mode == "llm":
                seg_llm_config = self.config.segmentation_llm
                if seg_llm_config is None:
                    raise ValueError(
                        "[llm.segmentation] config is required for llm segmentation mode."
                    )
                llm_client = OpenAICompatibleLLM(seg_llm_config)
                segmenter = LLMSegmenter(
                    llm_client,
                    self.config.segment,
                    seg_llm_config,
                    progress=progress,
                )
                segmented_chunks = asyncio.run(segmenter.segment_chunks_async(segment_inputs))
            elif segment_mode == "hybrid":
                seg_llm_config = self.config.segmentation_llm
                if seg_llm_config is None:
                    raise ValueError(
                        "[llm.segmentation] config is required for hybrid segmentation mode."
                    )
                llm_client = OpenAICompatibleLLM(seg_llm_config)
                segmenter = HybridSegmenter(
                    LocalSegmenter(self.config.segment, refine_mode="soft"),
                    LLMSegmenter(
                        llm_client,
                        self.config.segment,
                        seg_llm_config,
                        progress=progress,
                    ),
                    self.config.segment,
                    progress=progress,
                )
                segmented_chunks = asyncio.run(segmenter.segment_chunks_async(segment_inputs))
            else:
                segmenter = LocalSegmenter(
                    self.config.segment,
                    refine_mode="hard" if segment_mode == "local" else "none",
                )
                segmented_chunks = [
                    segmenter.segment(
                        chunk.tokens,
                        context_text=chunk.context_text,
                        language=chunk.language,
                    )
                    for chunk in segment_inputs
                ]
            subtitles = [
                subtitle for segmented_chunk in segmented_chunks for subtitle in segmented_chunk
            ]
        else:
            subtitles = _subtitles_from_transcript_chunks(chunks)
        subtitles = _renumber_subtitles(subtitles)
        _report(progress, f"subtitles: {len(subtitles)}")
        _report_limit_summary(progress, subtitles, self.config.segment.max_duration)

        if options.translate:
            llm_client = OpenAICompatibleLLM(self.config.llm)
            _report(progress, "proofreading")
            subtitles = SubtitleProofreader(
                llm_client,
                self.config.llm,
                progress=progress,
            ).proofread(subtitles)
            corrected_count = sum(1 for s in subtitles if s.proofread is not None)
            _report(progress, f"proofread: {corrected_count} corrected")
            _report(progress, f"translating to {options.translate}")
            subtitles = SubtitleTranslator(
                llm_client,
                self.config.llm,
                progress=progress,
            ).translate(subtitles, target_language=options.translate)
            _report(progress, "translation complete")

        return PipelineResult(subtitles=subtitles, transcript_chunks=chunks)


def _subtitles_from_transcript_chunks(chunks: list[TranscriptChunk]) -> list[SubtitleItem]:
    subtitles: list[SubtitleItem] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        subtitles.append(
            SubtitleItem(
                id=len(subtitles) + 1,
                start=chunk.start,
                end=chunk.end,
                text=text,
            )
        )
    return subtitles


def _renumber_subtitles(subtitles: list[SubtitleItem]) -> list[SubtitleItem]:
    return [
        SubtitleItem(
            id=index,
            start=subtitle.start,
            end=subtitle.end,
            text=subtitle.text,
            translation=subtitle.translation,
            proofread=subtitle.proofread,
        )
        for index, subtitle in enumerate(subtitles, start=1)
    ]


def _report_limit_summary(
    progress: ProgressCallback | None,
    subtitles: list[SubtitleItem],
    max_duration: float,
) -> None:
    if progress is None or not subtitles or max_duration <= 0:
        return
    over_duration = sum(1 for subtitle in subtitles if subtitle.end - subtitle.start > max_duration)
    if over_duration:
        _report(progress, f"notice: {over_duration} subtitle(s) exceed {max_duration:.1f}s")
