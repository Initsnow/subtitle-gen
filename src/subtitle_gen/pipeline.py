from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .aligner import QwenForcedAligner
from .asr import QwenASR
from .cache import (
    asr_cache_path,
    cleanup_cache,
    load_cached_alignment_tokens,
    load_cached_asr_result,
    media_cache_id,
    save_alignment_tokens,
    save_asr_result,
)
from .config import AppConfig
from .hybrid_segmenter import HybridSegmenter
from .llm import OpenAICompatibleLLM
from .llm_segmenter import LLMSegmenter, SegmentChunkInput
from .local_segmenter import LocalSegmenter
from .media import cut_wav_window, extract_wav, probe_duration
from .segmenter import project_context_punctuation
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
        chunk_cache = cache_root / "chunks" / media_id
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

        asr: QwenASR | None = None
        aligner: QwenForcedAligner | None = None

        chunks: list[TranscriptChunk] = []
        aligned_chunks: list[AlignedChunk] = []
        total_windows = len(windows)
        for window_position, window in enumerate(windows, start=1):
            _report(
                progress,
                (
                    f"chunk {window_position}/{total_windows} "
                    f"{_format_duration(window.start)}-{_format_duration(window.end)}"
                ),
            )
            chunk_path = cut_wav_window(
                wav_path,
                window,
                chunk_cache,
                overwrite=options.overwrite_cache or not cache_enabled,
            )
            chunk_asr_cache_path = asr_cache_path(cache_root, media_id, window)
            asr_result = None
            if cache_enabled and not options.overwrite_cache:
                asr_result = load_cached_asr_result(
                    chunk_asr_cache_path,
                    model_id=self.config.model.asr_model,
                    language_hint=self.config.model.language,
                )
            if asr_result is None:
                if asr is None:
                    _report(progress, f"loading ASR model: {self.config.model.asr_model}")
                    asr = QwenASR(
                        model_id=self.config.model.asr_model,
                        device_map=self.config.model.device_map,
                        dtype=self.config.model.dtype,
                        compile_model=self.config.performance.compile_asr,
                    )
                _report(progress, f"chunk {window_position}/{total_windows}: transcribing")
                asr_result = asr.transcribe(chunk_path, language=self.config.model.language)
                if cache_enabled:
                    save_asr_result(
                        chunk_asr_cache_path,
                        window=window,
                        model_id=self.config.model.asr_model,
                        language_hint=self.config.model.language,
                        result=asr_result,
                    )
            else:
                _report(progress, f"chunk {window_position}/{total_windows}: ASR cache hit")
            chunk_language = self.config.model.language or asr_result.language
            transcript_chunk = TranscriptChunk(
                index=window.index,
                start=window.start,
                end=window.end,
                text=asr_result.text,
                language=chunk_language,
            )
            chunks.append(transcript_chunk)

            if segment_mode != "none":
                alignment_language = chunk_language or "English"
                local_tokens = None
                if cache_enabled and not options.overwrite_cache:
                    local_tokens = load_cached_alignment_tokens(
                        chunk_asr_cache_path,
                        model_id=self.config.model.aligner_model,
                        language=alignment_language,
                        transcript=asr_result.text,
                    )
                if local_tokens is None:
                    if aligner is None:
                        _report(
                            progress,
                            f"loading aligner model: {self.config.model.aligner_model}",
                        )
                        aligner = QwenForcedAligner(
                            model_id=self.config.model.aligner_model,
                            device_map=self.config.model.device_map,
                            dtype=self.config.model.dtype,
                            compile_model=self.config.performance.compile_aligner,
                        )
                    _report(progress, f"chunk {window_position}/{total_windows}: aligning")
                    local_tokens = aligner.align(
                        chunk_path,
                        asr_result.text,
                        language=alignment_language,
                    )
                    if cache_enabled:
                        save_alignment_tokens(
                            chunk_asr_cache_path,
                            window=window,
                            model_id=self.config.model.aligner_model,
                            language=alignment_language,
                            transcript=asr_result.text,
                            tokens=local_tokens,
                        )
                else:
                    _report(
                        progress,
                        (
                            f"chunk {window_position}/{total_windows}: "
                            f"alignment cache hit ({len(local_tokens)} tokens)"
                        ),
                    )
                global_tokens = project_context_punctuation(
                    [token.shifted(window.start) for token in local_tokens],
                    asr_result.text,
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
                llm_client = OpenAICompatibleLLM(self.config.llm)
                segmenter = LLMSegmenter(
                    llm_client,
                    self.config.segment,
                    self.config.llm,
                    progress=progress,
                )
                segmented_chunks = asyncio.run(segmenter.segment_chunks_async(segment_inputs))
            elif segment_mode == "hybrid":
                llm_client = OpenAICompatibleLLM(self.config.llm)
                segmenter = HybridSegmenter(
                    LocalSegmenter(self.config.segment, refine_mode="soft"),
                    LLMSegmenter(
                        llm_client,
                        self.config.segment,
                        self.config.llm,
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
                subtitle
                for segmented_chunk in segmented_chunks
                for subtitle in segmented_chunk
            ]
        else:
            subtitles = _subtitles_from_transcript_chunks(chunks)
        subtitles = _renumber_subtitles(subtitles)
        _report(progress, f"subtitles: {len(subtitles)}")
        _report_limit_summary(progress, subtitles, self.config.segment.max_duration)

        if options.translate:
            _report(progress, f"translating to {options.translate}")
            llm_client = OpenAICompatibleLLM(self.config.llm)
            subtitles = SubtitleTranslator(
                llm_client,
                self.config.llm,
                progress=progress,
            ).translate(
                subtitles,
                target_language=options.translate,
                source_language=self.config.model.language,
            )
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
        )
        for index, subtitle in enumerate(subtitles, start=1)
    ]


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"


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
