from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .aligner import QwenForcedAligner
from .asr import QwenASR
from .cache import (
    asr_cache_path,
    load_cached_alignment_tokens,
    load_cached_asr_result,
    media_cache_id,
    save_alignment_tokens,
    save_asr_result,
)
from .config import AppConfig
from .media import cut_wav_window
from .types import AudioWindow, TimedToken

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ProcessedWindow:
    """ASR/alignment output for one audio window."""

    window: AudioWindow
    transcript: str
    language: str | None
    local_tokens: list[TimedToken] | None


@dataclass(frozen=True)
class ProcessedAudio:
    """ASR/alignment output for all audio windows of one media file."""

    windows: list[ProcessedWindow]
    detected_language: str | None
    aligner: QwenForcedAligner | None


def process_windows(
    wav_path: str | Path,
    cache_root: str | Path,
    config: AppConfig,
    windows: list[AudioWindow],
    *,
    language: str | None = None,
    progress: ProgressCallback | None = None,
    overwrite_cache: bool = False,
    with_alignment: bool = True,
) -> ProcessedAudio:
    """Run ASR (and optionally forced alignment) over every audio window.

    Shared by the subtitle pipeline and the ``align`` command so both paths use
    identical lazy model loading, cache keys, and progress reporting.
    """
    use_cache = config.cache.enabled
    regenerate = overwrite_cache or not use_cache
    media_id = media_cache_id(Path(wav_path))
    chunk_cache = Path(cache_root) / "chunks" / media_id

    asr: QwenASR | None = None
    aligner: QwenForcedAligner | None = None
    results: list[ProcessedWindow] = []
    detected_language: str | None = None
    total_windows = len(windows)

    for window_position, window in enumerate(windows, start=1):
        report(
            progress,
            (
                f"chunk {window_position}/{total_windows} "
                f"{format_duration(window.start)}-{format_duration(window.end)}"
            ),
        )
        chunk_path = cut_wav_window(
            Path(wav_path),
            window,
            chunk_cache,
            overwrite=regenerate,
        )
        cache_path = asr_cache_path(Path(cache_root), media_id, window)

        asr_result = None
        if use_cache and not regenerate:
            asr_result = load_cached_asr_result(
                cache_path,
                model_id=config.model.asr_model,
                language_hint=language,
            )
        if asr_result is None:
            if asr is None:
                report(progress, f"loading ASR model: {config.model.asr_model}")
                asr = QwenASR(
                    model_id=config.model.asr_model,
                    device_map=config.model.device_map,
                    dtype=config.model.dtype,
                    compile_model=config.performance.compile_asr,
                )
            report(progress, f"chunk {window_position}/{total_windows}: transcribing")
            asr_result = asr.transcribe(chunk_path, language=language)
            if use_cache:
                save_asr_result(
                    cache_path,
                    window=window,
                    model_id=config.model.asr_model,
                    language_hint=language,
                    result=asr_result,
                )
        else:
            report(progress, f"chunk {window_position}/{total_windows}: ASR cache hit")

        chunk_language = language or asr_result.language
        if detected_language is None and chunk_language:
            detected_language = chunk_language

        local_tokens: list[TimedToken] | None = None
        if with_alignment:
            alignment_language = chunk_language or "English"
            if use_cache and not regenerate:
                local_tokens = load_cached_alignment_tokens(
                    cache_path,
                    model_id=config.model.aligner_model,
                    language=alignment_language,
                    transcript=asr_result.text,
                )
            if local_tokens is None:
                if aligner is None:
                    report(
                        progress,
                        f"loading aligner model: {config.model.aligner_model}",
                    )
                    aligner = QwenForcedAligner(
                        model_id=config.model.aligner_model,
                        device_map=config.model.device_map,
                        dtype=config.model.dtype,
                        compile_model=config.performance.compile_aligner,
                    )
                report(progress, f"chunk {window_position}/{total_windows}: aligning")
                local_tokens = aligner.align(
                    chunk_path,
                    asr_result.text,
                    language=alignment_language,
                )
                if use_cache:
                    save_alignment_tokens(
                        cache_path,
                        window=window,
                        model_id=config.model.aligner_model,
                        language=alignment_language,
                        transcript=asr_result.text,
                        tokens=local_tokens,
                    )
            else:
                report(
                    progress,
                    (
                        f"chunk {window_position}/{total_windows}: "
                        f"alignment cache hit ({len(local_tokens)} tokens)"
                    ),
                )

        results.append(
            ProcessedWindow(
                window=window,
                transcript=asr_result.text,
                language=chunk_language,
                local_tokens=local_tokens,
            )
        )

    return ProcessedAudio(
        windows=results,
        detected_language=detected_language,
        aligner=aligner,
    )


def report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"


__all__ = [
    "ProcessedAudio",
    "ProcessedWindow",
    "format_duration",
    "process_windows",
    "report",
]
