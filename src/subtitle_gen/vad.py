from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import VADConfig
from .media import read_mono_wav_array
from .types import AudioWindow


class VADError(RuntimeError):
    pass


def plan_audio_windows(wav_path: str | Path, duration: float, config: VADConfig) -> list[AudioWindow]:
    if not config.enabled or duration <= config.short_audio_threshold:
        return [AudioWindow(index=1, start=0.0, end=max(0.0, duration))]
    if config.backend != "silero":
        raise VADError(f"Unsupported VAD backend: {config.backend}")

    speech_ranges = detect_speech_silero(wav_path, config)
    return build_windows_from_speech(speech_ranges, duration, config)


def detect_speech_silero(wav_path: str | Path, config: VADConfig) -> list[tuple[float, float]]:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise VADError(
            "silero-vad is not installed. Run `uv sync` or disable VAD."
        ) from exc

    model = load_silero_vad()
    wav = read_wav_tensor(wav_path, sampling_rate=16_000)
    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16_000,
        min_speech_duration_ms=int(config.min_speech_duration * 1000),
        min_silence_duration_ms=int(config.min_silence_duration * 1000),
        speech_pad_ms=int(config.speech_padding * 1000),
        return_seconds=True,
    )
    return [_coerce_speech_range(item) for item in timestamps]


def read_wav_tensor(wav_path: str | Path, sampling_rate: int = 16_000):
    try:
        import torch
    except ImportError as exc:
        raise VADError("torch is required for VAD. Run `uv sync`.") from exc

    mono = read_mono_wav_array(wav_path, sample_rate=sampling_rate)
    return torch.from_numpy(mono.copy())


def build_windows_from_speech(
    speech_ranges: Iterable[tuple[float, float]], duration: float, config: VADConfig
) -> list[AudioWindow]:
    padded = _normalize_speech_ranges(speech_ranges, duration, config.speech_padding)
    if not padded:
        return _split_span(0.0, duration, config.hard_max_chunk_duration)

    windows: list[tuple[float, float]] = []
    current_start: float | None = None
    current_end: float | None = None

    for speech_start, speech_end in padded:
        if speech_end <= speech_start:
            continue
        for part_start, part_end in _split_tuple(
            speech_start, speech_end, config.hard_max_chunk_duration
        ):
            if current_start is None or current_end is None:
                current_start, current_end = part_start, part_end
                continue

            silence_gap = part_start - current_end
            would_duration = part_end - current_start
            should_flush = (
                silence_gap > config.skip_silence_longer_than
                or would_duration > config.max_chunk_duration
                or would_duration > config.hard_max_chunk_duration
            )
            if should_flush:
                windows.append((current_start, current_end))
                current_start, current_end = part_start, part_end
            else:
                current_end = max(current_end, part_end)

    if current_start is not None and current_end is not None:
        windows.append((current_start, current_end))

    normalized = windows
    return [
        AudioWindow(index=index, start=round(start, 3), end=round(end, 3))
        for index, (start, end) in enumerate(normalized, start=1)
        if end > start
    ]


def _coerce_speech_range(item: Any) -> tuple[float, float]:
    if isinstance(item, dict):
        start = item.get("start")
        end = item.get("end")
    else:
        start, end = item
    if start is None or end is None:
        raise VADError(f"Invalid VAD timestamp: {item!r}")
    return float(start), float(end)


def _normalize_speech_ranges(
    speech_ranges: Iterable[tuple[float, float]], duration: float, padding: float
) -> list[tuple[float, float]]:
    normalized = [
        (max(0.0, float(start) - padding), min(duration, float(end) + padding))
        for start, end in speech_ranges
        if float(end) > float(start)
    ]
    normalized.sort(key=lambda item: item[0])
    if not normalized:
        return []

    merged: list[tuple[float, float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _split_tuple(start: float, end: float, max_duration: float) -> list[tuple[float, float]]:
    if max_duration <= 0:
        return [(start, end)]
    parts: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        next_end = min(end, cursor + max_duration)
        parts.append((cursor, next_end))
        cursor = next_end
    return parts


def _split_span(start: float, end: float, max_duration: float) -> list[AudioWindow]:
    parts = _split_tuple(start, end, max_duration)
    return [
        AudioWindow(index=index, start=round(part_start, 3), end=round(part_end, 3))
        for index, (part_start, part_end) in enumerate(parts, start=1)
        if part_end > part_start
    ]
