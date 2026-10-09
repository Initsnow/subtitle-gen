from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import VADConfig
from .media import read_mono_wav_array
from .types import AudioWindow


class VADError(RuntimeError):
    pass


# The Silero VAD model is small but was previously reloaded for every media
# file; caching it keeps batch runs free of repeated load overhead.
_vad_model: Any | None = None

# Hard splits land on nominal boundaries (multiples of hard_max_chunk_duration)
# which can chop a word in half. Each interior boundary is snapped to the
# quietest frame within this tolerance so cuts land in natural pauses.
_SNAP_TOLERANCE_SECONDS = 4.0
_SNAP_FRAME_SECONDS = 0.05
# Never move a boundary closer than this to its neighbours when rebuilding.
_SNAP_MIN_PART_SECONDS = 5.0


def get_silero_vad_model() -> Any:
    """Return the shared Silero VAD model, loading it on first use."""
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise VADError(
                "silero-vad is not installed. Reinstall subtitle-gen or disable VAD."
            ) from exc
        _vad_model = load_silero_vad()
    return _vad_model


def plan_audio_windows(
    wav_path: str | Path,
    duration: float,
    config: VADConfig,
    samples: Any | None = None,
) -> list[AudioWindow]:
    """Plan audio windows, reusing ``samples`` when the caller already holds them.

    ``samples`` is an optional mono float32 waveform array at 16 kHz. Passing
    it avoids a second full-file disk read inside the VAD stage.
    """
    if not config.enabled or duration <= config.short_audio_threshold:
        return [AudioWindow(index=1, start=0.0, end=max(0.0, duration))]
    if config.backend != "silero":
        raise VADError(f"Unsupported VAD backend: {config.backend}")

    speech_ranges = detect_speech_silero(wav_path, config, samples=samples)
    return build_windows_from_speech(speech_ranges, duration, config, samples=samples)


def detect_speech_silero(
    wav_path: str | Path, config: VADConfig, samples: Any | None = None
) -> list[tuple[float, float]]:
    try:
        from silero_vad import get_speech_timestamps
    except ImportError as exc:
        raise VADError(
            "silero-vad is not installed. Reinstall subtitle-gen or disable VAD."
        ) from exc

    model = get_silero_vad_model()
    wav = read_wav_tensor(wav_path, sampling_rate=16_000, samples=samples)
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


def read_wav_tensor(
    wav_path: str | Path, sampling_rate: int = 16_000, samples: Any | None = None
):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise VADError(
            "torch is required for VAD. Reinstall subtitle-gen to restore its dependencies."
        ) from exc

    if samples is None:
        samples = read_mono_wav_array(wav_path, sample_rate=sampling_rate)
    # Share memory with the caller's array instead of copying: Silero VAD only
    # reads the waveform. ascontiguousarray is a no-op on already-contiguous
    # input (the normal case) and only copies for exotic views/strides.
    return torch.from_numpy(np.ascontiguousarray(samples))


def build_windows_from_speech(
    speech_ranges: Iterable[tuple[float, float]],
    duration: float,
    config: VADConfig,
    samples: Any | None = None,
) -> list[AudioWindow]:
    padded = _normalize_speech_ranges(speech_ranges, duration, config.speech_padding)
    if not padded:
        return _split_span(0.0, duration, config.hard_max_chunk_duration, samples=samples)

    windows: list[tuple[float, float]] = []
    current_start: float | None = None
    current_end: float | None = None

    for speech_start, speech_end in padded:
        if speech_end <= speech_start:
            continue
        range_parts = _split_tuple(speech_start, speech_end, config.hard_max_chunk_duration)
        range_parts = _snap_boundaries_to_quiet(range_parts, samples)
        for part_start, part_end in range_parts:
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


def _split_span(
    start: float, end: float, max_duration: float, samples: Any | None = None
) -> list[AudioWindow]:
    parts = _snap_boundaries_to_quiet(_split_tuple(start, end, max_duration), samples)
    return [
        AudioWindow(index=index, start=round(part_start, 3), end=round(part_end, 3))
        for index, (part_start, part_end) in enumerate(parts, start=1)
        if part_end > part_start
    ]


def _snap_boundaries_to_quiet(
    parts: list[tuple[float, float]], samples: Any | None, sample_rate: int = 16_000
) -> list[tuple[float, float]]:
    """Snap interior hard-split boundaries to the quietest nearby frame.

    Nominal cuts fall on fixed time multiples and can chop a word in half.
    For each interior boundary, search a +/- tolerance window for the minimum
    RMS frame and cut there instead. Parts stay contiguous and monotonic; with
    the default 60 s spacing and 4 s tolerance, part lengths stay within
    roughly [52, 68] s. Without ``samples`` the parts are returned unchanged.
    """
    if samples is None or len(parts) < 2:
        return parts
    try:
        import numpy as np
    except ImportError:
        return parts

    data = np.asarray(samples, dtype=np.float32)
    total = int(data.shape[0])
    if total == 0:
        return parts
    frame = max(1, int(_SNAP_FRAME_SECONDS * sample_rate))
    tolerance = _SNAP_TOLERANCE_SECONDS

    def quietest_boundary(low: float, high: float) -> float:
        lo = max(0, int(low * sample_rate))
        hi = min(total, int(high * sample_rate) + frame)
        span = hi - lo
        if span < frame:
            return (lo + min(hi, total)) / 2 / sample_rate
        usable = (span // frame) * frame
        blocks = data[lo : lo + usable].reshape(-1, frame)
        rms = np.sqrt(np.mean(np.square(blocks, dtype=np.float64), axis=1))
        return (lo + int(np.argmin(rms)) * frame + frame / 2) / sample_rate

    snapped: list[float] = [parts[0][0]]
    for nominal in (part[0] for part in parts[1:]):
        low = max(nominal - tolerance, snapped[-1] + 1e-6)
        high = min(nominal + tolerance, parts[-1][1] - _SNAP_MIN_PART_SECONDS)
        if high <= low:
            snapped.append(nominal)
            continue
        candidate = min(max(quietest_boundary(low, high), low), high)
        snapped.append(max(candidate, snapped[-1] + 1e-6))
    snapped.append(parts[-1][1])

    rebuilt = [(a, b) for a, b in zip(snapped, snapped[1:]) if b > a]
    return rebuilt if rebuilt else parts
