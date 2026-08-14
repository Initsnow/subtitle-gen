from __future__ import annotations

import tempfile
import unicodedata
from collections.abc import Callable
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
from .formats import parse_lrc
from .media import cut_wav_window, extract_wav, probe_duration
from .types import AudioWindow, SubtitleItem, TimedToken
from .vad import plan_audio_windows

ProgressCallback = Callable[[str], None]

# Fixed-window size used when VAD is disabled (e.g. for songs, where Silero VAD
# misses sung vocals). VAD stays the default for speech-like audio.
ALIGN_WINDOW_SECONDS = 30.0
_REFINE_PADDING = 0.5
# Tokens further than this from both neighbours are treated as ASR
# hallucinations on instrumental/silent stretches and dropped before matching.
_SPURIOUS_GAP = 3.0


def load_text_input(path: str | Path) -> tuple[list[str], list[str]]:
    """Load untimed subtitle text from a file.

    Returns ``(metadata_lines, text_lines)``. LRC files keep their metadata
    tags (``[ti:...]`` etc.); any other file is treated as plain text with one
    cue per line.
    """
    input_path = Path(path)
    if input_path.suffix.lower() == ".lrc":
        data = parse_lrc(input_path)
        return data.metadata, data.lines

    content = (
        input_path.read_text(encoding="utf-8-sig")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    return [], lines


def normalize_for_alignment(text: str, language: str | None = None) -> str:
    """Lowercase, fold full-width forms, drop punctuation/whitespace, and keep
    letters/digits/CJK. Katakana is folded to hiragana for Japanese to make the
    reference text and ASR output more comparable."""
    folded = unicodedata.normalize("NFKC", text).lower()
    result = "".join(char for char in folded if char.isalnum())
    if language and language.lower().startswith(("ja", "jp")):
        result = _katakana_to_hiragana(result)
    return result


def align_lines_to_audio(
    config: AppConfig,
    audio_path: str | Path,
    lines: list[str],
    *,
    language: str | None = None,
    progress: ProgressCallback | None = None,
    overwrite_cache: bool = False,
    refine: bool = True,
    use_vad: bool = True,
) -> list[SubtitleItem]:
    """Add timestamps to untimed text lines by aligning them to audio.

    A first pass runs ASR + forced alignment over speech windows to get a rough
    time map; a second pass then force-aligns each line's exact text to its own
    audio window for precise boundaries. ``refine`` toggles the second pass;
    ``use_vad`` selects VAD windows (speech) versus fixed-size windows (songs).
    """
    if not lines:
        return []

    input_path = Path(audio_path)
    if config.cache.enabled:
        cache_root = Path(config.cache_dir)
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="subtitle-gen-align-")
        cache_root = Path(temp_dir.name)

    try:
        wav_path, duration = _prepare_wav(
            input_path,
            cache_root,
            config,
            progress=progress,
            overwrite_cache=overwrite_cache,
        )
        tokens, detected_language, aligner = _transcribe_and_align(
            wav_path,
            cache_root,
            config,
            duration,
            language=language,
            progress=progress,
            overwrite_cache=overwrite_cache,
            use_vad=use_vad,
        )
        match_language = language or detected_language
        viable_tokens = _filter_spurious_tokens(tokens)
        spans = match_lines_to_tokens(lines, viable_tokens, language=match_language)
        rough = _finalize_spans(spans, duration)

        if refine:
            return _refine_lines(
                wav_path,
                config,
                lines,
                rough,
                language=match_language,
                duration=duration,
                progress=progress,
                aligner=aligner,
            )

        return [
            SubtitleItem(
                id=index,
                start=round(start, 3),
                end=round(end, 3),
                text=line,
            )
            for index, (line, (start, end)) in enumerate(zip(lines, rough), start=1)
        ]
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def match_lines_to_tokens(
    lines: list[str],
    tokens: list[TimedToken],
    language: str | None = None,
) -> list[tuple[float, float] | None]:
    """Map each reference line to a ``(start, end)`` span in the timed token
    stream, or ``None`` when no token corresponds to the line.

    Uses a global character-level alignment between the concatenated reference
    text and the concatenated token text, then projects each line onto the
    aligned token range.
    """
    if not lines:
        return []
    if not tokens:
        return [None] * len(lines)

    ref_chars: list[str] = []
    ref_char_line: list[int] = []
    for line_index, line in enumerate(lines):
        for char in normalize_for_alignment(line, language):
            ref_chars.append(char)
            ref_char_line.append(line_index)

    asr_chars: list[str] = []
    asr_char_token: list[int] = []
    for token_index, token in enumerate(tokens):
        for char in normalize_for_alignment(token.text, language):
            asr_chars.append(char)
            asr_char_token.append(token_index)

    mapping = _align_chars("".join(ref_chars), "".join(asr_chars))

    line_min_token: list[int | None] = [None] * len(lines)
    line_max_token: list[int | None] = [None] * len(lines)
    for ref_index, asr_index in enumerate(mapping):
        if asr_index is None:
            continue
        line_index = ref_char_line[ref_index]
        token_index = asr_char_token[asr_index]
        if line_min_token[line_index] is None or token_index < line_min_token[line_index]:
            line_min_token[line_index] = token_index
        if line_max_token[line_index] is None or token_index > line_max_token[line_index]:
            line_max_token[line_index] = token_index

    spans: list[tuple[float, float] | None] = []
    for line_index in range(len(lines)):
        min_token = line_min_token[line_index]
        max_token = line_max_token[line_index]
        if min_token is None or max_token is None:
            spans.append(None)
            continue
        spans.append((tokens[min_token].start, tokens[max_token].end))
    return spans


def _prepare_wav(
    input_path: Path,
    cache_root: Path,
    config: AppConfig,
    *,
    progress: ProgressCallback | None,
    overwrite_cache: bool,
) -> tuple[Path, float]:
    _report(progress, f"input: {input_path}")
    _report(progress, "preparing 16 kHz mono audio")
    wav_path = extract_wav(
        input_path,
        cache_root / "audio",
        overwrite=overwrite_cache or not config.cache.enabled,
    )
    _report(progress, "probing audio duration")
    duration = probe_duration(wav_path)
    return wav_path, duration


def _transcribe_and_align(
    wav_path: Path,
    cache_root: Path,
    config: AppConfig,
    duration: float,
    *,
    language: str | None,
    progress: ProgressCallback | None,
    overwrite_cache: bool,
    use_vad: bool,
) -> tuple[list[TimedToken], str | None, QwenForcedAligner | None]:
    use_cache = config.cache.enabled
    regenerate = overwrite_cache or not use_cache
    media_id = media_cache_id(wav_path)
    chunk_cache = cache_root / "chunks" / media_id
    if use_vad:
        windows = plan_audio_windows(wav_path, duration, config.vad)
    else:
        windows = _plan_fixed_windows(duration, ALIGN_WINDOW_SECONDS)
    _report(
        progress,
        f"audio: {_format_duration(duration)}; chunks: {len(windows)}; media id: {media_id}",
    )

    asr: QwenASR | None = None
    aligner: QwenForcedAligner | None = None
    tokens: list[TimedToken] = []
    detected_language: str | None = None
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
            overwrite=regenerate,
        )
        chunk_cache_path = asr_cache_path(cache_root, media_id, window)

        asr_result = None
        if use_cache and not regenerate:
            asr_result = load_cached_asr_result(
                chunk_cache_path,
                model_id=config.model.asr_model,
                language_hint=language,
            )
        if asr_result is None:
            if asr is None:
                _report(progress, f"loading ASR model: {config.model.asr_model}")
                asr = QwenASR(
                    model_id=config.model.asr_model,
                    device_map=config.model.device_map,
                    dtype=config.model.dtype,
                    compile_model=config.performance.compile_asr,
                )
            _report(progress, f"chunk {window_position}/{total_windows}: transcribing")
            asr_result = asr.transcribe(chunk_path, language=language)
            if use_cache:
                save_asr_result(
                    chunk_cache_path,
                    window=window,
                    model_id=config.model.asr_model,
                    language_hint=language,
                    result=asr_result,
                )
        else:
            _report(progress, f"chunk {window_position}/{total_windows}: ASR cache hit")

        chunk_language = language or asr_result.language
        if detected_language is None and chunk_language:
            detected_language = chunk_language
        alignment_language = chunk_language or "English"

        local_tokens = None
        if use_cache and not regenerate:
            local_tokens = load_cached_alignment_tokens(
                chunk_cache_path,
                model_id=config.model.aligner_model,
                language=alignment_language,
                transcript=asr_result.text,
            )
        if local_tokens is None:
            if aligner is None:
                _report(
                    progress,
                    f"loading aligner model: {config.model.aligner_model}",
                )
                aligner = QwenForcedAligner(
                    model_id=config.model.aligner_model,
                    device_map=config.model.device_map,
                    dtype=config.model.dtype,
                    compile_model=config.performance.compile_aligner,
                )
            _report(progress, f"chunk {window_position}/{total_windows}: aligning")
            local_tokens = aligner.align(
                chunk_path,
                asr_result.text,
                language=alignment_language,
            )
            if use_cache:
                save_alignment_tokens(
                    chunk_cache_path,
                    window=window,
                    model_id=config.model.aligner_model,
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
        tokens.extend(token.shifted(window.start) for token in local_tokens)
    return tokens, detected_language, aligner


def _refine_lines(
    wav_path: Path,
    config: AppConfig,
    lines: list[str],
    rough: list[tuple[float, float]],
    *,
    language: str | None,
    duration: float,
    progress: ProgressCallback | None,
    aligner: QwenForcedAligner | None,
) -> list[SubtitleItem]:
    """Force-align each line's exact text to its own short audio window for
    precise boundaries, falling back to the rough span when alignment fails."""
    if aligner is None:
        _report(progress, f"loading aligner model: {config.model.aligner_model}")
        aligner = QwenForcedAligner(
            model_id=config.model.aligner_model,
            device_map=config.model.device_map,
            dtype=config.model.dtype,
            compile_model=config.performance.compile_aligner,
        )

    refine_dir = wav_path.parent / "refine"
    alignment_language = language or "English"
    total = len(lines)
    items: list[SubtitleItem] = []
    for index, (line, (rough_start, rough_end)) in enumerate(zip(lines, rough), start=1):
        start, end = rough_start, rough_end
        if line.strip():
            window = _refine_window(index, rough, duration)
            chunk_path = cut_wav_window(
                wav_path,
                window,
                refine_dir,
                overwrite=True,
            )
            try:
                local_tokens = aligner.align(
                    chunk_path,
                    line,
                    language=alignment_language,
                )
            except Exception:
                local_tokens = []
            viable = [token for token in local_tokens if token.end - token.start > 0]
            if viable:
                refined_start = viable[0].start + window.start
                refined_end = viable[-1].end + window.start
                if refined_end > refined_start:
                    start, end = refined_start, refined_end
        items.append(
            SubtitleItem(
                id=index,
                start=round(start, 3),
                end=round(end, 3),
                text=line,
            )
        )
        _report(progress, f"refining line {index}/{total}")
    return items


def _refine_window(
    index: int,
    rough: list[tuple[float, float]],
    duration: float,
) -> AudioWindow:
    start, end = rough[index - 1]
    low = 0.0
    if index > 1:
        low = (rough[index - 2][1] + start) / 2
    high = duration
    if index < len(rough):
        high = (end + rough[index][0]) / 2

    window_start = max(low, start - _REFINE_PADDING)
    window_end = min(high, end + _REFINE_PADDING)
    if window_end <= window_start:
        window_start, window_end = start, end
    return AudioWindow(
        index=index,
        start=round(window_start, 3),
        end=round(window_end, 3),
    )


def _filter_spurious_tokens(tokens: list[TimedToken]) -> list[TimedToken]:
    """Drop isolated tokens that are almost certainly ASR hallucinations on
    instrumental stretches.

    A token is isolated when both its neighbours are more than ``_SPURIOUS_GAP``
    seconds away (or it sits alone at the start/end of the stream). Dense runs —
    even ones with zero-duration tokens, which the aligner produces for fast
    sung passages — are kept.
    """
    if not tokens:
        return []
    kept: list[TimedToken] = []
    total = len(tokens)
    for index, token in enumerate(tokens):
        previous_gap = token.start - tokens[index - 1].end if index > 0 else None
        next_gap = tokens[index + 1].start - token.end if index + 1 < total else None
        isolated = (previous_gap is None or previous_gap > _SPURIOUS_GAP) and (
            next_gap is None or next_gap > _SPURIOUS_GAP
        )
        if isolated:
            continue
        kept.append(token)
    return kept


def _plan_fixed_windows(duration: float, window_size: float) -> list[AudioWindow]:
    if window_size <= 0:
        window_size = duration or 1.0
    windows: list[AudioWindow] = []
    cursor = 0.0
    index = 1
    while cursor < duration:
        end = min(duration, cursor + window_size)
        if end > cursor:
            windows.append(
                AudioWindow(
                    index=index,
                    start=round(cursor, 3),
                    end=round(end, 3),
                )
            )
            index += 1
        cursor = end
    if not windows:
        windows.append(AudioWindow(index=1, start=0.0, end=max(0.0, duration)))
    return windows


def _align_chars(ref: str, asr: str) -> list[int | None]:
    """Global (Needleman-Wunsch) alignment mapping each ref char index to an
    ASR char index, or ``None`` when the ref char aligns to a gap."""
    n, m = len(ref), len(asr)
    if n == 0:
        return []
    if m == 0:
        return [None] * n

    match_score, mismatch_score, gap_score = 2, -1, -2
    width = m + 1
    # Direction per cell: 1=diag, 2=up (gap in ASR), 3=left (gap in ref).
    directions = bytearray((n + 1) * width)

    prev = [0.0] * width
    for j in range(1, width):
        prev[j] = j * gap_score
        directions[j] = 3

    for i in range(1, n + 1):
        curr = [0.0] * width
        curr[0] = i * gap_score
        directions[i * width] = 2
        row_base = i * width
        ref_char = ref[i - 1]
        for j in range(1, width):
            diag = prev[j - 1] + (
                match_score if ref_char == asr[j - 1] else mismatch_score
            )
            up = prev[j] + gap_score
            left = curr[j - 1] + gap_score
            if diag >= up and diag >= left:
                curr[j] = diag
                directions[row_base + j] = 1
            elif up >= left:
                curr[j] = up
                directions[row_base + j] = 2
            else:
                curr[j] = left
                directions[row_base + j] = 3
        prev = curr

    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        direction = directions[i * width + j]
        if direction == 1:
            i -= 1
            j -= 1
            mapping[i] = j
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            break
    return mapping


def _finalize_spans(
    spans: list[tuple[float, float] | None],
    duration: float,
) -> list[tuple[float, float]]:
    """Turn possibly-None spans into concrete monotonic ``(start, end)`` pairs.

    Lines with no matched tokens are interpolated between neighbouring matched
    anchors so every input line still gets a timestamp.
    """
    total = len(spans)
    starts: list[float | None] = []
    ends: list[float | None] = []
    for span in spans:
        if span is None:
            starts.append(None)
            ends.append(None)
            continue
        start, end = span
        if end <= start:
            starts.append(None)
            ends.append(None)
        else:
            starts.append(start)
            ends.append(end)

    index = 0
    while index < total:
        if starts[index] is not None:
            index += 1
            continue
        run_end = index
        while run_end < total and starts[run_end] is None:
            run_end += 1
        count = run_end - index
        prev_end = ends[index - 1] if index > 0 else None
        next_start = starts[run_end] if run_end < total else None

        if prev_end is not None and next_start is not None:
            step = (next_start - prev_end) / (count + 1)
            for offset in range(count):
                position = prev_end + step * (offset + 1)
                starts[index + offset] = position
                ends[index + offset] = position + max(0.0, min(step, 1.0))
        elif prev_end is not None:
            for offset in range(count):
                starts[index + offset] = prev_end + 0.5 * (offset + 1)
                ends[index + offset] = starts[index + offset] + 1.0
        elif next_start is not None:
            for offset in range(count):
                starts[index + offset] = max(0.0, next_start - 0.5 * (count - offset))
                ends[index + offset] = starts[index + offset] + 0.5
        else:
            for offset in range(count):
                starts[index + offset] = 0.0
                ends[index + offset] = 1.0
        index = run_end

    # Enforce monotonic starts and end >= start.
    for position in range(1, total):
        if starts[position] < starts[position - 1]:
            starts[position] = starts[position - 1]
    for position in range(total):
        if ends[position] <= starts[position]:
            ends[position] = starts[position] + 0.5

    # Avoid overlaps by clamping each end to the next line's start.
    for position in range(total - 1):
        if ends[position] > starts[position + 1]:
            ends[position] = starts[position + 1]

    # Clamp to media duration.
    for position in range(total):
        if starts[position] > duration:
            starts[position] = duration
        if ends[position] > duration:
            ends[position] = duration
        if ends[position] <= starts[position]:
            ends[position] = min(duration, starts[position] + 0.5)

    return [(start, end) for start, end in zip(starts, ends)]


def _katakana_to_hiragana(text: str) -> str:
    characters: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0x30A1 <= codepoint <= 0x30F6:
            characters.append(chr(codepoint - 0x60))
        else:
            characters.append(char)
    return "".join(characters)


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _format_duration(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes:02d}:{secs:04.1f}"


__all__ = [
    "align_lines_to_audio",
    "load_text_input",
    "match_lines_to_tokens",
    "normalize_for_alignment",
]
