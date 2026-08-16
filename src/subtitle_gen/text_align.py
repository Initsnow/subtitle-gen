from __future__ import annotations

import re
import tempfile
import unicodedata
from collections.abc import Callable
from pathlib import Path

from .aligner import AlignmentError, QwenForcedAligner
from .cache import (
    chunk_cache_stem,
    cleanup_cache,
    load_cached_alignment_tokens,
    media_cache_id,
    save_alignment_tokens,
)
from .config import AppConfig
from .formats import parse_lrc
from .media import cut_wav_window, extract_wav, probe_duration
from .processing import (
    ProcessedWindow,
    format_duration as _format_duration,
    process_windows,
    report as _report,
)
from .types import AudioWindow, SubtitleItem, TimedToken
from .vad import plan_audio_windows

ProgressCallback = Callable[[str], None]

# Fixed-window size used when VAD is disabled (e.g. for songs, where Silero VAD
# misses sung vocals). VAD stays the default for speech-like audio.
ALIGN_WINDOW_SECONDS = 30.0
# Adjacent fixed windows overlap by this amount so words sitting exactly on a
# window boundary are recognized by at least one full chunk.
_FIXED_WINDOW_OVERLAP = 1.0
_REFINE_PADDING = 0.5
# Tokens further than this from both neighbours are treated as ASR
# hallucinations on instrumental/silent stretches and dropped before matching.
_SPURIOUS_GAP = 3.0
# Full Needleman-Wunsch is exact but quadratic. Above this cell count we switch
# to the banded implementation, whose half-width is chosen below and capped so
# long audio stays bounded in both time and memory.
_ALIGN_FULL_MAX_CELLS = 2_000_000
_ALIGN_BAND_MAX_CELLS = 24_000_000
_ALIGN_BAND_MIN = 64
_ALIGN_BAND_RATIO = 0.05
# LRC tags whose values describe the *previous* timing. Regenerating timestamps
# makes them stale (offset would be applied twice by players), so they are not
# carried into aligned output.
_LRC_TIMING_METADATA_TAGS = frozenset({"length", "offset"})
_LRC_TAG_RE = re.compile(r"^\[([A-Za-z]+):")


def load_text_input(path: str | Path) -> tuple[list[str], list[str]]:
    """Load untimed subtitle text from a file.

    Returns ``(metadata_lines, text_lines)``. LRC files keep descriptive
    metadata tags (``[ti:...]``, ``[ar:...]``, ...); timing-affecting tags such
    as ``[offset:...]`` and ``[length:...]`` are dropped because alignment
    regenerates timestamps. Any other file is treated as plain text with one
    cue per line.
    """
    input_path = Path(path)
    if input_path.suffix.lower() == ".lrc":
        data = parse_lrc(input_path)
        metadata = [line for line in data.metadata if not _is_timing_lrc_metadata(line)]
        return metadata, data.lines

    content = input_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    return [], lines


def _is_timing_lrc_metadata(line: str) -> bool:
    tag = _LRC_TAG_RE.match(line)
    return tag is not None and tag.group(1).lower() in _LRC_TIMING_METADATA_TAGS


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
        if duration <= 0:
            raise AlignmentError(f"audio has no usable duration: {duration:g} seconds")
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
        if not any(span is not None for span in spans) and tokens:
            # The hallucination filter may have removed every candidate for a
            # short, sparse utterance. Retry once with the unfiltered stream
            # before giving up, so a single real word is not discarded.
            _report(
                progress,
                "spurious token filter removed all matches; retrying with unfiltered tokens",
            )
            spans = match_lines_to_tokens(lines, tokens, language=match_language)

        if not any(span is not None for span in spans):
            raise AlignmentError(
                "no recognized speech tokens could be matched to the input text; "
                "check --language or try a different audio file"
            )

        matched_lines = sum(span is not None for span in spans)
        _report(
            progress,
            f"matched {matched_lines}/{len(lines)} line(s) to recognized speech",
        )
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
                overwrite_cache=overwrite_cache,
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
    media_id = media_cache_id(wav_path)
    if use_vad:
        windows = plan_audio_windows(wav_path, duration, config.vad)
    else:
        windows = _plan_fixed_windows(duration, ALIGN_WINDOW_SECONDS)
    _report(
        progress,
        f"audio: {_format_duration(duration)}; chunks: {len(windows)}; media id: {media_id}",
    )

    if use_cache and config.cache.cleanup_enabled:
        _report(progress, "cleaning stale cache entries")
        cleanup_cache(
            cache_root,
            keep_media_ids={media_id},
            active_windows_by_media={media_id: windows},
            max_media_entries=config.cache.max_media_entries,
        )

    processed = process_windows(
        wav_path,
        cache_root,
        config,
        windows,
        language=language,
        progress=progress,
        overwrite_cache=overwrite_cache,
        with_alignment=True,
    )
    tokens = _merge_window_tokens(processed.windows)
    return tokens, processed.detected_language, processed.aligner


def _merge_window_tokens(processed_windows: list[ProcessedWindow]) -> list[TimedToken]:
    """Shift local tokens to media time and drop duplicates from overlapping
    fixed windows.

    When windows overlap, the tail of the previous window that starts inside
    the overlap is removed and the current window's copy is kept. This keeps
    every word once while still letting boundary words be recognized by a
    window that contains their continuation.
    """
    tokens: list[TimedToken] = []
    previous_window_end: float | None = None
    for processed_window in processed_windows:
        local_tokens = processed_window.local_tokens or []
        cleaned = sorted(
            (token for token in local_tokens if token.text.strip() and token.end >= token.start),
            key=lambda token: (token.start, token.end),
        )
        shifted = [token.shifted(processed_window.window.start) for token in cleaned]
        if (
            tokens
            and previous_window_end is not None
            and processed_window.window.start < previous_window_end - 1e-9
        ):
            while tokens and tokens[-1].start >= processed_window.window.start - 1e-9:
                tokens.pop()
        tokens.extend(shifted)
        previous_window_end = processed_window.window.end
    return tokens


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
    overwrite_cache: bool,
) -> list[SubtitleItem]:
    """Force-align each line's exact text to its own short audio window for
    precise boundaries, falling back to the rough span when alignment fails.

    Refine chunks and alignments live under ``<cache>/refine/<media-id>/`` and
    are reused on subsequent runs when the persistent cache is enabled.
    """
    use_cache = config.cache.enabled
    media_id = media_cache_id(wav_path)
    cache_root = wav_path.parent.parent
    refine_dir = cache_root / "refine" / media_id
    alignment_language = language or "English"
    total = len(lines)
    refine_windows = [_refine_window(index, rough, duration) for index in range(1, total + 1)]

    if use_cache and config.cache.cleanup_enabled:
        _report(progress, "cleaning stale refine cache entries")
        cleanup_cache(
            cache_root,
            keep_media_ids={media_id},
            active_refine_windows_by_media={media_id: refine_windows},
            max_media_entries=config.cache.max_media_entries,
        )

    items: list[SubtitleItem] = []
    for index, (line, (rough_start, rough_end), window) in enumerate(
        zip(lines, rough, refine_windows), start=1
    ):
        start, end = rough_start, rough_end
        if line.strip():
            chunk_path = cut_wav_window(
                wav_path,
                window,
                refine_dir,
                overwrite=overwrite_cache or not use_cache,
            )
            refine_cache_path = refine_dir / f"{chunk_cache_stem(window)}.json"

            local_tokens = None
            if use_cache and not overwrite_cache:
                local_tokens = load_cached_alignment_tokens(
                    refine_cache_path,
                    model_id=config.model.aligner_model,
                    language=alignment_language,
                    transcript=line,
                )
            if local_tokens is None:
                if aligner is None:
                    _report(progress, f"loading aligner model: {config.model.aligner_model}")
                    aligner = QwenForcedAligner(
                        model_id=config.model.aligner_model,
                        device_map=config.model.device_map,
                        dtype=config.model.dtype,
                        compile_model=config.performance.compile_aligner,
                    )
                alignment_failed = False
                try:
                    local_tokens = aligner.align(
                        chunk_path,
                        line,
                        language=alignment_language,
                    )
                except Exception:
                    local_tokens = []
                    alignment_failed = True
                if use_cache and not alignment_failed:
                    save_alignment_tokens(
                        refine_cache_path,
                        window=window,
                        model_id=config.model.aligner_model,
                        language=alignment_language,
                        transcript=line,
                        tokens=local_tokens,
                    )

            viable = [
                token
                for token in local_tokens
                if token.text.strip() and token.end - token.start > 0
            ]
            if viable:
                refined_start = viable[0].start + window.start
                refined_end = viable[-1].end + window.start
                refined_start = min(max(0.0, refined_start), duration)
                refined_end = min(max(0.0, refined_end), duration)
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


def _plan_fixed_windows(
    duration: float,
    window_size: float,
    overlap: float | None = None,
) -> list[AudioWindow]:
    """Tile ``duration`` with fixed-size windows.

    Adjacent windows overlap by ``_FIXED_WINDOW_OVERLAP`` seconds so words at
    chunk boundaries are still seen in a full context by one of the chunks.
    Callers are responsible for de-duplicating tokens from overlapping windows.
    """
    if window_size <= 0:
        window_size = duration or 1.0
    if overlap is None:
        overlap = _FIXED_WINDOW_OVERLAP
    overlap = min(max(0.0, overlap), max(0.0, window_size - 0.001))

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
        cursor = end if end >= duration else end - overlap
    if not windows:
        windows.append(AudioWindow(index=1, start=0.0, end=max(0.0, duration)))
    return windows


def _align_chars(ref: str, asr: str) -> list[int | None]:
    """Global (Needleman-Wunsch) alignment mapping each ref char index to an
    ASR char index, or ``None`` when the ref char aligns to a gap.

    Small inputs use the exact full-matrix implementation. Large inputs switch
    to a bounded-width corridor so the cost stays ``O(n * band)`` instead of
    ``O(n * m)``; if no corridor fits the budget the mapping falls back to all
    gaps rather than allocating an unbounded matrix.
    """
    n, m = len(ref), len(asr)
    if n == 0:
        return []
    if m == 0:
        return [None] * n
    if n * m <= _ALIGN_FULL_MAX_CELLS:
        return _align_chars_full(ref, asr)

    band = _choose_alignment_band(n, m)
    if band is None:
        return [None] * n
    return _align_chars_banded(ref, asr, band) or [None] * n


def _choose_alignment_band(n: int, m: int) -> int | None:
    """Pick a DP half-width for the corridor through an ``n`` by ``m`` matrix."""
    band = max(_ALIGN_BAND_MIN, int(max(n, m) * _ALIGN_BAND_RATIO))
    if m > n:
        # The corridor must be wide enough for the horizontal slope; otherwise
        # consecutive row bands become disconnected.
        band = max(band, (m + n - 1) // n + 1)

    max_by_cells = max(1, (_ALIGN_BAND_MAX_CELLS // n - 1) // 2)
    if band > max_by_cells:
        band = max_by_cells
        if m > n and band < (m + n - 1) // n + 1:
            return None
    return max(1, band)


def _align_chars_full(ref: str, asr: str) -> list[int | None]:
    """Exact quadratic-space global alignment for small inputs."""
    n, m = len(ref), len(asr)
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
            diag = prev[j - 1] + (match_score if ref_char == asr[j - 1] else mismatch_score)
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


def _align_chars_banded(ref: str, asr: str, band: int) -> list[int | None] | None:
    """Cellular corridor alignment around the diagonal from ``(0, 0)`` to
    ``(n, m)``. Returns ``None`` only if the corridor happened to disconnect."""
    n, m = len(ref), len(asr)
    match_score, mismatch_score, gap_score = 2, -1, -2
    neg = -1e18
    rows: list[bytearray] = []
    prev: tuple[int, int, list[float]] | None = None

    for i in range(n + 1):
        lo, hi = _band_bounds(i, n, m, band)
        width = hi - lo + 1
        curr = [0.0] * width
        directions = bytearray(width)

        if i == 0:
            for index, j in enumerate(range(lo, hi + 1)):
                curr[index] = j * gap_score
                if j > 0:
                    directions[index] = 3
        else:
            assert prev is not None
            plo, phi, prev_scores = prev
            ref_char = ref[i - 1]
            for index, j in enumerate(range(lo, hi + 1)):
                diag = neg
                if plo <= j - 1 <= phi:
                    diag = prev_scores[j - 1 - plo] + (
                        match_score if ref_char == asr[j - 1] else mismatch_score
                    )
                up = neg
                if plo <= j <= phi:
                    up = prev_scores[j - plo] + gap_score
                left = neg
                if index > 0:
                    left = curr[index - 1] + gap_score

                if diag >= up and diag >= left:
                    curr[index] = diag
                    directions[index] = 1
                elif up >= left:
                    curr[index] = up
                    directions[index] = 2
                else:
                    curr[index] = left
                    directions[index] = 3

        rows.append(directions)
        prev = (lo, hi, curr)

    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        lo, hi = _band_bounds(i, n, m, band)
        if not lo <= j <= hi:
            return None
        direction = rows[i][j - lo]
        if direction == 1:
            i -= 1
            j -= 1
            mapping[i] = j
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            return None
    return mapping


def _band_bounds(i: int, n: int, m: int, band: int) -> tuple[int, int]:
    center = (i * m + n // 2) // n
    return max(0, center - band), min(m, center + band)


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

    # Enforce monotonic starts.
    for position in range(1, total):
        if starts[position] < starts[position - 1]:
            starts[position] = starts[position - 1]

    # Clamp to media duration and guarantee a positive cue duration. Moving a
    # start back from the media edge can break monotonicity, so that is
    # re-checked below.
    for position in range(total):
        if starts[position] >= duration:
            starts[position] = max(0.0, duration - 0.5)
        if ends[position] > duration:
            ends[position] = duration
        if ends[position] <= starts[position]:
            ends[position] = min(duration, starts[position] + 0.5)

    for position in range(1, total):
        if starts[position] < starts[position - 1]:
            starts[position] = starts[position - 1]
            if ends[position] <= starts[position]:
                ends[position] = min(duration, starts[position] + 0.5)

    # Avoid overlaps by clamping each end to the next line's start.
    for position in range(total - 1):
        if ends[position] > starts[position + 1]:
            ends[position] = starts[position + 1]

    # The overlap clamp above may turn a tiny cue into a zero-length cue.
    for position in range(total):
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


__all__ = [
    "align_lines_to_audio",
    "load_text_input",
    "match_lines_to_tokens",
    "normalize_for_alignment",
]
