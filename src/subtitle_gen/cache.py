from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .asr import ASRResult
from .types import AudioWindow, TimedToken


CACHE_SCHEMA_VERSION = 1
CHUNK_AUDIO_SUFFIX = ".wav"
ASR_CACHE_SUFFIX = ".json"
AUDIO_CACHE_SUFFIX = ".16k-mono.wav"


def media_cache_id(wav_path: str | Path) -> str:
    name = Path(wav_path).name
    if name.endswith(AUDIO_CACHE_SUFFIX):
        return name[: -len(AUDIO_CACHE_SUFFIX)]
    return Path(wav_path).stem


def chunk_cache_stem(window: AudioWindow) -> str:
    return f"chunk-{window.index:05d}-{window.start:.3f}-{window.end:.3f}"


def asr_cache_path(cache_root: str | Path, media_id: str, window: AudioWindow) -> Path:
    return Path(cache_root) / "asr" / media_id / f"{chunk_cache_stem(window)}.json"


def load_cached_asr_result(
    path: str | Path,
    *,
    model_id: str,
    language_hint: str | None,
) -> ASRResult | None:
    data = _read_cache(path)
    if data is None:
        return None

    asr_data = data.get("asr")
    if not isinstance(asr_data, dict):
        return None
    if asr_data.get("model_id") != model_id:
        return None
    if asr_data.get("language_hint") != language_hint:
        return None

    text = asr_data.get("text")
    if not isinstance(text, str):
        return None
    language = asr_data.get("language")
    return ASRResult(text=text, language=language if isinstance(language, str) else None)


def save_asr_result(
    path: str | Path,
    *,
    window: AudioWindow,
    model_id: str,
    language_hint: str | None,
    result: ASRResult,
) -> None:
    data = _read_cache(path) or _empty_cache(window)
    previous_text = None
    previous_asr = data.get("asr")
    if isinstance(previous_asr, dict) and isinstance(previous_asr.get("text"), str):
        previous_text = previous_asr["text"]

    data["schema_version"] = CACHE_SCHEMA_VERSION
    data["window"] = _window_to_dict(window)
    data["asr"] = {
        "model_id": model_id,
        "language_hint": language_hint,
        "text": result.text,
        "language": result.language,
    }
    if previous_text is not None and previous_text != result.text:
        data.pop("alignment", None)
    _write_cache(path, data)


def load_cached_alignment_tokens(
    path: str | Path,
    *,
    model_id: str,
    language: str,
    transcript: str,
) -> list[TimedToken] | None:
    data = _read_cache(path)
    if data is None:
        return None

    alignment_data = data.get("alignment")
    if not isinstance(alignment_data, dict):
        return None
    if alignment_data.get("model_id") != model_id:
        return None
    if alignment_data.get("language") != language:
        return None
    if alignment_data.get("transcript_sha1") != _text_sha1(transcript):
        return None

    raw_tokens = alignment_data.get("tokens")
    if not isinstance(raw_tokens, list):
        return None

    tokens: list[TimedToken] = []
    for raw_token in raw_tokens:
        if not isinstance(raw_token, dict):
            return None
        text = raw_token.get("text")
        start = raw_token.get("start")
        end = raw_token.get("end")
        if (
            not isinstance(text, str)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            return None
        tokens.append(TimedToken(text=text, start=float(start), end=float(end)))
    return tokens


def save_alignment_tokens(
    path: str | Path,
    *,
    window: AudioWindow,
    model_id: str,
    language: str,
    transcript: str,
    tokens: list[TimedToken],
) -> None:
    data = _read_cache(path) or _empty_cache(window)
    data["schema_version"] = CACHE_SCHEMA_VERSION
    data["window"] = _window_to_dict(window)
    data["alignment"] = {
        "model_id": model_id,
        "language": language,
        "transcript_sha1": _text_sha1(transcript),
        "tokens": [
            {"text": token.text, "start": token.start, "end": token.end}
            for token in tokens
        ],
    }
    _write_cache(path, data)


def cleanup_cache(
    cache_root: str | Path,
    *,
    keep_media_ids: set[str],
    active_windows_by_media: dict[str, list[AudioWindow]] | None = None,
    max_media_entries: int = 12,
) -> None:
    root = Path(cache_root)
    if not root.exists():
        return

    for media_id, active_windows in (active_windows_by_media or {}).items():
        active_stems = {chunk_cache_stem(window) for window in active_windows}
        _remove_stale_children(
            root / "chunks" / media_id,
            active_stems=active_stems,
            suffix=CHUNK_AUDIO_SUFFIX,
            cache_root=root,
        )
        _remove_stale_children(
            root / "asr" / media_id,
            active_stems=active_stems,
            suffix=ASR_CACHE_SUFFIX,
            cache_root=root,
        )

    if max_media_entries <= 0:
        return

    media_ids = _collect_media_ids(root)
    removable = [
        media_id
        for media_id in sorted(
            media_ids - keep_media_ids,
            key=lambda item: (_media_cache_mtime(root, item), item),
        )
    ]
    excess_count = max(0, len(media_ids) - max_media_entries)
    for media_id in removable[:excess_count]:
        _remove_media_cache(root, media_id)


def _empty_cache(window: AudioWindow) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "window": _window_to_dict(window),
    }


def _window_to_dict(window: AudioWindow) -> dict[str, int | float]:
    return {
        "index": window.index,
        "start": window.start,
        "end": window.end,
    }


def _read_cache(path: str | Path) -> dict[str, Any] | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return data


def _write_cache(path: str | Path, data: dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _remove_stale_children(
    directory: Path,
    *,
    active_stems: set[str],
    suffix: str,
    cache_root: Path,
) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if not child.is_file() or child.suffix.lower() != suffix:
            continue
        if child.stem in active_stems:
            continue
        _safe_unlink(child, cache_root)


def _collect_media_ids(root: Path) -> set[str]:
    media_ids: set[str] = set()
    audio_dir = root / "audio"
    if audio_dir.exists():
        for child in audio_dir.iterdir():
            if child.is_file() and child.name.endswith(AUDIO_CACHE_SUFFIX):
                media_ids.add(media_cache_id(child))

    for section in ("chunks", "asr"):
        section_dir = root / section
        if not section_dir.exists():
            continue
        for child in section_dir.iterdir():
            if child.is_dir():
                media_ids.add(child.name)
    return media_ids


def _media_cache_mtime(root: Path, media_id: str) -> float:
    candidates = [
        root / "audio" / f"{media_id}{AUDIO_CACHE_SUFFIX}",
        root / "chunks" / media_id,
        root / "asr" / media_id,
    ]
    mtimes: list[float] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        mtimes.append(candidate.stat().st_mtime)
        if candidate.is_dir():
            mtimes.extend(child.stat().st_mtime for child in candidate.rglob("*") if child.exists())
    return max(mtimes, default=0.0)


def _remove_media_cache(root: Path, media_id: str) -> None:
    _safe_unlink(root / "audio" / f"{media_id}{AUDIO_CACHE_SUFFIX}", root)
    _safe_rmtree(root / "chunks" / media_id, root)
    _safe_rmtree(root / "asr" / media_id, root)


def _safe_unlink(path: Path, cache_root: Path) -> None:
    if not path.exists() or not _is_within(path, cache_root):
        return
    path.unlink()


def _safe_rmtree(path: Path, cache_root: Path) -> None:
    if not path.exists() or not path.is_dir() or not _is_within(path, cache_root):
        return
    shutil.rmtree(path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
