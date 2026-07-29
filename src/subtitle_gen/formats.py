from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from .types import SubtitleItem


SubtitleMode = Literal["original", "translation", "bilingual", "proofread"]

_SRT_TIMING_RE = re.compile(r"\d\d:\d\d:\d\d[,.]\d\d\d\s+-->\s+\d\d:\d\d:\d\d[,.]\d\d\d")
_SRT_TIMESTAMP_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
_STRIP_PUNCTUATION_CHARS = set(",.;:!?，。；：！？、")


class FormatError(ValueError):
    pass


def format_srt_timestamp(seconds: float) -> str:
    hours, minutes, secs, millis = _split_timestamp(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    hours, minutes, secs, millis = _split_timestamp(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def render_srt(
    items: list[SubtitleItem],
    mode: SubtitleMode = "original",
    *,
    strip_punctuation: bool = False,
) -> str:
    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        text = render_subtitle_text(item, mode, strip_punctuation=strip_punctuation)
        if not text:
            continue
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(item.start)} --> {format_srt_timestamp(item.end)}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(
    items: list[SubtitleItem],
    mode: SubtitleMode = "original",
    *,
    strip_punctuation: bool = False,
) -> str:
    blocks = ["WEBVTT", ""]
    for item in items:
        text = render_subtitle_text(item, mode, strip_punctuation=strip_punctuation)
        if not text:
            continue
        blocks.append(f"{format_vtt_timestamp(item.start)} --> {format_vtt_timestamp(item.end)}")
        blocks.append(text)
        blocks.append("")
    return "\n".join(blocks)


def render_json(
    items: list[SubtitleItem],
    mode: SubtitleMode = "original",
    *,
    strip_punctuation: bool = False,
) -> str:
    if mode == "original":
        data = [item.to_dict() for item in items]
    elif mode == "proofread":
        data = [
            {
                "id": item.id,
                "start": round(item.start, 3),
                "end": round(item.end, 3),
                "text": _strip_punctuation(item.proofread or item.text) if strip_punctuation else (item.proofread or item.text),
            }
            for item in items
        ]
    elif mode == "translation":
        data = [
            {
                "id": item.id,
                "start": round(item.start, 3),
                "end": round(item.end, 3),
                "text": _strip_punctuation(item.translation or "") if strip_punctuation else (item.translation or ""),
            }
            for item in items
        ]
    elif mode == "bilingual":
        data = [item.to_dict() for item in items]
        if strip_punctuation:
            for entry in data:
                if "text" in entry and isinstance(entry["text"], str):
                    entry["text"] = _strip_punctuation(entry["text"])
                if "translation" in entry and isinstance(entry["translation"], str):
                    entry["translation"] = _strip_punctuation(entry["translation"])
                if "proofread" in entry and isinstance(entry["proofread"], str):
                    entry["proofread"] = _strip_punctuation(entry["proofread"])
    else:
        raise FormatError(f"Unsupported mode: {mode}")
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_subtitles(
    path: str | Path,
    items: list[SubtitleItem],
    mode: SubtitleMode = "original",
    *,
    strip_punctuation: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower().lstrip(".")
    if extension == "srt":
        content = render_srt(items, mode, strip_punctuation=strip_punctuation)
    elif extension == "vtt":
        content = render_vtt(items, mode, strip_punctuation=strip_punctuation)
    elif extension == "json":
        content = render_json(items, mode, strip_punctuation=strip_punctuation)
    else:
        raise FormatError(f"Unsupported subtitle format: {extension}")
    path.write_text(content, encoding="utf-8")
    return path


def write_output_set(
    out_dir: str | Path,
    stem: str,
    items: list[SubtitleItem],
    formats: list[str] | tuple[str, ...],
    include_translation: bool = False,
    include_bilingual: bool = False,
    include_proofread: bool = False,
    *,
    strip_punctuation: bool = False,
) -> list[Path]:
    out_dir = Path(out_dir)
    written: list[Path] = []
    for subtitle_format in formats:
        normalized_format = subtitle_format.lower().lstrip(".")
        if normalized_format not in {"srt", "vtt", "json"}:
            raise FormatError(f"Unsupported subtitle format: {subtitle_format}")
        written.append(
            write_subtitles(
                out_dir / f"{stem}.original.{normalized_format}",
                items,
                "original",
                strip_punctuation=strip_punctuation,
            )
        )
        if include_proofread:
            written.append(
                write_subtitles(
                    out_dir / f"{stem}.proofread.{normalized_format}",
                    items,
                    "proofread",
                    strip_punctuation=strip_punctuation,
                )
            )
        if include_translation:
            written.append(
                write_subtitles(
                    out_dir / f"{stem}.translation.{normalized_format}",
                    items,
                    "translation",
                    strip_punctuation=strip_punctuation,
                )
            )
        if include_bilingual:
            written.append(
                write_subtitles(
                    out_dir / f"{stem}.bilingual.{normalized_format}",
                    items,
                    "bilingual",
                    strip_punctuation=strip_punctuation,
                )
            )
    return written


def render_subtitle_text(
    item: SubtitleItem,
    mode: SubtitleMode = "original",
    *,
    strip_punctuation: bool = False,
) -> str:
    if mode == "original":
        text = item.text
    elif mode == "proofread":
        text = item.proofread or item.text
    elif mode == "translation":
        text = item.translation or ""
    elif mode == "bilingual":
        base = item.proofread or item.text
        text = base if not item.translation else f"{base}\n{item.translation}"
    else:
        raise FormatError(f"Unsupported mode: {mode}")
    if strip_punctuation:
        text = _strip_punctuation(text)
    return text


def _strip_punctuation(text: str) -> str:
    return "\n".join(line.rstrip("".join(_STRIP_PUNCTUATION_CHARS)) for line in text.splitlines())


def _split_timestamp(seconds: float) -> tuple[int, int, int, int]:
    total_millis = max(0, int(round(seconds * 1000)))
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return hours, minutes, secs, millis


def parse_srt(path: str | Path) -> list[SubtitleItem]:
    """Parse an SRT file into SubtitleItem objects."""
    content = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if not content.strip():
        return []

    items: list[SubtitleItem] = []
    seen_ids: set[int] = set()
    for ordinal, raw_block in enumerate(re.split(r"\n{2,}", content.strip("\n")), start=1):
        lines = raw_block.split("\n")
        timing_idx = next((i for i, line in enumerate(lines) if _SRT_TIMING_RE.search(line)), -1)
        if timing_idx < 0:
            continue

        cue_id = ordinal
        if timing_idx > 0 and lines[0].strip().isdigit():
            cue_id = int(lines[0].strip())
        if cue_id in seen_ids:
            raise FormatError(f"Duplicate cue id in SRT: {cue_id}")
        seen_ids.add(cue_id)

        start, end = _parse_srt_timing_line(lines[timing_idx])
        text = "\n".join(lines[timing_idx + 1:]).strip()
        if not text:
            continue
        items.append(SubtitleItem(id=cue_id, start=start, end=end, text=text))

    return items


def _parse_srt_timing_line(line: str) -> tuple[float, float]:
    parts = line.split("-->")
    if len(parts) != 2:
        raise FormatError(f"Invalid SRT timing line: {line}")
    return _parse_srt_timestamp(parts[0].strip()), _parse_srt_timestamp(parts[1].strip())


def _parse_srt_timestamp(ts: str) -> float:
    m = _SRT_TIMESTAMP_RE.match(ts)
    if not m:
        raise FormatError(f"Invalid SRT timestamp: {ts}")
    hours, minutes, secs, millis = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    return hours * 3600.0 + minutes * 60.0 + secs + millis / 1000.0
