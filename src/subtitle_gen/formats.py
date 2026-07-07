from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .types import SubtitleItem


SubtitleMode = Literal["original", "translation", "bilingual"]


class FormatError(ValueError):
    pass


def format_srt_timestamp(seconds: float) -> str:
    hours, minutes, secs, millis = _split_timestamp(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    hours, minutes, secs, millis = _split_timestamp(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def render_srt(items: list[SubtitleItem], mode: SubtitleMode = "original") -> str:
    blocks: list[str] = []
    for index, item in enumerate(items, start=1):
        text = render_subtitle_text(item, mode)
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


def render_vtt(items: list[SubtitleItem], mode: SubtitleMode = "original") -> str:
    blocks = ["WEBVTT", ""]
    for item in items:
        text = render_subtitle_text(item, mode)
        if not text:
            continue
        blocks.append(f"{format_vtt_timestamp(item.start)} --> {format_vtt_timestamp(item.end)}")
        blocks.append(text)
        blocks.append("")
    return "\n".join(blocks)


def render_json(items: list[SubtitleItem], mode: SubtitleMode = "original") -> str:
    if mode == "original":
        data = [item.to_dict() for item in items]
    elif mode == "translation":
        data = [
            {
                "id": item.id,
                "start": round(item.start, 3),
                "end": round(item.end, 3),
                "text": item.translation or "",
            }
            for item in items
        ]
    elif mode == "bilingual":
        data = [item.to_dict() for item in items]
    else:
        raise FormatError(f"Unsupported mode: {mode}")
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def write_subtitles(path: str | Path, items: list[SubtitleItem], mode: SubtitleMode = "original") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower().lstrip(".")
    if extension == "srt":
        content = render_srt(items, mode)
    elif extension == "vtt":
        content = render_vtt(items, mode)
    elif extension == "json":
        content = render_json(items, mode)
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
) -> list[Path]:
    out_dir = Path(out_dir)
    written: list[Path] = []
    for subtitle_format in formats:
        normalized_format = subtitle_format.lower().lstrip(".")
        if normalized_format not in {"srt", "vtt", "json"}:
            raise FormatError(f"Unsupported subtitle format: {subtitle_format}")
        written.append(write_subtitles(out_dir / f"{stem}.original.{normalized_format}", items, "original"))
        if include_translation:
            written.append(
                write_subtitles(
                    out_dir / f"{stem}.translation.{normalized_format}",
                    items,
                    "translation",
                )
            )
        if include_bilingual:
            written.append(
                write_subtitles(
                    out_dir / f"{stem}.bilingual.{normalized_format}",
                    items,
                    "bilingual",
                )
            )
    return written


def render_subtitle_text(item: SubtitleItem, mode: SubtitleMode = "original") -> str:
    if mode == "original":
        return item.text
    if mode == "translation":
        return item.translation or ""
    if mode == "bilingual":
        return item.text if not item.translation else f"{item.text}\n{item.translation}"
    raise FormatError(f"Unsupported mode: {mode}")


def _split_timestamp(seconds: float) -> tuple[int, int, int, int]:
    total_millis = max(0, int(round(seconds * 1000)))
    millis = total_millis % 1000
    total_seconds = total_millis // 1000
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return hours, minutes, secs, millis
