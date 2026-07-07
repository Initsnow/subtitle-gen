import json

from subtitle_gen.formats import (
    format_srt_timestamp,
    format_vtt_timestamp,
    render_json,
    render_srt,
    render_vtt,
)
from subtitle_gen.types import SubtitleItem


def test_timestamp_formatting_rounds_millis():
    assert format_srt_timestamp(3661.2345) == "01:01:01,234"
    assert format_vtt_timestamp(0.9996) == "00:00:01.000"


def test_render_srt_bilingual():
    items = [SubtitleItem(1, 0.0, 1.2, "Hello", "你好")]

    content = render_srt(items, "bilingual")

    assert content == "1\n00:00:00,000 --> 00:00:01,200\nHello\n你好\n"


def test_render_vtt_original():
    items = [SubtitleItem(1, 0.0, 1.2, "Hello")]

    content = render_vtt(items)

    assert content.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.200\nHello" in content


def test_render_json_includes_optional_translation():
    items = [SubtitleItem(1, 0.0, 1.2, "Hello", "你好")]

    data = json.loads(render_json(items))

    assert data == [{"id": 1, "start": 0.0, "end": 1.2, "text": "Hello", "translation": "你好"}]
