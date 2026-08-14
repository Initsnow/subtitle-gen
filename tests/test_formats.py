import json

from subtitle_gen.formats import (
    format_lrc_timestamp,
    format_srt_timestamp,
    format_vtt_timestamp,
    parse_lrc,
    render_json,
    render_lrc,
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


def test_format_lrc_timestamp_centiseconds():
    assert format_lrc_timestamp(0.0) == "[00:00.00]"
    assert format_lrc_timestamp(65.789) == "[01:05.79]"


def test_render_lrc_keeps_metadata_and_uses_start_times():
    items = [
        SubtitleItem(1, 5.0, 7.0, "first line"),
        SubtitleItem(2, 9.25, 11.0, "second line"),
    ]

    content = render_lrc(items, metadata=["[ti:Title]", "[ar:Artist]"])

    assert content == (
        "[ti:Title]\n[ar:Artist]\n[00:05.00]first line\n[00:09.25]second line\n"
    )


def test_parse_lrc_strips_timestamps_and_keeps_metadata(tmp_path):
    path = tmp_path / "song.lrc"
    path.write_text(
        "[ti:きっとそう]\n[ar:ルサンチマン]\n[00:12.34]ああ、きっとそう\n[00:15.00]間違ってないような\n",
        encoding="utf-8",
    )

    data = parse_lrc(path)

    assert data.metadata == ["[ti:きっとそう]", "[ar:ルサンチマン]"]
    assert data.lines == ["ああ、きっとそう", "間違ってないような"]


def test_parse_lrc_strips_inline_word_tags(tmp_path):
    path = tmp_path / "song.lrc"
    path.write_text("[00:01.00]<00:01.00>あ<00:01.50>あ\n", encoding="utf-8")

    assert parse_lrc(path).lines == ["ああ"]
