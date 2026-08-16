from argparse import Namespace
import pytest

from subtitle_gen.cli import _cmd_align, _write_align_outputs
from subtitle_gen.types import SubtitleItem


def test_cmd_align_rejects_out_and_format_together():
    with pytest.raises(SystemExit) as excinfo:
        _cmd_align(["song.flac", "lyrics.txt", "--out", "out.lrc", "--format", "srt"])

    assert excinfo.value.code == 2


def test_write_align_outputs_deduplicates_formats(tmp_path):
    text_path = tmp_path / "lyrics.txt"
    args = Namespace(out=None, text=text_path, formats=["lrc", "srt", "lrc"])
    items = [SubtitleItem(1, 0.0, 1.0, "hello")]

    written = _write_align_outputs(args, items, [])

    assert written == [tmp_path / "lyrics.timed.lrc", tmp_path / "lyrics.timed.srt"]
    assert all(path.exists() for path in written)


def test_write_align_outputs_single_out_path(tmp_path):
    out_path = tmp_path / "result.lrc"
    args = Namespace(out=out_path, text=tmp_path / "lyrics.txt", formats=None)
    items = [SubtitleItem(1, 0.0, 1.0, "hello")]

    written = _write_align_outputs(args, items, ["[ti:Title]"])

    assert written == [out_path]
    assert out_path.read_text(encoding="utf-8") == "[ti:Title]\n[00:00.00]hello\n"
