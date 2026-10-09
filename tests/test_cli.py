from argparse import Namespace
from pathlib import Path

import pytest

import subtitle_gen.cli as cli
from subtitle_gen.cli import _cmd_align, _cmd_pipeline, _write_align_outputs
from subtitle_gen.config import AppConfig
from subtitle_gen.pipeline import PipelineOptions, PipelineResult
from subtitle_gen.types import SubtitleItem


class _NullProgress:
    def __enter__(self):
        return lambda message: None

    def __exit__(self, *exc_info):
        return False


class _FakePipeline:
    instances = 0
    fail_inputs: set = set()

    def __init__(self, config):
        _FakePipeline.instances += 1
        self.calls: list[Path] = []

    def run(self, options: PipelineOptions) -> PipelineResult:
        self.calls.append(options.input_path)
        if str(options.input_path) in _FakePipeline.fail_inputs:
            raise RuntimeError("boom")
        return PipelineResult(
            subtitles=[SubtitleItem(1, 0.0, 1.0, f"text {options.input_path.stem}")],
            transcript_chunks=[],
        )


@pytest.fixture(autouse=True)
def _reset_fake_pipeline():
    _FakePipeline.instances = 0
    _FakePipeline.fail_inputs = set()


def _patch_pipeline(monkeypatch):
    monkeypatch.setattr(cli, "load_config", lambda path=None: AppConfig())
    monkeypatch.setattr(cli, "SubtitlePipeline", _FakePipeline)
    monkeypatch.setattr(cli, "RichProgressReporter", _NullProgress)


def test_cmd_align_rejects_out_and_format_together():
    with pytest.raises(SystemExit) as excinfo:
        _cmd_align(["song.flac", "lyrics.txt", "--out", "out.lrc", "--format", "srt"])

    assert excinfo.value.code == 2


def test_cmd_pipeline_rejects_out_with_multiple_inputs():
    with pytest.raises(SystemExit) as excinfo:
        _cmd_pipeline(["a.mp4", "b.mp4", "--out", "out.srt"])

    assert excinfo.value.code == 2


def test_cmd_pipeline_processes_multiple_files_with_one_pipeline(monkeypatch, tmp_path, capsys):
    _patch_pipeline(monkeypatch)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    out_dir = tmp_path / "out"

    exit_code = _cmd_pipeline([str(first), str(second), "--out-dir", str(out_dir)])

    assert exit_code == 0
    assert _FakePipeline.instances == 1
    assert (out_dir / "first.original.srt").exists()
    assert (out_dir / "second.original.srt").exists()
    written = capsys.readouterr().out.splitlines()
    assert [Path(line).name for line in written] == [
        "first.original.srt",
        "second.original.srt",
    ]


def test_cmd_pipeline_continues_after_single_file_failure(monkeypatch, tmp_path, capsys):
    _patch_pipeline(monkeypatch)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    _FakePipeline.fail_inputs = {str(first)}
    out_dir = tmp_path / "out"

    exit_code = _cmd_pipeline([str(first), str(second), "--out-dir", str(out_dir)])

    assert exit_code == 1
    assert _FakePipeline.instances == 1
    assert not (out_dir / "first.original.srt").exists()
    assert (out_dir / "second.original.srt").exists()
    stderr = capsys.readouterr().err
    assert "1/2 input(s) failed" in stderr
    assert "first.mp4" in stderr


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
