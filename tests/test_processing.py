import numpy as np
import pytest

import subtitle_gen.processing as processing
from subtitle_gen.asr import ASRResult
from subtitle_gen.config import AppConfig
from subtitle_gen.types import AudioWindow, TimedToken

SAMPLES = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeASR.instances = 0
    FakeASR.calls = 0
    FakeAligner.instances = 0
    FakeAligner.calls = 0


class FakeASR:
    instances = 0
    calls = 0

    def __init__(self, *args, **kwargs):
        FakeASR.instances += 1

    def transcribe(self, audio, language=None):
        FakeASR.calls += 1
        return ASRResult(text="hello", language="English")


class FakeAligner:
    instances = 0
    calls = 0

    def __init__(self, *args, **kwargs):
        FakeAligner.instances += 1

    def align(self, audio, transcript, language=None):
        FakeAligner.calls += 1
        assert len(audio) > 0
        return [TimedToken(transcript, 0.1, 0.9)]


def _config(tmp_path):
    config = AppConfig()
    return AppConfig(
        model=config.model,
        vad=config.vad,
        segment=config.segment,
        cache=type(config.cache)(enabled=True, cleanup_enabled=False),
        performance=config.performance,
        llm=config.llm,
        segmentation_llm=config.segmentation_llm,
        output=config.output,
        cache_dir=str(tmp_path / "cache"),
    )


def _patch_models(monkeypatch):
    monkeypatch.setattr(processing, "get_asr", lambda **kwargs: FakeASR())
    monkeypatch.setattr(processing, "get_aligner", lambda **kwargs: FakeAligner())
    monkeypatch.setattr(processing, "read_mono_wav_array", lambda path: SAMPLES)


def test_process_windows_runs_and_reuses_cache(monkeypatch, tmp_path):
    wav_path = tmp_path / "audio" / "media.16k-mono.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"fake")
    windows = [AudioWindow(index=1, start=0.0, end=2.0)]
    _patch_models(monkeypatch)

    config = _config(tmp_path)
    first = processing.process_windows(
        wav_path, config.cache_dir, config, windows, language="English"
    )
    second = processing.process_windows(
        wav_path, config.cache_dir, config, windows, language="English"
    )

    assert first.windows[0].transcript == "hello"
    assert first.windows[0].local_tokens == [TimedToken("hello", 0.1, 0.9)]
    assert first.detected_language == "English"
    assert second.windows[0].local_tokens == first.windows[0].local_tokens
    assert FakeASR.instances == 1
    assert FakeASR.calls == 1
    assert FakeAligner.instances == 1
    assert FakeAligner.calls == 1


def test_process_windows_slices_windows_in_memory(monkeypatch, tmp_path):
    wav_path = tmp_path / "audio" / "media.16k-mono.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"fake")
    windows = [AudioWindow(index=1, start=0.5, end=1.5)]
    seen = {}

    def fake_transcribe(audio, language=None):
        seen["audio"] = audio
        return ASRResult(text="hello", language="English")

    class SlicingFakeASR(FakeASR):
        def transcribe(self, audio, language=None):
            return fake_transcribe(audio, language)

    monkeypatch.setattr(processing, "get_asr", lambda **kwargs: SlicingFakeASR())
    monkeypatch.setattr(processing, "get_aligner", lambda **kwargs: FakeAligner())
    monkeypatch.setattr(processing, "read_mono_wav_array", lambda path: SAMPLES)

    processing.process_windows(
        wav_path, _config(tmp_path).cache_dir, _config(tmp_path), windows, language="English"
    )

    # 0.5s..1.5s at 16 kHz -> samples 8000..24000 of the full waveform.
    expected = SAMPLES[8000:24000]
    assert len(seen["audio"]) == len(expected)
    assert list(seen["audio"]) == list(expected)


def test_process_windows_skips_alignment_when_not_requested(monkeypatch, tmp_path):
    wav_path = tmp_path / "audio" / "media.16k-mono.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"fake")
    windows = [AudioWindow(index=1, start=0.0, end=2.0)]
    _patch_models(monkeypatch)

    result = processing.process_windows(
        wav_path,
        _config(tmp_path).cache_dir,
        _config(tmp_path),
        windows,
        language="English",
        with_alignment=False,
    )

    assert result.windows[0].transcript == "hello"
    assert result.windows[0].local_tokens is None
    assert FakeAligner.instances == 0
