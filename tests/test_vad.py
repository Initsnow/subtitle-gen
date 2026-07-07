from subtitle_gen.config import VADConfig
from subtitle_gen.vad import build_windows_from_speech, plan_audio_windows


def test_short_audio_skips_vad_backend():
    windows = plan_audio_windows("missing.wav", 30.0, VADConfig(enabled=True))

    assert [(window.start, window.end) for window in windows] == [(0.0, 30.0)]


def test_build_windows_splits_on_long_silence():
    config = VADConfig(
        speech_padding=0.0,
        skip_silence_longer_than=8.0,
        max_chunk_duration=120.0,
        hard_max_chunk_duration=240.0,
    )

    windows = build_windows_from_speech([(1.0, 4.0), (20.0, 22.0)], 30.0, config)

    assert [(window.start, window.end) for window in windows] == [(1.0, 4.0), (20.0, 22.0)]


def test_build_windows_splits_by_max_chunk_duration():
    config = VADConfig(
        speech_padding=0.0,
        skip_silence_longer_than=999.0,
        max_chunk_duration=10.0,
        hard_max_chunk_duration=10.0,
    )

    windows = build_windows_from_speech([(0.0, 25.0)], 25.0, config)

    assert [(window.start, window.end) for window in windows] == [
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 25.0),
    ]
