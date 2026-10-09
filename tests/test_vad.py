import numpy as np

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


def test_hard_split_snaps_boundaries_to_quiet_frames():
    sample_rate = 16_000
    samples = np.ones(150 * sample_rate, dtype="float32")
    # Quiet pauses near (but not on) the nominal 60 s hard-cut boundaries.
    samples[57 * sample_rate : 59 * sample_rate] = 0.0
    samples[118 * sample_rate : 119 * sample_rate] = 0.0
    config = VADConfig(
        speech_padding=0.0,
        skip_silence_longer_than=999.0,
        max_chunk_duration=60.0,
        hard_max_chunk_duration=60.0,
    )

    windows = build_windows_from_speech([(0.0, 150.0)], 150.0, config, samples=samples)

    boundaries = [window.end for window in windows[:-1]]
    assert len(boundaries) == 2
    # Each cut moved into the nearby pause instead of landing on the multiple.
    assert abs(boundaries[0] - 57.0) < 0.5
    assert abs(boundaries[1] - 118.0) < 0.5
    # Snap tolerance keeps part lengths within hard_max +/- tolerance.
    assert all(window.end - window.start <= 68.0 for window in windows)


def test_hard_split_without_samples_keeps_nominal_boundaries():
    config = VADConfig(
        speech_padding=0.0,
        skip_silence_longer_than=999.0,
        max_chunk_duration=60.0,
        hard_max_chunk_duration=60.0,
    )

    windows = build_windows_from_speech([(0.0, 150.0)], 150.0, config)

    assert [(window.start, window.end) for window in windows] == [
        (0.0, 60.0),
        (60.0, 120.0),
        (120.0, 150.0),
    ]
