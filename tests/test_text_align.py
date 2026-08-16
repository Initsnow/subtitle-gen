from pathlib import Path

import pytest

import subtitle_gen.text_align as text_align
from subtitle_gen.aligner import AlignmentError
from subtitle_gen.config import AppConfig, CacheConfig
from subtitle_gen.processing import ProcessedAudio, ProcessedWindow
from subtitle_gen.text_align import (
    ALIGN_WINDOW_SECONDS,
    _align_chars,
    _filter_spurious_tokens,
    _finalize_spans,
    _merge_window_tokens,
    _plan_fixed_windows,
    _refine_window,
    align_lines_to_audio,
    load_text_input,
    match_lines_to_tokens,
    normalize_for_alignment,
)
from subtitle_gen.types import AudioWindow, TimedToken


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_for_alignment("Hello, World!") == "helloworld"


def test_normalize_folds_fullwidth_and_katakana():
    assert normalize_for_alignment("ABC１２３") == "abc123"
    assert normalize_for_alignment("ベタな愛", "ja") == "べたな愛"
    assert normalize_for_alignment("ベタな愛", "en") == "ベタな愛"


def test_align_chars_matches_identical_sequences():
    assert _align_chars("abc", "abc") == [0, 1, 2]


def test_align_chars_handles_gaps_on_either_side():
    assert _align_chars("ac", "abc") == [0, 2]
    assert _align_chars("abc", "ac") == [0, None, 1]


def test_align_chars_empty_inputs():
    assert _align_chars("", "abc") == []
    assert _align_chars("abc", "") == [None, None, None]


def test_align_chars_switches_to_banded_implementation_for_large_inputs():
    text = "ab" * 1200

    assert _align_chars(text, text) == list(range(len(text)))


def test_match_lines_to_tokens_assigns_token_spans():
    tokens = [
        TimedToken("Hello", 0.0, 0.5),
        TimedToken("world", 0.5, 1.0),
        TimedToken("goodbye", 2.0, 2.5),
        TimedToken("moon", 2.5, 3.0),
    ]

    spans = match_lines_to_tokens(["Hello world", "Goodbye moon"], tokens)

    assert spans == [(0.0, 1.0), (2.0, 3.0)]


def test_match_lines_to_tokens_gaps_unmatched_lines():
    tokens = [
        TimedToken("hello", 0.0, 0.5),
        TimedToken("world", 1.0, 1.5),
    ]

    spans = match_lines_to_tokens(["hello", "missing", "world"], tokens)

    assert spans == [(0.0, 0.5), None, (1.0, 1.5)]


def test_match_lines_to_tokens_repeated_lines_stay_in_order():
    tokens = [
        TimedToken("hello", 0.0, 0.5),
        TimedToken("again", 1.0, 1.5),
        TimedToken("hello", 2.0, 2.5),
    ]

    spans = match_lines_to_tokens(["hello", "hello"], tokens)

    assert spans == [(0.0, 0.5), (2.0, 2.5)]


def test_finalize_spans_interpolates_between_anchors():
    finalized = _finalize_spans([(0.0, 1.0), None, None, (4.0, 5.0)], 10.0)

    assert finalized == [(0.0, 1.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0)]


def test_finalize_spans_clamps_overlaps_and_duration():
    finalized = _finalize_spans([(0.0, 8.0), (2.0, 3.0)], 5.0)

    assert finalized[0][1] == finalized[1][0]
    assert all(end <= 5.0 for _start, end in finalized)


def test_finalize_spans_keeps_positive_duration_at_media_edge():
    finalized = _finalize_spans([(0.3, 0.4)], 0.1)

    assert finalized == [(0.0, 0.1)]


def test_load_text_input_plain_text(tmp_path):
    path = tmp_path / "lyrics.txt"
    path.write_text("line one\n\nline two\r\n", encoding="utf-8")

    metadata, lines = load_text_input(path)

    assert metadata == []
    assert lines == ["line one", "line two"]


def test_load_text_input_lrc_returns_metadata(tmp_path):
    path = tmp_path / "song.lrc"
    path.write_text("[ti:Title]\n[00:01.00]hello\n", encoding="utf-8")

    metadata, lines = load_text_input(path)

    assert metadata == ["[ti:Title]"]
    assert lines == ["hello"]


def test_load_text_input_lrc_drops_stale_timing_metadata(tmp_path):
    path = tmp_path / "song.lrc"
    path.write_text(
        "[ti:Title]\n[offset:+500]\n[length:03:00]\n[00:01.00]hello\n",
        encoding="utf-8",
    )

    metadata, lines = load_text_input(path)

    assert metadata == ["[ti:Title]"]
    assert lines == ["hello"]


def test_plan_fixed_windows_overlaps_by_default():
    windows = _plan_fixed_windows(75.0, ALIGN_WINDOW_SECONDS)

    assert [(window.start, window.end) for window in windows] == [
        (0.0, 30.0),
        (29.0, 59.0),
        (58.0, 75.0),
    ]


def test_plan_fixed_windows_can_tile_without_overlap():
    windows = _plan_fixed_windows(75.0, 30.0, overlap=0.0)

    assert [(window.start, window.end) for window in windows] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 75.0),
    ]


def test_plan_fixed_windows_empty_duration_returns_single_window():
    windows = _plan_fixed_windows(0.0, 30.0)

    assert [(window.start, window.end) for window in windows] == [(0.0, 0.0)]


def test_merge_window_tokens_deduplicates_overlap():
    processed = [
        ProcessedWindow(
            window=AudioWindow(index=1, start=0.0, end=30.0),
            transcript="a b",
            language="English",
            local_tokens=[
                TimedToken("a", 0.0, 1.0),
                TimedToken("b", 29.0, 30.0),
            ],
        ),
        ProcessedWindow(
            window=AudioWindow(index=2, start=29.0, end=59.0),
            transcript="b c",
            language="English",
            local_tokens=[
                TimedToken("b", 0.0, 1.0),
                TimedToken("c", 1.0, 2.0),
            ],
        ),
    ]

    tokens = _merge_window_tokens(processed)

    assert [(token.text, token.start, token.end) for token in tokens] == [
        ("a", 0.0, 1.0),
        ("b", 29.0, 30.0),
        ("c", 30.0, 31.0),
    ]


def test_refine_window_clamps_to_neighbour_midpoints():
    rough = [(0.0, 2.0), (4.0, 6.0), (10.0, 12.0)]

    window = _refine_window(2, rough, duration=20.0)

    assert window.start == 3.5
    assert window.end == 6.5


def test_filter_spurious_tokens_drops_isolated_hallucinations():
    tokens = [
        TimedToken("好き", 0.0, 0.0),
        TimedToken("に", 14.72, 14.88),
        TimedToken("あて", 23.28, 23.52),
        TimedToken("ない", 23.76, 24.08),
        TimedToken("明かり", 24.08, 25.84),
    ]

    kept = _filter_spurious_tokens(tokens)

    assert [token.text for token in kept] == ["あて", "ない", "明かり"]


def test_filter_spurious_tokens_keeps_dense_zero_duration_tokens():
    tokens = [
        TimedToken("すてき", 10.0, 10.0),
        TimedToken("な", 10.1, 10.1),
        TimedToken("って", 10.2, 10.2),
        TimedToken("おもう", 10.3, 11.0),
    ]

    kept = _filter_spurious_tokens(tokens)

    assert [token.text for token in kept] == ["すてき", "な", "って", "おもう"]


def test_filter_spurious_tokens_keeps_dense_clusters():
    tokens = [
        TimedToken("a", 0.0, 0.5),
        TimedToken("b", 0.6, 1.0),
        TimedToken("c", 1.1, 1.5),
    ]

    kept = _filter_spurious_tokens(tokens)

    assert [token.text for token in kept] == ["a", "b", "c"]


def _fake_media(monkeypatch, tmp_path):
    def fake_extract_wav(input_path, output_dir, **kwargs):
        path = Path(output_dir) / "fake.16k-mono.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        return path

    def fake_probe_duration(path):
        return 10.0

    monkeypatch.setattr(text_align, "extract_wav", fake_extract_wav)
    monkeypatch.setattr(text_align, "probe_duration", fake_probe_duration)
    return Path("song.flac")


def _config_with_cache(cache_dir):
    config = AppConfig()
    return AppConfig(
        model=config.model,
        vad=config.vad,
        segment=config.segment,
        cache=CacheConfig(enabled=True, cleanup_enabled=True),
        performance=config.performance,
        llm=config.llm,
        segmentation_llm=config.segmentation_llm,
        output=config.output,
        cache_dir=str(cache_dir),
    )


def test_align_lines_to_audio_raises_when_nothing_can_be_matched(monkeypatch, tmp_path):
    audio = _fake_media(monkeypatch, tmp_path)
    window = AudioWindow(index=1, start=0.0, end=10.0)
    monkeypatch.setattr(
        text_align,
        "process_windows",
        lambda *args, **kwargs: ProcessedAudio(
            windows=[ProcessedWindow(window, "", "English", [])],
            detected_language="English",
            aligner=None,
        ),
    )

    with pytest.raises(AlignmentError):
        align_lines_to_audio(AppConfig(), audio, ["hello"], refine=False)


def test_align_lines_to_audio_falls_back_to_unfiltered_single_token(monkeypatch, tmp_path):
    audio = _fake_media(monkeypatch, tmp_path)
    window = AudioWindow(index=1, start=0.0, end=10.0)
    monkeypatch.setattr(
        text_align,
        "process_windows",
        lambda *args, **kwargs: ProcessedAudio(
            windows=[
                ProcessedWindow(
                    window,
                    "hello",
                    "English",
                    [TimedToken("hello", 0.5, 1.0)],
                )
            ],
            detected_language="English",
            aligner=None,
        ),
    )

    items = align_lines_to_audio(AppConfig(), audio, ["hello"], refine=False)

    assert [(item.start, item.end, item.text) for item in items] == [(0.5, 1.0, "hello")]


def test_align_lines_to_audio_caches_refine_alignments(monkeypatch, tmp_path):
    audio = _fake_media(monkeypatch, tmp_path)
    window = AudioWindow(index=1, start=0.0, end=10.0)

    class FakeAligner:
        instances = 0
        calls = 0

        def __init__(self, *args, **kwargs):
            FakeAligner.instances += 1

        def align(self, path, transcript, language=None):
            FakeAligner.calls += 1
            return [TimedToken(transcript, 0.2, 0.8)]

    def fake_cut_wav_window(wav_path, window, output_dir, **kwargs):
        path = (
            Path(output_dir) / f"chunk-{window.index:05d}-{window.start:.3f}-{window.end:.3f}.wav"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        return path

    def fake_process_windows(*args, **kwargs):
        return ProcessedAudio(
            windows=[
                ProcessedWindow(
                    window,
                    "hello",
                    "English",
                    [TimedToken("hello", 0.5, 1.0)],
                )
            ],
            detected_language="English",
            aligner=None,
        )

    monkeypatch.setattr(text_align, "QwenForcedAligner", FakeAligner)
    monkeypatch.setattr(text_align, "cut_wav_window", fake_cut_wav_window)
    monkeypatch.setattr(text_align, "process_windows", fake_process_windows)

    config = _config_with_cache(tmp_path / "cache")
    first = align_lines_to_audio(config, audio, ["hello"], refine=True)
    second = align_lines_to_audio(config, audio, ["hello"], refine=True)

    assert [(item.start, item.end, item.text) for item in first] == [(0.2, 0.8, "hello")]
    assert second == first
    assert FakeAligner.instances == 1
    assert FakeAligner.calls == 1
