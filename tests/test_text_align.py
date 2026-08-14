from subtitle_gen.text_align import (
    _align_chars,
    _filter_spurious_tokens,
    _finalize_spans,
    _plan_fixed_windows,
    _refine_window,
    load_text_input,
    match_lines_to_tokens,
    normalize_for_alignment,
)
from subtitle_gen.types import TimedToken


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


def test_plan_fixed_windows_tiles_full_duration():
    windows = _plan_fixed_windows(75.0, 30.0)

    assert [(window.start, window.end) for window in windows] == [
        (0.0, 30.0),
        (30.0, 60.0),
        (60.0, 75.0),
    ]


def test_plan_fixed_windows_empty_duration_returns_single_window():
    windows = _plan_fixed_windows(0.0, 30.0)

    assert [(window.start, window.end) for window in windows] == [(0.0, 0.0)]


def test_refine_window_clamps_to_neighbour_midpoints():
    rough = [(0.0, 2.0), (4.0, 6.0), (10.0, 12.0)]

    window = _refine_window(2, rough, duration=20.0)

    # Padded window stays inside the neighbouring lines' midpoints.
    assert window.start == 3.5
    assert window.end == 6.5


def test_filter_spurious_tokens_drops_isolated_hallucinations():
    tokens = [
        TimedToken("好き", 0.0, 0.0),       # isolated hallucination
        TimedToken("に", 14.72, 14.88),      # isolated hallucination
        TimedToken("あて", 23.28, 23.52),    # start of real vocals
        TimedToken("ない", 23.76, 24.08),
        TimedToken("明かり", 24.08, 25.84),
    ]

    kept = _filter_spurious_tokens(tokens)

    assert [token.text for token in kept] == ["あて", "ない", "明かり"]


def test_filter_spurious_tokens_keeps_dense_zero_duration_tokens():
    tokens = [
        TimedToken("すてき", 10.0, 10.0),    # zero duration but dense
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
