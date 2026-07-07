import os

from subtitle_gen.asr import ASRResult
from subtitle_gen.cache import (
    asr_cache_path,
    cleanup_cache,
    load_cached_alignment_tokens,
    load_cached_asr_result,
    save_alignment_tokens,
    save_asr_result,
)
from subtitle_gen.types import AudioWindow, TimedToken


def test_asr_cache_round_trips_and_checks_model(tmp_path):
    window = AudioWindow(index=1, start=0.0, end=2.0)
    path = asr_cache_path(tmp_path, "media", window)

    save_asr_result(
        path,
        window=window,
        model_id="asr-a",
        language_hint="English",
        result=ASRResult(text="Hello.", language="English"),
    )

    cached = load_cached_asr_result(path, model_id="asr-a", language_hint="English")

    assert cached == ASRResult(text="Hello.", language="English")
    assert load_cached_asr_result(path, model_id="asr-b", language_hint="English") is None


def test_alignment_cache_uses_transcript_hash(tmp_path):
    window = AudioWindow(index=1, start=0.0, end=2.0)
    path = asr_cache_path(tmp_path, "media", window)
    tokens = [TimedToken("Hello.", 0.0, 0.8)]

    save_alignment_tokens(
        path,
        window=window,
        model_id="aligner-a",
        language="English",
        transcript="Hello.",
        tokens=tokens,
    )

    cached = load_cached_alignment_tokens(
        path,
        model_id="aligner-a",
        language="English",
        transcript="Hello.",
    )

    assert cached == tokens
    assert (
        load_cached_alignment_tokens(
            path,
            model_id="aligner-a",
            language="English",
            transcript="Changed.",
        )
        is None
    )


def test_cleanup_cache_removes_stale_chunks_for_current_media(tmp_path):
    active = AudioWindow(index=1, start=0.0, end=2.0)
    stale = AudioWindow(index=2, start=2.0, end=4.0)
    active_chunk = tmp_path / "chunks" / "media" / "chunk-00001-0.000-2.000.wav"
    stale_chunk = tmp_path / "chunks" / "media" / "chunk-00002-2.000-4.000.wav"
    active_asr = asr_cache_path(tmp_path, "media", active)
    stale_asr = asr_cache_path(tmp_path, "media", stale)
    for path in (active_chunk, stale_chunk, active_asr, stale_asr):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cache", encoding="utf-8")

    cleanup_cache(
        tmp_path,
        keep_media_ids={"media"},
        active_windows_by_media={"media": [active]},
        max_media_entries=12,
    )

    assert active_chunk.exists()
    assert active_asr.exists()
    assert not stale_chunk.exists()
    assert not stale_asr.exists()


def test_cleanup_cache_prunes_old_media_entries(tmp_path):
    for index, media_id in enumerate(("old", "current", "new"), start=1):
        audio = tmp_path / "audio" / f"{media_id}.16k-mono.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_text("audio", encoding="utf-8")
        os.utime(audio, (index, index))

    cleanup_cache(
        tmp_path,
        keep_media_ids={"current"},
        max_media_entries=2,
    )

    assert not (tmp_path / "audio" / "old.16k-mono.wav").exists()
    assert (tmp_path / "audio" / "current.16k-mono.wav").exists()
    assert (tmp_path / "audio" / "new.16k-mono.wav").exists()
