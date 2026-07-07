from subtitle_gen.pipeline import _subtitles_from_transcript_chunks
from subtitle_gen.types import TranscriptChunk


def test_transcript_chunks_become_unsegmented_subtitles():
    chunks = [
        TranscriptChunk(index=1, start=0.0, end=3.0, text="First. Still first."),
        TranscriptChunk(index=2, start=4.0, end=6.0, text="Second."),
    ]

    subtitles = _subtitles_from_transcript_chunks(chunks)

    assert [subtitle.text for subtitle in subtitles] == ["First. Still first.", "Second."]
    assert [(subtitle.start, subtitle.end) for subtitle in subtitles] == [(0.0, 3.0), (4.0, 6.0)]
