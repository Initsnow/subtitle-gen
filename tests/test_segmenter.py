import asyncio

from subtitle_gen.config import LLMConfig, SegmentConfig
from subtitle_gen.hybrid_segmenter import HybridSegmenter
from subtitle_gen.llm import LLMError
from subtitle_gen.llm_segmenter import LLMSegmenter, groups_from_delimited_text
from subtitle_gen.llm_segmenter import SegmentChunkInput
from subtitle_gen.local_segmenter import LocalSegmenter
from subtitle_gen.segmenter import (
    SegmentAtom,
    join_text_parts,
    project_context_punctuation,
)
from subtitle_gen.types import TimedToken


def test_join_text_parts_handles_cjk_and_punctuation():
    assert join_text_parts(["你", "好", "，", "world", "!"]) == "你好，world!"


def test_local_segmenter_uses_blingfire_sentence_boundaries():
    tokens = [
        TimedToken("Hello", 0.0, 0.4),
        TimedToken("world.", 0.4, 1.0),
        TimedToken("Next", 1.2, 1.6),
        TimedToken("one.", 1.6, 2.2),
    ]

    segments = LocalSegmenter(SegmentConfig(min_duration=0.2)).segment(tokens, language="English")

    assert [segment.text for segment in segments] == ["Hello world.", "Next one."]
    assert [(segment.start, segment.end) for segment in segments] == [(0.0, 1.0), (1.2, 2.2)]


def test_local_segmenter_projects_asr_punctuation_onto_aligned_tokens():
    tokens = [
        TimedToken("Hello", 0.0, 0.4),
        TimedToken("world", 0.4, 1.0),
        TimedToken("Next", 1.2, 1.6),
        TimedToken("one", 1.6, 2.2),
    ]

    segments = LocalSegmenter(SegmentConfig(min_duration=0.2)).segment(
        tokens,
        context_text="Hello world. Next one.",
        language="English",
    )

    assert [segment.text for segment in segments] == ["Hello world.", "Next one."]


def test_project_context_punctuation_keeps_token_timing():
    tokens = [
        TimedToken("Hello", 0.0, 0.4),
        TimedToken("world", 0.4, 1.0),
    ]

    projected = project_context_punctuation(tokens, "Hello world!")

    assert projected == [
        TimedToken("Hello", 0.0, 0.4),
        TimedToken("world!", 0.4, 1.0),
    ]


def test_project_context_punctuation_does_not_duplicate_partial_suffixes():
    projected = project_context_punctuation(
        [TimedToken("Hello.", 0.0, 0.4)],
        'Hello."',
    )

    assert projected == [TimedToken('Hello."', 0.0, 0.4)]


def test_local_segmenter_splits_overlong_sentence_on_soft_boundary():
    tokens = [
        TimedToken("This", 0.0, 0.3),
        TimedToken("is", 0.3, 0.6),
        TimedToken("quite", 0.6, 0.9),
        TimedToken("long,", 0.9, 1.2),
        TimedToken("but", 1.2, 1.5),
        TimedToken("it", 1.5, 1.8),
        TimedToken("can", 1.8, 2.1),
        TimedToken("split.", 2.1, 2.6),
    ]

    segments = LocalSegmenter(
        SegmentConfig(
            mode="local",
            min_duration=0.2,
            max_chars_en=20,
            max_lines=1,
        )
    ).segment(tokens, language="English")

    assert [segment.text for segment in segments] == [
        "This is quite long,",
        "but it can split.",
    ]


def test_local_segmenter_keeps_blingfire_sentence_without_refinement():
    tokens = [
        TimedToken("This", 0.0, 0.3),
        TimedToken("is", 0.3, 0.6),
        TimedToken("quite", 0.6, 0.9),
        TimedToken("long,", 0.9, 1.2),
        TimedToken("but", 1.2, 1.5),
        TimedToken("it", 1.5, 1.8),
        TimedToken("stays", 1.8, 2.1),
        TimedToken("together.", 2.1, 2.6),
    ]

    segments = LocalSegmenter(
        SegmentConfig(mode="blingfire", max_chars_en=20, max_lines=1)
    ).segment(tokens, language="English")

    assert [segment.text for segment in segments] == [
        "This is quite long, but it stays together."
    ]


def test_hybrid_segmenter_soft_splits_before_overlong_llm():
    class SplitLLM:
        config = LLMConfig(concurrency=1)

        def __init__(self) -> None:
            self.calls = 0

        async def complete_text_async(self, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            return "Bravo Charlie|Delta Echo Foxtrot."

    config = SegmentConfig(mode="hybrid", min_duration=0.2, max_chars_en=12, max_lines=1)
    llm = SplitLLM()
    tokens = [
        TimedToken("Alpha,", 0.0, 0.5),
        TimedToken("Bravo", 0.5, 1.0),
        TimedToken("Charlie", 1.0, 1.5),
        TimedToken("Delta", 1.5, 2.0),
        TimedToken("Echo", 2.0, 2.5),
        TimedToken("Foxtrot.", 2.5, 3.0),
    ]

    chunks = asyncio.run(
        HybridSegmenter(
            LocalSegmenter(config, refine_mode="soft"),
            LLMSegmenter(llm, config, LLMConfig(concurrency=1)),
            config,
        ).segment_chunks_async([SegmentChunkInput(tokens, language="English")])
    )

    assert llm.calls == 1
    assert [segment.text for segment in chunks[0]] == [
        "Alpha,",
        "Bravo Charlie",
        "Delta Echo Foxtrot.",
    ]


def test_local_segmenter_merges_too_short_sentences():
    tokens = [
        TimedToken("Yes.", 0.0, 0.3),
        TimedToken("Go", 0.3, 0.8),
        TimedToken("now.", 0.8, 1.5),
    ]

    segments = LocalSegmenter(
        SegmentConfig(mode="local", min_duration=0.8)
    ).segment(tokens, language="English")

    assert [segment.text for segment in segments] == ["Yes. Go now."]


def test_llm_delimiter_output_maps_back_to_token_ids():
    atoms = [
        SegmentAtom(1, "I", 0.0, 0.2),
        SegmentAtom(2, "think", 0.2, 0.7),
        SegmentAtom(3, "this", 0.7, 1.0),
        SegmentAtom(4, "works.", 1.0, 1.5),
        SegmentAtom(5, "Next", 2.5, 3.0),
        SegmentAtom(6, "one.", 3.0, 3.8),
    ]

    groups = groups_from_delimited_text(
        "I think this works.|Next one.",
        atoms,
        SegmentConfig(min_duration=1.0),
    )

    assert groups == [[1, 2, 3, 4], [5, 6]]


def test_llm_delimiter_output_rejects_text_changes():
    atoms = [
        SegmentAtom(1, "hello", 0.0, 0.3),
        SegmentAtom(2, "world", 0.3, 0.8),
    ]

    groups = groups_from_delimited_text("hello|there", atoms, SegmentConfig())

    assert groups == []


def test_llm_delimiter_output_keeps_over_limit_groups():
    atoms = [
        SegmentAtom(1, "This", 0.0, 1.0),
        SegmentAtom(2, "segment", 1.0, 4.0),
        SegmentAtom(3, "stays.", 4.0, 8.0),
        SegmentAtom(4, "Next.", 8.2, 8.8),
    ]

    groups = groups_from_delimited_text(
        "This segment stays.|Next.",
        atoms,
        SegmentConfig(max_duration=1.0, max_chars_en=5),
    )

    assert groups == [[1, 2, 3], [4]]


def test_llm_delimiter_output_snaps_inside_token_split_to_token_boundary():
    atoms = [
        SegmentAtom(1, "こんにちは", 0.0, 1.0),
        SegmentAtom(2, "世界。", 1.0, 2.0),
    ]

    groups = groups_from_delimited_text("こん|にちは世界。", atoms, SegmentConfig())

    assert groups == [[1], [2]]


def test_llm_segmenter_invalid_output_falls_back_to_unsegmented_subtitle():
    class BadLLM:
        config = LLMConfig(concurrency=1)

        async def complete_text_async(self, system_prompt: str, user_prompt: str) -> str:
            return "changed text"

    tokens = [
        TimedToken("Hello", 0.0, 0.3),
        TimedToken("world.", 0.3, 0.8),
    ]
    progress: list[str] = []

    segments = LLMSegmenter(
        BadLLM(),
        SegmentConfig(),
        LLMConfig(concurrency=1),
        progress=progress.append,
    ).segment(tokens)

    assert [segment.text for segment in segments] == ["Hello world."]
    assert [(segment.start, segment.end) for segment in segments] == [(0.0, 0.8)]
    assert progress == [
        "LLM segmentation requests: 1",
        "LLM segmentation 1/1: fallback (invalid output: text changed outside delimiters)",
    ]


def test_llm_segmenter_reports_request_failure_reason():
    class FailingLLM:
        config = LLMConfig(concurrency=1)

        async def complete_text_async(self, system_prompt: str, user_prompt: str) -> str:
            raise LLMError("Missing LLM API key.")

    progress: list[str] = []

    segments = LLMSegmenter(
        FailingLLM(),
        SegmentConfig(),
        LLMConfig(concurrency=1),
        progress=progress.append,
    ).segment([TimedToken("Hello", 0.0, 0.5)])

    assert [segment.text for segment in segments] == ["Hello"]
    assert progress == [
        "LLM segmentation requests: 1",
        "LLM segmentation 1/1: fallback (request error: LLMError: Missing LLM API key.)",
    ]


def test_llm_segmenter_can_require_a_useful_split():
    class NoSplitLLM:
        config = LLMConfig(concurrency=1)

        async def complete_text_async(self, system_prompt: str, user_prompt: str) -> str:
            return "Hello world."

    progress: list[str] = []

    chunks = asyncio.run(
        LLMSegmenter(
            NoSplitLLM(),
            SegmentConfig(),
            LLMConfig(concurrency=1),
            progress=progress.append,
        ).segment_chunks_async(
            [
                SegmentChunkInput(
                    [TimedToken("Hello", 0.0, 0.3), TimedToken("world.", 0.3, 0.8)],
                    require_split=True,
                )
            ]
        )
    )

    assert [segment.text for segment in chunks[0]] == ["Hello world."]
    assert progress == [
        "LLM segmentation requests: 1",
        "LLM segmentation 1/1: fallback (invalid output: no usable split)",
    ]
