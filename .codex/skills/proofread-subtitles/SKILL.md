---
name: proofread-subtitles
description: Text-only proofreading for subtitles. Use when asked to 校对字幕, proofread SRT subtitles, correct ASR/transcription errors, clean generated subtitles, or improve subtitle readability without changing meaning. For SRT, work through the project extract/apply script instead of reading raw timestamped content.
---

# Proofread Subtitles

## Workflow

1. For SRT files, extract JSONL first; do not read raw SRT content for proofreading.
2. Proofread from extracted text and neighboring cues only. Do not browse, inspect transcripts, or open media unless the user explicitly asks for source verification.
3. Edit only each JSONL `text` value and keep its `id`.
4. Use neighboring cues to fix ASR errors, homophones, missing punctuation, and obvious transcription artifacts.
5. Keep the original meaning. Do not invent unheard content; if uncertain, leave the safest text or flag it briefly in the final note.
6. Preserve names, terminology, speaker labels, sound markers, markup, and bilingual line order unless clearly wrong.
7. Validate with `uv run python .codex/scripts/srt_text.py check output.srt`.

## SRT Text Script

Use the shared project script from the repository root:

```powershell
uv run python .codex/scripts/srt_text.py extract input.srt work.jsonl
uv run python .codex/scripts/srt_text.py apply input.srt corrected.jsonl output.srt --mode replace
uv run python .codex/scripts/srt_text.py apply input.srt partial-corrections.jsonl output.srt --mode replace --allow-partial
```

Edit JSONL records like this:

```jsonl
{"id":1,"text":"Corrected subtitle text."}
```

Apply requires all cue ids by default. For targeted fixes, create JSONL or TSV with only changed ids and pass `--allow-partial`; unchanged cues remain as-is. For full-file proofreading, keep all cue ids in the corrected JSONL, or apply contiguous ranges with `--allow-partial` to a working SRT.

Use `--format text` only for context-only transcript output, or `--format tsv` for compact id/text data. Keep scratch files under `.codex/work/`.

Reading raw SRT is only for parser or encoding diagnostics. After diagnosis, return to the extract/edit/apply flow.

## Editing Rules

- Prefer minimal, high-confidence corrections over stylistic rewriting.
- Keep subtitles concise and readable, but do not remove meaningful disfluencies unless they are obvious ASR noise or the user asked for polishing.
