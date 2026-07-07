---
name: proofread-subtitles
description: Proofread and correct subtitle text while preserving subtitle structure. Use when asked to 校对字幕, correct ASR/transcription errors, clean generated SRT subtitles, remove obvious ASR artifacts, or improve source subtitle readability without changing meaning; prefer the project SRT text extraction/apply script to avoid spending tokens on timestamps.
---

# Proofread Subtitles

## Workflow

1. For SRT files, run `.codex/scripts/srt_text.py` to extract cue text before editing; do not load timestamps into the prompt unless timing is relevant.
2. Preserve ids, timestamps, cue order, blank-line rules, metadata, and encoding.
3. Edit subtitle text only unless the user explicitly asks for timing, splitting, or merging changes.
4. Use neighboring cues as context to fix ASR errors, homophones.
5. Keep the original meaning. Do not invent unheard content; if a correction is uncertain, leave the safest text or flag it briefly in the final note.
6. Preserve names, terminology, speaker labels, music/sound markers, markup, and bilingual line order unless clearly wrong.
7. Validate the result by checking that SRT blocks remain parseable and cue counts/order are unchanged unless intentionally changed.

## SRT Text Script

Use the shared project script from the repository root:

```powershell
uv run python .codex/scripts/srt_text.py extract input.srt work.jsonl
uv run python .codex/scripts/srt_text.py apply input.srt corrected.jsonl output.srt --mode replace
```

Prefer JSONL for editable data because it can be applied back safely:

```jsonl
{"id":1,"text":"Corrected subtitle text."}
```

Use `--format text` only for context-only pure transcript output, or `--format tsv` when compact id/text data is enough. For long files, process contiguous id ranges in batches, then concatenate the corrected JSONL before applying.

## Editing Rules

- Prefer minimal, high-confidence corrections over stylistic rewriting.
- Keep subtitles concise and readable, but do not remove meaningful disfluencies unless they are obvious ASR noise or the user asked for polishing.
