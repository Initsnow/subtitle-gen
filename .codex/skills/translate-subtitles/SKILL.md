---
name: translate-subtitles
description: Text-only subtitle translation. Use when asked to 翻译字幕, produce translated or bilingual SRT subtitles, translate generated subtitle outputs, or adapt subtitle text to a target language. For SRT, work through the project extract/apply script instead of reading raw timestamped content.
---

# Translate Subtitles

## Workflow

1. Determine source and target languages from the request or file context; ask only if the target language is truly ambiguous.
2. For SRT files, extract JSONL first; do not read raw SRT content for translation.
3. Translate from extracted text and neighboring cues only. Do not browse, inspect transcripts, or open media unless the user explicitly asks for source verification.
4. Translate each JSONL `text` value and keep its `id`.

## SRT Text Script

Use the shared project script from the repository root:

```powershell
uv run python .codex/scripts/srt_text.py extract input.srt source.jsonl
uv run python .codex/scripts/srt_text.py apply input.srt translated.jsonl output.translation.srt --mode translation
uv run python .codex/scripts/srt_text.py apply input.srt translated.jsonl output.bilingual.srt --mode bilingual
```

Edit JSONL records like this:

```jsonl
{"id":1,"text":"Translated subtitle text."}
```

Apply requires all cue ids by default. Do not use `--allow-partial` for translation outputs: partial apply can silently mix source-language cues into the translated SRT. For long files, translate contiguous ranges in separate JSONL files, then concatenate one complete translated JSONL before applying.

Use `--format text` for transcript context, or `--format tsv` for compact id/text data. Keep scratch files under `.codex/work/`.

Reading raw SRT is only for parser or encoding diagnostics. After diagnosis, return to the extract/translate/apply flow.

## Style Rules

- Translate meaning, not word order; avoid over-literal phrasing.
- Keep repeated terms and proper nouns consistent across the file.
- Preserve intentional slang, register, jokes, and emotional intensity when possible.
- If a source line is likely an ASR error, make the best local context-aware translation and mention any material uncertainty in the final note.
