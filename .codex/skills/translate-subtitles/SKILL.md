---
name: translate-subtitles
description: Translate subtitle files while preserving timing and subtitle structure. Use when asked to 翻译字幕, produce translated or bilingual SRT subtitles, translate generated subtitle outputs, or adapt subtitle text to a target language without merging, splitting, reordering, or retiming cues; prefer the project SRT text extraction/apply script to avoid spending tokens on timestamps.
---

# Translate Subtitles

## Workflow

1. Determine source and target languages from the request or file context; ask only if the target language is truly ambiguous.
2. For SRT files, run `.codex/scripts/srt_text.py` to extract cue text before translation; do not load timestamps into the prompt unless timing is relevant.
3. Translate cue by cue with surrounding context. Do not merge, split, remove, reorder, or retime cues unless explicitly requested.

## SRT Text Script

Use the shared project script from the repository root:

```powershell
uv run python .codex/scripts/srt_text.py extract input.srt source.jsonl
uv run python .codex/scripts/srt_text.py apply input.srt translated.jsonl output.translation.srt --mode translation
uv run python .codex/scripts/srt_text.py apply input.srt translated.jsonl output.bilingual.srt --mode bilingual
```

Translate only the `text` value and preserve every `id`:

```jsonl
{"id":1,"text":"Translated subtitle text."}
```

Use `--format text` for pure transcript context, or `--format tsv` for compact id/text data. For long files, translate contiguous id ranges in batches, then concatenate the translated JSONL before applying.

## Style Rules

- Translate meaning, not word order; avoid over-literal phrasing.
- Keep repeated terms and proper nouns consistent across the file.
- Preserve intentional slang, register, jokes, and emotional intensity when possible.
- If a source line is likely an ASR error, make the best context-aware translation and mention any material uncertainty in the final note.
