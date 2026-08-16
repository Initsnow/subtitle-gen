# subtitle-gen

Generate SRT, VTT, or JSON subtitles from audio/video with Qwen3 ASR, forced
alignment, local or LLM-assisted segmentation, proofreading, and optional translation.
Also supports proofreading and translating existing SRT files directly, and adding
timestamps to untimed lyrics/subtitle text by aligning it to audio.

## Requirements

- Python 3.11-3.13
- `uv`
- `ffmpeg` / `ffprobe` on `PATH`
- A GPU is recommended. Windows and Linux installs use PyTorch CUDA 12.8 wheels
  by default; adjust the `uv` source/index settings for CPU-only installs.

## Setup

```powershell
uv sync
Copy-Item config.example.toml config.toml
```

Edit `config.toml` for model, cache, language, and LLM provider settings. The
local file is ignored by git and is loaded automatically when present.

## Usage

### Pipeline (audio/video → subtitles)

```powershell
uv run subtitle-gen input.mp4 --out output.srt
uv run subtitle-gen input.mp4 --out-dir outputs --format srt --format vtt
uv run subtitle-gen input.mp4 --segment-mode local --out-dir outputs
uv run subtitle-gen input.mp4 --segment-mode hybrid --out-dir outputs
uv run subtitle-gen input.mp4 --translate zh --out-dir outputs
```

Pipeline options:

- `--segment-mode none|blingfire|local|hybrid|llm`
- `--language LANG` to hint the source language
- `--translate LANG` — also runs proofread before translation, outputs original/proofread/translation/bilingual
- `--no-bilingual` to skip bilingual output when translating
- `--overwrite-cache` to regenerate cached artifacts
- `--no-cache` to run without persistent cache

### Align untimed text (add timestamps to existing subtitles/lyrics)

```powershell
uv run subtitle-gen align "song.flac" "lyrics.lrc" --no-vad
uv run subtitle-gen align "song.flac" "lyrics.txt" --out "lyrics.timed.lrc"
uv run subtitle-gen align "song.flac" "lyrics.txt" --format lrc --format srt
uv run subtitle-gen align "song.flac" "lyrics.txt" --language Japanese
```

`align` runs ASR + forced alignment over the audio, then maps each input line
onto the resulting timed words with a global sequence alignment. The text file
may be plain text (one line per cue) or an LRC file; descriptive LRC metadata
tags (`[ti:...]`, `[ar:...]`, ...) are preserved in the output, while stale
timing tags (`[offset:...]`, `[length:...]`) are dropped because timestamps are
regenerated. By default it writes `<text>.timed.lrc`; use `--out` (single file,
format inferred from its extension) or `--format lrc|srt|vtt|json` (repeatable)
to control output. `--out` and `--format` cannot be combined.

VAD is used by default (good for speech). Pass `--no-vad` for songs/music, where
Silero VAD misses sung vocals — this aligns the whole audio with slightly
overlapping fixed-size windows so boundary words are not cut off.

It accepts the same model/cache flags as the pipeline (`--asr-model`,
`--low-vram`, `--language`, `--device-map`, `--cache`/`--no-cache`,
`--cache-dir`, `--overwrite-cache`, `--[no-]compile-aligner`,
`--[no-]compile-asr`), plus `--no-refine` to skip the per-line forced-alignment
refinement pass. With persistent cache enabled, both the rough pass and the
per-line refinement alignments are cached, so unchanged inputs are reused on
subsequent runs.

### Subcommands (work on existing SRT)

```powershell
uv run subtitle-gen proofread subs.srt
uv run subtitle-gen proofread subs.srt --out corrected.srt
uv run subtitle-gen translate subs.srt --target "Chinese"
uv run subtitle-gen translate subs.srt --target zh --no-bilingual
```

Both subcommands accept `--llm-model`, `--llm-concurrency`, `--batch-size` overrides.

LLM segmentation uses a separate `[llm.segmentation]` config; translation and
proofreading use `[llm]`. Set `api_key` (or `OPENAI_API_KEY`) and model for each.

## Development

```powershell
uv sync --dev
uv run pytest
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).
