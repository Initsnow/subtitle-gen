# subtitle-gen

Generate SRT, VTT, LRC, or JSON subtitles from audio/video with Qwen3 ASR, forced
alignment, local or LLM-assisted segmentation, proofreading, and optional translation.
Also supports proofreading and translating existing SRT files directly, and adding
timestamps to untimed lyrics/subtitle text by aligning it to audio.

Usable both as a command line tool and as a Python library.

## Requirements

- Python 3.11+
- `ffmpeg` / `ffprobe` on `PATH`
- A GPU is recommended, but CPU-only runs work (slower)

## Install

```console
pip install subtitle-gen
```

This pulls the default `torch` build from PyPI: CUDA-enabled on Linux (several
GB of NVIDIA libraries) and CPU-only on macOS and Windows. To pick a different
build — a smaller CPU-only install, or CUDA on Windows — install torch first and
let the package reuse it:

```console
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install subtitle-gen
```

Or run it without touching your environment:

```console
uv tool install subtitle-gen
```

`uv` projects resolve torch from the CUDA 12.8 index on Linux and Windows, as
[pyproject.toml](pyproject.toml) configures under `[tool.uv.sources]`.

## Configuration

Everything works with built-in defaults; a config file only overrides them.
`subtitle-gen` loads `./config.toml` when present, or the path given via
`--config`. [config.example.toml](config.example.toml) lists every option; a
minimal one looks like this:

```toml
[model]
asr_model = "Qwen/Qwen3-ASR-0.6B-hf"
aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
device_map = "auto"
language = "Chinese"          # optional hint; omit to auto-detect

[segment]
mode = "hybrid"               # none | blingfire | local | hybrid | llm

[llm]                         # proofreading + translation
base_url = "https://api.deepseek.com/v1"
model = "deepseek-v4-pro"
api_key = ""                  # or set OPENAI_API_KEY

[llm.segmentation]            # used by mode = "llm" / "hybrid"
model = "deepseek-v4-flash"
```

LLM features need an OpenAI-compatible endpoint; `SUBTITLE_GEN_LLM_MODEL`
overrides the configured model.

## CLI usage

### Pipeline (audio/video → subtitles)

```console
subtitle-gen input.mp4 --out output.srt
subtitle-gen input.mp4 --out-dir outputs --format srt --format vtt
subtitle-gen input.mp4 --out-dir outputs --format txt
subtitle-gen input.mp4 --segment-mode local --out-dir outputs
subtitle-gen input.mp4 --segment-mode hybrid --out-dir outputs
subtitle-gen input.mp4 --translate zh --out-dir outputs
```

### Batch (multiple files, one model load)

Pass several inputs to process them in one invocation — the ASR, forced
aligner, and VAD weights load once and are shared across every file, which is
much faster than looping `subtitle-gen` in a shell (each shell invocation
reloads the weights):

```console
subtitle-gen ep01.mp4 ep02.mp4 ep03.mp4 --out-dir outputs
subtitle-gen *.mkv --out-dir outputs --translate zh
```

Each file gets its own output set named after its stem. `--out` only works
with a single input. A file that fails is reported to stderr and does not
abort the batch; the exit code is non-zero if any input failed.

Pipeline options:

- `--segment-mode none|blingfire|local|hybrid|llm`
- `--language LANG` to hint the source language
- `--translate LANG` — also runs proofread before translation, outputs original/proofread/translation
- `--bilingual` to also write a bilingual output when translating
- `--overwrite-cache` to regenerate cached artifacts
- `--no-cache` to run without persistent cache

### Align untimed text (add timestamps to existing subtitles/lyrics)

```console
subtitle-gen align song.flac lyrics.lrc --no-vad
subtitle-gen align song.flac lyrics.txt --out lyrics.timed.lrc
subtitle-gen align song.flac lyrics.txt --format lrc --format srt
subtitle-gen align song.flac lyrics.txt --language Japanese
```

`align` runs ASR + forced alignment over the audio, then maps each input line
onto the resulting timed words with a global sequence alignment. The text file
may be plain text (one line per cue) or an LRC file; descriptive LRC metadata
tags (`[ti:...]`, `[ar:...]`, ...) are preserved in the output, while stale
timing tags (`[offset:...]`, `[length:...]`) are dropped because timestamps are
regenerated. By default it writes `<text>.timed.lrc`; use `--out` (single file,
format inferred from its extension) or `--format lrc|srt|vtt|json|txt` (repeatable)
to control output. `--out` and `--format` cannot be combined.

VAD is used by default (good for speech). Pass `--no-vad` for songs/music, where
Silero VAD misses sung vocals — this aligns the whole audio with slightly
overlapping fixed-size windows so boundary words are not cut off.

It accepts the same model/cache flags as the pipeline (`--asr-model`,
`--language`, `--device-map`, `--cache`/`--no-cache`,
`--cache-dir`, `--overwrite-cache`, `--[no-]compile-aligner`,
`--[no-]compile-asr`), plus `--no-refine` to skip the per-line forced-alignment
refinement pass. With persistent cache enabled, both the rough pass and the
per-line refinement alignments are cached, so unchanged inputs are reused on
subsequent runs.

### Subcommands (work on existing SRT)

```console
subtitle-gen proofread subs.srt
subtitle-gen proofread subs.srt --out corrected.srt
subtitle-gen translate subs.srt --target "Chinese"
subtitle-gen translate subs.srt --target zh --bilingual
```

Both subcommands accept `--llm-model`, `--llm-concurrency`, `--batch-size` overrides.

LLM segmentation uses a separate `[llm.segmentation]` config; translation and
proofreading use `[llm]`. Set `api_key` (or `OPENAI_API_KEY`) and model for each.

## Library usage

```python
from pathlib import Path

from subtitle_gen import PipelineOptions, SubtitlePipeline, load_config

config = load_config()  # ./config.toml when present, otherwise defaults
result = SubtitlePipeline(config).run(
    PipelineOptions(input_path=Path("input.mp4"), translate="zh")
)
for item in result.subtitles:
    print(item.start, item.end, item.text, item.translation)
```

Other entry points exported from `subtitle_gen`:

- `align_lines_to_audio(config, audio_path, lines, ...)` — timestamp untimed text
- `SubtitleProofreader(llm, llm_config).proofread(items)`
- `SubtitleTranslator(llm, llm_config).translate(items, target_language)`
- `parse_srt` / `parse_lrc` / `write_subtitles` / `write_lrc` — standalone file I/O
- `OpenAICompatibleLLM` — the OpenAI-compatible client used by the above

Model weights are cached process-wide and keyed by model id, device map,
dtype, and compile flags, so looping over inputs reuses one load. Call
`subtitle_gen.models.clear_model_cache()` to release them.

```python
from pathlib import Path

from subtitle_gen import PipelineOptions, SubtitlePipeline, load_config

config = load_config()
pipeline = SubtitlePipeline(config)
for media in Path("inputs").iterdir():
    result = pipeline.run(PipelineOptions(input_path=media, translate="zh"))
```

## Development

```console
uv sync
uv run pytest
uv run ruff check .
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).
