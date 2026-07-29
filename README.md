# subtitle-gen

Generate SRT, VTT, or JSON subtitles from audio/video with Qwen3 ASR, forced
alignment, local or LLM-assisted segmentation, proofreading, and optional translation.
Also supports proofreading and translating existing SRT files directly.

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
