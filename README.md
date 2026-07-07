# subtitle-gen

Generate SRT, VTT, or JSON subtitles from audio/video with Qwen3 ASR, forced
alignment, local or LLM-assisted segmentation, and optional translation.

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

```powershell
uv run subtitle-gen input.mp4 --out output.srt
uv run subtitle-gen input.mp4 --out-dir outputs --format srt --format vtt
uv run subtitle-gen input.mp4 --segment-mode local --out-dir outputs
uv run subtitle-gen input.mp4 --segment-mode hybrid --llm-model deepseek-v4-flash --out-dir outputs
uv run subtitle-gen input.mp4 --translate zh --out-dir outputs
```

Useful options:

- `--segment-mode none|blingfire|local|hybrid|llm`
- `--language LANG` to hint the source language
- `--translate LANG` to write translated and bilingual subtitles
- `--no-bilingual` to skip bilingual output when translating
- `--overwrite-cache` to regenerate cached audio, ASR, and alignment artifacts
- `--no-cache` to run without persistent cache

For LLM segmentation or translation, set `[llm].model` and `[llm].api_key` in
`config.toml`, or use `SUBTITLE_GEN_LLM_MODEL` and `OPENAI_API_KEY`.

## Development

```powershell
uv sync --dev
uv run pytest
```

## License

GPL-3.0-only. See [LICENSE](LICENSE).
