from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .types import AudioWindow


class MediaError(RuntimeError):
    pass


def probe_duration(path: str | Path) -> float:
    input_path = Path(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = _run(command)
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise MediaError(f"Could not read duration for {input_path}") from exc


def extract_wav(
    input_path: str | Path,
    output_dir: str | Path,
    sample_rate: int = 16_000,
    overwrite: bool = False,
) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_stable_media_id(input_path)}.16k-mono.wav"
    if output_path.exists() and not overwrite:
        return output_path

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    _run(command)
    return output_path


def cut_wav_window(
    wav_path: str | Path,
    window: AudioWindow,
    output_dir: str | Path,
    sample_rate: int = 16_000,
    overwrite: bool = False,
) -> Path:
    wav_path = Path(wav_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"chunk-{window.index:05d}-{window.start:.3f}-{window.end:.3f}.wav"
    if output_path.exists() and not overwrite:
        return output_path

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{window.start:.3f}",
        "-t",
        f"{window.duration:.3f}",
        "-i",
        str(wav_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    _run(command)
    return output_path


def read_mono_wav_array(wav_path: str | Path, sample_rate: int = 16_000):
    try:
        import soundfile as sf
    except ImportError as exc:
        raise MediaError("soundfile is required to read extracted WAV audio. Run `uv sync`.") from exc

    data, actual_sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if actual_sample_rate != sample_rate:
        raise MediaError(
            f"Expected {sample_rate} Hz WAV, got {actual_sample_rate} Hz: {wav_path}"
        )
    if data.size == 0:
        return data[:, 0]
    return data.mean(axis=1)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise MediaError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip()
        raise MediaError(f"{command[0]} failed: {stderr}") from exc


def _stable_media_id(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]
