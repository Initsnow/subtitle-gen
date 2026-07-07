from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
LOW_VRAM_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
DEFAULT_CONFIG_FILE = "config.toml"


@dataclass(frozen=True)
class ModelConfig:
    asr_model: str = DEFAULT_ASR_MODEL
    low_vram_asr_model: str = LOW_VRAM_ASR_MODEL
    aligner_model: str = DEFAULT_ALIGNER_MODEL
    device_map: str = "auto"
    dtype: str = "auto"
    language: str | None = None


@dataclass(frozen=True)
class VADConfig:
    enabled: bool = True
    backend: str = "silero"
    short_audio_threshold: float = 180.0
    min_speech_duration: float = 0.25
    min_silence_duration: float = 0.6
    speech_padding: float = 0.25
    max_chunk_duration: float = 120.0
    hard_max_chunk_duration: float = 240.0
    skip_silence_longer_than: float = 8.0


@dataclass(frozen=True)
class SegmentConfig:
    mode: str = "blingfire"
    min_duration: float = 0.833
    max_duration: float = 7.0
    max_lines: int = 2
    max_chars_en: int = 42
    max_chars_zh: int = 16
    max_chars_ja: int = 13
    max_cps_en: float = 20.0
    max_cps_zh: float = 9.0
    max_cps_ja: float = 4.0
    pause_threshold: float = 0.65


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    cleanup_enabled: bool = True
    max_media_entries: int = 12


@dataclass(frozen=True)
class PerformanceConfig:
    compile_aligner: bool = True
    compile_asr: bool = False


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    batch_size: int = 40
    concurrency: int = 2


@dataclass(frozen=True)
class OutputConfig:
    default_formats: tuple[str, ...] = ("srt",)
    bilingual_separator: str = "\n"


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    vad: VADConfig = VADConfig()
    segment: SegmentConfig = SegmentConfig()
    cache: CacheConfig = CacheConfig()
    performance: PerformanceConfig = PerformanceConfig()
    llm: LLMConfig = LLMConfig()
    output: OutputConfig = OutputConfig()
    cache_dir: str = ".subtitle-gen-cache"


T = TypeVar("T")


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is None:
        config_path = Path(DEFAULT_CONFIG_FILE)
        if not config_path.exists():
            return config
    else:
        config_path = Path(path)

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Config {config_path} must contain a TOML table.")
    return _merge_app_config(config, data)


def apply_overrides(config: AppConfig, **overrides: Any) -> AppConfig:
    model = config.model
    vad = config.vad
    segment = config.segment
    cache = config.cache
    performance = config.performance
    llm = config.llm

    if overrides.get("asr_model"):
        model = replace(model, asr_model=overrides["asr_model"])
    if overrides.get("low_vram"):
        model = replace(model, asr_model=model.low_vram_asr_model)
    if overrides.get("language"):
        model = replace(model, language=overrides["language"])
    if overrides.get("device_map"):
        model = replace(model, device_map=overrides["device_map"])
    if overrides.get("segment_mode") is not None:
        segment = _validate_segment_config(replace(segment, mode=str(overrides["segment_mode"])))
    if overrides.get("cache_enabled") is not None:
        cache = replace(cache, enabled=bool(overrides["cache_enabled"]))
    if overrides.get("compile_aligner") is not None:
        performance = replace(performance, compile_aligner=bool(overrides["compile_aligner"]))
    if overrides.get("compile_asr") is not None:
        performance = replace(performance, compile_asr=bool(overrides["compile_asr"]))
    if overrides.get("llm_model"):
        llm = replace(llm, model=overrides["llm_model"])
    if overrides.get("llm_concurrency") is not None:
        llm = replace(llm, concurrency=int(overrides["llm_concurrency"]))
    llm = _validate_llm_config(llm)
    if overrides.get("cache_dir"):
        cache_dir = str(overrides["cache_dir"])
    else:
        cache_dir = config.cache_dir

    return AppConfig(
        model=model,
        vad=vad,
        segment=segment,
        cache=cache,
        performance=performance,
        llm=llm,
        output=config.output,
        cache_dir=cache_dir,
    )


def env_llm_model(config: LLMConfig) -> str | None:
    return config.model or os.environ.get("SUBTITLE_GEN_LLM_MODEL")


class ConfigError(ValueError):
    pass


def _merge_app_config(config: AppConfig, data: dict[str, Any]) -> AppConfig:
    allowed_top = {field.name for field in fields(AppConfig)}
    unknown_top = sorted(set(data) - allowed_top)
    if unknown_top:
        raise ConfigError(f"Unknown config section(s): {', '.join(unknown_top)}")

    return AppConfig(
        model=_merge_dataclass(config.model, data.get("model", {}), "model"),
        vad=_merge_dataclass(config.vad, data.get("vad", {}), "vad"),
        segment=_merge_segment(config.segment, data.get("segment", {})),
        cache=_merge_dataclass(config.cache, data.get("cache", {}), "cache"),
        performance=_merge_dataclass(
            config.performance, data.get("performance", {}), "performance"
        ),
        llm=_merge_llm(config.llm, data.get("llm", {})),
        output=_merge_output(config.output, data.get("output", {})),
        cache_dir=str(data.get("cache_dir", config.cache_dir)),
    )


def _merge_dataclass(instance: T, data: Any, section: str) -> T:
    if data is None:
        return instance
    if not isinstance(data, dict):
        raise ConfigError(f"[{section}] must be a TOML table.")
    valid = {field.name for field in fields(instance)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{section}]: {', '.join(unknown)}")
    return replace(instance, **data)


def _merge_segment(instance: SegmentConfig, data: Any) -> SegmentConfig:
    return _validate_segment_config(_merge_dataclass(instance, data, "segment"))


def _merge_llm(instance: LLMConfig, data: Any) -> LLMConfig:
    return _validate_llm_config(_merge_dataclass(instance, data, "llm"))


def _validate_segment_config(config: SegmentConfig) -> SegmentConfig:
    valid_modes = {"none", "blingfire", "local", "hybrid", "llm"}
    if config.mode not in valid_modes:
        raise ConfigError(
            f"Unsupported [segment].mode: {config.mode}. "
            f"Expected one of: {', '.join(sorted(valid_modes))}."
        )
    return config


def _validate_llm_config(config: LLMConfig) -> LLMConfig:
    _require_positive_number("[llm].timeout", config.timeout)
    _require_non_negative_int("[llm].max_retries", config.max_retries)
    _require_positive_int("[llm].batch_size", config.batch_size)
    _require_positive_int("[llm].concurrency", config.concurrency)
    _require_number("[llm].temperature", config.temperature)
    if not 0.0 <= float(config.temperature) <= 2.0:
        raise ConfigError("[llm].temperature must be between 0.0 and 2.0.")
    return config


def _require_number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number.")


def _require_positive_number(name: str, value: Any) -> None:
    _require_number(name, value)
    if value <= 0:
        raise ConfigError(f"{name} must be greater than 0.")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer.")
    if value <= 0:
        raise ConfigError(f"{name} must be greater than 0.")


def _require_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer.")
    if value < 0:
        raise ConfigError(f"{name} must be greater than or equal to 0.")


def _merge_output(instance: OutputConfig, data: Any) -> OutputConfig:
    if data is None:
        return instance
    if not isinstance(data, dict):
        raise ConfigError("[output] must be a TOML table.")
    valid = {field.name for field in fields(instance)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [output]: {', '.join(unknown)}")

    if "default_formats" in data:
        data = dict(data)
        data["default_formats"] = tuple(data["default_formats"])
    return replace(instance, **data)
