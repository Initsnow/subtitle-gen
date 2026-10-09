"""Generate subtitles from audio or video, and work with existing subtitle files."""

__version__ = "0.2.0"

from .config import AppConfig, ConfigError, load_config
from .formats import FormatError, parse_lrc, parse_srt, write_lrc, write_subtitles
from .llm import LLMError, OpenAICompatibleLLM
from .pipeline import PipelineOptions, PipelineResult, SubtitlePipeline
from .proofreader import SubtitleProofreader
from .text_align import align_lines_to_audio, load_text_input
from .translator import SubtitleTranslator
from .types import AudioWindow, SubtitleItem, TimedToken, TranscriptChunk

__all__ = [
    "__version__",
    "AppConfig",
    "AudioWindow",
    "ConfigError",
    "FormatError",
    "LLMError",
    "OpenAICompatibleLLM",
    "PipelineOptions",
    "PipelineResult",
    "SubtitleItem",
    "SubtitlePipeline",
    "SubtitleProofreader",
    "SubtitleTranslator",
    "TimedToken",
    "TranscriptChunk",
    "align_lines_to_audio",
    "load_config",
    "load_text_input",
    "parse_lrc",
    "parse_srt",
    "write_lrc",
    "write_subtitles",
]
