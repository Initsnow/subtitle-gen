from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, LLMConfig, apply_overrides, load_config
from .formats import FormatError, parse_srt
from .llm import OpenAICompatibleLLM
from .pipeline import PipelineOptions, SubtitlePipeline
from .progress import RichProgressReporter
from .proofreader import SubtitleProofreader
from .translator import SubtitleTranslator
from .types import SubtitleItem


def main(argv: list[str] | None = None) -> int:
    _prefer_utf8_stdio()
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "proofread":
        return _cmd_proofread(argv[1:])
    if argv and argv[0] == "translate":
        return _cmd_translate(argv[1:])
    return _cmd_pipeline(argv)


# ---------------------------------------------------------------------------
# pipeline (default)
# ---------------------------------------------------------------------------

def _cmd_pipeline(argv: list[str]) -> int:
    parser = _build_pipeline_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        config = apply_overrides(
            config,
            asr_model=args.asr_model,
            low_vram=args.low_vram,
            language=args.language,
            device_map=args.device_map,
            segment_mode=args.segment_mode,
            compile_aligner=args.compile_aligner,
            compile_asr=args.compile_asr,
            llm_model=args.llm_model,
            llm_concurrency=args.llm_concurrency,
            cache_enabled=args.cache_enabled,
            cache_dir=args.cache_dir,
        )
        with RichProgressReporter() as progress:
            result = SubtitlePipeline(config).run(
                PipelineOptions(
                    input_path=Path(args.input),
                    translate=args.translate,
                    segment_mode=args.segment_mode,
                    overwrite_cache=args.overwrite_cache,
                    progress=progress,
                )
            )
            progress("writing subtitle output")
            written = _write_outputs(args, result.subtitles, config.output.strip_punctuation)
            progress(f"wrote {len(written)} file(s)")
    except (ConfigError, Exception) as exc:
        print(f"subtitle-gen: {exc}", file=sys.stderr)
        return 1

    for path in written:
        print(path)
    return 0


def _build_pipeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle-gen",
        description="Generate subtitles from audio or video files.",
        epilog="subcommands:\n  proofread     proofread an existing SRT file\n  translate     translate an existing SRT file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Input audio/video path.")
    parser.add_argument("--config", help="TOML config path. Defaults to ./config.toml if present.")
    parser.add_argument("--out", help="Single output file path. Format is inferred from extension.")
    parser.add_argument("--out-dir", help="Directory for output subtitle set.")
    parser.add_argument(
        "--format",
        action="append",
        choices=["srt", "vtt", "json"],
        dest="formats",
        help="Output format for --out-dir. Can be passed multiple times.",
    )
    parser.add_argument(
        "--segment-mode",
        choices=["none", "blingfire", "local", "hybrid", "llm"],
        help="Subtitle segmentation mode. Defaults to [segment].mode.",
    )
    parser.add_argument("--translate", metavar="LANG", help="Translate subtitles to target language.")
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Also write a bilingual subtitle file alongside the translation.",
    )
    parser.add_argument("--asr-model", help="Override ASR model id.")
    parser.add_argument("--low-vram", action="store_true", help="Use the configured low-VRAM ASR model.")
    parser.add_argument("--language", help="Source language hint for ASR and forced alignment.")
    parser.add_argument("--device-map", default=None, help="Transformers device_map override.")
    parser.add_argument("--llm-model", help="LLM model name for segmentation/translation.")
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        help="Concurrent LLM requests.",
    )
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--cache",
        action="store_true",
        default=None,
        dest="cache_enabled",
        help="Enable persistent audio, ASR, and alignment cache.",
    )
    cache.add_argument(
        "--no-cache",
        action="store_false",
        dest="cache_enabled",
        help="Disable persistent audio, ASR, and alignment cache for this run.",
    )
    parser.add_argument("--cache-dir", help="Cache directory.")
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Regenerate cached audio chunks, ASR transcripts, and alignments.",
    )

    compile_aligner = parser.add_mutually_exclusive_group()
    compile_aligner.add_argument(
        "--compile-aligner",
        action="store_true",
        default=None,
        help="Compile the forced aligner forward pass.",
    )
    compile_aligner.add_argument(
        "--no-compile-aligner",
        action="store_false",
        dest="compile_aligner",
        help="Disable forced aligner torch.compile.",
    )

    compile_asr = parser.add_mutually_exclusive_group()
    compile_asr.add_argument(
        "--compile-asr",
        action="store_true",
        default=None,
        help="Compile ASR forward pass. Not recommended by default for generate().",
    )
    compile_asr.add_argument(
        "--no-compile-asr",
        action="store_false",
        dest="compile_asr",
        help="Disable ASR torch.compile.",
    )
    return parser


# ---------------------------------------------------------------------------
# proofread
# ---------------------------------------------------------------------------

def _cmd_proofread(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="subtitle-gen proofread",
        description="Proofread an existing SRT subtitle file using LLM.",
    )
    parser.add_argument("input", type=Path, help="Input SRT file.")
    parser.add_argument("--out", type=Path, help="Output SRT path (default: input.proofread.srt).")
    parser.add_argument("--config", help="TOML config path.")
    parser.add_argument("--llm-model", help="Override LLM model.")
    parser.add_argument("--llm-concurrency", type=int, help="Concurrent LLM requests.")
    parser.add_argument("--batch-size", type=int, help="Subtitles per LLM request.")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        llm_config = _llm_config_with_overrides(config.llm, args.llm_model, args.llm_concurrency, args.batch_size)
        items = parse_srt(args.input)
        if not items:
            print("subtitle-gen proofread: no cues found in input file.", file=sys.stderr)
            return 1

        out_path = args.out or _default_out(args.input, "proofread")
        with RichProgressReporter() as progress:
            progress(f"proofreading {len(items)} cue(s)")
            llm = OpenAICompatibleLLM(llm_config)
            corrected = SubtitleProofreader(llm, llm_config, progress=progress).proofread(items)
            progress("writing output")
            _write_srt_output(out_path, corrected, "proofread", strip_punctuation=config.output.strip_punctuation)
            progress("done")
        print(out_path)
        return 0
    except (ConfigError, FormatError, Exception) as exc:
        print(f"subtitle-gen proofread: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# translate (standalone SRT)
# ---------------------------------------------------------------------------

def _cmd_translate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="subtitle-gen translate",
        description="Translate an existing SRT subtitle file using LLM.",
    )
    parser.add_argument("input", type=Path, help="Input SRT file.")
    parser.add_argument("--target", metavar="LANG", required=True, help="Target language.")
    parser.add_argument("--out", type=Path, help="Output SRT path (default: input.<target>.srt).")
    parser.add_argument("--config", help="TOML config path.")
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Write bilingual SRT (original + translation).",
    )
    parser.add_argument("--llm-model", help="Override LLM model.")
    parser.add_argument("--llm-concurrency", type=int, help="Concurrent LLM requests.")
    parser.add_argument("--batch-size", type=int, help="Subtitles per LLM request.")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        llm_config = _llm_config_with_overrides(config.llm, args.llm_model, args.llm_concurrency, args.batch_size)
        items = parse_srt(args.input)
        if not items:
            print("subtitle-gen translate: no cues found in input file.", file=sys.stderr)
            return 1

        suffix = args.target.lower().replace(" ", "-")
        out_path = args.out or _default_out(args.input, suffix)

        with RichProgressReporter() as progress:
            progress(f"translating {len(items)} cue(s) to {args.target}")
            llm = OpenAICompatibleLLM(llm_config)
            translated = SubtitleTranslator(llm, llm_config, progress=progress).translate(
                items, target_language=args.target
            )
            progress("writing output")
            _write_srt_output(out_path, translated, "translation", strip_punctuation=config.output.strip_punctuation)
            if args.bilingual:
                bilingual_path = _bilingual_out(out_path)
                _write_srt_output(bilingual_path, translated, "bilingual", strip_punctuation=config.output.strip_punctuation)
                progress("done")
                print(out_path)
                print(bilingual_path)
            else:
                progress("done")
                print(out_path)
        return 0
    except (ConfigError, FormatError, Exception) as exc:
        print(f"subtitle-gen translate: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_srt_output(
    path: Path, items: list[SubtitleItem], mode: str, *, strip_punctuation: bool = False
) -> Path:
    from .formats import write_subtitles

    return write_subtitles(path, items, mode, strip_punctuation=strip_punctuation)


def _default_out(input_path: Path, suffix: str) -> Path:
    return input_path.parent / f"{input_path.stem}.{suffix}.srt"


def _bilingual_out(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}.bilingual{out_path.suffix}"


def _llm_config_with_overrides(
    base: LLMConfig,
    model: str | None,
    concurrency: int | None,
    batch_size: int | None,
) -> LLMConfig:
    from dataclasses import replace

    if model is not None:
        base = replace(base, model=model)
    if concurrency is not None:
        base = replace(base, concurrency=concurrency)
    if batch_size is not None:
        base = replace(base, batch_size=batch_size)
    return base


def _write_outputs(
    args: argparse.Namespace, subtitles: list, strip_punctuation: bool = False
) -> list[Path]:
    input_path = Path(args.input)
    formats = tuple(args.formats or ["srt"])

    if args.out:
        out_path = Path(args.out)
        written = [_write_srt_output(out_path, subtitles, "translation" if args.translate else "original", strip_punctuation=strip_punctuation)]
        if args.translate and args.bilingual:
            written.append(_write_srt_output(_bilingual_out(out_path), subtitles, "bilingual", strip_punctuation=strip_punctuation))
        return written

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    from .formats import write_output_set

    return write_output_set(
        out_dir,
        input_path.stem,
        subtitles,
        formats=formats,
        include_translation=bool(args.translate),
        include_bilingual=bool(args.translate and args.bilingual),
        include_proofread=bool(args.translate),
        strip_punctuation=strip_punctuation,
    )


def _prefer_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
