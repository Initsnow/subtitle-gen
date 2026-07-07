from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, apply_overrides, load_config
from .formats import write_output_set, write_subtitles
from .pipeline import PipelineOptions, SubtitlePipeline
from .progress import RichProgressReporter


def main(argv: list[str] | None = None) -> int:
    _prefer_utf8_stdio()
    parser = build_parser()
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
            written = _write_outputs(args, result.subtitles)
            progress(f"wrote {len(written)} file(s)")
    except (ConfigError, Exception) as exc:
        print(f"subtitle-gen: {exc}", file=sys.stderr)
        return 1

    for path in written:
        print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle-gen",
        description="Generate subtitles from audio or video files.",
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
        "--no-bilingual",
        action="store_true",
        help="Do not write bilingual subtitle files when translating with --out-dir.",
    )
    parser.add_argument("--asr-model", help="Override ASR model id.")
    parser.add_argument("--low-vram", action="store_true", help="Use the configured low-VRAM ASR model.")
    parser.add_argument("--language", help="Source language hint for ASR and forced alignment.")
    parser.add_argument("--device-map", default=None, help="Transformers device_map override.")
    parser.add_argument("--llm-model", help="LLM model name for segmentation/translation.")
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        help="Concurrent LLM requests for segmentation/translation-capable providers.",
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


def _write_outputs(args: argparse.Namespace, subtitles: list) -> list[Path]:
    input_path = Path(args.input)
    formats = tuple(args.formats or ["srt"])

    if args.out:
        out_path = Path(args.out)
        mode = "bilingual" if args.translate and not args.no_bilingual else "translation"
        if not args.translate:
            mode = "original"
        return [write_subtitles(out_path, subtitles, mode)]

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    return write_output_set(
        out_dir,
        input_path.stem,
        subtitles,
        formats=formats,
        include_translation=bool(args.translate),
        include_bilingual=bool(args.translate and not args.no_bilingual),
    )


def _prefer_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

if __name__ == "__main__":
    raise SystemExit(main())
