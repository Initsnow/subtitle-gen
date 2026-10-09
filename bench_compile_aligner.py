"""Benchmark QwenForcedAligner with torch.compile on/off.

Usage:
    python bench_compile_aligner.py --wav <path> --compile 1|0

Loads the project's own aligner class, aligns audio windows of several
different lengths (mirroring real workloads where chunk length varies and
triggers recompiles), and reports per-call wall time plus CUDA memory stats.
"""
from __future__ import annotations

import argparse
import time
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "src"))

import torch

from subtitle_gen.aligner import QwenForcedAligner
from subtitle_gen.media import read_mono_wav_array

LENGTHS = [15, 30, 60, 90, 120, 30, 60]  # seconds; repeats hit cached shapes
SENTENCE = "This is a benchmark sentence used to exercise the forced aligner. "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--compile", type=int, default=0)
    args = parser.parse_args()

    samples = read_mono_wav_array(args.wav)
    sr = 16_000
    total_seconds = samples.shape[0] / sr
    print(f"wav duration: {total_seconds:.0f}s | compile={bool(args.compile)}", flush=True)

    t0 = time.perf_counter()
    aligner = QwenForcedAligner(
        device_map="auto",
        dtype="auto",
        compile_model=bool(args.compile),
    )
    # Force load now so load time is separate from first-call compile time.
    aligner._load()
    load_s = time.perf_counter() - t0
    print(f"model load: {load_s:.1f}s", flush=True)

    torch.cuda.synchronize()
    free0, total0 = torch.cuda.mem_get_info()
    used0 = total0 - free0

    for length in LENGTHS:
        n = min(length, int(total_seconds))
        if n <= 0:
            continue
        audio = samples[: n * sr]
        words = max(1, n // 3)  # ~3 words/sec, proportional to chunk length
        transcript = (SENTENCE * ((words // len(SENTENCE.split())) + 1)).strip()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()

        start = time.perf_counter()
        try:
            tokens = aligner.align(audio, transcript, language="English")
            ok = len(tokens)
        except Exception as exc:  # noqa: BLE001 - report and continue
            ok = f"ERROR: {type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start

        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.memory_reserved() / 2**20
        ctx_used = (total - free_after) / 2**20
        print(
            f"chunk {n:>4}s | time {elapsed:7.2f}s | peak-alloc {peak_alloc:7.0f} MiB | "
            f"allocator-reserved {reserved:6.0f} MiB | cuda-total-used {ctx_used:6.0f} MiB "
            f"(delta {(free_before - free_after) / 2**20:+6.0f}) | tokens {ok}",
            flush=True,
        )

    free_end, _ = torch.cuda.mem_get_info()
    print(
        f"summary | load {load_s:.1f}s | steady cuda-total-used {(total0 - free_end) / 2**20:.0f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
