#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path


TIMING_RE = re.compile(r"\d\d:\d\d:\d\d[,.]\d\d\d\s+-->\s+\d\d:\d\d:\d\d[,.]\d\d\d")


@dataclass
class Cue:
    id: int
    lines: list[str]
    text_start: int

    @property
    def text(self) -> str:
        return "\n".join(self.lines[self.text_start :])

    def with_text(self, text: str, mode: str) -> "Cue":
        new_lines = self.lines[: self.text_start]
        edited_lines = text.splitlines() or [""]
        if mode == "bilingual":
            new_lines.extend(self.lines[self.text_start :] + edited_lines)
        else:
            new_lines.extend(edited_lines)
        return Cue(id=self.id, lines=new_lines, text_start=self.text_start)


Block = Cue | list[str]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


def parse_srt(path: Path) -> list[Block]:
    content = read_text(path).strip("\n")
    if not content:
        return []

    blocks: list[Block] = []
    seen_ids: set[int] = set()
    for ordinal, raw_block in enumerate(re.split(r"\n{2,}", content), start=1):
        lines = raw_block.split("\n")
        timing_index = next((i for i, line in enumerate(lines) if TIMING_RE.search(line)), -1)
        if timing_index < 0:
            blocks.append(lines)
            continue

        cue_id = ordinal
        if timing_index > 0 and lines[0].strip().isdigit():
            cue_id = int(lines[0].strip())
        if cue_id in seen_ids:
            raise ValueError(f"duplicate cue id: {cue_id}")
        seen_ids.add(cue_id)
        blocks.append(Cue(id=cue_id, lines=lines, text_start=timing_index + 1))
    return blocks


def iter_cues(blocks: list[Block]) -> list[Cue]:
    return [block for block in blocks if isinstance(block, Cue)]


def write_srt(path: Path, blocks: list[Block]) -> None:
    serialized = ["\n".join(block.lines if isinstance(block, Cue) else block) for block in blocks]
    path.write_text("\n\n".join(serialized) + ("\n" if serialized else ""), encoding="utf-8")


def escape_tsv(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def unescape_tsv(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            nxt = text[index + 1]
            if nxt == "n":
                result.append("\n")
                index += 2
                continue
            if nxt == "t":
                result.append("\t")
                index += 2
                continue
            if nxt == "\\":
                result.append("\\")
                index += 2
                continue
        result.append(char)
        index += 1
    return "".join(result)


def extract(args: argparse.Namespace) -> None:
    cues = iter_cues(parse_srt(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "jsonl":
        lines = [
            json.dumps({"id": cue.id, "text": cue.text}, ensure_ascii=False, separators=(",", ":"))
            for cue in cues
        ]
    elif args.format == "tsv":
        lines = [f"{cue.id}\t{escape_tsv(cue.text)}" for cue in cues]
    else:
        lines = [cue.text.replace("\n", " ") for cue in cues]
    args.output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_edits(path: Path, input_format: str, field: str) -> dict[int, str]:
    content = read_text(path).strip()
    if not content:
        return {}

    edits: dict[int, str] = {}
    if input_format == "jsonl":
        try:
            parsed = json.loads(content)
        except JSONDecodeError:
            raw_items = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                raw_items = parsed["items"]
            elif isinstance(parsed, list):
                raw_items = parsed
            else:
                raw_items = [parsed]
        for item in raw_items:
            if not isinstance(item, dict):
                raise ValueError("JSON edits must be objects")
            raw_id = item.get("id")
            if isinstance(raw_id, str) and raw_id.isdigit():
                raw_id = int(raw_id)
            raw_text = item.get(field)
            if raw_text is None and field != "text":
                raw_text = item.get("text")
            if not isinstance(raw_id, int) or not isinstance(raw_text, str):
                raise ValueError(f"edit item must include integer id and string {field!r}: {item!r}")
            edits[raw_id] = raw_text
    else:
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            raw_id, sep, raw_text = line.partition("\t")
            if not sep or not raw_id.isdigit():
                raise ValueError(f"invalid TSV edit at line {line_number}")
            edits[int(raw_id)] = unescape_tsv(raw_text)
    return edits


def apply_edits(args: argparse.Namespace) -> None:
    blocks = parse_srt(args.input)
    cue_ids = {cue.id for cue in iter_cues(blocks)}
    edits = load_edits(args.edits, args.input_format, args.field)
    edit_ids = set(edits)

    missing = cue_ids - edit_ids
    extra = edit_ids - cue_ids
    if not args.allow_partial and (missing or extra):
        details = []
        if missing:
            details.append(f"missing ids: {sorted(missing)[:10]}")
        if extra:
            details.append(f"extra ids: {sorted(extra)[:10]}")
        raise ValueError("; ".join(details))

    updated: list[Block] = []
    for block in blocks:
        if isinstance(block, Cue) and block.id in edits:
            updated.append(block.with_text(edits[block.id], args.mode))
        else:
            updated.append(block)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_srt(args.output, updated)


def check(args: argparse.Namespace) -> None:
    print(f"{len(iter_cues(parse_srt(args.input)))} cue(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract/apply SRT cue text without timestamps.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("input", type=Path)
    extract_parser.add_argument("output", type=Path)
    extract_parser.add_argument("--format", choices=["jsonl", "tsv", "text"], default="jsonl")
    extract_parser.set_defaults(func=extract)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("input", type=Path)
    apply_parser.add_argument("edits", type=Path)
    apply_parser.add_argument("output", type=Path)
    apply_parser.add_argument("--input-format", choices=["jsonl", "tsv"], default="jsonl")
    apply_parser.add_argument("--field", default="text")
    apply_parser.add_argument("--mode", choices=["replace", "translation", "bilingual"], default="replace")
    apply_parser.add_argument("--allow-partial", action="store_true")
    apply_parser.set_defaults(func=apply_edits)

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("input", type=Path)
    check_parser.set_defaults(func=check)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"srt_text.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
