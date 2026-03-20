#!/usr/bin/env python3
"""Split a JSON/JSONL dataset into named subsets with no overlap."""

from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a dataset into named JSON files. "
            "Input may be a JSON array or a JSONL file."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file.")
    parser.add_argument(
        "--splits",
        required=True,
        help='Python dict of split sizes, e.g. \'{"train": 1000, "val": 100, "test": 100}\'',
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where split files will be written.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Shuffle data before splitting. Default is input order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --random is set.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent used when writing output JSON files.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return data

    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSONL record at line {line_number} in {path}: {exc}"
            ) from exc
    return records


def parse_splits(raw_splits: str) -> dict[str, int]:
    try:
        split_map = ast.literal_eval(raw_splits)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Failed to parse --splits: {raw_splits}") from exc

    if not isinstance(split_map, dict) or not split_map:
        raise ValueError("--splits must be a non-empty dict of {name: count}")

    normalized: dict[str, int] = {}
    for name, count in split_map.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Split names must be non-empty strings")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Split count for {name!r} must be a non-negative int")
        normalized[name] = count
    return normalized


def write_split(path: Path, records: list[Any], indent: int) -> None:
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    records = load_records(input_path)
    splits = parse_splits(args.splits)

    total_requested = sum(splits.values())
    if total_requested > len(records):
        raise ValueError(
            f"Requested {total_requested} records, but input only has {len(records)}"
        )

    pool = list(records)
    if args.random:
        rng = random.Random(args.seed)
        rng.shuffle(pool)

    output_dir.mkdir(parents=True, exist_ok=True)

    start = 0
    for split_name, split_size in splits.items():
        end = start + split_size
        write_split(output_dir / f"{split_name}.json", pool[start:end], args.indent)
        start = end

    print(
        f"Wrote {len(splits)} splits to {output_dir} "
        f"from {len(records)} input records."
    )


if __name__ == "__main__":
    main()
