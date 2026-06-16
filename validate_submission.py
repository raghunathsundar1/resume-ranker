#!/usr/bin/env python3
"""Validate submission.csv against judging.md auto-validator rules.

Usage:
    python validate_submission.py submission.csv [--pool candidates.jsonl]

Exits 0 on pass, 1 on any failure. Run before every submission.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from pathlib import Path

import orjson


def load_pool_ids(pool_path: Path) -> set[str]:
    ids: set[str] = set()
    opener = gzip.open if pool_path.suffix == ".gz" else open
    with opener(pool_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = orjson.loads(line)
                cid = rec.get("candidate_id")
                if cid:
                    ids.add(cid)
    return ids


def validate(csv_path: Path, pool_ids: set[str] | None = None) -> list[str]:
    """Return list of failure messages; empty = pass."""
    failures: list[str] = []

    if not csv_path.exists():
        return [f"File not found: {csv_path}"]
    if csv_path.suffix.lower() != ".csv":
        failures.append(f"File must be .csv, got {csv_path.suffix}")

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return ["File is empty or has no header"]
        required = {"candidate_id", "rank", "score"}
        missing = required - set(reader.fieldnames)
        if missing:
            failures.append(f"Missing required columns: {missing}")
            return failures

        rows = list(reader)

    # Exactly 100 data rows
    if len(rows) != 100:
        failures.append(f"Expected 100 rows, got {len(rows)}")

    ranks_seen: set[int] = set()
    ids_seen: set[str] = set()
    prev_score: float | None = None
    prev_rank: int = 0

    for i, row in enumerate(rows):
        line = i + 2  # 1-indexed + header

        # rank must be integer 1..100
        try:
            rank = int(row["rank"])
        except (ValueError, KeyError):
            failures.append(f"Line {line}: rank is not an integer: {row.get('rank')!r}")
            continue
        if rank < 1 or rank > 100:
            failures.append(f"Line {line}: rank {rank} out of range 1–100")
        if rank in ranks_seen:
            failures.append(f"Line {line}: duplicate rank {rank}")
        ranks_seen.add(rank)

        # score must be a float
        try:
            score = float(row["score"])
        except (ValueError, KeyError):
            failures.append(f"Line {line}: score is not a number: {row.get('score')!r}")
            score = None

        # scores non-increasing with rank
        if score is not None and prev_score is not None:
            if score > prev_score + 1e-9:
                failures.append(
                    f"Line {line}: score {score:.6f} increases vs previous "
                    f"{prev_score:.6f} (rank {rank})"
                )
        if score is not None:
            prev_score = score

        # All scores identical check (deferred; check after loop)
        cid = row.get("candidate_id", "")
        if not cid:
            failures.append(f"Line {line}: empty candidate_id")
        if cid in ids_seen:
            failures.append(f"Line {line}: duplicate candidate_id {cid!r}")
        ids_seen.add(cid)

        # candidate_id must be in pool (if pool provided)
        if pool_ids is not None and cid and cid not in pool_ids:
            failures.append(f"Line {line}: candidate_id {cid!r} not in pool")

    # Ranks 1..N each exactly once (N = min(100, rows))
    expected_ranks = set(range(1, len(rows) + 1))
    if ranks_seen != expected_ranks:
        missing_r = sorted(expected_ranks - ranks_seen)
        extra_r = sorted(ranks_seen - expected_ranks)
        if missing_r:
            failures.append(f"Missing ranks: {missing_r[:10]}")
        if extra_r:
            failures.append(f"Unexpected ranks: {extra_r[:10]}")

    # All scores identical?
    scores = []
    for row in rows:
        try:
            scores.append(float(row["score"]))
        except (ValueError, KeyError):
            pass
    if len(set(scores)) == 1 and len(scores) > 1:
        failures.append("All scores are identical — ranking is not meaningful")

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate submission.csv")
    ap.add_argument("csv", help="submission CSV to validate")
    ap.add_argument("--pool", default=None,
                    help="candidates.jsonl or .jsonl.gz to validate IDs against")
    args = ap.parse_args()

    pool_ids: set[str] | None = None
    if args.pool:
        pool_path = Path(args.pool)
        if not pool_path.exists():
            sys.exit(f"Pool not found: {pool_path}")
        print(f"Loading pool IDs from {pool_path} …")
        pool_ids = load_pool_ids(pool_path)
        print(f"  {len(pool_ids):,} IDs loaded")

    failures = validate(Path(args.csv), pool_ids)
    if failures:
        print(f"\nFAIL — {len(failures)} issue(s):")
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)
    else:
        print(f"\nPASS — {args.csv} is valid")


if __name__ == "__main__":
    main()
