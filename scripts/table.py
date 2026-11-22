#!/usr/bin/env python3
"""Render results/bench.csv (stdin or a path) as a markdown table.

Kept as a script rather than inline YAML so it can be run by hand:
    ./build/bench | python3 scripts/table.py
"""
import csv
import sys

COLUMNS = [
    ("kernel", lambda r: r["kernel"]),
    ("variant", lambda r: r["variant"]),
    ("n", lambda r: f'{int(r["n"]):,}'),
    ("best", lambda r: f'{float(r["best_s"]) * 1e3:.2f} ms'),
    ("median", lambda r: f'{float(r["median_s"]) * 1e3:.2f} ms'),
    ("metric", lambda r: f'{r["metric"]} {r["unit"]}'),
    ("spread", lambda r: f'{r["spread_pct"]}%'),
]


def main() -> int:
    stream = open(sys.argv[1], newline="") if len(sys.argv) > 1 else sys.stdin
    rows = list(csv.DictReader(stream))
    if not rows:
        print("no rows: the benchmark produced an empty CSV", file=sys.stderr)
        return 1

    print("| " + " | ".join(name for name, _ in COLUMNS) + " |")
    print("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for r in rows:
        print("| " + " | ".join(get(r) for _, get in COLUMNS) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
