#!/usr/bin/env python3
"""Agrège les fichiers ``ux_scenario_*.jsonl`` (issue-020) pour des deltas ``monotonic_ns``.

Lit un répertoire (ex. celui pointé par ``DATASET_STYLE_UX_TELEMETRY_DIR``), fusionne
les lignes JSON ``kind=ux_milestone``, groupe par ``run_id`` + ``surface``, trie par
``monotonic_ns`` et affiche les écarts entre jalons successifs (même série qu'en
``docs/migration_parity_matrix.md``).

Usage::

    python scripts/aggregate_ux_baseline_jsonl.py --input /tmp/dataset_style_ux_telemetry

Sortie : TSV sur stdout (colonnes : run_id, surface, from_code, to_code, delta_ns).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _iter_scenario_rows(input_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(input_dir.glob("ux_scenario_*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("kind") == "ux_milestone":
                    yield row


def aggregate_rows(rows: list[dict[str, Any]]) -> list[tuple[str, str, str, str, int]]:
    """Retourne des tuples ``(run_id, surface, from_milestone, to_milestone, delta_ns)``."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = str(row.get("run_id") or "")
        surface = str(row.get("surface") or "")
        by_key[(rid, surface)].append(row)
    out: list[tuple[str, str, str, str, int]] = []
    for (rid, surface), seq in sorted(by_key.items()):
        seq.sort(key=lambda r: int(r.get("monotonic_ns") or 0))
        for i in range(1, len(seq)):
            prev, cur = seq[i - 1], seq[i]
            d = int(cur["monotonic_ns"]) - int(prev["monotonic_ns"])
            out.append(
                (
                    rid,
                    surface,
                    str(prev.get("milestone_code") or ""),
                    str(cur.get("milestone_code") or ""),
                    d,
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agrège les jalons UX scenario JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Répertoire contenant ux_scenario_YYYYMMDD.jsonl",
    )
    args = parser.parse_args(argv)
    input_dir: Path = args.input
    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}", file=sys.stderr)
        return 2
    rows = list(_iter_scenario_rows(input_dir))
    if not rows:
        print("run_id\tsurface\tfrom_milestone\tto_milestone\tdelta_ns")
        return 0
    deltas = aggregate_rows(rows)
    print("run_id\tsurface\tfrom_milestone\tto_milestone\tdelta_ns")
    for rid, surface, a, b, d in deltas:
        print(f"{rid}\t{surface}\t{a}\t{b}\t{d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
