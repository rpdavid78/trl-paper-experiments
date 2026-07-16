#!/usr/bin/env python
"""Aggregate JSONL experiment records into mean/std tables.

Usage:
  python scripts/aggregate_results.py --input 'results/*.jsonl' --out tables/main.csv

For nested stochastic runs, average within each independent experimental unit
before computing the reported across-unit standard deviation:

  python scripts/aggregate_results.py --input results/boost_1d.jsonl \
    --group experiment split boost beta_perp --independent-unit map_seed \
    --metrics acc nll ece brier --out tables/boost_1d.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

METRICS_DEFAULT = ["acc", "nll", "ece", "brier", "auroc", "runtime_total_sec", "peak_vram_gb"]
GROUP_DEFAULT = ["dataset", "architecture", "method"]


def require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit("pandas is required; install requirements.txt first") from exc
    return pd


def load_jsonl(paths: List[str]) -> pd.DataFrame:
    pd = require_pandas()
    rows = []
    for pattern in paths:
        matched = sorted(glob.glob(pattern)) or [pattern]
        for path in matched:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        raise SystemExit("No rows found. Check --input paths.")
    return pd.DataFrame(rows)


def mean_std_table(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    pd = require_pandas()
    present_metrics = [m for m in metrics if m in df.columns]
    if not present_metrics:
        raise SystemExit(f"None of the requested metrics are present: {metrics}")
    grouped = df.groupby(group_cols, dropna=False)[present_metrics]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = grouped.count().iloc[:, [0]].rename(columns={present_metrics[0]: "n"})
    out = pd.concat([count, mean, std], axis=1).reset_index()
    return out


def collapse_within_units(
    df: pd.DataFrame,
    group_cols: List[str],
    unit_cols: List[str],
    metrics: List[str],
) -> pd.DataFrame:
    """Average stochastic repeats inside each independent unit.

    For example, Table 17 first averages posterior-sampling seeds within a MAP
    checkpoint and then reports mean/std across MAP checkpoint seeds.
    """
    present_metrics = [m for m in metrics if m in df.columns]
    present_units = [c for c in unit_cols if c in df.columns]
    if not present_units:
        raise SystemExit(f"No independent-unit columns found in data: {unit_cols}")
    return (
        df.groupby(group_cols + present_units, dropna=False)[present_metrics]
        .mean()
        .reset_index()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", "--output", dest="out", required=True)
    p.add_argument("--group", nargs="+", default=GROUP_DEFAULT)
    p.add_argument("--metrics", nargs="+", default=METRICS_DEFAULT)
    p.add_argument(
        "--independent-unit",
        nargs="+",
        default=None,
        help="Columns defining independent units; stochastic repeats are averaged within each unit first.",
    )
    args = p.parse_args()

    df = load_jsonl(args.input)
    group_cols = [c for c in args.group if c in df.columns]
    if not group_cols:
        raise SystemExit("No grouping columns found in data.")
    if args.independent_unit:
        df = collapse_within_units(df, group_cols, args.independent_unit, args.metrics)
    table = mean_std_table(df, group_cols, args.metrics)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.endswith(".tex"):
        table.to_latex(args.out, index=False, float_format="%.4f")
    else:
        table.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(table)


if __name__ == "__main__":
    main()
