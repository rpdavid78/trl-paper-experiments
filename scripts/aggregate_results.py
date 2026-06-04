#!/usr/bin/env python
"""Aggregate JSONL experiment records into mean/std tables.

Usage:
  python -m trl_iclr_utils.aggregate_results --input results/*.jsonl --out tables/main.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import pandas as pd

METRICS_DEFAULT = ["acc", "nll", "ece", "brier", "auroc", "runtime_total_sec", "peak_vram_gb"]
GROUP_DEFAULT = ["dataset", "architecture", "method"]


def load_jsonl(paths: List[str]) -> pd.DataFrame:
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
    present_metrics = [m for m in metrics if m in df.columns]
    if not present_metrics:
        raise SystemExit(f"None of the requested metrics are present: {metrics}")
    grouped = df.groupby(group_cols, dropna=False)[present_metrics]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = grouped.count().iloc[:, [0]].rename(columns={present_metrics[0]: "n"})
    out = pd.concat([count, mean, std], axis=1).reset_index()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", "--output", dest="out", required=True)
    p.add_argument("--group", nargs="+", default=GROUP_DEFAULT)
    p.add_argument("--metrics", nargs="+", default=METRICS_DEFAULT)
    args = p.parse_args()

    df = load_jsonl(args.input)
    group_cols = [c for c in args.group if c in df.columns]
    if not group_cols:
        raise SystemExit("No grouping columns found in data.")
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
