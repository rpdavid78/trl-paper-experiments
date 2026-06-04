#!/usr/bin/env python
"""Simple plotting utilities for ablations and stale-eigenspace studies."""
from __future__ import annotations
import argparse, glob, json, os
import pandas as pd
import matplotlib.pyplot as plt


def load(paths):
    rows=[]
    for pat in paths:
        for p in glob.glob(pat) or [pat]:
            if os.path.exists(p):
                for line in open(p, encoding='utf-8'):
                    if line.strip(): rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_ablation(df, x, y, out):
    g = df.groupby(x)[y].agg(['mean','std']).reset_index().sort_values(x)
    plt.figure()
    plt.errorbar(g[x], g['mean'], yerr=g['std'], marker='o', capsize=3)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plt.savefig(out, dpi=200)
    print('wrote', out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--x', required=True)
    p.add_argument('--y', default='ece')
    p.add_argument('--out', required=True)
    args=p.parse_args()
    df=load(args.input)
    plot_ablation(df,args.x,args.y,args.out)

if __name__=='__main__': main()
