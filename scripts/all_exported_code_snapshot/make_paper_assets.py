#!/usr/bin/env python
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/mnt/hd2/rpdavid/trl_results")
OUT = ROOT / "paper_assets"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
CSVS = OUT / "csv"

for d in [TABLES, FIGS, CSVS]:
    d.mkdir(parents=True, exist_ok=True)

METRICS = ["acc", "nll", "ece", "brier"]
DIRECTION = {
    "acc": "max",
    "nll": "min",
    "ece": "min",
    "brier": "min",
    "auroc": "max",
    "runtime_total_sec": "min",
    "fixbn_overhead_sec": "min",
    "peak_vram_gb": "min",
}

DISPLAY = {
    "acc": "Acc $\\uparrow$",
    "nll": "NLL $\\downarrow$",
    "ece": "ECE $\\downarrow$",
    "brier": "Brier $\\downarrow$",
    "auroc": "AUROC $\\uparrow$",
    "runtime_total_sec": "Runtime (s) $\\downarrow$",
    "fixbn_overhead_sec": "FixBN (s) $\\downarrow$",
    "peak_vram_gb": "VRAM (GB) $\\downarrow$",
}

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"

def fmt_ms(m, s, nd=3, bold=False):
    text = f"{fmt(m, nd)} $\\pm$ {fmt(s, nd)}"
    return f"\\textbf{{{text}}}" if bold else text

def latex_escape(s):
    s = str(s)
    return (
        s.replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
    )

def best_indices(rows, metrics):
    best = {}
    for metric in metrics:
        vals = []
        for i, r in enumerate(rows):
            vals.append((i, safe_float(r.get(f"{metric}_mean", "nan"))))
        vals = [(i, v) for i, v in vals if not math.isnan(v)]
        if not vals:
            continue
        if DIRECTION.get(metric, "min") == "max":
            b = max(v for _, v in vals)
        else:
            b = min(v for _, v in vals)
        best[metric] = {i for i, v in vals if abs(v - b) <= 1e-12}
    return best

def summary_table_to_latex(
    csv_path,
    out_path,
    label_col,
    metrics,
    caption,
    label,
    label_name=None,
    sort_key=None,
):
    rows = read_csv(csv_path)
    if sort_key is not None:
        rows.sort(key=sort_key)

    best = best_indices(rows, metrics)

    label_name = label_name or label_col
    header = [label_name] + [DISPLAY.get(m, m) for m in metrics]
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for i, r in enumerate(rows):
        vals = [latex_escape(r[label_col])]
        for m in metrics:
            vals.append(fmt_ms(
                safe_float(r.get(f"{m}_mean", "nan")),
                safe_float(r.get(f"{m}_std", "nan")),
                nd=3,
                bold=i in best.get(m, set()),
            ))
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    out_path.write_text("\n".join(lines))
    print("Wrote", out_path)

def aggregate_jsonl(paths, group_cols, metrics, out_csv):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    dedup = {}
    for r in rows:
        key = tuple(r.get(c) for c in group_cols + ["seed"])
        dedup[key] = r
    rows = list(dedup.values())

    grouped = defaultdict(list)
    for r in rows:
        key = tuple(r.get(c) for c in group_cols)
        grouped[key].append(r)

    out = []
    fields = list(group_cols) + ["n"]
    for m in metrics:
        fields += [f"{m}_mean", f"{m}_std"]

    for key, rs in sorted(grouped.items(), key=lambda kv: kv[0]):
        rec = {c: v for c, v in zip(group_cols, key)}
        rec["n"] = len(rs)
        for m in metrics:
            xs = [safe_float(r[m]) for r in rs if m in r and not math.isnan(safe_float(r[m]))]
            rec[f"{m}_mean"] = mean(xs) if xs else float("nan")
            rec[f"{m}_std"] = stdev(xs) if len(xs) > 1 else 0.0
        out.append(rec)

    write_csv(out_csv, out, fields)
    print("Wrote", out_csv)

def build_cifar100_clean():
    paths = sorted(glob.glob(str(ROOT / "cifar100_seed*_core.jsonl")))
    if not paths:
        print("Skip CIFAR-100 clean: no cifar100_seed*_core.jsonl")
        return

    out_csv = CSVS / "cifar100_clean_5seeds_summary.csv"
    aggregate_jsonl(
        paths,
        group_cols=["dataset", "architecture", "method"],
        metrics=["acc", "nll", "ece", "brier", "auroc"],
        out_csv=out_csv,
    )

    summary_table_to_latex(
        out_csv,
        TABLES / "table_cifar100_clean.tex",
        label_col="method",
        metrics=["acc", "nll", "ece", "brier", "auroc"],
        caption="CIFAR-100 clean test performance. Mean $\\pm$ std over seeds.",
        label="tab:cifar100-clean",
        label_name="Method",
    )

def build_cifar100c_main():
    src = ROOT / "cifar100c_main_ts4_3seeds_summary.csv"
    if not src.exists():
        print("Skip CIFAR-100-C main: missing", src)
        return

    dst = CSVS / "cifar100c_main_ts4_3seeds_summary.csv"
    dst.write_text(src.read_text())

    summary_table_to_latex(
        dst,
        TABLES / "table_cifar100c_main.tex",
        label_col="method",
        metrics=["acc", "nll", "ece", "brier"],
        caption="CIFAR-100-C performance across all corruptions and severities. Mean $\\pm$ std over three seeds.",
        label="tab:cifar100c-main",
        label_name="Method",
    )

def build_cifar100c_by_severity():
    paths = sorted(glob.glob(str(ROOT / "cifar100c_seed*_main_ts4.jsonl")))
    if not paths:
        print("Skip CIFAR-100-C severity: no cifar100c_seed*_main_ts4.jsonl")
        return

    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    dedup = {}
    for r in rows:
        key = (r["seed"], r["method"], r["corruption"], r["severity"])
        dedup[key] = r
    rows = list(dedup.values())

    by_seed = defaultdict(list)
    for r in rows:
        by_seed[(r["method"], int(r["severity"]), int(r["seed"]))].append(r)

    seed_means = []
    for (method, sev, seed), rs in by_seed.items():
        rec = {"method": method, "severity": sev, "seed": seed}
        for m in METRICS:
            rec[m] = mean(float(r[m]) for r in rs)
        seed_means.append(rec)

    grouped = defaultdict(list)
    for r in seed_means:
        grouped[(r["method"], r["severity"])].append(r)

    out = []
    fields = ["method", "severity", "n"]
    for m in METRICS:
        fields += [f"{m}_mean", f"{m}_std"]

    for (method, sev), rs in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rec = {"method": method, "severity": sev, "n": len(rs)}
        for m in METRICS:
            xs = [r[m] for r in rs]
            rec[f"{m}_mean"] = mean(xs)
            rec[f"{m}_std"] = stdev(xs) if len(xs) > 1 else 0.0
        out.append(rec)

    out_csv = CSVS / "cifar100c_by_severity_3seeds_summary.csv"
    write_csv(out_csv, out, fields)
    print("Wrote", out_csv)

    # LaTeX compact: TRL vs MC-Dropout vs DeepEns only, since full table is long.
    compact = [r for r in out if r["method"] in {"TRL", "MC-Dropout", "DeepEns"}]
    compact_csv = CSVS / "cifar100c_by_severity_compact.csv"
    write_csv(compact_csv, compact, fields)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Severity & Method & Acc $\\uparrow$ & NLL $\\downarrow$ & ECE $\\downarrow$ & Brier $\\downarrow$ \\\\")
    lines.append("\\midrule")
    for sev in sorted(set(int(r["severity"]) for r in compact)):
        rows_sev = [r for r in compact if int(r["severity"]) == sev]
        best = best_indices(rows_sev, METRICS)
        for i, r in enumerate(rows_sev):
            vals = [
                str(sev) if i == 0 else "",
                latex_escape(r["method"]),
            ]
            for m in METRICS:
                vals.append(fmt_ms(
                    safe_float(r[f"{m}_mean"]),
                    safe_float(r[f"{m}_std"]),
                    nd=3,
                    bold=i in best.get(m, set()),
                ))
            lines.append(" & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\caption{CIFAR-100-C by severity for the main methods. Mean $\\pm$ std over three seeds.}")
    lines.append("\\label{tab:cifar100c-severity}")
    lines.append("\\end{table}")
    (TABLES / "table_cifar100c_by_severity_compact.tex").write_text("\n".join(lines))
    print("Wrote", TABLES / "table_cifar100c_by_severity_compact.tex")

def copy_and_table_existing():
    specs = [
        (
            ROOT / "ablation_k_3seeds_summary.csv",
            TABLES / "table_ablation_k.tex",
            "trl_k_perp",
            ["acc", "nll", "ece", "brier", "auroc", "runtime_total_sec"],
            "TRL sensitivity to transverse rank $k$. Mean $\\pm$ std over three seeds.",
            "tab:ablation-k",
            "$k$",
            lambda r: int(float(r["trl_k_perp"])),
        ),
        (
            ROOT / "ablation_T_3seeds_summary.csv",
            TABLES / "table_ablation_T.tex",
            "trl_steps",
            ["acc", "nll", "ece", "brier", "auroc", "runtime_total_sec"],
            "TRL sensitivity to the number of spine steps $T$. Mean $\\pm$ std over three seeds.",
            "tab:ablation-T",
            "$T$",
            lambda r: int(float(r["trl_steps"])),
        ),
        (
            ROOT / "ablation_fixbn_3seeds_summary.csv",
            TABLES / "table_ablation_fixbn.tex",
            "trl_fixbn_batches",
            ["acc", "nll", "ece", "brier", "auroc", "runtime_total_sec"],
            "Effect of BatchNorm recalibration batches in TRL. Mean $\\pm$ std over three seeds.",
            "tab:ablation-fixbn",
            "FixBN batches",
            lambda r: int(float(r["trl_fixbn_batches"])),
        ),
        (
            ROOT / "tube_scale_sensitivity_3seeds_summary.csv",
            TABLES / "table_ablation_tube_scale.tex",
            "tube_scale",
            ["acc", "nll", "ece", "brier", "auroc"],
            "TRL sensitivity to tube scale. Mean $\\pm$ std over three seeds.",
            "tab:ablation-tube-scale",
            "Tube scale",
            lambda r: float(r["tube_scale"]),
        ),
        (
            ROOT / "trl_basis_ablation_3seeds_summary.csv",
            TABLES / "table_ablation_basis.tex",
            "method",
            ["acc", "nll", "ece", "brier"],
            "Ablation comparing transported transverse basis against a fixed MAP basis. Mean $\\pm$ std over three seeds.",
            "tab:ablation-basis",
            "Variant",
            None,
        ),
    ]

    for src, tex, label_col, metrics, caption, label, label_name, sort_key in specs:
        if not src.exists():
            print("Skip missing", src)
            continue
        dst = CSVS / src.name
        dst.write_text(src.read_text())
        summary_table_to_latex(
            dst, tex, label_col, metrics, caption, label,
            label_name=label_name,
            sort_key=sort_key,
        )

def stale_subspace_table():
    src = ROOT / "stale_subspace_3seeds_summary.csv"
    if not src.exists():
        print("Skip stale subspace: missing", src)
        return
    rows = read_csv(src)
    rows.sort(key=lambda r: float(r["spine_fraction"]))
    dst = CSVS / src.name
    dst.write_text(src.read_text())

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{ccccc}")
    lines.append("\\toprule")
    lines.append("Spine fraction & Trans. overlap $\\uparrow$ & MAP overlap $\\uparrow$ & Trans. angle $\\downarrow$ & Eig. drift $\\downarrow$ \\\\")
    lines.append("\\midrule")
    for r in rows:
        vals = [
            fmt(float(r["spine_fraction"]), 2),
            fmt_ms(safe_float(r["transport_vs_fresh_subspace_overlap_mean"]), safe_float(r["transport_vs_fresh_subspace_overlap_std"]), 3),
            fmt_ms(safe_float(r["map_vs_fresh_subspace_overlap_mean"]), safe_float(r["map_vs_fresh_subspace_overlap_std"]), 3),
            fmt_ms(safe_float(r["transport_vs_fresh_mean_angle_deg_mean"]), safe_float(r["transport_vs_fresh_mean_angle_deg_std"]), 1),
            fmt_ms(safe_float(r["eigenvalue_relative_drift_mean_mean"]), safe_float(r["eigenvalue_relative_drift_mean_std"]), 3),
        ]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Subspace overlap and principal-angle diagnostic along the TRL spine. Mean $\\pm$ std over three seeds.}")
    lines.append("\\label{tab:stale-subspace}")
    lines.append("\\end{table}")
    (TABLES / "table_stale_subspace.tex").write_text("\n".join(lines))
    print("Wrote", TABLES / "table_stale_subspace.tex")

def make_figures():
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Skip figures: matplotlib unavailable:", e)
        return

    def plot_xy(csv_path, x_col, y_col, out_name, xlabel, ylabel, title):
        if not csv_path.exists():
            return
        rows = read_csv(csv_path)
        rows = [r for r in rows if f"{y_col}_mean" in r]
        rows.sort(key=lambda r: float(r[x_col]))
        x = [float(r[x_col]) for r in rows]
        y = [float(r[f"{y_col}_mean"]) for r in rows]
        yerr = [float(r.get(f"{y_col}_std", 0.0)) for r in rows]

        plt.figure()
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out = FIGS / out_name
        plt.savefig(out)
        plt.close()
        print("Wrote", out)

    plot_xy(CSVS / "ablation_k_3seeds_summary.csv", "trl_k_perp", "nll", "fig_ablation_k_nll.pdf", "Transverse rank k", "NLL", "TRL rank ablation")
    plot_xy(CSVS / "ablation_k_3seeds_summary.csv", "trl_k_perp", "ece", "fig_ablation_k_ece.pdf", "Transverse rank k", "ECE", "TRL rank ablation")
    plot_xy(CSVS / "ablation_T_3seeds_summary.csv", "trl_steps", "nll", "fig_ablation_T_nll.pdf", "Spine steps T", "NLL", "TRL spine-length ablation")
    plot_xy(CSVS / "ablation_T_3seeds_summary.csv", "trl_steps", "ece", "fig_ablation_T_ece.pdf", "Spine steps T", "ECE", "TRL spine-length ablation")
    plot_xy(CSVS / "ablation_fixbn_3seeds_summary.csv", "trl_fixbn_batches", "nll", "fig_ablation_fixbn_nll.pdf", "FixBN batches", "NLL", "FixBN ablation")
    plot_xy(CSVS / "ablation_fixbn_3seeds_summary.csv", "trl_fixbn_batches", "ece", "fig_ablation_fixbn_ece.pdf", "FixBN batches", "ECE", "FixBN ablation")
    plot_xy(CSVS / "tube_scale_sensitivity_3seeds_summary.csv", "tube_scale", "nll", "fig_ablation_tube_scale_nll.pdf", "Tube scale", "NLL", "Tube-scale sensitivity")
    plot_xy(CSVS / "tube_scale_sensitivity_3seeds_summary.csv", "tube_scale", "ece", "fig_ablation_tube_scale_ece.pdf", "Tube scale", "ECE", "Tube-scale sensitivity")

    stale = CSVS / "stale_subspace_3seeds_summary.csv"
    if stale.exists():
        rows = read_csv(stale)
        rows.sort(key=lambda r: float(r["spine_fraction"]))
        x = [float(r["spine_fraction"]) for r in rows]
        y1 = [float(r["transport_vs_fresh_subspace_overlap_mean"]) for r in rows]
        e1 = [float(r["transport_vs_fresh_subspace_overlap_std"]) for r in rows]
        y2 = [float(r["map_vs_fresh_subspace_overlap_mean"]) for r in rows]
        e2 = [float(r["map_vs_fresh_subspace_overlap_std"]) for r in rows]

        plt.figure()
        plt.errorbar(x, y1, yerr=e1, marker="o", capsize=3, label="Transported vs fresh")
        plt.errorbar(x, y2, yerr=e2, marker="s", capsize=3, label="MAP vs fresh")
        plt.xlabel("Spine fraction")
        plt.ylabel("Subspace overlap")
        plt.title("Fresh eigenspace drift along the spine")
        plt.legend()
        plt.tight_layout()
        out = FIGS / "fig_stale_subspace_overlap.pdf"
        plt.savefig(out)
        plt.close()
        print("Wrote", out)

    sev_csv = CSVS / "cifar100c_by_severity_3seeds_summary.csv"
    if sev_csv.exists():
        rows = read_csv(sev_csv)
        methods = ["DeepEns", "TRL", "MC-Dropout", "MAP", "SWAG"]
        for metric in ["nll", "ece"]:
            plt.figure()
            for method in methods:
                rs = [r for r in rows if r["method"] == method]
                if not rs:
                    continue
                rs.sort(key=lambda r: int(r["severity"]))
                x = [int(r["severity"]) for r in rs]
                y = [float(r[f"{metric}_mean"]) for r in rs]
                plt.plot(x, y, marker="o", label=method)
            plt.xlabel("Severity")
            plt.ylabel(metric.upper())
            plt.title(f"CIFAR-100-C by severity: {metric.upper()}")
            plt.legend()
            plt.tight_layout()
            out = FIGS / f"fig_cifar100c_severity_{metric}.pdf"
            plt.savefig(out)
            plt.close()
            print("Wrote", out)

def write_manifest():
    text = f"""# Paper assets

Generated under:

`{OUT}`

## Suggested placement

### Main paper
- `tables/table_cifar100_clean.tex`
- `tables/table_cifar100c_main.tex`
- Optional main figure: `figures/fig_cifar100c_severity_nll.pdf` or `figures/fig_cifar100c_severity_ece.pdf`

### Appendix
- `tables/table_cifar100c_by_severity_compact.tex`
- `tables/table_ablation_k.tex`
- `tables/table_ablation_T.tex`
- `tables/table_ablation_fixbn.tex`
- `tables/table_ablation_tube_scale.tex`
- `tables/table_ablation_basis.tex`
- `tables/table_stale_subspace.tex`

### Appendix figures
- `figures/fig_ablation_k_nll.pdf`
- `figures/fig_ablation_k_ece.pdf`
- `figures/fig_ablation_T_nll.pdf`
- `figures/fig_ablation_T_ece.pdf`
- `figures/fig_ablation_fixbn_nll.pdf`
- `figures/fig_ablation_fixbn_ece.pdf`
- `figures/fig_ablation_tube_scale_nll.pdf`
- `figures/fig_ablation_tube_scale_ece.pdf`
- `figures/fig_stale_subspace_overlap.pdf`

## Recommended narrative

Main text should emphasize:
1. TRL is the best non-ensemble method in NLL/ECE/Brier on CIFAR-100-C.
2. `tube_scale=4.0` is selected by validation and is supported by sensitivity analysis.
3. Ablations show the method is more sensitive to transverse rank `k` and FixBN than to long spine length `T`.

Appendix should include:
1. Full severity breakdown.
2. Geometric stale-subspace diagnostic.
3. Transported vs fixed-MAP basis ablation.
4. Sensitivity to `k`, `T`, FixBN, and tube scale.
"""
    (OUT / "MANIFEST.md").write_text(text)
    print("Wrote", OUT / "MANIFEST.md")

def main():
    build_cifar100_clean()
    build_cifar100c_main()
    build_cifar100c_by_severity()
    copy_and_table_existing()
    stale_subspace_table()
    make_figures()
    write_manifest()
    print()
    print("Done. Assets in:", OUT)

if __name__ == "__main__":
    main()
