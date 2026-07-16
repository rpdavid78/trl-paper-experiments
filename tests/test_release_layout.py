#!/usr/bin/env python3
"""Dependency-free structural checks for the academic code release."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    ".github/workflows/release-smoke.yml",
    "README.md",
    "RELEASE_MANIFEST.md",
    "requirements.txt",
    "scripts/cifar100_all_methods_iclr.py",
    "scripts/cifar100_arch_sensitivity_iclr.py",
    "scripts/cifar100c_eval_iclr.py",
    "scripts/vgg_all_methods_iclr.py",
    "scripts/vgg_bn_cifar.py",
    "scripts/cifar100_laplace_prior_grid_iclr.py",
    "scripts/cifar100_temperature_scaling_iclr.py",
    "scripts/cifar100_random_rank30_baseline.py",
    "scripts/imagenet_marglik_fit.py",
    "scripts/imagenet_resnet50_scalecheck.py",
    "scripts/aggregate_results.py",
    "scripts/make_paper_assets.py",
    "ablation_scripts/stale_eigenspace_study_cifar100.py",
    "ablation_scripts/trl_fixed_basis_ablation_cifar100.py",
    "ablation_scripts/trl_refresh_single_ablation_cifar100.py",
    "ablation_scripts/trl_tube_scale_sensitivity_cifar100.py",
    "diagnostics/spine_functional_disagreement_cifar100.py",
    "finetune/finetune_cifar10_spine_smoke.py",
    "toy/run_final_toy_tables.sh",
    "docs/laplace_grid_temperature_scaling.md",
    "docs/table17_boost_ablation.md",
    "docs/tube_scale_sameood_rerun.md",
    "docs/imagenet_resnet50_scalecheck.md",
    "phase1_prereg/PREREGISTRATION_phase1.md",
    "phase1_prereg/README.md",
]


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing release files: {missing}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manifest = (ROOT / "RELEASE_MANIFEST.md").read_text(encoding="utf-8")
    main_script = (ROOT / "scripts/cifar100_all_methods_iclr.py").read_text(encoding="utf-8")
    snapshot_sweep = (
        ROOT / "scripts/all_exported_code_snapshot/boost_betaperp_sweep_2d.py"
    ).read_text(encoding="utf-8")

    assertions = {
        "README documents ImageNet exclusion": "ImageNet is not redistributed" in readme,
        "README maps calibration Table 7": "Table 7" in readme,
        "README maps boost Table 17": "Table 17" in readme,
        "manifest documents excluded checkpoints": "checkpoints" in manifest.lower(),
        "1D boost CLI exists": "--run-boost-ablation" in main_script,
        "2D boost CLI exists": "--run-boost-betaperp-sweep" in main_script,
        "historical sweep has no unresolved hooks": "NotImplementedError" not in snapshot_sweep,
    }
    failed = [label for label, ok in assertions.items() if not ok]
    if failed:
        raise SystemExit(f"Release-layout assertions failed: {failed}")

    print(f"Release layout OK ({len(REQUIRED_PATHS)} required files).")


if __name__ == "__main__":
    main()
