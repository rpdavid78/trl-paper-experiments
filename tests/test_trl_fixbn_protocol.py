from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "cifar100_all_methods_iclr.py"
DIAGNOSTIC_PATH = ROOT / "diagnostics" / "trl_fixbn_ab_cifar100.py"


def load_module(name: str, path: Path):
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_module("trl_fixbn_runner_test", RUNNER_PATH)
DIAGNOSTIC = load_module("trl_fixbn_diagnostic_test", DIAGNOSTIC_PATH)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class TRLModeTests(unittest.TestCase):
    def make_trl(self):
        model = TinyClassifier()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trl = RUNNER.PracticalTRLStage2(
            map_model=model,
            prior_vec=torch.ones(parameter_count),
            clean_loader=[],
            steps=1,
            k_perp=1,
            step_size=0.01,
            eta=0.001,
            tube_scale=4.0,
            max_delta_norm=0.02,
            hvp_batches=1,
        )
        theta = parameters_to_vector(model.parameters()).detach()
        trl.spine = [{
            "theta": theta,
            "N": torch.zeros(theta.numel(), 1),
            "inv_sqrt_prec": torch.ones(1),
        }]
        return trl

    def test_predict_forwards_explicit_reset_mode(self):
        old_device = RUNNER.DEVICE
        RUNNER.DEVICE = torch.device("cpu")
        trl = self.make_trl()
        loader = [(torch.tensor([[1.0, -1.0]]), torch.tensor([0]))]
        try:
            with mock.patch.object(RUNNER, "fix_bn", return_value=0.0) as mocked:
                trl.predict(loader, [], n_samples=1, fix_bn_batches=1, fix_bn_mode="reset")
        finally:
            RUNNER.DEVICE = old_device
        self.assertEqual(mocked.call_args.kwargs["mode"], "reset")

    def test_predict_internal_default_remains_historical_rolling(self):
        old_device = RUNNER.DEVICE
        RUNNER.DEVICE = torch.device("cpu")
        trl = self.make_trl()
        loader = [(torch.tensor([[1.0, -1.0]]), torch.tensor([0]))]
        try:
            with mock.patch.object(RUNNER, "fix_bn", return_value=0.0) as mocked:
                trl.predict(loader, [], n_samples=1, fix_bn_batches=1)
        finally:
            RUNNER.DEVICE = old_device
        self.assertEqual(mocked.call_args.kwargs["mode"], "rolling")

    def test_cli_and_metrics_record_trl_fixbn_mode(self):
        argv = ["runner", "--methods", "trl", "--trl-fixbn-mode", "rolling"]
        with mock.patch.object(sys, "argv", argv):
            cfg = RUNNER.cfg_from_args(RUNNER.parse_args())
        self.assertEqual(cfg.trl_fixbn_mode, "rolling")

        probs = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        row = RUNNER._metrics_row(
            "toy", "toy", "TRL", 0, probs, None, torch.tensor([0, 1]),
            RUNNER.CFG(device="cpu", num_classes=2, trl_fixbn_mode="rolling"), {},
        )
        self.assertEqual(row["trl_fixbn_mode"], "rolling")

    def test_default_cfg_selects_corrected_reset(self):
        self.assertEqual(RUNNER.CFG().trl_fixbn_mode, "reset")


class DiagnosticHelperTests(unittest.TestCase):
    def test_probability_aggregation_is_canonical_by_draw_id(self):
        forward = {
            0: torch.tensor([1.0e20]),
            1: torch.tensor([-1.0e20]),
            2: torch.tensor([3.0]),
        }
        reverse = {key: forward[key] for key in reversed(forward)}
        self.assertTrue(torch.equal(
            DIAGNOSTIC.canonical_draw_mean(forward),
            DIAGNOSTIC.canonical_draw_mean(reverse),
        ))

    def test_draw_bank_is_reproducible_and_reversible_by_id(self):
        spine = [{"N": torch.zeros(4, 2)} for _ in range(3)]
        left = DIAGNOSTIC.build_draw_bank(spine, samples=5, seed=1000)
        right = DIAGNOSTIC.build_draw_bank(spine, samples=5, seed=1000)
        self.assertEqual([draw.anchor_index for draw in left], [draw.anchor_index for draw in right])
        for lhs, rhs in zip(left, right):
            self.assertTrue(torch.equal(lhs.z, rhs.z))
        self.assertEqual(
            sorted(draw.draw_id for draw in reversed(left)),
            list(range(5)),
        )

    def test_beta_grid_order_effect_detects_only_changed_arm(self):
        def metrics(nll):
            return {"acc": 0.5, "nll": nll, "ece": 0.1, "brier": 0.8}
        forward = {
            "rows": [
                {"beta": 2.0, "rolling": metrics(1.0), "reset": metrics(1.1)},
                {"beta": 4.0, "rolling": metrics(0.9), "reset": metrics(1.0)},
            ],
            "selected": {
                "rolling": {"beta": 4.0, "val_nll": 0.9},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        reverse = {
            "rows": [
                {"beta": 4.0, "rolling": metrics(1.2), "reset": metrics(1.0)},
                {"beta": 2.0, "rolling": metrics(0.8), "reset": metrics(1.1)},
            ],
            "selected": {
                "rolling": {"beta": 2.0, "val_nll": 0.8},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        effect = DIAGNOSTIC.beta_grid_order_effect(forward, reverse)
        self.assertTrue(effect["selected_beta_changed"]["rolling"])
        self.assertFalse(effect["selected_beta_changed"]["reset"])

    def test_final_decision_requires_pipeline_and_order_control(self):
        metrics = {"acc": 0.0, "nll": 0.0, "ece": 0.0, "brier": 0.0,
                   "entropy_auroc": 0.0}
        summary = {"delta_reset_minus_rolling": metrics}
        sweep = {
            "rows": [
                {"beta": 4.0,
                 "rolling": {"acc": 0.5, "nll": 1.0, "ece": 0.1, "brier": 0.8},
                 "reset": {"acc": 0.5, "nll": 1.0, "ece": 0.1, "brier": 0.8}}
            ],
            "selected": {
                "rolling": {"beta": 4.0, "val_nll": 1.0},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        args = SimpleNamespace(
            acc_threshold=0.002,
            nll_threshold=0.01,
            ece_threshold=0.005,
            brier_threshold=0.005,
            auroc_threshold=0.01,
        )
        partial = DIAGNOSTIC.threshold_decision(
            summary, None, sweep, None, None, args
        )
        self.assertFalse(partial["complete_for_final_decision"])
        self.assertIsNone(partial["rerun_seeds_0_to_4"])

        zero_order_arm = {
            "delta_reverse_minus_forward": metrics,
            "id_probability_mae": 0.0,
            "ood_probability_mae": 0.0,
        }
        complete = DIAGNOSTIC.threshold_decision(
            summary,
            summary,
            sweep,
            sweep,
            {"rolling": zero_order_arm, "reset": zero_order_arm},
            args,
        )
        self.assertTrue(complete["complete_for_final_decision"])
        self.assertTrue(complete["audit_valid"])
        self.assertFalse(complete["rerun_seeds_0_to_4"])

    def test_final_decision_rejects_noninvariant_reset_beta_sweep(self):
        metrics = {"acc": 0.0, "nll": 0.0, "ece": 0.0, "brier": 0.0,
                   "entropy_auroc": 0.0}
        summary = {"delta_reset_minus_rolling": metrics}
        forward = {
            "rows": [{"beta": 4.0, "rolling": metrics, "reset": metrics}],
            "selected": {
                "rolling": {"beta": 4.0, "val_nll": 1.0},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        changed_reset = dict(metrics, ece=0.001)
        reverse = {
            "rows": [{"beta": 4.0, "rolling": metrics, "reset": changed_reset}],
            "selected": forward["selected"],
        }
        zero_order_arm = {
            "delta_reverse_minus_forward": metrics,
            "id_probability_mae": 0.0,
            "ood_probability_mae": 0.0,
        }
        args = SimpleNamespace(
            acc_threshold=0.002, nll_threshold=0.01, ece_threshold=0.005,
            brier_threshold=0.005, auroc_threshold=0.01,
        )
        decision = DIAGNOSTIC.threshold_decision(
            summary, summary, forward, reverse,
            {"rolling": zero_order_arm, "reset": zero_order_arm}, args,
        )
        self.assertFalse(decision["audit_valid"])
        self.assertFalse(decision["complete_for_final_decision"])
        self.assertIn("reset_beta_grid_order_invariance_failed", decision["triggered"])

    def test_beta_only_change_requires_another_seed0_bank(self):
        metrics = {"acc": 0.0, "nll": 0.0, "ece": 0.0, "brier": 0.0,
                   "entropy_auroc": 0.0}
        summary = {"delta_reset_minus_rolling": metrics}
        base = {"acc": 0.5, "ece": 0.1, "brier": 0.8}
        forward = {
            "rows": [
                {"beta": 2.0, "rolling": dict(base, nll=1.1),
                 "reset": dict(base, nll=1.1)},
                {"beta": 4.0, "rolling": dict(base, nll=1.0),
                 "reset": dict(base, nll=1.0)},
            ],
            "selected": {
                "rolling": {"beta": 4.0, "val_nll": 1.0},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        reverse = {
            "rows": [
                {"beta": 4.0, "rolling": dict(base, nll=1.2),
                 "reset": dict(base, nll=1.0)},
                {"beta": 2.0, "rolling": dict(base, nll=0.9),
                 "reset": dict(base, nll=1.1)},
            ],
            "selected": {
                "rolling": {"beta": 2.0, "val_nll": 0.9},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        zero_order_arm = {
            "delta_reverse_minus_forward": metrics,
            "id_probability_mae": 0.0,
            "ood_probability_mae": 0.0,
        }
        args = SimpleNamespace(
            acc_threshold=0.002, nll_threshold=0.01, ece_threshold=0.005,
            brier_threshold=0.005, auroc_threshold=0.01,
        )
        decision = DIAGNOSTIC.threshold_decision(
            summary, summary, forward, reverse,
            {"rolling": zero_order_arm, "reset": zero_order_arm}, args,
        )
        self.assertTrue(decision["phases_complete"])
        self.assertTrue(decision["audit_valid"])
        self.assertFalse(decision["complete_for_final_decision"])
        self.assertTrue(decision["repeat_seed0_posterior_banks"])
        self.assertIsNone(decision["rerun_seeds_0_to_4"])

    def test_nonfinite_delta_invalidates_audit(self):
        metrics = {"acc": 0.0, "nll": float("nan"), "ece": 0.0,
                   "brier": 0.0, "entropy_auroc": 0.0}
        summary = {"delta_reset_minus_rolling": metrics}
        sweep_metrics = {"acc": 0.5, "nll": 1.0, "ece": 0.1, "brier": 0.8}
        sweep = {
            "rows": [{"beta": 4.0, "rolling": sweep_metrics,
                      "reset": sweep_metrics}],
            "selected": {
                "rolling": {"beta": 4.0, "val_nll": 1.0},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        zero_order = {
            "delta_reverse_minus_forward": dict(metrics, nll=0.0),
            "id_probability_mae": 0.0,
            "ood_probability_mae": 0.0,
        }
        args = SimpleNamespace(
            acc_threshold=0.002, nll_threshold=0.01, ece_threshold=0.005,
            brier_threshold=0.005, auroc_threshold=0.01,
        )
        decision = DIAGNOSTIC.threshold_decision(
            summary, summary, sweep, sweep,
            {"rolling": zero_order, "reset": zero_order}, args,
        )
        self.assertFalse(decision["audit_valid"])
        self.assertFalse(decision["complete_for_final_decision"])
        self.assertIn("fixed:nll:nonfinite", decision["triggered"])

    def test_nonfinite_reset_order_delta_invalidates_audit(self):
        metric_delta = {"acc": 0.0, "nll": 0.0, "ece": 0.0,
                        "brier": 0.0, "entropy_auroc": 0.0}
        summary = {"delta_reset_minus_rolling": metric_delta}
        val_metrics = {"acc": 0.5, "nll": 1.0, "ece": 0.1, "brier": 0.8}
        sweep = {
            "rows": [{"beta": 4.0, "rolling": val_metrics, "reset": val_metrics}],
            "selected": {
                "rolling": {"beta": 4.0, "val_nll": 1.0},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        rolling_order = {
            "delta_reverse_minus_forward": metric_delta,
            "id_probability_mae": 0.0,
            "ood_probability_mae": 0.0,
        }
        reset_order = {
            "delta_reverse_minus_forward": dict(metric_delta, ece=float("nan")),
            "id_probability_mae": 0.0,
            "ood_probability_mae": 0.0,
        }
        args = SimpleNamespace(
            acc_threshold=0.002, nll_threshold=0.01, ece_threshold=0.005,
            brier_threshold=0.005, auroc_threshold=0.01,
        )
        decision = DIAGNOSTIC.threshold_decision(
            summary, summary, sweep, sweep,
            {"rolling": rolling_order, "reset": reset_order}, args,
        )
        self.assertFalse(decision["audit_valid"])
        self.assertFalse(decision["draw_order_checks"]["reset_all_finite"])

    def test_nonfinite_reset_beta_grid_delta_invalidates_audit(self):
        delta = {"acc": 0.0, "nll": 0.0, "ece": 0.0,
                 "brier": 0.0, "entropy_auroc": 0.0}
        summary = {"delta_reset_minus_rolling": delta}
        val_metrics = {"acc": 0.5, "nll": 1.0, "ece": 0.1, "brier": 0.8}
        changed = dict(val_metrics, ece=float("nan"))
        forward = {
            "rows": [{"beta": 4.0, "rolling": val_metrics, "reset": val_metrics}],
            "selected": {
                "rolling": {"beta": 4.0, "val_nll": 1.0},
                "reset": {"beta": 4.0, "val_nll": 1.0},
            },
        }
        reverse = {
            "rows": [{"beta": 4.0, "rolling": val_metrics, "reset": changed}],
            "selected": forward["selected"],
        }
        zero_order = {
            "delta_reverse_minus_forward": delta,
            "id_probability_mae": 0.0,
            "ood_probability_mae": 0.0,
        }
        args = SimpleNamespace(
            acc_threshold=0.002, nll_threshold=0.01, ece_threshold=0.005,
            brier_threshold=0.005, auroc_threshold=0.01,
        )
        decision = DIAGNOSTIC.threshold_decision(
            summary, summary, forward, reverse,
            {"rolling": zero_order, "reset": zero_order}, args,
        )
        self.assertFalse(decision["audit_valid"])
        self.assertFalse(decision["reset_beta_grid_invariant"])
        self.assertFalse(
            decision["beta_grid_order_effect"]["reset_all_metrics_finite"]
        )

    def test_seed_helper_enables_deterministic_cudnn(self):
        DIAGNOSTIC.seed_everything(123)
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)


class ConsumerModeTests(unittest.TestCase):
    def test_main_runner_consumers_pass_and_record_fixbn_mode(self):
        paths = [
            ROOT / "scripts" / "cifar100c_eval_iclr.py",
            ROOT / "scripts" / "cifar100_random_rank30_baseline.py",
            ROOT / "ablation_scripts" / "trl_fixed_basis_ablation_cifar100.py",
            ROOT / "ablation_scripts" / "trl_tube_scale_sensitivity_cifar100.py",
            ROOT / "ablation_scripts" / "trl_refresh_single_ablation_cifar100.py",
            ROOT / "scripts" / "all_exported_code_snapshot" / "cifar100c_eval_iclr.py",
            ROOT / "scripts" / "all_exported_code_snapshot" / "trl_fixed_basis_ablation_cifar100.py",
            ROOT / "scripts" / "all_exported_code_snapshot" / "trl_tube_scale_sensitivity_cifar100.py",
        ]
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                self.assertIn("fix_bn_mode=", source)
                self.assertRegex(source, r'fixbn_mode|trl_fixbn_mode')

    def test_exported_main_snapshot_has_no_implicit_predict_mode(self):
        path = ROOT / "scripts" / "all_exported_code_snapshot" / "cifar100_all_methods_iclr.py"
        source = path.read_text(encoding="utf-8")
        calls = source.split("trl.predict(")[1:]
        self.assertGreaterEqual(len(calls), 5)
        for call in calls:
            argument_block = call.split(")", 1)[0]
            self.assertIn("fix_bn_mode=", argument_block)

    def test_canonical_main_has_no_implicit_predict_mode(self):
        path = ROOT / "scripts" / "cifar100_all_methods_iclr.py"
        source = path.read_text(encoding="utf-8")
        calls = source.split("trl.predict(")[1:]
        self.assertEqual(len(calls), 5)
        for call in calls:
            argument_block = call.split(")", 1)[0]
            self.assertIn("fix_bn_mode=", argument_block)

    def test_spine_diagnostic_makes_fixbn_mode_explicit(self):
        path = ROOT / "diagnostics" / "spine_functional_disagreement_cifar100.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn('"--fixbn-mode"', source)
        self.assertIn("mode=args.fixbn_mode", source)
        self.assertIn('"fixbn_mode"', source)


if __name__ == "__main__":
    unittest.main()
