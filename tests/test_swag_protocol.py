from __future__ import annotations

import ast
import importlib.util
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "cifar100_all_methods_iclr.py"


def load_runner():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("cifar100_all_methods_iclr_test", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


class BatchNormOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2, momentum=0.1)

    def forward(self, x):
        return self.bn(x)


def calibration_batches():
    labels = torch.zeros(2, dtype=torch.long)
    return [
        (torch.tensor([[1.0, 3.0], [3.0, 5.0]]), labels),
        (torch.tensor([[5.0, 7.0], [7.0, 9.0]]), labels),
    ]


class FixBNTests(unittest.TestCase):
    def test_reset_is_independent_and_cumulative(self):
        left = BatchNormOnly()
        right = BatchNormOnly()
        left.bn.running_mean.copy_(torch.tensor([100.0, -100.0]))
        left.bn.running_var.fill_(25.0)
        left.bn.num_batches_tracked.fill_(17)
        right.bn.running_mean.copy_(torch.tensor([-20.0, 40.0]))
        right.bn.running_var.fill_(3.0)
        right.bn.num_batches_tracked.fill_(9)

        RUNNER.fix_bn(left, calibration_batches(), torch.device("cpu"), 2, mode="reset")
        RUNNER.fix_bn(right, calibration_batches(), torch.device("cpu"), 2, mode="reset")

        expected_mean = torch.tensor([4.0, 6.0])
        expected_var = torch.tensor([2.0, 2.0])
        self.assertTrue(torch.allclose(left.bn.running_mean, expected_mean))
        self.assertTrue(torch.allclose(left.bn.running_var, expected_var))
        self.assertTrue(torch.equal(left.bn.running_mean, right.bn.running_mean))
        self.assertTrue(torch.equal(left.bn.running_var, right.bn.running_var))
        self.assertEqual(left.bn.num_batches_tracked.item(), 2)
        self.assertEqual(right.bn.num_batches_tracked.item(), 2)
        self.assertEqual(left.bn.momentum, 0.1)
        self.assertEqual(right.bn.momentum, 0.1)
        self.assertFalse(left.training)
        self.assertFalse(right.training)

    def test_rolling_preserves_buffer_history(self):
        model = BatchNormOnly()
        model.bn.running_mean.copy_(torch.tensor([10.0, -10.0]))
        model.bn.running_var.fill_(4.0)
        model.bn.num_batches_tracked.fill_(7)

        RUNNER.fix_bn(model, calibration_batches(), torch.device("cpu"), 2, mode="rolling")

        self.assertTrue(
            torch.allclose(model.bn.running_mean, torch.tensor([8.88, -6.94]), atol=1e-6)
        )
        self.assertEqual(model.bn.num_batches_tracked.item(), 9)
        self.assertEqual(model.bn.momentum, 0.1)
        self.assertFalse(model.training)

    def test_invalid_mode_and_batch_count_are_rejected(self):
        model = BatchNormOnly()
        with self.assertRaises(ValueError):
            RUNNER.fix_bn(model, calibration_batches(), torch.device("cpu"), 2, mode="rest")
        with self.assertRaises(ValueError):
            RUNNER.fix_bn(model, calibration_batches(), torch.device("cpu"), 0, mode="reset")


class SWAGInitializerTests(unittest.TestCase):
    def test_main_flow_passes_map_to_swag_after_ensemble(self):
        map_model = object()
        ensemble_member = object()
        captured = {}
        probs = torch.full((2, 100), 0.01)

        def fake_swag(_train, _test, _ood, initializer, _cfg, timings=None):
            captured["initializer"] = initializer
            return probs, probs

        def fake_row(_dataset, _architecture, method, seed, *_args, **_kwargs):
            return {
                "method": method,
                "seed": seed,
                "acc": 0.0,
                "nll": 0.0,
                "ece": 0.0,
                "brier": 0.0,
                "auroc": 0.0,
                "runtime_total_sec": 0.0,
            }

        cfg = RUNNER.CFG(device="cpu", seed=0, ckpt_dir="unused")
        dummy_loader = object()
        with (
            mock.patch.object(RUNNER, "set_seed"),
            mock.patch.object(RUNNER, "ensure_dir"),
            mock.patch.object(RUNNER, "cleanup"),
            mock.patch.object(
                RUNNER,
                "get_data",
                return_value=(dummy_loader, dummy_loader, dummy_loader, dummy_loader, dummy_loader),
            ),
            mock.patch.object(RUNNER, "get_targets", return_value=torch.tensor([0, 1])),
            mock.patch.object(RUNNER, "load_or_train_map", return_value=map_model),
            mock.patch.object(
                RUNNER,
                "deep_ensemble",
                return_value=(probs, probs, ensemble_member, torch.tensor([0, 1])),
            ),
            mock.patch.object(RUNNER, "run_swag", side_effect=fake_swag),
            mock.patch.object(RUNNER, "_metrics_row", side_effect=fake_row),
        ):
            RUNNER.main_iclr(cfg, ["deepens", "swag"])

        self.assertIs(captured["initializer"], map_model)
        self.assertIsNot(captured["initializer"], ensemble_member)

    def test_all_release_runners_use_map_initializer(self):
        offenders = []
        for path in sorted((ROOT / "scripts").rglob("*.py")):
            if path.name.startswith("._"):
                continue
            source = path.read_text(encoding="utf-8")
            if "run_swag(" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "run_swag"
                    and len(node.args) >= 4
                ):
                    continue
                initializer = ast.unparse(node.args[3])
                if "model_map" not in initializer or "last_model" in initializer:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} -> {initializer}")
        self.assertEqual(offenders, [], "\n".join(offenders))


class SWAGCacheTests(unittest.TestCase):
    @staticmethod
    def _cfg():
        return RUNNER.CFG(
            device="cpu",
            seed=7,
            swag_epochs=3,
            swag_lr=0.0125,
            momentum=0.83,
        )

    @staticmethod
    def _valid_payload(model, cfg):
        fingerprint = RUNNER.model_state_sha256(model)
        return {
            "schema_version": 2,
            "base_model_source": "MAP",
            "base_model_state_sha256": fingerprint,
            "swag_variant": "diagonal",
            "map_seed": int(cfg.seed),
            "swag_epochs": int(cfg.swag_epochs),
            "swag_lr": float(cfg.swag_lr),
            "swag_momentum": float(cfg.momentum),
            "swag_batch_size": int(cfg.batch_size),
            "swag_num_workers": int(cfg.num_workers),
        }

    def test_state_fingerprint_includes_batchnorm_buffers(self):
        model = BatchNormOnly()
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
        fingerprint_before = RUNNER.model_state_sha256(model)

        model.bn.running_mean.add_(torch.tensor([1.0, -2.0]))
        fingerprint_after = RUNNER.model_state_sha256(model)

        self.assertNotEqual(fingerprint_before, fingerprint_after)
        for before, after in zip(parameters_before, model.parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_cache_provenance_accepts_exact_state_and_protocol(self):
        model = BatchNormOnly()
        cfg = self._cfg()
        payload = self._valid_payload(model, cfg)
        fingerprint = payload["base_model_state_sha256"]

        self.assertEqual(
            RUNNER.validate_swag_cache_provenance(payload, model, cfg),
            fingerprint,
        )

    def test_cache_provenance_rejects_legacy_without_opt_in(self):
        model = BatchNormOnly()
        cfg = self._cfg()
        with self.assertRaises(RuntimeError):
            RUNNER.validate_swag_cache_provenance({}, model, cfg)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RUNNER.validate_swag_cache_provenance(
                {}, model, cfg, allow_legacy=True
            )
        self.assertEqual(len(caught), 1)

    def test_cache_provenance_rejects_missing_required_fields(self):
        model = BatchNormOnly()
        cfg = self._cfg()
        valid = self._valid_payload(model, cfg)

        for field in valid:
            with self.subTest(field=field):
                payload = dict(valid)
                payload.pop(field)
                with self.assertRaises(RuntimeError):
                    RUNNER.validate_swag_cache_provenance(payload, model, cfg)

    def test_cache_provenance_rejects_hash_mismatch(self):
        model = BatchNormOnly()
        cfg = self._cfg()
        payload = self._valid_payload(model, cfg)
        payload["base_model_state_sha256"] = "not-the-map-state"

        with self.assertRaises(RuntimeError):
            RUNNER.validate_swag_cache_provenance(payload, model, cfg)
        with self.assertRaises(RuntimeError):
            RUNNER.validate_swag_cache_provenance(
                payload, model, cfg, allow_legacy=True
            )

    def test_cache_provenance_rejects_protocol_mismatches(self):
        model = BatchNormOnly()
        cfg = self._cfg()
        valid = self._valid_payload(model, cfg)
        mismatches = {
            "schema_version": 1,
            "base_model_source": "ensemble-member-4",
            "swag_variant": "full",
            "map_seed": cfg.seed + 1,
            "swag_epochs": cfg.swag_epochs + 1,
            "swag_lr": cfg.swag_lr * 2.0,
            "swag_momentum": cfg.momentum - 0.1,
            "swag_batch_size": cfg.batch_size // 2,
            "swag_num_workers": cfg.num_workers + 1,
        }

        for field, wrong_value in mismatches.items():
            with self.subTest(field=field):
                payload = dict(valid)
                payload[field] = wrong_value
                with self.assertRaises(RuntimeError):
                    RUNNER.validate_swag_cache_provenance(payload, model, cfg)
                # The legacy escape hatch must never accept contradictory
                # provenance supplied by a versioned cache.
                with self.assertRaises(RuntimeError):
                    RUNNER.validate_swag_cache_provenance(
                        payload, model, cfg, allow_legacy=True
                    )

    def test_cli_records_published_protocol_explicitly(self):
        argv = [
            "runner",
            "--methods",
            "swag",
            "--swag-samples",
            "20",
            "--swag-fixbn-batches",
            "20",
            "--swag-fixbn-mode",
            "rolling",
            "--swag-stats",
            "c100_swag_stats.pth",
            "--allow-legacy-swag-cache",
        ]
        with mock.patch.object(sys, "argv", argv):
            cfg = RUNNER.cfg_from_args(RUNNER.parse_args())
        self.assertEqual(cfg.swag_samples, 20)
        self.assertEqual(cfg.swag_fixbn_batches, 20)
        self.assertEqual(cfg.swag_fixbn_mode, "rolling")
        self.assertEqual(cfg.swag_stats, "c100_swag_stats.pth")
        self.assertTrue(cfg.allow_legacy_swag_cache)


class SnapshotTests(unittest.TestCase):
    def test_canonical_files_match_exported_snapshot(self):
        pairs = [
            ("cifar100_all_methods_iclr.py", "cifar100_all_methods_iclr.py"),
            ("cifar100_all_methods_base.py", "cifar100_all_methods_base.py"),
            ("cifar100c_eval_iclr.py", "cifar100c_eval_iclr.py"),
            ("make_paper_assets.py", "make_paper_assets.py"),
        ]
        snapshot = ROOT / "scripts" / "all_exported_code_snapshot"
        for canonical_name, snapshot_name in pairs:
            canonical = ROOT / "scripts" / canonical_name
            exported = snapshot / snapshot_name
            self.assertEqual(canonical.read_bytes(), exported.read_bytes(), canonical_name)


if __name__ == "__main__":
    unittest.main()
