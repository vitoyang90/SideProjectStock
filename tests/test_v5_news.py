import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from train_v5_news import _update_registry, evaluate_gate


class V5NewsGateTests(unittest.TestCase):
    def test_registry_update_accepts_null_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "registry.json"
            registry_path.write_text(
                json.dumps({"candidate": None, "rejected": []}),
                encoding="utf-8",
            )
            decision = {
                "passes_gate": False,
                "gate_checks": {"mean_macro_f1_improved": False},
            }

            with patch("train_v5_news.REGISTRY_PATH", registry_path):
                _update_registry(decision)

            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertIsNone(registry["candidate"])
            self.assertEqual(registry["rejected"][0]["version"], "v5_news")

    def test_gate_passes_only_when_all_predeclared_checks_pass(self):
        folds = pd.DataFrame({
            "validation_year": [2022, 2022, 2023, 2023, 2024, 2024],
            "version": [
                "v2_matched", "v5_news", "v2_matched",
                "v5_news", "v2_matched", "v5_news",
            ],
            "balanced_accuracy": [0.35, 0.36, 0.36, 0.38, 0.37, 0.36],
            "macro_f1": [0.34, 0.35, 0.35, 0.37, 0.36, 0.35],
        })
        summary = pd.DataFrame({
            "version": ["v2_matched", "v5_news"],
            "delta_balanced_accuracy_vs_v2": [0.0, 0.006],
            "delta_macro_f1_vs_v2": [0.0, 0.007],
        })
        backtest = pd.DataFrame({
            "strategy": ["v2_matched", "v5_news"],
            "net_total_return": [0.10, 0.12],
        })

        checks, deltas = evaluate_gate(folds, summary, backtest)

        self.assertTrue(all(checks.values()))
        self.assertEqual(deltas["delta_balanced_accuracy"].gt(0).sum(), 2)

    def test_gate_rejects_large_single_year_degradation(self):
        folds = pd.DataFrame({
            "validation_year": [2022, 2022, 2023, 2023, 2024, 2024],
            "version": [
                "v2_matched", "v5_news", "v2_matched",
                "v5_news", "v2_matched", "v5_news",
            ],
            "balanced_accuracy": [0.35, 0.40, 0.36, 0.41, 0.42, 0.38],
            "macro_f1": [0.34, 0.39, 0.35, 0.40, 0.41, 0.37],
        })
        summary = pd.DataFrame({
            "version": ["v2_matched", "v5_news"],
            "delta_balanced_accuracy_vs_v2": [0.0, 0.02],
            "delta_macro_f1_vs_v2": [0.0, 0.02],
        })
        backtest = pd.DataFrame({
            "strategy": ["v2_matched", "v5_news"],
            "net_total_return": [0.10, 0.20],
        })

        checks, _ = evaluate_gate(folds, summary, backtest)

        self.assertFalse(
            checks["worst_balanced_accuracy_delta_at_least_minus_0_03"]
        )
        self.assertFalse(checks["worst_macro_f1_delta_at_least_minus_0_03"])


if __name__ == "__main__":
    unittest.main()
