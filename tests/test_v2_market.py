import unittest

import pandas as pd

from train_v2_market import evaluate_gate


class V2MarketGateTests(unittest.TestCase):
    def test_gate_requires_three_winning_folds_and_better_return(self):
        rows = []
        for year, baseline, candidate in (
            (2021, 0.35, 0.36),
            (2022, 0.36, 0.38),
            (2023, 0.37, 0.39),
            (2024, 0.38, 0.37),
        ):
            rows.extend([
                {
                    "validation_year": year,
                    "version": "v2",
                    "balanced_accuracy": baseline,
                    "macro_f1": baseline,
                },
                {
                    "validation_year": year,
                    "version": "v2_market",
                    "balanced_accuracy": candidate,
                    "macro_f1": candidate,
                },
            ])
        folds = pd.DataFrame(rows)
        summary = pd.DataFrame({
            "version": ["v2", "v2_market"],
            "delta_balanced_accuracy_vs_v2": [0.0, 0.01],
            "delta_macro_f1_vs_v2": [0.0, 0.01],
        })
        backtest = pd.DataFrame({
            "strategy": ["v2", "v2_market"],
            "net_total_return": [0.10, 0.12],
        })

        checks, deltas = evaluate_gate(folds, summary, backtest)

        self.assertTrue(all(checks.values()))
        self.assertEqual(deltas["delta_macro_f1"].gt(0).sum(), 3)


if __name__ == "__main__":
    unittest.main()
