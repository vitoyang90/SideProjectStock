import unittest

import pandas as pd

from train_v6_disclosures import evaluate_gate


class V6DisclosureGateTests(unittest.TestCase):
    def test_gate_requires_broad_stock_improvement(self) -> None:
        rows = []
        backtest = []
        for index, stock_id in enumerate(("2330", "2454", "2317", "2308", "2881")):
            delta = 0.01 if index < 2 else -0.001
            rows.extend(
                [
                    {
                        "stock_id": stock_id,
                        "version": "v2_matched",
                        "mean_balanced_accuracy": 0.35,
                        "mean_macro_f1": 0.34,
                    },
                    {
                        "stock_id": stock_id,
                        "version": "v6_disclosure",
                        "mean_balanced_accuracy": 0.35 + delta,
                        "mean_macro_f1": 0.34 + delta,
                    },
                ]
            )
            backtest.extend(
                [
                    {"stock_id": stock_id, "version": "v2_matched", "net_total_return": 0.1},
                    {"stock_id": stock_id, "version": "v6_disclosure", "net_total_return": 0.11},
                ]
            )
        checks, _ = evaluate_gate(pd.DataFrame(rows), pd.DataFrame(backtest))
        self.assertFalse(checks["balanced_accuracy_wins_at_least_3_of_5_stocks"])
        self.assertFalse(checks["macro_f1_wins_at_least_3_of_5_stocks"])


if __name__ == "__main__":
    unittest.main()
