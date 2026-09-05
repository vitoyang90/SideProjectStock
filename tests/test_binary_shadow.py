import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analyze_binary_candidate import record_shadow_prediction
from evaluate_v22_forward import summarize


class BinaryShadowTests(unittest.TestCase):
    def test_record_is_idempotent(self) -> None:
        report = {
            "stock_id": "2330",
            "prediction_date": "2026-09-04",
            "model": {"artifact_id": "candidate"},
            "market_snapshot": {"close": 100.0},
            "prediction": {
                "probability_up": 0.6,
                "threshold": 0.45,
                "emitted": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.csv"
            record_shadow_prediction(report, path)
            record_shadow_prediction(report, path)
            self.assertEqual(len(pd.read_csv(path)), 1)

    def test_summary_requires_forward_sample(self) -> None:
        result = summarize(pd.DataFrame())
        self.assertFalse(result["promotion_ready"])
        self.assertEqual(result["status"], "waiting_for_five_trading_days")


if __name__ == "__main__":
    unittest.main()
