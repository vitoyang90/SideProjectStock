import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from evaluate_forward import evaluate_prediction_file, summarize


class ForwardEvaluationTests(unittest.TestCase):
    def test_scores_only_after_fifth_future_trading_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "2330_predictions.csv"
            price_path = root / "2330_price.csv"
            pd.DataFrame(
                [
                    {
                        "prediction_date": "2026-01-02",
                        "recorded_at": "2026-01-02T18:00:00+08:00",
                        "stock_id": "2330",
                        "model_version": "test",
                        "close": 100,
                        "abstained": False,
                        "probability_down": 0.1,
                        "probability_neutral": 0.2,
                        "probability_up": 0.7,
                    }
                ]
            ).to_csv(prediction_path, index=False)
            pd.DataFrame(
                {
                    "date": pd.bdate_range("2026-01-02", periods=6),
                    "close": [100, 101, 102, 103, 104, 106],
                }
            ).to_csv(price_path, index=False)
            with patch(
                "evaluate_forward._cached_paths",
                return_value={"price": price_path},
            ):
                rows = evaluate_prediction_file(
                    prediction_path, datetime.fromisoformat("2026-01-09T20:00:00")
                )
            self.assertTrue(bool(rows.loc[0, "matured"]))
            self.assertAlmostEqual(float(rows.loc[0, "actual_return_5d"]), 0.06)
            self.assertEqual(int(rows.loc[0, "actual_class"]), 2)
            self.assertTrue(bool(rows.loc[0, "correct_emitted_signal"]))

    def test_empty_summary_waits_for_outcomes(self) -> None:
        result = summarize(pd.DataFrame())
        self.assertEqual(result["status"], "waiting_for_five_trading_days")
        self.assertEqual(result["matured_records"], 0)


if __name__ == "__main__":
    unittest.main()
