import unittest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from analyze import (
    _prediction_label,
    _view_from_score,
    record_forward_prediction,
    render_markdown,
    summarize_factors,
)


class AnalyzeTests(unittest.TestCase):
    def test_forward_record_is_idempotent_per_model_and_date(self):
        report = {
            "stock_id": "2330",
            "prediction_date": "2026-09-04",
            "model": {"version": "v2"},
            "market_snapshot": {"close": 100.0},
            "prediction": {
                "raw_top_class": "上漲",
                "trend": "不確定",
                "abstained": True,
                "confidence": 0.34,
                "probabilities": {"down": 0.33, "neutral": 0.33, "up": 0.34},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"

            record_forward_prediction(report, path)
            record_forward_prediction(report, path)

            recorded = pd.read_csv(path)
            self.assertEqual(len(recorded), 1)

    def test_factor_summary_separates_observations_from_model_prediction(self):
        row = pd.Series({
            "ma20_ratio": 0.03,
            "macd_hist": 1.0,
            "return_5d_past": 0.02,
            "foreign_5d": 1000,
            "trust_5d": 100,
            "institutional_total_net": 500,
            "rsi_14": 75,
            "atr_ratio": 0.02,
            "margin_change": 10,
        })

        summary = summarize_factors(row)

        self.assertEqual(summary["technical_view"], "偏多")
        self.assertEqual(summary["chip_view"], "偏多")
        self.assertTrue(any("過熱" in risk for risk in summary["risks"]))

    def test_view_thresholds(self):
        self.assertEqual(_view_from_score(2), "偏多")
        self.assertEqual(_view_from_score(1), "中性")
        self.assertEqual(_view_from_score(-2), "偏空")

    def test_prediction_abstains_when_probabilities_are_nearly_tied(self):
        class_id, trend, abstained, margin = _prediction_label(
            np.array([0.336, 0.327, 0.337])
        )

        self.assertEqual(class_id, 2)
        self.assertEqual(trend, "不確定")
        self.assertTrue(abstained)
        self.assertAlmostEqual(margin, 0.001)

    def test_prediction_emits_direction_when_confident(self):
        class_id, trend, abstained, _ = _prediction_label(
            np.array([0.15, 0.20, 0.65])
        )

        self.assertEqual(class_id, 2)
        self.assertEqual(trend, "上漲")
        self.assertFalse(abstained)

    def test_markdown_warns_when_data_is_stale(self):
        report = {
            "stock_id": "2330",
            "prediction_date": "2026-08-24",
            "model": {"target": "未來五個交易日"},
            "prediction": {
                "trend": "上漲",
                "confidence": 0.6,
                "probabilities": {"up": 0.6, "neutral": 0.3, "down": 0.1},
            },
            "factor_summary": {
                "technical_view": "偏多",
                "chip_view": "中性",
                "technical_observations": ["技術觀察"],
                "chip_observations": ["籌碼觀察"],
                "risks": ["測試風險"],
            },
            "stale_data_warning": True,
            "data_staleness_calendar_days": 11,
            "limitations": ["測試限制"],
        }

        markdown = render_markdown(report)

        self.assertIn("資料已落後請求日期 11 天", markdown)
        self.assertIn("上漲 60.0%", markdown)


if __name__ == "__main__":
    unittest.main()
