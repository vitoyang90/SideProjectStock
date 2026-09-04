import unittest

import numpy as np
import pandas as pd

from train_v21 import (
    _policy_labels,
    _promotion_checks,
    _selection_table,
)


class V21Tests(unittest.TestCase):
    def test_quantile_threshold_comes_from_training_data(self):
        frame = pd.DataFrame({
            "return_5d": [-0.03, -0.01, 0.0, 0.02],
            "volatility_20": [0.01] * 4,
        })

        labels, thresholds = _policy_labels(
            frame, "train_quantile", quantile_threshold=0.015,
        )

        self.assertEqual(labels.tolist(), [0, 1, 1, 2])
        self.assertTrue((thresholds == 0.015).all())

    def test_selection_penalizes_unstable_candidate(self):
        rows = pd.DataFrame({
            "candidate_id": ["stable", "stable", "unstable", "unstable"],
            "label_policy": ["fixed"] * 4,
            "model_name": ["a", "a", "b", "b"],
            "family": ["xgboost"] * 4,
            "validation_year": [2021, 2022, 2021, 2022],
            "balanced_accuracy": [0.38, 0.38, 0.30, 0.48],
            "macro_f1": [0.37, 0.37, 0.29, 0.47],
            "mlogloss": [1.0, 1.0, 1.0, 1.0],
        })

        selected = _selection_table(rows).iloc[0]

        self.assertEqual(selected["candidate_id"], "stable")

    def test_promotion_requires_return_and_classification_checks(self):
        nested = pd.DataFrame({
            "delta_balanced_accuracy": [0.02, 0.01, -0.01],
            "delta_macro_f1": [0.02, 0.01, -0.01],
            "selected_net_return": [0.03, 0.02, 0.01],
            "baseline_net_return": [0.01, 0.01, 0.0],
        })

        checks = _promotion_checks(nested)

        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
