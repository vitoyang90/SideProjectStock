import unittest

import pandas as pd

from replay_v22_holdout import eligible_training_rows, replay_gate


class V22HoldoutReplayTests(unittest.TestCase):
    def test_training_excludes_labels_maturing_on_or_after_prediction(self) -> None:
        data = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=12, freq="D"),
                "return_5d": range(12),
            }
        )
        training = eligible_training_rows(data, prediction_position=10)
        self.assertEqual(len(training), 5)
        self.assertEqual(training.iloc[-1]["date"], pd.Timestamp("2025-01-05"))

    def test_replay_gate_requires_positive_candidate_return(self) -> None:
        candidate = {
            "rows": 40,
            "balanced_accuracy": 0.60,
            "f1": 0.50,
            "precision": 0.55,
            "net_return": -0.01,
            "signals": 20,
            "coverage": 0.5,
        }
        baseline = {
            "balanced_accuracy": 0.50,
            "f1": 0.30,
            "precision": 0.40,
            "net_return": -0.10,
        }
        naive = {
            "balanced_accuracy": 0.50,
            "net_return": 0.05,
        }
        annual = pd.DataFrame({"delta_balanced_accuracy": [0.02, 0.03]})
        checks = replay_gate(candidate, baseline, naive, annual)
        self.assertFalse(checks["candidate_net_return_positive"])

    def test_replay_gate_rejects_always_on_signal(self) -> None:
        candidate = {
            "rows": 40,
            "balanced_accuracy": 0.60,
            "f1": 0.50,
            "precision": 0.55,
            "net_return": 0.10,
            "signals": 40,
            "coverage": 1.0,
        }
        baseline = {
            "balanced_accuracy": 0.50,
            "f1": 0.30,
            "precision": 0.40,
            "net_return": 0.02,
        }
        naive = {
            "balanced_accuracy": 0.50,
            "net_return": 0.10,
        }
        annual = pd.DataFrame({"delta_balanced_accuracy": [0.02, 0.03]})
        checks = replay_gate(candidate, baseline, naive, annual)
        self.assertFalse(checks["candidate_emits_both_signal_classes"])
        self.assertFalse(checks["candidate_coverage_between_5_and_95_percent"])
        self.assertFalse(checks["net_return_beats_naive_always_up"])


if __name__ == "__main__":
    unittest.main()
