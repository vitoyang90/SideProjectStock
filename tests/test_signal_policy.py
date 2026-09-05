import unittest

import numpy as np
import pandas as pd

from optimize_signal_policy import apply_policy, promotion_checks


class SignalPolicyTests(unittest.TestCase):
    def test_apply_policy_uses_confidence_and_margin(self) -> None:
        probabilities = np.array(
            [[0.50, 0.30, 0.20], [0.40, 0.39, 0.21], [0.34, 0.33, 0.33]]
        )
        predicted, emitted, margins = apply_policy(probabilities, 0.40, 0.05)
        self.assertEqual(predicted.tolist(), [0, 0, 0])
        self.assertEqual(emitted.tolist(), [True, False, False])
        self.assertAlmostEqual(float(margins[0]), 0.20)

    def test_promotion_requires_enough_calls_and_return(self) -> None:
        nested = pd.DataFrame(
            {
                "selected_calls": [10, 10, 10],
                "selected_accuracy": [0.60, 0.60, 0.60],
                "selected_net_return": [0.01, 0.01, 0.01],
                "baseline_calls": [10, 10, 10],
                "baseline_accuracy": [0.40, 0.40, 0.40],
                "baseline_net_return": [0.02, 0.02, 0.02],
                "validation_rows": [200, 200, 200],
                "delta_accuracy": [0.20, 0.20, 0.20],
            }
        )
        checks = promotion_checks(nested)
        self.assertFalse(checks["at_least_40_emitted_signals"])
        self.assertFalse(checks["net_return_not_worse"])


if __name__ == "__main__":
    unittest.main()
