import unittest

import pandas as pd

from train_v22_binary import promotion_checks


class V22BinaryTests(unittest.TestCase):
    def test_gate_rejects_large_single_year_f1_loss(self) -> None:
        nested = pd.DataFrame(
            {
                "delta_balanced_accuracy": [0.02, 0.02, 0.02],
                "delta_f1": [0.03, 0.03, -0.08],
                "binary_net_return": [0.05, 0.05, 0.05],
                "baseline_net_return": [0.01, 0.01, 0.01],
            }
        )
        checks = promotion_checks(nested)
        self.assertFalse(checks["worst_f1_delta_at_least_minus_0_05"])


if __name__ == "__main__":
    unittest.main()
