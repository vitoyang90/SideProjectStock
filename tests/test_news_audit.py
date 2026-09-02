import unittest

import pandas as pd

from audit_news import find_zero_streaks, population_stability_index


class NewsAuditTests(unittest.TestCase):
    def test_finds_only_long_zero_streaks(self) -> None:
        features = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=8, freq="D"),
                "news_count": [0, 0, 1, 0, 0, 0, 1, 0],
            }
        )
        streaks = find_zero_streaks(features, minimum_length=3)
        self.assertEqual(len(streaks), 1)
        self.assertEqual(streaks.iloc[0]["start"], "2026-01-04")
        self.assertEqual(int(streaks.iloc[0]["trading_days"]), 3)

    def test_psi_is_zero_for_identical_distributions(self) -> None:
        values = pd.Series(range(1, 101), dtype=float)
        psi = population_stability_index(values, values)
        self.assertIsNotNone(psi)
        self.assertAlmostEqual(psi, 0.0, places=10)

    def test_psi_detects_distribution_shift(self) -> None:
        reference = pd.Series(range(1, 101), dtype=float)
        shifted = pd.Series(range(101, 201), dtype=float)
        psi = population_stability_index(reference, shifted)
        self.assertIsNotNone(psi)
        self.assertGreater(psi, 0.25)


if __name__ == "__main__":
    unittest.main()
