import unittest

import pandas as pd

from audit_disclosures import REQUIRED_HISTORY_YEARS, evaluate_readiness
from collectors.disclosures import normalize_twse_disclosures, parse_roc_date
from features.disclosures import build_daily_disclosure_features


def sample_row() -> dict:
    return {
        "出表日期": "1150904",
        "發言日期": "1150903",
        "發言時間": "183824",
        "公司代號": "2454",
        "公司名稱": "聯發科",
        "主旨 ": "公告營收成長",
        "符合條款": "第36款",
        "事實發生日": "1150903",
        "說明": "營收增加 12%，但未來仍可能有風險。",
    }


class DisclosureCollectorTests(unittest.TestCase):
    def test_parses_roc_date(self) -> None:
        self.assertEqual(parse_roc_date("115/09/04"), pd.Timestamp("2026-09-04"))

    def test_normalizes_trimmed_title_key_and_deduplicates_stably(self) -> None:
        observed = pd.Timestamp("2026-09-05 10:00:00")
        first = normalize_twse_disclosures([sample_row()], observed, {"2454"})
        second = normalize_twse_disclosures([sample_row()], observed, {"2454"})
        self.assertEqual(first.loc[0, "title"], "公告營收成長")
        self.assertEqual(first.loc[0, "disclosure_id"], second.loc[0, "disclosure_id"])
        self.assertEqual(first.loc[0, "published_at"], pd.Timestamp("2026-09-03 18:38:24"))


class DisclosureFeatureTests(unittest.TestCase):
    def test_uses_first_observation_to_prevent_retrospective_leakage(self) -> None:
        normalized = normalize_twse_disclosures(
            [sample_row()], pd.Timestamp("2026-09-05 10:00:00"), {"2454"}
        )
        trading = pd.DataFrame(
            {
                "date": ["2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09", "2026-09-10"],
                "stock_id": ["2454"] * 5,
            }
        )
        features, assigned = build_daily_disclosure_features(normalized, trading)
        self.assertEqual(
            assigned.loc[0, "disclosure_available_date"], pd.Timestamp("2026-09-07")
        )
        counts = features.set_index(features["date"].dt.strftime("%Y-%m-%d"))[
            "disclosure_count"
        ].to_dict()
        self.assertEqual(counts["2026-09-04"], 0)
        self.assertEqual(counts["2026-09-07"], 1)
        event_day = features[features["date"].eq(pd.Timestamp("2026-09-07"))].iloc[0]
        self.assertGreater(int(event_day["disclosure_positive_term_count"]), 0)
        self.assertGreater(int(event_day["disclosure_uncertainty_term_count"]), 0)


class DisclosureAuditTests(unittest.TestCase):
    def test_requires_every_history_year_even_when_row_count_is_high(self) -> None:
        normalized = normalize_twse_disclosures(
            [sample_row()], pd.Timestamp("2026-09-05 10:00:00"), {"2454"}
        )
        repeated = pd.concat([normalized] * 60, ignore_index=True)
        repeated["disclosure_id"] = [str(index) for index in range(len(repeated))]
        repeated["availability_basis"] = "official_archive_published_at"
        result = evaluate_readiness(
            {"2454": repeated},
            pd.DataFrame(),
            {"runs": [{"stock_id": "2454", "year": REQUIRED_HISTORY_YEARS[0], "completed": True}]},
        )
        self.assertFalse(result["per_stock"]["2454"]["ready"])


if __name__ == "__main__":
    unittest.main()
