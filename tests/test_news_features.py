import unittest

import pandas as pd

from features.news import build_daily_news_features, normalize_news


class NewsFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trading = pd.DataFrame(
            {
                "date": ["2026-08-28", "2026-08-31", "2026-09-01"],
                "stock_id": ["2330", "2330", "2330"],
            }
        )

    def test_deduplicates_tracking_links_and_cleans_html(self) -> None:
        news = pd.DataFrame(
            {
                "date": ["2026-08-28 09:00:00", "2026-08-28 09:01:00"],
                "stock_id": ["2330", "2330"],
                "description": ["<b>內容</b>", "內容"],
                "link": [
                    "https://example.com/a?utm_source=x",
                    "https://example.com/a",
                ],
                "source": ["來源", "來源"],
                "title": ["<b>標題</b>", "標題"],
            }
        )
        normalized = normalize_news(news)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.loc[0, "title"], "標題")
        self.assertEqual(normalized.loc[0, "link"], "https://example.com/a")

    def test_moves_after_close_and_weekend_news_forward(self) -> None:
        news = pd.DataFrame(
            {
                "date": [
                    "2026-08-28 13:30:00",
                    "2026-08-28 13:31:00",
                    "2026-08-29 08:00:00",
                ],
                "stock_id": ["2330"] * 3,
                "description": ["a", "b", "c"],
                "link": ["https://x.test/1", "https://x.test/2", "https://x.test/3"],
                "source": ["s1", "s2", "s3"],
                "title": ["t1", "t2", "t3"],
            }
        )
        features, assigned = build_daily_news_features(news, self.trading)
        available = assigned["news_available_date"].dt.strftime("%Y-%m-%d").tolist()
        self.assertEqual(available, ["2026-08-28", "2026-08-31", "2026-08-31"])
        counts = features.set_index(features["date"].dt.strftime("%Y-%m-%d"))[
            "news_count"
        ].to_dict()
        self.assertEqual(counts["2026-08-28"], 1)
        self.assertEqual(counts["2026-08-31"], 2)
        self.assertEqual(counts["2026-09-01"], 0)
        self.assertEqual(int(features.iloc[-1]["news_count_3d"]), 3)
        self.assertTrue(pd.isna(features.iloc[0]["news_count_3d"]))
        self.assertTrue(pd.isna(features.iloc[-1]["news_count_5d"]))


if __name__ == "__main__":
    unittest.main()
