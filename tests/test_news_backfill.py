import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backfill_news import find_contiguous_cache_start, next_backfill_chunk


class NewsBackfillTests(unittest.TestCase):
    def test_finds_gap_free_suffix_and_previous_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for item in pd.date_range("2026-06-17", "2026-08-17", freq="D"):
                (cache / f"{item:%Y-%m-%d}.csv").write_text(
                    "date,stock_id,description,link,source,title\n",
                    encoding="utf-8",
                )
            target_start = pd.Timestamp("2020-04-01")
            target_end = pd.Timestamp("2026-08-17")
            self.assertEqual(
                find_contiguous_cache_start(cache, target_start, target_end),
                pd.Timestamp("2026-06-17"),
            )
            self.assertEqual(
                next_backfill_chunk(cache, target_start, target_end, 31),
                (pd.Timestamp("2026-05-17"), pd.Timestamp("2026-06-16")),
            )

    def test_starts_from_target_end_when_cache_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            self.assertEqual(
                next_backfill_chunk(
                    cache,
                    pd.Timestamp("2026-01-01"),
                    pd.Timestamp("2026-01-31"),
                    10,
                ),
                (pd.Timestamp("2026-01-22"), pd.Timestamp("2026-01-31")),
            )


if __name__ == "__main__":
    unittest.main()
