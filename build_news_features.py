import argparse
import json
import os
from pathlib import Path

import pandas as pd

from collectors.news import fetch_stock_news
from features.news import build_daily_news_features, merge_news_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the V5 leakage-aware Taiwan stock news data layer."
    )
    parser.add_argument("--stock-id", default="2330")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--market-close-time", default="13:30")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-large-range", action="store_true")
    parser.add_argument("--max-request-days", type=int, default=31)
    parser.add_argument(
        "--merge-dataset",
        action="store_true",
        help="Write a separate *_dataset_v5_ready.csv without replacing V2.",
    )
    return parser.parse_args()


def write_news_outputs(
    stock_id: str,
    start_date: str,
    end_date: str,
    news: pd.DataFrame,
    market_close_time: str = "13:30",
    merge_dataset: bool = False,
) -> dict:
    """Build V5 artifacts from a complete, already-fetched news range."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    price_path = raw_dir / f"{stock_id}_price.csv"
    if not price_path.exists():
        raise FileNotFoundError(
            f"Missing {price_path}. Run main.py to build the trading calendar first."
        )
    trading = pd.read_csv(price_path, dtype={"stock_id": str})
    trading_dates = pd.to_datetime(trading["date"], errors="raise").dt.normalize()
    coverage_start = pd.Timestamp(start_date).normalize()
    coverage_end = pd.Timestamp(end_date).normalize()
    next_trading_dates = trading_dates[trading_dates > coverage_end]
    feature_end = (
        next_trading_dates.min() if not next_trading_dates.empty else coverage_end
    )
    trading_window = trading[
        (trading_dates >= coverage_start) & (trading_dates <= feature_end)
    ].copy()
    if trading_window.empty:
        raise ValueError("The requested range does not overlap the price calendar")

    features, normalized = build_daily_news_features(
        news,
        trading_window,
        market_close_time=market_close_time,
    )

    raw_output = raw_dir / f"{stock_id}_news_normalized.csv"
    feature_output = processed_dir / f"{stock_id}_news_features.csv"
    normalized.to_csv(raw_output, index=False, encoding="utf-8-sig")
    features.to_csv(feature_output, index=False, encoding="utf-8-sig")
    quality_output = processed_dir / f"{stock_id}_news_data_quality.json"
    usable_articles = int(normalized["news_available_date"].notna().sum())
    quality = {
        "stock_id": str(stock_id),
        "requested_news_start": coverage_start.strftime("%Y-%m-%d"),
        "requested_news_end": coverage_end.strftime("%Y-%m-%d"),
        "feature_start": features["date"].min().strftime("%Y-%m-%d"),
        "feature_end": features["date"].max().strftime("%Y-%m-%d"),
        "market_close_time": market_close_time,
        "raw_articles": int(len(news)),
        "normalized_unique_articles": int(len(normalized)),
        "articles_with_usable_date": usable_articles,
        "trading_days_covered": int(len(features)),
        "sentiment_features": "not_built",
        "outside_coverage_policy": "missing_not_zero",
    }
    quality_output.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Normalized news saved to {raw_output} ({len(normalized)} rows)")
    print(f"Daily news features saved to {feature_output} ({len(features)} rows)")
    print(f"News data quality report saved to {quality_output}")

    if merge_dataset:
        dataset_path = processed_dir / f"{stock_id}_dataset.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Missing model dataset: {dataset_path}")
        dataset = pd.read_csv(dataset_path, dtype={"stock_id": str})
        merged = merge_news_features(dataset, features)
        merged_output = processed_dir / f"{stock_id}_dataset_v5_ready.csv"
        merged.to_csv(merged_output, index=False, encoding="utf-8-sig")
        print(
            f"V5-ready dataset saved to {merged_output}. "
            "The V2 champion and registry were not changed."
        )
    return quality


def main() -> None:
    args = parse_args()
    news = fetch_stock_news(
        stock_id=args.stock_id,
        start_date=args.start_date,
        end_date=args.end_date,
        token=os.environ.get("FINMIND_TOKEN", ""),
        refresh=args.refresh,
        max_request_days=args.max_request_days,
        allow_large_range=args.allow_large_range,
    )
    write_news_outputs(
        stock_id=args.stock_id,
        start_date=args.start_date,
        end_date=args.end_date,
        news=news,
        market_close_time=args.market_close_time,
        merge_dataset=args.merge_dataset,
    )


if __name__ == "__main__":
    main()
