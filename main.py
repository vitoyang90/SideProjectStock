from pathlib import Path

from collectors.chip import (
    fetch_institutional_investors,
    fetch_margin_purchase_short_sale,
)
from collectors.fundamental import fetch_month_revenue, fetch_valuation
from collectors.market import (
    fetch_taiex,
    fetch_us_10y_yield,
    fetch_us_stock_price,
    fetch_usd_twd,
)
from collectors.price import fetch_stock_price
from features.chip import (
    build_institutional_features,
    build_margin_features,
    merge_chip_features,
)
from features.fundamental import (
    build_month_revenue_features,
    build_valuation_features,
    merge_fundamental_features,
)
from features.market import (
    build_taiex_features,
    build_us_10y_features,
    build_us_market_features,
    build_usd_twd_features,
    merge_market_features,
)
from features.technical import add_technical_features
from labeling.target import (
    add_5day_direction_label,
    add_5day_three_class_label,
)


stock_id = "2330"
start_date = "2018-01-01"
end_date = "2026-08-24"
revenue_history_start_date = "2016-01-01"
market_history_start_date = "2017-12-01"

price_df = fetch_stock_price(
    stock_id=stock_id,
    start_date=start_date,
    end_date=end_date,
)
institutional_df = fetch_institutional_investors(
    stock_id=stock_id,
    start_date=start_date,
    end_date=end_date,
)
margin_df = fetch_margin_purchase_short_sale(
    stock_id=stock_id,
    start_date=start_date,
    end_date=end_date,
)
month_revenue_df = fetch_month_revenue(
    stock_id=stock_id,
    start_date=revenue_history_start_date,
    end_date=end_date,
)
valuation_df = fetch_valuation(
    stock_id=stock_id,
    start_date=start_date,
    end_date=end_date,
)
taiex_df = fetch_taiex(
    start_date=market_history_start_date,
    end_date=end_date,
)
us_market_frames = {
    symbol: fetch_us_stock_price(
        symbol=symbol,
        start_date=market_history_start_date,
        end_date=end_date,
    )
    for symbol in ("SPY", "QQQ", "SOXX")
}
usd_twd_df = fetch_usd_twd(
    start_date=market_history_start_date,
    end_date=end_date,
)
us_10y_df = fetch_us_10y_yield(
    start_date=market_history_start_date,
    end_date=end_date,
)

raw_output_dir = Path("data/raw")
raw_output_dir.mkdir(parents=True, exist_ok=True)

raw_frames = {
    "price": price_df,
    "institutional": institutional_df,
    "margin": margin_df,
    "month_revenue": month_revenue_df,
    "valuation": valuation_df,
    "taiex": taiex_df,
    "us_spy": us_market_frames["SPY"],
    "us_qqq": us_market_frames["QQQ"],
    "us_soxx": us_market_frames["SOXX"],
    "usd_twd": usd_twd_df,
    "us_10y": us_10y_df,
}
for name, raw_df in raw_frames.items():
    raw_output_path = raw_output_dir / f"{stock_id}_{name}.csv"
    raw_df.to_csv(raw_output_path, index=False, encoding="utf-8-sig")
    print(f"Raw {name} data saved to {raw_output_path}")

institutional_features = build_institutional_features(institutional_df)
margin_features = build_margin_features(margin_df)
dataset_df = merge_chip_features(
    price_df,
    institutional_features,
    margin_features,
)
month_revenue_features = build_month_revenue_features(month_revenue_df)
valuation_features = build_valuation_features(valuation_df)
dataset_df = merge_fundamental_features(
    dataset_df,
    month_revenue_features,
    valuation_features,
)
taiex_features = build_taiex_features(taiex_df)
us_market_features = build_us_market_features(us_market_frames)
usd_twd_features = build_usd_twd_features(usd_twd_df)
us_10y_features = build_us_10y_features(us_10y_df)
dataset_df = merge_market_features(
    dataset_df,
    taiex_features,
    us_market_features,
    usd_twd_features,
    us_10y_features,
)
dataset_df = add_technical_features(dataset_df)
dataset_df = add_5day_direction_label(dataset_df)
dataset_df = add_5day_three_class_label(dataset_df, threshold=0.02)

processed_output_path = Path("data/processed") / f"{stock_id}_dataset.csv"
processed_output_path.parent.mkdir(parents=True, exist_ok=True)

preview_columns = [
    "date",
    "close",
    "foreign_net",
    "foreign_5d",
    "trust_net",
    "trust_5d",
    "dealer_net",
    "margin_balance",
    "margin_change",
    "short_balance",
    "revenue",
    "revenue_mom",
    "revenue_yoy",
    "pe",
    "pb",
    "dividend_yield",
    "taiex_return_1d",
    "taiex_return_5d",
    "spy_return_1d",
    "qqq_return_1d",
    "soxx_return_1d",
    "usd_twd",
    "usd_twd_change_1d",
    "us10y_yield",
    "us10y_change_1d",
    "rsi_14",
    "macd",
    "ma20_ratio",
    "atr_ratio",
    "volatility_20",
    "volume_ratio_20",
    "return_5d",
    "label",
    "target_3class",
]
print(dataset_df[preview_columns].tail())
print(dataset_df["label"].value_counts().sort_index())

processed_output_dir = Path("data/processed")
processed_output_dir.mkdir(parents=True, exist_ok=True)

processed_output_path = processed_output_dir / f"{stock_id}_dataset.csv"
dataset_df.to_csv(
    processed_output_path,
    index=False,
    encoding="utf-8-sig",
)

print(f"Model dataset saved to {processed_output_path}")
