import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add backward-looking technical indicators without future data."""
    required_columns = {
        "stock_id",
        "date",
        "open",
        "max",
        "min",
        "close",
        "Trading_Volume",
    }
    _require_columns(df, required_columns)

    technical = df.copy().sort_values(["stock_id", "date"])
    numeric_columns = [
        "open",
        "max",
        "min",
        "close",
        "Trading_Volume",
    ]
    technical[numeric_columns] = technical[numeric_columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    grouped = technical.groupby("stock_id", group_keys=False)
    technical["return_1d"] = grouped["close"].pct_change(1)
    technical["return_5d_past"] = grouped["close"].pct_change(5)

    ma5 = grouped["close"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    ma20 = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    technical["ma5_ratio"] = technical["close"] / ma5 - 1
    technical["ma20_ratio"] = technical["close"] / ma20 - 1

    delta = grouped["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.groupby(technical["stock_id"]).transform(
        lambda values: values.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()
    )
    average_loss = losses.groupby(technical["stock_id"]).transform(
        lambda values: values.ewm(
            alpha=1 / 14,
            min_periods=14,
            adjust=False,
        ).mean()
    )
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    technical["rsi_14"] = 100 - (100 / (1 + relative_strength))
    technical.loc[
        (average_loss == 0) & (average_gain > 0),
        "rsi_14",
    ] = 100

    ema12 = grouped["close"].transform(
        lambda values: values.ewm(span=12, adjust=False).mean()
    )
    ema26 = grouped["close"].transform(
        lambda values: values.ewm(span=26, adjust=False).mean()
    )
    technical["macd"] = ema12 - ema26
    technical["macd_signal"] = technical.groupby("stock_id")[
        "macd"
    ].transform(lambda values: values.ewm(span=9, adjust=False).mean())
    technical["macd_hist"] = (
        technical["macd"] - technical["macd_signal"]
    )

    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            technical["max"] - technical["min"],
            (technical["max"] - previous_close).abs(),
            (technical["min"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    technical["atr_14"] = true_range.groupby(
        technical["stock_id"]
    ).transform(lambda values: values.rolling(14, min_periods=14).mean())
    technical["atr_ratio"] = technical["atr_14"] / technical["close"]

    technical["volatility_20"] = technical.groupby("stock_id")[
        "return_1d"
    ].transform(lambda values: values.rolling(20, min_periods=20).std())
    average_volume_20 = grouped["Trading_Volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    technical["volume_ratio_20"] = (
        technical["Trading_Volume"] / average_volume_20
    )

    return technical.reset_index(drop=True)
