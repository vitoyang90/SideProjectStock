import pandas as pd


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")


def build_taiex_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create TAIEX level and return features."""
    _require_columns(df, {"date", "price"})

    taiex = df[["date", "price"]].copy()
    taiex = taiex.rename(
        columns={"date": "taiex_source_date", "price": "taiex_index"}
    )
    taiex["taiex_source_date"] = pd.to_datetime(
        taiex["taiex_source_date"],
        errors="raise",
    )
    taiex["taiex_index"] = pd.to_numeric(
        taiex["taiex_index"],
        errors="raise",
    )
    taiex = taiex.sort_values("taiex_source_date").drop_duplicates(
        "taiex_source_date",
        keep="last",
    )
    taiex["taiex_return_1d"] = taiex["taiex_index"].pct_change(1)
    taiex["taiex_return_5d"] = taiex["taiex_index"].pct_change(5)

    return taiex.reset_index(drop=True)


def build_us_market_features(
    market_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create US ETF price/return features available in Taiwan next day."""
    combined = None

    for symbol, frame in market_frames.items():
        _require_columns(frame, {"date", "Adj_Close"})
        prefix = symbol.lower()

        market = frame[["date", "Adj_Close"]].copy()
        market = market.rename(
            columns={
                "date": "us_market_date",
                "Adj_Close": f"{prefix}_close",
            }
        )
        market["us_market_date"] = pd.to_datetime(
            market["us_market_date"],
            errors="raise",
        )
        market[f"{prefix}_close"] = pd.to_numeric(
            market[f"{prefix}_close"],
            errors="raise",
        )
        market = market.sort_values("us_market_date").drop_duplicates(
            "us_market_date",
            keep="last",
        )
        market[f"{prefix}_return_1d"] = market[
            f"{prefix}_close"
        ].pct_change(1)
        market[f"{prefix}_return_5d"] = market[
            f"{prefix}_close"
        ].pct_change(5)
        market["us_available_date"] = (
            market["us_market_date"] + pd.Timedelta(days=1)
        )

        if combined is None:
            combined = market
        else:
            combined = combined.merge(
                market,
                on=["us_market_date", "us_available_date"],
                how="outer",
                validate="one_to_one",
            )

    if combined is None:
        raise ValueError("At least one US market DataFrame is required")

    return combined.sort_values("us_available_date").reset_index(drop=True)


def build_usd_twd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create USD/TWD midpoint and daily-change features."""
    _require_columns(df, {"date", "spot_buy", "spot_sell"})

    exchange = df[["date", "spot_buy", "spot_sell"]].copy()
    exchange["spot_buy"] = pd.to_numeric(
        exchange["spot_buy"],
        errors="raise",
    )
    exchange["spot_sell"] = pd.to_numeric(
        exchange["spot_sell"],
        errors="raise",
    )
    exchange["usd_twd_source_date"] = pd.to_datetime(
        exchange["date"],
        errors="raise",
    )
    exchange["usd_twd"] = (
        exchange["spot_buy"] + exchange["spot_sell"]
    ) / 2
    exchange = exchange.sort_values("usd_twd_source_date").drop_duplicates(
        "usd_twd_source_date",
        keep="last",
    )
    exchange["usd_twd_change_1d"] = exchange["usd_twd"].pct_change(1)

    return exchange[
        ["usd_twd_source_date", "usd_twd", "usd_twd_change_1d"]
    ].reset_index(drop=True)


def build_us_10y_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create US 10-year yield features available in Taiwan next day."""
    _require_columns(df, {"date", "value"})

    bonds = df[["date", "value"]].copy()
    bonds = bonds.rename(
        columns={"date": "us10y_source_date", "value": "us10y_yield"}
    )
    bonds["us10y_source_date"] = pd.to_datetime(
        bonds["us10y_source_date"],
        errors="raise",
    )
    bonds["us10y_yield"] = pd.to_numeric(
        bonds["us10y_yield"],
        errors="raise",
    )
    bonds = bonds.sort_values("us10y_source_date").drop_duplicates(
        "us10y_source_date",
        keep="last",
    )
    bonds["us10y_change_1d"] = bonds["us10y_yield"].diff(1)
    bonds["us10y_available_date"] = (
        bonds["us10y_source_date"] + pd.Timedelta(days=1)
    )

    return bonds.reset_index(drop=True)


def merge_market_features(
    daily_df: pd.DataFrame,
    taiex_features: pd.DataFrame,
    us_market_features: pd.DataFrame,
    usd_twd_features: pd.DataFrame,
    us_10y_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge market and macro features using their known-at dates."""
    _require_columns(daily_df, {"date", "stock_id"})

    merged = daily_df.copy()
    merged["date"] = pd.to_datetime(merged["date"], errors="raise")

    merged = merged.merge(
        taiex_features,
        left_on="date",
        right_on="taiex_source_date",
        how="left",
        validate="many_to_one",
    )

    asof_frames = (
        (us_market_features, "us_available_date"),
        (usd_twd_features, "usd_twd_source_date"),
        (us_10y_features, "us10y_available_date"),
    )
    for frame, available_column in asof_frames:
        merged = pd.merge_asof(
            merged.sort_values("date"),
            frame.sort_values(available_column),
            left_on="date",
            right_on=available_column,
            direction="backward",
            allow_exact_matches=True,
        )

    datetime_columns = [
        "date",
        "taiex_source_date",
        "us_market_date",
        "us_available_date",
        "usd_twd_source_date",
        "us10y_source_date",
        "us10y_available_date",
    ]
    for column in datetime_columns:
        merged[column] = merged[column].dt.strftime("%Y-%m-%d")

    return merged.sort_values(["stock_id", "date"]).reset_index(drop=True)
