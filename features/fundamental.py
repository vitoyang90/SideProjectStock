import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")


def build_month_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create monthly revenue growth features and a conservative usable date."""
    required_columns = {
        "date",
        "stock_id",
        "revenue",
        "revenue_month",
        "revenue_year",
    }
    _require_columns(df, required_columns)

    revenue = df[list(required_columns)].copy()
    revenue["revenue"] = pd.to_numeric(revenue["revenue"], errors="raise")
    revenue["revenue_month"] = pd.to_numeric(
        revenue["revenue_month"],
        errors="raise",
    )
    revenue["revenue_year"] = pd.to_numeric(
        revenue["revenue_year"],
        errors="raise",
    )
    revenue["revenue_period_date"] = revenue["date"]

    revenue = (
        revenue.sort_values(["stock_id", "revenue_year", "revenue_month"])
        .drop_duplicates(
            subset=["stock_id", "revenue_year", "revenue_month"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    grouped_revenue = revenue.groupby("stock_id")["revenue"]
    revenue["revenue_mom"] = grouped_revenue.pct_change(periods=1)
    revenue["revenue_yoy"] = grouped_revenue.pct_change(periods=12)
    revenue[["revenue_mom", "revenue_yoy"]] = revenue[
        ["revenue_mom", "revenue_yoy"]
    ].replace([np.inf, -np.inf], np.nan)

    # Historical FinMind rows do not contain the official announcement date.
    # Use the 11th calendar day as a conservative availability assumption.
    revenue["revenue_available_date"] = (
        pd.to_datetime(revenue["date"], errors="raise")
        + pd.Timedelta(days=10)
    )

    return revenue[
        [
            "stock_id",
            "revenue_period_date",
            "revenue_available_date",
            "revenue_year",
            "revenue_month",
            "revenue",
            "revenue_mom",
            "revenue_yoy",
        ]
    ]


def build_valuation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and normalize daily valuation features."""
    _require_columns(
        df,
        {"date", "stock_id", "PER", "PBR", "dividend_yield"},
    )

    valuation = df[
        ["date", "stock_id", "PER", "PBR", "dividend_yield"]
    ].copy()
    valuation = valuation.rename(columns={"PER": "pe", "PBR": "pb"})

    for column in ("pe", "pb", "dividend_yield"):
        valuation[column] = pd.to_numeric(
            valuation[column],
            errors="coerce",
        )

    return (
        valuation.drop_duplicates(subset=["date", "stock_id"], keep="last")
        .sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )


def merge_fundamental_features(
    daily_df: pd.DataFrame,
    revenue_features: pd.DataFrame,
    valuation_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily valuation and point-in-time monthly revenue features."""
    keys = ["date", "stock_id"]
    for frame in (daily_df, valuation_features):
        _require_columns(frame, set(keys))
    _require_columns(
        revenue_features,
        {"stock_id", "revenue_available_date"},
    )

    merged = daily_df.merge(
        valuation_features,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    merged["date"] = pd.to_datetime(merged["date"], errors="raise")

    revenue = revenue_features.copy()
    revenue["revenue_available_date"] = pd.to_datetime(
        revenue["revenue_available_date"],
        errors="raise",
    )

    merged = pd.merge_asof(
        merged.sort_values(["date", "stock_id"]),
        revenue.sort_values(["revenue_available_date", "stock_id"]),
        left_on="date",
        right_on="revenue_available_date",
        by="stock_id",
        direction="backward",
        allow_exact_matches=True,
    )

    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    merged["revenue_available_date"] = merged[
        "revenue_available_date"
    ].dt.strftime("%Y-%m-%d")

    return merged.sort_values(keys).reset_index(drop=True)
