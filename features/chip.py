import pandas as pd


INSTITUTION_CATEGORIES = {
    "Foreign_Investor": "foreign",
    "Foreign_Dealer_Self": "foreign",
    "Investment_Trust": "trust",
    "Dealer": "dealer",
    "Dealer_self": "dealer",
    "Dealer_Hedging": "dealer",
}


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")


def build_institutional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create institutional net-buy and rolling-sum features."""
    _require_columns(df, {"date", "stock_id", "name", "buy", "sell"})

    institutional = df.copy()
    institutional["category"] = institutional["name"].map(
        INSTITUTION_CATEGORIES
    )

    unknown_names = sorted(
        institutional.loc[institutional["category"].isna(), "name"]
        .dropna()
        .unique()
    )
    if unknown_names:
        raise ValueError(
            f"Unknown institutional investor categories: {unknown_names}"
        )

    institutional["net"] = (
        pd.to_numeric(institutional["buy"], errors="raise")
        - pd.to_numeric(institutional["sell"], errors="raise")
    )

    features = (
        institutional.pivot_table(
            index=["date", "stock_id"],
            columns="category",
            values="net",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=["foreign", "trust", "dealer"], fill_value=0)
        .rename(
            columns={
                "foreign": "foreign_net",
                "trust": "trust_net",
                "dealer": "dealer_net",
            }
        )
        .reset_index()
        .sort_values(["stock_id", "date"])
    )

    net_columns = ["foreign_net", "trust_net", "dealer_net"]
    features["institutional_total_net"] = features[net_columns].sum(axis=1)

    for column in net_columns:
        prefix = column.removesuffix("_net")
        for window in (3, 5):
            features[f"{prefix}_{window}d"] = features.groupby(
                "stock_id"
            )[column].transform(
                lambda values: values.rolling(
                    window=window,
                    min_periods=window,
                ).sum()
            )

    return features.reset_index(drop=True)


def build_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create margin and short balance/change features."""
    required_columns = {
        "date",
        "stock_id",
        "MarginPurchaseTodayBalance",
        "MarginPurchaseYesterdayBalance",
        "ShortSaleTodayBalance",
        "ShortSaleYesterdayBalance",
    }
    _require_columns(df, required_columns)

    margin = df[list(required_columns)].copy()
    balance_columns = required_columns.difference({"date", "stock_id"})
    margin[list(balance_columns)] = margin[list(balance_columns)].apply(
        pd.to_numeric,
        errors="raise",
    )

    margin["margin_balance"] = margin["MarginPurchaseTodayBalance"]
    margin["short_balance"] = margin["ShortSaleTodayBalance"]
    margin["margin_change"] = (
        margin["MarginPurchaseTodayBalance"]
        - margin["MarginPurchaseYesterdayBalance"]
    )
    margin["short_change"] = (
        margin["ShortSaleTodayBalance"]
        - margin["ShortSaleYesterdayBalance"]
    )

    return (
        margin[
            [
                "date",
                "stock_id",
                "margin_balance",
                "short_balance",
                "margin_change",
                "short_change",
            ]
        ]
        .sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )


def merge_chip_features(
    price_df: pd.DataFrame,
    institutional_features: pd.DataFrame,
    margin_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge price and chip features by trading date and stock ID."""
    keys = ["date", "stock_id"]
    for frame in (price_df, institutional_features, margin_features):
        _require_columns(frame, set(keys))

    merged = price_df.merge(
        institutional_features,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        margin_features,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    return merged.sort_values(keys).reset_index(drop=True)
