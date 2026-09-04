import re

import numpy as np
import pandas as pd


POSITIVE_TERMS = ("成長", "增加", "提升", "創新高", "獲利", "轉盈", "上調", "優於")
NEGATIVE_TERMS = ("下滑", "減少", "衰退", "虧損", "轉虧", "減損", "停工", "違約", "罰款", "下調")
UNCERTAINTY_TERMS = ("可能", "預計", "尚未", "不確定", "風險", "估計", "視情況", "尚待")
FINANCIAL_TERMS = ("營收", "毛利", "獲利", "盈餘", "每股", "資本支出", "財務報告", "現金流")
DISCLOSURE_FEATURE_COLUMNS = [
    "disclosure_count",
    "disclosure_after_close_count",
    "disclosure_text_chars_mean",
    "disclosure_numeric_token_count",
    "disclosure_positive_term_count",
    "disclosure_negative_term_count",
    "disclosure_uncertainty_term_count",
    "disclosure_financial_term_count",
    "disclosure_net_tone",
    "disclosure_count_5d",
    "disclosure_count_20d",
    "disclosure_net_tone_5d",
    "disclosure_days_since_update",
]
_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d[\d,]*)(?:\.\d+)?%?")


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")


def _term_count(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term.lower()) for term in terms)


def assign_disclosure_available_dates(
    disclosures: pd.DataFrame,
    trading_dates: pd.Series,
    market_close_time: str = "13:30",
) -> pd.DataFrame:
    """Map disclosures using max(source publish time, first observed time)."""
    _require_columns(disclosures, {"published_at", "first_observed_at"})
    hours, minutes = map(int, market_close_time.split(":"))
    if hours not in range(24) or minutes not in range(60):
        raise ValueError("market_close_time must use valid HH:MM notation")
    output = disclosures.copy()
    published = pd.to_datetime(output["published_at"], errors="raise")
    observed = pd.to_datetime(output["first_observed_at"], errors="raise")
    output["known_at"] = pd.concat([published, observed], axis=1).max(axis=1)

    calendar = np.array(
        sorted(pd.to_datetime(trading_dates, errors="raise").dt.normalize().unique()),
        dtype="datetime64[ns]",
    )
    if not len(calendar):
        raise ValueError("At least one trading date is required")
    known_day = output["known_at"].dt.normalize()
    close_delta = pd.Timedelta(hours=hours, minutes=minutes)
    output["disclosure_after_close"] = output["known_at"] - known_day > close_delta
    earliest_day = known_day + pd.to_timedelta(
        output["disclosure_after_close"].astype(int), unit="D"
    )
    positions = np.searchsorted(
        calendar, earliest_day.to_numpy(dtype="datetime64[ns]"), side="left"
    )
    available = np.full(len(output), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = positions < len(calendar)
    available[valid] = calendar[positions[valid]]
    output["disclosure_available_date"] = pd.to_datetime(available)
    return output


def build_daily_disclosure_features(
    disclosure_df: pd.DataFrame,
    trading_df: pd.DataFrame,
    market_close_time: str = "13:30",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic, leakage-aware text features for a covered window."""
    _require_columns(trading_df, {"date", "stock_id"})
    _require_columns(
        disclosure_df,
        {
            "disclosure_id",
            "published_at",
            "first_observed_at",
            "stock_id",
            "title",
            "body",
        },
    )
    trading = trading_df[["date", "stock_id"]].copy()
    trading["date"] = pd.to_datetime(trading["date"], errors="raise").dt.normalize()
    trading["stock_id"] = trading["stock_id"].astype(str)
    trading = trading.drop_duplicates().sort_values(["stock_id", "date"])
    disclosures = disclosure_df.drop_duplicates("disclosure_id", keep="first").copy()
    disclosures["stock_id"] = disclosures["stock_id"].astype(str)

    assigned_parts: list[pd.DataFrame] = []
    aggregate_parts: list[pd.DataFrame] = []
    for stock_id, stock_days in trading.groupby("stock_id", sort=False):
        stock_rows = disclosures[disclosures["stock_id"] == stock_id].copy()
        if stock_rows.empty:
            continue
        assigned = assign_disclosure_available_dates(
            stock_rows, stock_days["date"], market_close_time
        )
        text = assigned["title"].fillna("") + " " + assigned["body"].fillna("")
        assigned["text_chars"] = text.str.len()
        assigned["numeric_token_count"] = text.map(lambda item: len(_NUMBER.findall(item)))
        assigned["positive_term_count"] = text.map(lambda item: _term_count(item, POSITIVE_TERMS))
        assigned["negative_term_count"] = text.map(lambda item: _term_count(item, NEGATIVE_TERMS))
        assigned["uncertainty_term_count"] = text.map(lambda item: _term_count(item, UNCERTAINTY_TERMS))
        assigned["financial_term_count"] = text.map(lambda item: _term_count(item, FINANCIAL_TERMS))
        tone_denominator = (
            assigned["positive_term_count"] + assigned["negative_term_count"]
        ).clip(lower=1)
        assigned["net_tone"] = (
            assigned["positive_term_count"] - assigned["negative_term_count"]
        ) / tone_denominator
        assigned_parts.append(assigned)
        usable = assigned.dropna(subset=["disclosure_available_date"])
        if usable.empty:
            continue
        aggregate = (
            usable.groupby("disclosure_available_date", as_index=False)
            .agg(
                disclosure_count=("disclosure_id", "size"),
                disclosure_after_close_count=("disclosure_after_close", "sum"),
                disclosure_text_chars_mean=("text_chars", "mean"),
                disclosure_numeric_token_count=("numeric_token_count", "sum"),
                disclosure_positive_term_count=("positive_term_count", "sum"),
                disclosure_negative_term_count=("negative_term_count", "sum"),
                disclosure_uncertainty_term_count=("uncertainty_term_count", "sum"),
                disclosure_financial_term_count=("financial_term_count", "sum"),
                disclosure_net_tone=("net_tone", "mean"),
            )
            .rename(columns={"disclosure_available_date": "date"})
        )
        aggregate["stock_id"] = stock_id
        aggregate_parts.append(aggregate)

    features = trading.copy()
    if aggregate_parts:
        features = features.merge(
            pd.concat(aggregate_parts, ignore_index=True),
            on=["date", "stock_id"],
            how="left",
            validate="one_to_one",
        )
    base_columns = DISCLOSURE_FEATURE_COLUMNS[:9]
    for column in base_columns:
        if column not in features:
            features[column] = np.nan
    count_columns = [column for column in base_columns if column != "disclosure_text_chars_mean" and column != "disclosure_net_tone"]
    features[count_columns] = features[count_columns].fillna(0).astype("int64")
    features[["disclosure_text_chars_mean", "disclosure_net_tone"]] = features[
        ["disclosure_text_chars_mean", "disclosure_net_tone"]
    ].fillna(0.0)

    grouped = features.groupby("stock_id", sort=False)
    features["disclosure_count_5d"] = grouped["disclosure_count"].transform(
        lambda values: values.rolling(5, min_periods=5).sum()
    )
    features["disclosure_count_20d"] = grouped["disclosure_count"].transform(
        lambda values: values.rolling(20, min_periods=20).sum()
    )
    features["disclosure_net_tone_5d"] = grouped["disclosure_net_tone"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )

    update_dates = features["date"].where(features["disclosure_count"].gt(0))
    last_update = update_dates.groupby(features["stock_id"], sort=False).ffill()
    features["disclosure_days_since_update"] = (
        features["date"] - last_update
    ).dt.days.astype("Float64")
    assigned_all = (
        pd.concat(assigned_parts, ignore_index=True)
        if assigned_parts
        else disclosures.assign(
            known_at=pd.Series(dtype="datetime64[ns]"),
            disclosure_after_close=pd.Series(dtype=bool),
            disclosure_available_date=pd.Series(dtype="datetime64[ns]"),
        )
    )
    return features.reset_index(drop=True), assigned_all.reset_index(drop=True)
