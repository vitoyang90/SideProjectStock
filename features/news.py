import hashlib
import html
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd


NEWS_FEATURE_COLUMNS = [
    "news_count",
    "news_source_count",
    "news_after_close_count",
    "news_title_chars_mean",
    "news_description_chars_mean",
    "news_count_3d",
    "news_count_5d",
    "news_has_update",
]

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def _require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    cleaned = html.unescape(str(value))
    cleaned = _HTML_TAG.sub(" ", cleaned)
    return _WHITESPACE.sub(" ", cleaned).strip()


def _canonicalize_link(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return ""
    parts = urlsplit(str(value).strip())
    query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            urlencode(query),
            "",
        )
    )


def normalize_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize FinMind news rows and remove deterministic duplicates."""
    required = {"date", "stock_id", "description", "link", "source", "title"}
    _require_columns(news_df, required)
    if news_df.empty:
        return pd.DataFrame(
            columns=[
                "news_id",
                "published_at",
                "stock_id",
                "title",
                "description",
                "source",
                "link",
            ]
        )

    news = news_df[list(required)].copy()
    news["published_at"] = pd.to_datetime(news.pop("date"), errors="coerce")
    news["stock_id"] = news["stock_id"].astype(str).str.strip()
    for column in ("title", "description", "source"):
        news[column] = news[column].map(_clean_text)
    news["link"] = news["link"].map(_canonicalize_link)
    news = news.dropna(subset=["published_at"])
    news = news[(news["title"] != "") | (news["link"] != "")].copy()

    def make_id(row: pd.Series) -> str:
        identity = row["link"] or "|".join(
            [row["source"], row["title"], row["published_at"].isoformat()]
        )
        raw_id = f"{row['stock_id']}|{identity}".encode("utf-8")
        return hashlib.sha256(raw_id).hexdigest()

    news["news_id"] = news.apply(make_id, axis=1)
    output_columns = [
            "news_id",
            "published_at",
            "stock_id",
            "title",
            "description",
            "source",
            "link",
    ]
    return (
        news.drop_duplicates("news_id", keep="first")
        .sort_values(["published_at", "news_id"])[output_columns]
        .reset_index(drop=True)
    )


def assign_news_available_dates(
    normalized_news: pd.DataFrame,
    trading_dates: pd.Series,
    market_close_time: str = "13:30",
) -> pd.DataFrame:
    """Map each publication timestamp to the first usable trading date.

    News published no later than the market close is usable on that date.
    After-close and non-trading-day news moves to the next trading date.
    Rows after the final supplied trading date are retained with NaT.
    """
    _require_columns(normalized_news, {"published_at"})
    hours, minutes = map(int, market_close_time.split(":"))
    if hours not in range(24) or minutes not in range(60):
        raise ValueError("market_close_time must use valid HH:MM notation")

    news = normalized_news.copy()
    news["published_at"] = pd.to_datetime(news["published_at"], errors="raise")
    normalized_trading_dates = pd.to_datetime(
        trading_dates,
        errors="raise",
    ).dt.normalize()
    calendar = np.array(
        sorted(normalized_trading_dates.unique()),
        dtype="datetime64[ns]",
    )
    if not len(calendar):
        raise ValueError("At least one trading date is required")

    close_delta = pd.Timedelta(hours=hours, minutes=minutes)
    published_day = news["published_at"].dt.normalize()
    news["published_after_close"] = (
        news["published_at"] - published_day > close_delta
    )
    earliest_day = published_day + pd.to_timedelta(
        news["published_after_close"].astype(int), unit="D"
    )
    positions = np.searchsorted(
        calendar,
        earliest_day.to_numpy(dtype="datetime64[ns]"),
        side="left",
    )
    available = np.full(
        len(news),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    valid = positions < len(calendar)
    available[valid] = calendar[positions[valid]]
    news["news_available_date"] = pd.to_datetime(available)
    return news


def build_daily_news_features(
    news_df: pd.DataFrame,
    trading_df: pd.DataFrame,
    market_close_time: str = "13:30",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-aware daily count/activity features.

    Returns the dense trading-date feature table and normalized article table.
    No sentiment model is used at this data-layer stage.
    """
    _require_columns(trading_df, {"date", "stock_id"})
    trading = trading_df[["date", "stock_id"]].copy()
    trading["date"] = pd.to_datetime(trading["date"], errors="raise").dt.normalize()
    trading["stock_id"] = trading["stock_id"].astype(str)
    trading = trading.drop_duplicates().sort_values(["stock_id", "date"])

    normalized = normalize_news(news_df)
    assigned_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    for stock_id, stock_days in trading.groupby("stock_id", sort=False):
        stock_news = normalized[normalized["stock_id"] == stock_id].copy()
        if stock_news.empty:
            continue
        assigned = assign_news_available_dates(
            stock_news,
            stock_days["date"],
            market_close_time=market_close_time,
        )
        assigned_frames.append(assigned)
        usable = assigned.dropna(subset=["news_available_date"]).copy()
        if usable.empty:
            continue
        usable["title_chars"] = usable["title"].str.len()
        usable["description_chars"] = usable["description"].str.len()
        aggregate = (
            usable.groupby("news_available_date", as_index=False)
            .agg(
                news_count=("news_id", "size"),
                news_source_count=("source", "nunique"),
                news_after_close_count=("published_after_close", "sum"),
                news_title_chars_mean=("title_chars", "mean"),
                news_description_chars_mean=("description_chars", "mean"),
            )
            .rename(columns={"news_available_date": "date"})
        )
        aggregate["stock_id"] = stock_id
        aggregate_frames.append(aggregate)

    if aggregate_frames:
        aggregate_all = pd.concat(aggregate_frames, ignore_index=True)
        features = trading.merge(
            aggregate_all,
            on=["date", "stock_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        features = trading.copy()
        for column in NEWS_FEATURE_COLUMNS[:5]:
            features[column] = np.nan

    count_columns = ["news_count", "news_source_count", "news_after_close_count"]
    features[count_columns] = features[count_columns].fillna(0).astype("int64")
    mean_columns = ["news_title_chars_mean", "news_description_chars_mean"]
    features[mean_columns] = features[mean_columns].fillna(0.0).astype(float)
    grouped_counts = features.groupby("stock_id", sort=False)["news_count"]
    features["news_count_3d"] = grouped_counts.transform(
        lambda values: values.rolling(3, min_periods=3).sum()
    )
    features["news_count_5d"] = grouped_counts.transform(
        lambda values: values.rolling(5, min_periods=5).sum()
    )
    features["news_has_update"] = (features["news_count"] > 0).astype("int8")

    assigned_all = (
        pd.concat(assigned_frames, ignore_index=True)
        if assigned_frames
        else normalized.assign(
            published_after_close=pd.Series(dtype=bool),
            news_available_date=pd.Series(dtype="datetime64[ns]"),
        )
    )
    return features.reset_index(drop=True), assigned_all.reset_index(drop=True)


def merge_news_features(
    daily_df: pd.DataFrame,
    news_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge a dense news feature table into a daily model dataset."""
    keys = {"date", "stock_id"}
    _require_columns(daily_df, keys)
    _require_columns(news_features, keys | set(NEWS_FEATURE_COLUMNS))
    daily = daily_df.copy()
    news = news_features.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    news["date"] = pd.to_datetime(news["date"], errors="raise")
    daily["stock_id"] = daily["stock_id"].astype(str)
    news["stock_id"] = news["stock_id"].astype(str)
    merged = daily.merge(
        news[["date", "stock_id", *NEWS_FEATURE_COLUMNS]],
        on=["date", "stock_id"],
        how="left",
        validate="one_to_one",
    )
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged
