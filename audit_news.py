import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.news import NEWS_FEATURE_COLUMNS


NEWS_PATH = Path("data/raw/2330_news_normalized.csv")
FEATURE_PATH = Path("data/processed/2330_news_features.csv")
PROGRESS_PATH = Path("data/processed/2330_news_backfill_progress.json")
REPORT_DIR = Path("reports")
REFERENCE_YEAR = 2021


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float | None:
    """Calculate PSI using quantile bins learned from the reference sample."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty or reference.nunique() < 2:
        return None
    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    ).astype(float)
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts = pd.cut(reference, bins=edges, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    current_counts = pd.cut(current, bins=edges, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    epsilon = 1e-6
    reference_share = np.clip(reference_counts.to_numpy(float), epsilon, None)
    current_share = np.clip(current_counts.to_numpy(float), epsilon, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def find_zero_streaks(
    features: pd.DataFrame,
    minimum_length: int = 3,
) -> pd.DataFrame:
    """Find consecutive trading-day runs with zero usable news."""
    ordered = features.sort_values("date").reset_index(drop=True).copy()
    is_zero = ordered["news_count"].eq(0)
    groups = is_zero.ne(is_zero.shift()).cumsum()
    rows: list[dict] = []
    for _, group in ordered[is_zero].groupby(groups[is_zero]):
        if len(group) >= minimum_length:
            rows.append(
                {
                    "start": group["date"].iloc[0].strftime("%Y-%m-%d"),
                    "end": group["date"].iloc[-1].strftime("%Y-%m-%d"),
                    "trading_days": int(len(group)),
                }
            )
    return pd.DataFrame(rows, columns=["start", "end", "trading_days"])


def _drift_level(psi: float | None) -> str:
    if psi is None or pd.isna(psi):
        return "not_available"
    if psi >= 0.25:
        return "high"
    if psi >= 0.10:
        return "moderate"
    return "low"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    for path in (NEWS_PATH, FEATURE_PATH, PROGRESS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required V5 artifact: {path}")
    news = pd.read_csv(NEWS_PATH, dtype={"stock_id": str})
    features = pd.read_csv(FEATURE_PATH, dtype={"stock_id": str})
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    news["published_at"] = pd.to_datetime(news["published_at"], errors="raise")
    news["news_available_date"] = pd.to_datetime(
        news["news_available_date"], errors="coerce"
    )
    features["date"] = pd.to_datetime(features["date"], errors="raise")
    return news, features, progress


def _hard_checks(news: pd.DataFrame, features: pd.DataFrame, progress: dict) -> dict:
    target_dates = pd.date_range(
        progress["target_start"], progress["target_end"], freq="D"
    )
    cache_root = Path("data/raw/news") / str(progress["stock_id"])
    missing_cache_dates = [
        item.strftime("%Y-%m-%d")
        for item in target_dates
        if not (cache_root / f"{item:%Y-%m-%d}.csv").exists()
    ]
    usable = news.dropna(subset=["news_available_date"])
    timing_ok = bool(
        (
            usable["news_available_date"].dt.normalize()
            >= usable["published_at"].dt.normalize()
        ).all()
    )
    after_close = usable[usable["published_after_close"].astype(bool)]
    after_close_timing_ok = bool(
        (
            after_close["news_available_date"].dt.normalize()
            > after_close["published_at"].dt.normalize()
        ).all()
    )
    return {
        "backfill_complete": progress.get("status") == "complete",
        "missing_cache_dates": len(missing_cache_dates),
        "missing_cache_date_examples": missing_cache_dates[:10],
        "news_id_unique": int(news["news_id"].nunique()) == len(news),
        "daily_aggregation_reconciled": int(features["news_count"].sum())
        == int(news["news_available_date"].notna().sum()),
        "available_date_not_before_publication": timing_ok,
        "after_close_moves_to_later_date": after_close_timing_ok,
        "feature_keys_unique": not features.duplicated(["date", "stock_id"]).any(),
    }


def _yearly_summary(news: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    daily = features.copy()
    daily["year"] = daily["date"].dt.year
    yearly_daily = (
        daily.groupby("year")
        .agg(
            trading_days=("date", "size"),
            news_days=("news_has_update", "sum"),
            total_news=("news_count", "sum"),
            news_count_mean=("news_count", "mean"),
            news_count_median=("news_count", "median"),
            news_count_p95=("news_count", lambda x: x.quantile(0.95)),
            news_count_p99=("news_count", lambda x: x.quantile(0.99)),
            news_count_max=("news_count", "max"),
            source_count_mean=("news_source_count", "mean"),
            after_close_news=("news_after_close_count", "sum"),
        )
        .reset_index()
    )
    yearly_daily["zero_news_days"] = (
        yearly_daily["trading_days"] - yearly_daily["news_days"]
    )
    yearly_daily["zero_news_rate"] = (
        yearly_daily["zero_news_days"] / yearly_daily["trading_days"]
    )
    yearly_daily["after_close_share"] = np.where(
        yearly_daily["total_news"] > 0,
        yearly_daily["after_close_news"] / yearly_daily["total_news"],
        np.nan,
    )

    articles = news.copy()
    articles["year"] = articles["published_at"].dt.year
    article_yearly = (
        articles.groupby("year")
        .agg(
            unique_articles=("news_id", "size"),
            unique_sources=("source", "nunique"),
            missing_title=("title", lambda x: x.fillna("").str.strip().eq("").sum()),
            missing_source=("source", lambda x: x.fillna("").str.strip().eq("").sum()),
            missing_link=("link", lambda x: x.fillna("").str.strip().eq("").sum()),
            replacement_char_titles=(
                "title",
                lambda x: x.fillna("").str.contains("�", regex=False).sum(),
            ),
        )
        .reset_index()
    )
    return yearly_daily.merge(article_yearly, on="year", how="left")


def _source_summary(news: pd.DataFrame) -> pd.DataFrame:
    sources = news.assign(
        source_clean=news["source"].fillna("").replace("", "<missing>")
    )
    summary = (
        sources.groupby("source_clean")
        .agg(
            article_count=("news_id", "size"),
            first_published=("published_at", "min"),
            last_published=("published_at", "max"),
            years_active=("published_at", lambda x: x.dt.year.nunique()),
        )
        .sort_values("article_count", ascending=False)
        .reset_index(names="source")
    )
    summary["article_share"] = summary["article_count"] / len(news)
    summary["cumulative_share"] = summary["article_share"].cumsum()
    for column in ("first_published", "last_published"):
        summary[column] = summary[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    return summary


def _feature_drift(features: pd.DataFrame) -> pd.DataFrame:
    data = features.copy()
    data["year"] = data["date"].dt.year
    if REFERENCE_YEAR not in data["year"].unique():
        raise ValueError(f"Reference year {REFERENCE_YEAR} is unavailable")
    reference = data[data["year"] == REFERENCE_YEAR]
    rows: list[dict] = []
    for feature in NEWS_FEATURE_COLUMNS:
        for year, current in data.groupby("year"):
            psi = population_stability_index(reference[feature], current[feature])
            rows.append(
                {
                    "feature": feature,
                    "year": int(year),
                    "reference_year": REFERENCE_YEAR,
                    "mean": float(current[feature].mean()),
                    "std": float(current[feature].std()),
                    "missing_rate": float(current[feature].isna().mean()),
                    "unique_values": int(current[feature].nunique(dropna=True)),
                    "psi_vs_reference": psi,
                    "drift_level": _drift_level(psi),
                    "partial_year": bool(
                        year in (data["year"].min(), data["year"].max())
                    ),
                }
            )
    return pd.DataFrame(rows)


def _daily_anomalies(features: pd.DataFrame) -> pd.DataFrame:
    data = features.copy()
    data["year"] = data["date"].dt.year
    rows: list[pd.DataFrame] = []
    for year, group in data.groupby("year"):
        median = group["news_count"].median()
        mad = (group["news_count"] - median).abs().median()
        robust_limit = median + 6 * 1.4826 * mad
        threshold = max(group["news_count"].quantile(0.99), robust_limit)
        spikes = group[group["news_count"] > threshold].copy()
        if spikes.empty:
            continue
        spikes["anomaly_type"] = "news_count_spike"
        spikes["year_threshold"] = float(threshold)
        rows.append(spikes)
    columns = [
        "date",
        "stock_id",
        "year",
        "anomaly_type",
        "news_count",
        "news_source_count",
        "news_after_close_count",
        "year_threshold",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    anomalies = pd.concat(rows, ignore_index=True)[columns]
    anomalies["date"] = anomalies["date"].dt.strftime("%Y-%m-%d")
    return anomalies.sort_values("news_count", ascending=False)


def _freeze_schema(features: pd.DataFrame, quality: dict) -> dict:
    excluded = {
        "news_description_chars_mean": (
            "excluded_constant: FinMind live response omitted description for all rows"
        ),
        "news_has_update": "excluded_redundant: exactly derivable from news_count > 0",
    }
    active = [feature for feature in NEWS_FEATURE_COLUMNS if feature not in excluded]
    schema = {
        "schema_version": "v5_news_activity_1",
        "status": "frozen_for_historical_walk_forward",
        "stock_id": "2330",
        "source_dataset": "FinMind TaiwanStockNews",
        "raw_coverage": {
            "start": quality["requested_news_start"],
            "end": quality["requested_news_end"],
        },
        "recommended_model_start": "2021-01-01",
        "availability_rule": {
            "market_close_time": "13:30",
            "timestamp_timezone_assumption": "Asia/Taipei",
            "after_close_policy": "next trading date",
            "non_trading_day_policy": "next trading date",
        },
        "active_features": active,
        "excluded_features": excluded,
        "rolling_windows": {
            "news_count_3d": {"trading_days": 3, "minimum_periods": 3},
            "news_count_5d": {"trading_days": 5, "minimum_periods": 5},
        },
        "outside_coverage_policy": "missing_not_zero",
        "selection_note": (
            "Feature definitions were frozen before running the V5 model comparison."
        ),
    }
    canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True).encode("utf-8")
    schema["schema_sha256"] = hashlib.sha256(canonical).hexdigest()
    return schema


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    news, features, progress = _load_inputs()
    checks = _hard_checks(news, features, progress)
    yearly = _yearly_summary(news, features)
    sources = _source_summary(news)
    drift = _feature_drift(features)
    anomalies = _daily_anomalies(features)
    zero_streaks = find_zero_streaks(features)

    description_nonempty = int(
        news["description"].fillna("").str.strip().ne("").sum()
    )
    replacement_titles = int(
        news["title"].fillna("").str.contains("�", regex=False).sum()
    )
    first_article = news["published_at"].min()
    hard_pass = all(
        bool(value)
        for key, value in checks.items()
        if key not in {"missing_cache_dates", "missing_cache_date_examples"}
    ) and checks["missing_cache_dates"] == 0
    warnings = []
    if description_nonempty == 0:
        warnings.append(
            "description is empty for every article; exclude news_description_chars_mean"
        )
    if replacement_titles:
        warnings.append(f"{replacement_titles} titles contain Unicode replacement characters")
    if first_article.normalize() > pd.Timestamp(progress["target_start"]):
        warnings.append(
            "first returned article is later than requested coverage start; "
            "treat 2020 as warm-up only"
        )
    if (drift["drift_level"] == "high").any():
        warnings.append("one or more yearly feature distributions have PSI >= 0.25")

    audit_status = (
        "fail"
        if not hard_pass
        else ("pass_with_warnings" if warnings else "pass")
    )
    summary = {
        "status": audit_status,
        "hard_checks": checks,
        "coverage": {
            "requested_start": progress["target_start"],
            "requested_end": progress["target_end"],
            "first_returned_article": first_article.strftime("%Y-%m-%d %H:%M:%S"),
            "last_returned_article": news["published_at"].max().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "trading_days": int(len(features)),
        },
        "articles": {
            "normalized_unique": int(len(news)),
            "unique_sources": int(news["source"].nunique()),
            "description_nonempty": description_nonempty,
            "replacement_character_titles": replacement_titles,
            "after_close_share": float(
                news["published_after_close"].astype(bool).mean()
            ),
        },
        "daily_features": {
            "zero_news_days": int(features["news_count"].eq(0).sum()),
            "zero_news_rate": float(features["news_count"].eq(0).mean()),
            "max_news_count": int(features["news_count"].max()),
            "spike_anomalies": int(len(anomalies)),
            "zero_streaks_at_least_3_days": int(len(zero_streaks)),
        },
        "warnings": warnings,
        "decision": {
            "data_layer_usable": hard_pass,
            "recommended_model_start": "2021-01-01",
            "ready_to_build_v5_walk_forward": hard_pass,
            "do_not_use_as_final_forward_proof": True,
        },
    }
    schema = _freeze_schema(
        features,
        {
            "requested_news_start": progress["target_start"],
            "requested_news_end": progress["target_end"],
        },
    )

    yearly.to_csv(REPORT_DIR / "news_audit_yearly.csv", index=False, encoding="utf-8-sig")
    sources.to_csv(REPORT_DIR / "news_audit_sources.csv", index=False, encoding="utf-8-sig")
    drift.to_csv(REPORT_DIR / "news_feature_drift.csv", index=False, encoding="utf-8-sig")
    anomalies.to_csv(
        REPORT_DIR / "news_audit_anomalies.csv", index=False, encoding="utf-8-sig"
    )
    zero_streaks.to_csv(
        REPORT_DIR / "news_audit_zero_streaks.csv", index=False, encoding="utf-8-sig"
    )
    (REPORT_DIR / "news_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "news_feature_schema_v5.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Frozen active features: {schema['active_features']}")
    print(f"Schema SHA-256: {schema['schema_sha256']}")


if __name__ == "__main__":
    main()
