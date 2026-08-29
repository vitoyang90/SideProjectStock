import time
from pathlib import Path

import pandas as pd
import requests


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
NEWS_COLUMNS = [
    "date",
    "stock_id",
    "description",
    "link",
    "source",
    "title",
]


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)


def fetch_stock_news_day(
    stock_id: str,
    news_date: str,
    token: str = "",
    session: requests.Session | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch one calendar day of FinMind TaiwanStockNews data."""
    request_session = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params = {
        "dataset": "TaiwanStockNews",
        "data_id": stock_id,
        "start_date": news_date,
    }
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = request_session.get(
                FINMIND_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                break
            response.raise_for_status()
        except requests.RequestException:
            if attempt >= max_retries:
                raise
            retry_after = response.headers.get("Retry-After") if response else None
            delay = float(retry_after) if retry_after else 2 ** (attempt + 1)
            time.sleep(min(delay, 30.0))

    if response is None:
        raise RuntimeError("FinMind request did not return a response")

    result = response.json()
    if "data" not in result:
        raise RuntimeError(f"FinMind response error: {result}")

    frame = pd.DataFrame(result["data"])
    if frame.empty:
        return _empty_news_frame()

    # FinMind's live endpoint may omit description even though older schema
    # documentation includes it. Title/link/source remain required; an absent
    # description is represented as an empty string instead of failing a run.
    required_columns = set(NEWS_COLUMNS).difference({"description"})
    missing = required_columns.difference(frame.columns)
    if missing:
        raise RuntimeError(
            f"TaiwanStockNews response is missing columns: {sorted(missing)}"
        )
    if "description" not in frame.columns:
        frame["description"] = ""

    frame = frame[NEWS_COLUMNS].copy()
    frame["stock_id"] = frame["stock_id"].astype(str)
    published_at = pd.to_datetime(frame["date"], errors="coerce")
    expected_date = pd.Timestamp(news_date).date()
    frame = frame[published_at.dt.date == expected_date].copy()
    return frame.reset_index(drop=True)


def fetch_stock_news(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str = "",
    cache_dir: str | Path = "data/raw/news",
    refresh: bool = False,
    request_delay_seconds: float = 0.2,
    max_request_days: int = 31,
    allow_large_range: bool = False,
) -> pd.DataFrame:
    """Fetch and cache a date range one day at a time.

    FinMind returns at most one day for TaiwanStockNews, so every calendar day
    is cached separately. Existing cache files never consume an API request.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must not be earlier than start_date")
    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds must be non-negative")

    dates = pd.date_range(start, end, freq="D")
    cache_root = Path(cache_dir) / str(stock_id)
    uncached_dates = [
        item
        for item in dates
        if refresh or not (cache_root / f"{item:%Y-%m-%d}.csv").exists()
    ]
    if len(uncached_dates) > max_request_days and not allow_large_range:
        raise ValueError(
            f"This run would make {len(uncached_dates)} daily API requests. "
            "Narrow the date range or pass allow_large_range=True."
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    session = requests.Session()
    try:
        for index, item in enumerate(dates):
            day = item.strftime("%Y-%m-%d")
            cache_path = cache_root / f"{day}.csv"
            if cache_path.exists() and not refresh:
                daily = pd.read_csv(
                    cache_path,
                    dtype={"stock_id": str},
                )
            else:
                daily = fetch_stock_news_day(
                    stock_id=str(stock_id),
                    news_date=day,
                    token=token,
                    session=session,
                )
                daily.to_csv(cache_path, index=False, encoding="utf-8-sig")
                if index < len(dates) - 1 and request_delay_seconds:
                    time.sleep(request_delay_seconds)
            if not daily.empty:
                frames.append(daily)
    finally:
        session.close()

    if not frames:
        return _empty_news_frame()
    return pd.concat(frames, ignore_index=True)[NEWS_COLUMNS]
