import pandas as pd
import requests


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def _fetch_fundamental_dataset(
    dataset: str,
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
    }

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(
        FINMIND_URL,
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    result = response.json()
    if "data" not in result:
        raise RuntimeError(f"FinMind response error: {result}")

    return pd.DataFrame(result["data"])


def fetch_month_revenue(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    """Fetch monthly company revenue data."""
    return _fetch_fundamental_dataset(
        dataset="TaiwanStockMonthRevenue",
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        token=token,
    )


def fetch_valuation(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    """Fetch daily PE, PB, and dividend-yield data."""
    return _fetch_fundamental_dataset(
        dataset="TaiwanStockPER",
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        token=token,
    )
