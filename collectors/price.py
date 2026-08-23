import pandas as pd
import requests


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def fetch_stock_price(
    stock_id: str,
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    params = {
        "dataset": "TaiwanStockPrice",
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
