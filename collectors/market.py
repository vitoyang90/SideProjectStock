import pandas as pd
import requests


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def _fetch_market_dataset(
    dataset: str,
    start_date: str,
    end_date: str,
    data_id: str = "",
    token: str = "",
) -> pd.DataFrame:
    params = {
        "dataset": dataset,
        "start_date": start_date,
        "end_date": end_date,
    }
    if data_id:
        params["data_id"] = data_id

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


def fetch_taiex(
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    """Fetch the TAIEX total-return index."""
    return _fetch_market_dataset(
        dataset="TaiwanStockTotalReturnIndex",
        data_id="TAIEX",
        start_date=start_date,
        end_date=end_date,
        token=token,
    )


def fetch_us_stock_price(
    symbol: str,
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    """Fetch daily US stock or ETF prices."""
    return _fetch_market_dataset(
        dataset="USStockPrice",
        data_id=symbol,
        start_date=start_date,
        end_date=end_date,
        token=token,
    )


def fetch_usd_twd(
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    """Fetch Bank of Taiwan USD exchange rates."""
    return _fetch_market_dataset(
        dataset="TaiwanExchangeRate",
        data_id="USD",
        start_date=start_date,
        end_date=end_date,
        token=token,
    )


def fetch_us_10y_yield(
    start_date: str,
    end_date: str,
    token: str = "",
) -> pd.DataFrame:
    """Fetch the United States 10-year government bond yield."""
    return _fetch_market_dataset(
        dataset="GovernmentBondsYield",
        data_id="United States 10-Year",
        start_date=start_date,
        end_date=end_date,
        token=token,
    )
