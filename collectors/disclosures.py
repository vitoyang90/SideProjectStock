import hashlib
import time
from datetime import datetime, timezone

import pandas as pd
import requests


TWSE_DISCLOSURE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
MOPS_HISTORY_URL = "https://mops.twse.com.tw/mops/api/t05st01"
MOPS_DETAIL_URL = "https://mops.twse.com.tw/mops/api/t05st01_detail"
FINANCIAL_TEXT_TITLE_TERMS = (
    "法人說明會",
    "財務報告",
    "每股盈餘",
    "營收報告",
    "自結",
    "損益",
    "財務業務資訊",
    "業績",
)
DISCLOSURE_COLUMNS = [
    "disclosure_id",
    "report_date",
    "published_at",
    "first_observed_at",
    "stock_id",
    "company_name",
    "title",
    "rule",
    "event_date",
    "body",
    "document_type",
    "availability_basis",
    "source_url",
]
_SOURCE_COLUMNS = {
    "出表日期": "report_date",
    "發言日期": "published_date",
    "發言時間": "published_time",
    "公司代號": "stock_id",
    "公司名稱": "company_name",
    "主旨": "title",
    "符合條款": "rule",
    "事實發生日": "event_date",
    "說明": "body",
}


def parse_roc_date(value: object) -> pd.Timestamp:
    """Parse a Minguo/ROC date such as 1150904 or 115/09/04."""
    digits = "".join(character for character in str(value).strip() if character.isdigit())
    if len(digits) != 7:
        raise ValueError(f"Invalid ROC date: {value!r}")
    year = int(digits[:3]) + 1911
    return pd.Timestamp(year=year, month=int(digits[3:5]), day=int(digits[5:7]))


def parse_roc_datetime(date_value: object, time_value: object) -> pd.Timestamp:
    day = parse_roc_date(date_value)
    time_digits = "".join(
        character for character in str(time_value).strip() if character.isdigit()
    ).zfill(6)
    if len(time_digits) != 6:
        raise ValueError(f"Invalid disclosure time: {time_value!r}")
    return day + pd.Timedelta(
        hours=int(time_digits[:2]),
        minutes=int(time_digits[2:4]),
        seconds=int(time_digits[4:6]),
    )


def _clean_source_row(row: dict) -> dict:
    return {str(key).strip(): value for key, value in row.items()}


def normalize_twse_disclosures(
    rows: list[dict],
    observed_at: datetime | pd.Timestamp | None = None,
    stock_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Normalize the official TWSE daily material-information payload."""
    if not rows:
        return pd.DataFrame(columns=DISCLOSURE_COLUMNS)
    cleaned_rows = [_clean_source_row(row) for row in rows]
    missing = set(_SOURCE_COLUMNS).difference(cleaned_rows[0])
    if missing:
        raise RuntimeError(f"TWSE disclosure response is missing columns: {sorted(missing)}")

    observed = pd.Timestamp(observed_at or datetime.now(timezone.utc))
    if observed.tzinfo is not None:
        observed = observed.tz_convert("Asia/Taipei").tz_localize(None)
    selected_ids = {str(item) for item in stock_ids} if stock_ids else None
    normalized: list[dict] = []
    for raw in cleaned_rows:
        stock_id = str(raw["公司代號"]).strip()
        if selected_ids is not None and stock_id not in selected_ids:
            continue
        published_at = parse_roc_datetime(raw["發言日期"], raw["發言時間"])
        title = str(raw.get("主旨", "")).strip()
        body = str(raw.get("說明", "")).strip()
        identity = "|".join(
            [stock_id, published_at.isoformat(), title, body]
        ).encode("utf-8")
        normalized.append(
            {
                "disclosure_id": hashlib.sha256(identity).hexdigest(),
                "report_date": parse_roc_date(raw["出表日期"]),
                "published_at": published_at,
                "first_observed_at": observed,
                "stock_id": stock_id,
                "company_name": str(raw.get("公司名稱", "")).strip(),
                "title": title,
                "rule": str(raw.get("符合條款", "")).strip(),
                "event_date": parse_roc_date(raw["事實發生日"]),
                "body": body,
                "document_type": classify_document_type(title),
                "availability_basis": "live_first_observed",
                "source_url": TWSE_DISCLOSURE_URL,
            }
        )
    if not normalized:
        return pd.DataFrame(columns=DISCLOSURE_COLUMNS)
    return (
        pd.DataFrame(normalized, columns=DISCLOSURE_COLUMNS)
        .sort_values(["published_at", "disclosure_id"])
        .reset_index(drop=True)
    )


def fetch_twse_daily_disclosures(
    stock_ids: set[str] | None = None,
    session: requests.Session | None = None,
    observed_at: datetime | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch one official daily snapshot and return normalized plus raw rows."""
    request_session = session or requests.Session()
    close_session = session is None
    try:
        response = request_session.get(TWSE_DISCLOSURE_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
    finally:
        if close_session:
            request_session.close()
    if not isinstance(payload, list):
        raise RuntimeError("TWSE disclosure response must be a JSON list")
    return normalize_twse_disclosures(payload, observed_at, stock_ids), payload


def classify_document_type(title: str) -> str:
    if "法人說明會" in title:
        return "investor_conference"
    if any(term in title for term in FINANCIAL_TEXT_TITLE_TERMS):
        return "financial_result"
    return "other_material_information"


def is_financial_text_title(title: object) -> bool:
    cleaned = str(title).strip()
    return any(term in cleaned for term in FINANCIAL_TEXT_TITLE_TERMS)


def _post_json(
    session: requests.Session,
    url: str,
    payload: dict,
    max_retries: int = 3,
) -> dict:
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = session.post(
                url,
                json=payload,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; stock-research/1.0)",
                    "Referer": "https://mops.twse.com.tw/mops/#/web/t05st01",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("code") != 200:
                raise RuntimeError(f"MOPS response error: {result}")
            return result
        except (requests.RequestException, ValueError):
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("MOPS request did not return a usable response")


def fetch_mops_history_index(
    stock_id: str,
    year: int,
    session: requests.Session,
) -> dict:
    """Fetch one Gregorian year's official MOPS material-information index."""
    roc_year = year - 1911
    if roc_year <= 0:
        raise ValueError("MOPS history year must be a Gregorian year after 1911")
    return _post_json(
        session,
        MOPS_HISTORY_URL,
        {
            "companyId": str(stock_id),
            "year": str(roc_year),
            "month": "all",
            "firstDay": "",
            "lastDay": "",
        },
    )


def fetch_mops_history_detail(parameters: dict, session: requests.Session) -> dict:
    return _post_json(session, MOPS_DETAIL_URL, parameters)


def normalize_mops_history_detail(
    detail_payload: dict,
    stock_id: str,
    company_name: str,
) -> pd.DataFrame:
    """Normalize one archived MOPS detail response.

    The MOPS archive states that rows are disclosed after company filing. For
    backtests, the archived source timestamp is therefore the point-in-time
    availability timestamp; later corrections remain separate disclosures.
    """
    rows = detail_payload.get("result", {}).get("data", [])
    if not rows:
        return pd.DataFrame(columns=DISCLOSURE_COLUMNS)
    normalized = []
    for row in rows:
        if len(row) < 10:
            raise RuntimeError("MOPS detail row has an unexpected schema")
        published_at = parse_roc_datetime(row[1], row[2])
        title = str(row[6]).strip()
        body = str(row[9]).strip()
        identity = "|".join(
            [str(stock_id), published_at.isoformat(), title, body]
        ).encode("utf-8")
        normalized.append(
            {
                "disclosure_id": hashlib.sha256(identity).hexdigest(),
                "report_date": published_at.normalize(),
                "published_at": published_at,
                "first_observed_at": published_at,
                "stock_id": str(stock_id),
                "company_name": str(company_name).strip(),
                "title": title,
                "rule": str(row[7]).strip(),
                "event_date": parse_roc_date(row[8]),
                "body": body,
                "document_type": classify_document_type(title),
                "availability_basis": "official_archive_published_at",
                "source_url": MOPS_DETAIL_URL,
            }
        )
    return pd.DataFrame(normalized, columns=DISCLOSURE_COLUMNS)
