import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from collect_disclosures import SUPPORTED_STOCKS
from collectors.disclosures import (
    DISCLOSURE_COLUMNS,
    fetch_mops_history_detail,
    fetch_mops_history_index,
    is_financial_text_title,
    normalize_mops_history_detail,
)


RAW_ROOT = Path("data/raw/disclosures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill official point-in-time MOPS financial/call text."
    )
    parser.add_argument("--stock-id", action="append", dest="stock_ids")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--request-delay-seconds", type=float, default=0.15)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def backfill_stock_year(
    stock_id: str,
    year: int,
    session: requests.Session,
    request_delay_seconds: float,
) -> tuple[pd.DataFrame, dict]:
    cache_dir = RAW_ROOT / "history_cache" / stock_id / str(year)
    index_path = cache_dir / "index.json"
    if index_path.exists():
        index_payload = _read_json(index_path)
    else:
        index_payload = fetch_mops_history_index(stock_id, year, session)
        _write_json(index_path, index_payload)
        time.sleep(request_delay_seconds)

    result = index_payload.get("result", {})
    company_name = str(result.get("companyAbbreviation", "")).strip()
    index_rows = result.get("data", [])
    selected = [row for row in index_rows if len(row) >= 6 and is_financial_text_title(row[4])]
    frames: list[pd.DataFrame] = []
    for row in selected:
        descriptor = row[5]
        parameters = descriptor.get("parameters", {}) if isinstance(descriptor, dict) else {}
        enter_date = str(parameters.get("enterDate", "unknown"))
        serial = str(parameters.get("serialNumber", "unknown"))
        detail_path = cache_dir / f"{enter_date}_{serial}.json"
        if detail_path.exists():
            detail = _read_json(detail_path)
        else:
            detail = fetch_mops_history_detail(parameters, session)
            _write_json(detail_path, detail)
            time.sleep(request_delay_seconds)
        frame = normalize_mops_history_detail(detail, stock_id, company_name)
        if not frame.empty:
            frames.append(frame)
    output = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=DISCLOSURE_COLUMNS)
    )
    return output, {
        "stock_id": stock_id,
        "year": year,
        "index_rows": len(index_rows),
        "selected_rows": len(selected),
        "normalized_rows": len(output),
        "completed": True,
    }


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("end-year must not be earlier than start-year")
    if args.request_delay_seconds < 0:
        raise ValueError("request-delay-seconds must be non-negative")
    selected_stocks = tuple(args.stock_ids or SUPPORTED_STOCKS)
    unsupported = set(selected_stocks).difference(SUPPORTED_STOCKS)
    if unsupported:
        raise ValueError(f"Unsupported stock ids: {sorted(unsupported)}")

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_ROOT / "backfill_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {"runs": []}
    session = requests.Session()
    try:
        for stock_id in selected_stocks:
            frames: list[pd.DataFrame] = []
            for year in range(args.start_year, args.end_year + 1):
                frame, result = backfill_stock_year(
                    stock_id, year, session, args.request_delay_seconds
                )
                frames.append(frame)
                manifest["runs"] = [
                    item
                    for item in manifest["runs"]
                    if not (item["stock_id"] == stock_id and item["year"] == year)
                ]
                manifest["runs"].append(result)
                _write_json(manifest_path, manifest)
                print(
                    f"{stock_id} {year}: selected {result['selected_rows']} "
                    f"of {result['index_rows']} disclosures"
                )

            archive_path = RAW_ROOT / f"{stock_id}.csv"
            combined = pd.concat(frames, ignore_index=True)
            if archive_path.exists():
                combined = pd.concat(
                    [pd.read_csv(archive_path, dtype={"stock_id": str}), combined],
                    ignore_index=True,
                )
            for column in (
                "report_date",
                "published_at",
                "first_observed_at",
                "event_date",
            ):
                combined[column] = pd.to_datetime(combined[column], errors="raise")
            combined = combined.sort_values("published_at").drop_duplicates(
                "disclosure_id", keep="first"
            )
            combined[DISCLOSURE_COLUMNS].to_csv(
                archive_path, index=False, encoding="utf-8-sig"
            )
            print(f"Saved {len(combined)} unique rows to {archive_path}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
