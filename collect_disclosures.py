import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from collectors.disclosures import (
    DISCLOSURE_COLUMNS,
    fetch_twse_daily_disclosures,
    parse_roc_date,
)


SUPPORTED_STOCKS = ("2330", "2454", "2317", "2308", "2881")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect an append-only V6 TWSE disclosure-text snapshot."
    )
    parser.add_argument("--stock-id", action="append", dest="stock_ids")
    return parser.parse_args()


def persist_snapshot(
    normalized: pd.DataFrame,
    raw_rows: list[dict],
    observed_at: pd.Timestamp,
    root: str | Path = "data/raw/disclosures",
) -> dict:
    root_path = Path(root)
    snapshot_dir = root_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = observed_at.strftime("%Y%m%dT%H%M%S%f")
    snapshot_path = snapshot_dir / f"{stamp}.json"
    raw_bytes = json.dumps(raw_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    snapshot_path.write_text(
        json.dumps(raw_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    if normalized.empty:
        raw_report_dates = [
            item.get("出表日期")
            for item in raw_rows
            if isinstance(item, dict) and item.get("出表日期")
        ]
        report_dates = sorted(
            {parse_roc_date(item).strftime("%Y-%m-%d") for item in raw_report_dates}
        )
    else:
        report_dates = sorted(
            pd.to_datetime(normalized["report_date"], errors="coerce")
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )
    log_path = root_path / "collection_log.csv"
    log_row = pd.DataFrame(
        [
            {
                "observed_at": observed_at.isoformat(),
                "report_dates": ";".join(report_dates),
                "source_row_count": len(raw_rows),
                "selected_row_count": len(normalized),
                "snapshot_path": snapshot_path.as_posix(),
                "snapshot_sha256": sha256,
            }
        ]
    )
    previous_log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    pd.concat([previous_log, log_row], ignore_index=True).to_csv(
        log_path, index=False, encoding="utf-8-sig"
    )

    for stock_id in sorted(normalized["stock_id"].unique()):
        archive_path = root_path / f"{stock_id}.csv"
        stock_rows = normalized[normalized["stock_id"] == stock_id].copy()
        if archive_path.exists():
            existing = pd.read_csv(archive_path, dtype={"stock_id": str})
            stock_rows = pd.concat([existing, stock_rows], ignore_index=True)
        stock_rows = stock_rows.sort_values("first_observed_at").drop_duplicates(
            "disclosure_id", keep="first"
        )
        stock_rows[DISCLOSURE_COLUMNS].to_csv(
            archive_path, index=False, encoding="utf-8-sig"
        )
    return {
        "snapshot_path": snapshot_path.as_posix(),
        "snapshot_sha256": sha256,
        "report_dates": report_dates,
        "source_row_count": len(raw_rows),
        "selected_row_count": len(normalized),
    }


def main() -> None:
    args = parse_args()
    selected = set(args.stock_ids or SUPPORTED_STOCKS)
    unsupported = selected.difference(SUPPORTED_STOCKS)
    if unsupported:
        raise ValueError(f"Unsupported stock ids: {sorted(unsupported)}")
    observed_at = pd.Timestamp(datetime.now(timezone.utc)).tz_convert("Asia/Taipei").tz_localize(None)
    normalized, raw_rows = fetch_twse_daily_disclosures(selected, observed_at=observed_at)
    result = persist_snapshot(normalized, raw_rows, observed_at)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("V6 remains forward-shadow only; this command does not train or promote a model.")


if __name__ == "__main__":
    main()
