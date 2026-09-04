import hashlib
import json
from pathlib import Path

import pandas as pd

from collect_disclosures import SUPPORTED_STOCKS
from features.disclosures import (
    DISCLOSURE_FEATURE_COLUMNS,
    FINANCIAL_TERMS,
    NEGATIVE_TERMS,
    POSITIVE_TERMS,
    UNCERTAINTY_TERMS,
)


RAW_ROOT = Path("data/raw/disclosures")
REPORT_DIR = Path("reports")
MIN_CALENDAR_DAYS = 730
MIN_ROWS_PER_STOCK = 50
REQUIRED_HISTORY_YEARS = tuple(range(2018, 2025))


def evaluate_readiness(
    archives: dict[str, pd.DataFrame],
    collection_log: pd.DataFrame,
    manifest: dict | None = None,
) -> dict:
    completed_pairs = {
        (str(item.get("stock_id")), int(item.get("year")))
        for item in (manifest or {}).get("runs", [])
        if item.get("completed")
    }
    per_stock: dict[str, dict] = {}
    all_ready = True
    for stock_id in SUPPORTED_STOCKS:
        frame = archives.get(stock_id, pd.DataFrame()).copy()
        if frame.empty:
            summary = {
                "unique_disclosures": 0,
                "first_observed_at": None,
                "last_observed_at": None,
                "calendar_span_days": 0,
                "duplicate_ids": 0,
                "ready": False,
            }
        else:
            archive_rows = frame[
                frame["availability_basis"].eq("official_archive_published_at")
            ].copy()
            published = pd.to_datetime(archive_rows["published_at"], errors="coerce")
            span = int((published.max() - published.min()).days) if published.notna().any() else 0
            unique_rows = int(archive_rows["disclosure_id"].nunique())
            completed_years = [
                year
                for year in REQUIRED_HISTORY_YEARS
                if (stock_id, year) in completed_pairs
            ]
            summary = {
                "unique_disclosures": unique_rows,
                "first_published_at": published.min().isoformat() if published.notna().any() else None,
                "last_published_at": published.max().isoformat() if published.notna().any() else None,
                "calendar_span_days": span,
                "completed_history_years": completed_years,
                "duplicate_ids": int(archive_rows["disclosure_id"].duplicated().sum()),
                "ready": bool(
                    span >= MIN_CALENDAR_DAYS
                    and unique_rows >= MIN_ROWS_PER_STOCK
                    and completed_years == list(REQUIRED_HISTORY_YEARS)
                    and not archive_rows["disclosure_id"].duplicated().any()
                ),
            }
        per_stock[stock_id] = summary
        all_ready &= summary["ready"]

    snapshots = int(len(collection_log))
    return {
        "version": "v6_disclosure_text",
        "status": "ready_for_nested_experiment" if all_ready else "collecting_forward_shadow",
        "ready_for_training": bool(all_ready),
        "source": (
            "MOPS official historical material-information API plus TWSE OpenAPI "
            "listed-company daily material information"
        ),
        "point_in_time_rule": "usable no earlier than max(published_at, first_observed_at), then next trading date when after 13:30",
        "training_gate": {
            "minimum_calendar_span_days_per_stock": MIN_CALENDAR_DAYS,
            "minimum_unique_disclosures_per_stock": MIN_ROWS_PER_STOCK,
            "required_complete_history_years": list(REQUIRED_HISTORY_YEARS),
            "all_supported_stocks_required": True,
            "duplicate_ids_required": 0,
        },
        "snapshots_collected": snapshots,
        "per_stock": per_stock,
        "decision": "do_not_train" if not all_ready else "freeze_features_then_run_nested_walk_forward",
        "reason": (
            "Historical point-in-time text is not yet available for every stock. "
            "Zero-filled retrospective rows would fabricate knowledge and are forbidden."
            if not all_ready
            else "The minimum forward history and row-count gates passed."
        ),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RAW_ROOT / "collection_log.csv"
    collection_log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    manifest_path = RAW_ROOT / "backfill_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    archives = {}
    for stock_id in SUPPORTED_STOCKS:
        path = RAW_ROOT / f"{stock_id}.csv"
        if path.exists():
            archives[stock_id] = pd.read_csv(path, dtype={"stock_id": str})

    readiness = evaluate_readiness(archives, collection_log, manifest)
    readiness_path = REPORT_DIR / "v6_disclosure_data_readiness.json"
    readiness_path.write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    schema_payload = {
        "version": "v6_disclosure_text",
        "feature_columns": DISCLOSURE_FEATURE_COLUMNS,
        "lexicons": {
            "positive": POSITIVE_TERMS,
            "negative": NEGATIVE_TERMS,
            "uncertainty": UNCERTAINTY_TERMS,
            "financial": FINANCIAL_TERMS,
        },
        "feature_policy": "deterministic counts only; no LLM-generated labels",
    }
    canonical = json.dumps(schema_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    schema_payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    schema_path = REPORT_DIR / "v6_disclosure_feature_schema.json"
    schema_path.write_text(
        json.dumps(schema_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    print(f"Feature schema saved to {schema_path}")


if __name__ == "__main__":
    main()
