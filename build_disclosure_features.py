import json
from pathlib import Path

import pandas as pd

from audit_disclosures import REPORT_DIR, evaluate_readiness
from collect_disclosures import SUPPORTED_STOCKS
from features.disclosures import DISCLOSURE_FEATURE_COLUMNS, build_daily_disclosure_features


RAW_ROOT = Path("data/raw/disclosures")
OUTPUT_ROOT = Path("data/processed/disclosures")


def _dataset_path(stock_id: str) -> Path:
    candidates = [
        Path("data/processed") / f"{stock_id}_dataset.csv",
        Path("data/stock_models") / f"{stock_id}_dataset.csv",
        Path("data/external_validation") / f"{stock_id}_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No V2 dataset found for {stock_id}")


def main() -> None:
    manifest_path = RAW_ROOT / "backfill_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_path = RAW_ROOT / "collection_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    archives = {
        stock_id: pd.read_csv(RAW_ROOT / f"{stock_id}.csv", dtype={"stock_id": str})
        for stock_id in SUPPORTED_STOCKS
    }
    readiness = evaluate_readiness(archives, log, manifest)
    if not readiness["ready_for_training"]:
        raise RuntimeError(
            "V6 disclosure history failed the data-readiness gate; run audit_disclosures.py"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for stock_id, disclosures in archives.items():
        disclosures = disclosures[
            disclosures["availability_basis"].eq("official_archive_published_at")
        ].copy()
        dataset_path = _dataset_path(stock_id)
        dataset = pd.read_csv(dataset_path, dtype={"stock_id": str})
        dataset["date"] = pd.to_datetime(dataset["date"], errors="raise")
        history_window = dataset[
            dataset["date"].between("2018-01-01", "2024-12-31")
        ].copy()
        features, assigned = build_daily_disclosure_features(disclosures, history_window)
        merged = history_window.merge(
            features[["date", "stock_id", *DISCLOSURE_FEATURE_COLUMNS]],
            on=["date", "stock_id"],
            how="left",
            validate="one_to_one",
        )
        feature_path = OUTPUT_ROOT / f"{stock_id}_features.csv"
        assigned_path = OUTPUT_ROOT / f"{stock_id}_assigned.csv"
        dataset_output = OUTPUT_ROOT / f"{stock_id}_dataset_v6_ready.csv"
        features.to_csv(feature_path, index=False, encoding="utf-8-sig")
        assigned.to_csv(assigned_path, index=False, encoding="utf-8-sig")
        merged.to_csv(dataset_output, index=False, encoding="utf-8-sig")
        summaries.append(
            {
                "stock_id": stock_id,
                "disclosures": int(len(assigned)),
                "trading_days": int(len(features)),
                "days_with_disclosures": int(features["disclosure_count"].gt(0).sum()),
                "first_feature_date": features["date"].min().strftime("%Y-%m-%d"),
                "last_feature_date": features["date"].max().strftime("%Y-%m-%d"),
            }
        )
    summary_path = REPORT_DIR / "v6_disclosure_feature_coverage.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("V6-ready datasets were created separately; the V2 champion was not changed.")


if __name__ == "__main__":
    main()
