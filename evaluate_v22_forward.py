"""Evaluate matured V2.2 binary shadow predictions."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import FORWARD_DIR, _cached_paths
from backtest import DEFAULT_COMMISSION_BPS, DEFAULT_SELL_TAX_BPS
from train import DEFAULT_LABEL_THRESHOLD, REPORT_DIR


ROW_PATH = REPORT_DIR / "v22_forward_evaluation_rows.csv"
SUMMARY_PATH = REPORT_DIR / "v22_forward_evaluation_summary.json"


def evaluate_file(path: Path, evaluated_at: datetime | None = None) -> pd.DataFrame:
    records = pd.read_csv(path, dtype={"stock_id": str})
    stock_id = str(records["stock_id"].iloc[0])
    prices = pd.read_csv(_cached_paths(stock_id)["price"])
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    calendar = prices["date"].tolist()
    positions = {day: index for index, day in enumerate(calendar)}
    closes = prices.set_index("date")["close"]
    now = pd.Timestamp(evaluated_at or datetime.now().astimezone())
    if now.tzinfo is not None:
        now = now.tz_localize(None)
    rows = []
    cost = (2 * DEFAULT_COMMISSION_BPS + DEFAULT_SELL_TAX_BPS) / 10_000
    for _, record in records.iterrows():
        origin = pd.Timestamp(record["prediction_date"]).normalize()
        position = positions.get(origin)
        maturity = calendar[position + 5] if position is not None and position + 5 < len(calendar) else pd.NaT
        recorded_at = pd.Timestamp(record["recorded_at"])
        if recorded_at.tzinfo is not None:
            recorded_at = recorded_at.tz_convert(None)
        valid = bool(pd.notna(maturity) and recorded_at < maturity)
        matured = bool(valid and maturity <= now.normalize())
        actual_return = float(closes.loc[maturity] / record["close"] - 1) if matured else np.nan
        actual_up = bool(actual_return > DEFAULT_LABEL_THRESHOLD) if matured else None
        signal = bool(record["signal_emitted"])
        rows.append(
            {
                **record.to_dict(),
                "maturity_date": maturity.strftime("%Y-%m-%d") if pd.notna(maturity) else None,
                "valid_forward_record": valid,
                "matured": matured,
                "actual_return_5d": actual_return,
                "actual_up": actual_up,
                "correct": bool(signal == actual_up) if matured else None,
                "net_strategy_return": actual_return - cost if matured and signal else (0.0 if matured else np.nan),
            }
        )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> dict:
    valid = rows[rows["valid_forward_record"].astype(bool)] if not rows.empty else rows
    matured = valid[valid["matured"].astype(bool)] if not valid.empty else valid
    signals = matured[matured["signal_emitted"].astype(bool)] if not matured.empty else matured
    return {
        "status": "has_matured_predictions" if len(matured) else "waiting_for_five_trading_days",
        "records": int(len(rows)),
        "valid_forward_records": int(len(valid)),
        "matured_records": int(len(matured)),
        "signals_emitted": int(len(signals)),
        "binary_accuracy": float(matured["correct"].astype(bool).mean()) if len(matured) else None,
        "signal_precision": float(signals["actual_up"].astype(bool).mean()) if len(signals) else None,
        "net_strategy_return": float(np.prod(1 + matured["net_strategy_return"]) - 1) if len(matured) else None,
        "promotion_ready": bool(len(matured) >= 60 and len(signals) >= 15),
        "promotion_minimums": {"matured_records": 60, "emitted_signals": 15},
    }


def main() -> None:
    frames = [evaluate_file(path) for path in FORWARD_DIR.glob("*_v22_binary_predictions.csv")]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(ROW_PATH, index=False, encoding="utf-8-sig")
    summary = summarize(rows)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
