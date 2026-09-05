"""Score immutable forward predictions only after five trading days mature."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from analyze import FORWARD_DIR, _cached_paths
from train import CLASS_NAMES, DEFAULT_LABEL_THRESHOLD, REPORT_DIR


ROW_REPORT_PATH = REPORT_DIR / "forward_evaluation_rows.csv"
SUMMARY_PATH = REPORT_DIR / "forward_evaluation_summary.json"


def _actual_class(value: float) -> int:
    if value < -DEFAULT_LABEL_THRESHOLD:
        return 0
    if value > DEFAULT_LABEL_THRESHOLD:
        return 2
    return 1


def evaluate_prediction_file(path: Path, evaluated_at: datetime | None = None) -> pd.DataFrame:
    predictions = pd.read_csv(path, dtype={"stock_id": str})
    if predictions.empty:
        return pd.DataFrame()
    stock_id = str(predictions["stock_id"].iloc[0])
    prices = pd.read_csv(_cached_paths(stock_id)["price"])
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.normalize()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["close"]).drop_duplicates("date").sort_values("date")
    price_by_date = prices.set_index("date")
    calendar = prices["date"].tolist()
    positions = {item: index for index, item in enumerate(calendar)}
    now = pd.Timestamp(evaluated_at or datetime.now().astimezone())
    rows = []
    for _, prediction in predictions.iterrows():
        origin = pd.Timestamp(prediction["prediction_date"]).normalize()
        position = positions.get(origin)
        maturity = calendar[position + 5] if position is not None and position + 5 < len(calendar) else pd.NaT
        recorded_at = pd.Timestamp(prediction["recorded_at"])
        if recorded_at.tzinfo is not None:
            recorded_at = recorded_at.tz_convert(None)
        valid_forward_record = bool(pd.notna(maturity) and recorded_at < maturity)
        matured = bool(valid_forward_record and maturity <= now.tz_localize(None).normalize())
        future_close = float(price_by_date.loc[maturity, "close"]) if matured else np.nan
        origin_close = float(prediction["close"])
        actual_return = future_close / origin_close - 1 if matured else np.nan
        actual_class = _actual_class(actual_return) if matured else np.nan
        probability_columns = ["probability_down", "probability_neutral", "probability_up"]
        probabilities = prediction[probability_columns].astype(float).to_numpy()
        rows.append(
            {
                **prediction.to_dict(),
                "maturity_date": maturity.strftime("%Y-%m-%d") if pd.notna(maturity) else None,
                "valid_forward_record": valid_forward_record,
                "matured": matured,
                "future_close_5d": future_close,
                "actual_return_5d": actual_return,
                "actual_class": actual_class,
                "correct_raw_class": bool(np.argmax(probabilities) == actual_class) if matured else None,
                "correct_emitted_signal": (
                    bool(np.argmax(probabilities) == actual_class)
                    if matured and not bool(prediction["abstained"])
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> dict:
    valid = rows[rows["valid_forward_record"].astype(bool)] if not rows.empty else rows
    matured = valid[valid["matured"].astype(bool)] if not valid.empty else valid
    emitted = matured[~matured["abstained"].astype(bool)] if not matured.empty else matured
    result = {
        "status": "has_matured_predictions" if len(matured) else "waiting_for_five_trading_days",
        "prediction_records": int(len(rows)),
        "valid_forward_records": int(len(valid)),
        "matured_records": int(len(matured)),
        "emitted_matured_signals": int(len(emitted)),
        "raw_accuracy": None,
        "raw_balanced_accuracy": None,
        "raw_log_loss": None,
        "emitted_signal_accuracy": None,
        "warning": "Promotion decisions require materially more than one five-day outcome per stock.",
    }
    if len(matured):
        truth = matured["actual_class"].astype(int)
        predicted = matured[
            ["probability_down", "probability_neutral", "probability_up"]
        ].to_numpy(float).argmax(axis=1)
        result["raw_accuracy"] = float(accuracy_score(truth, predicted))
        result["raw_balanced_accuracy"] = float(balanced_accuracy_score(truth, predicted))
        result["raw_log_loss"] = float(
            log_loss(
                truth,
                matured[["probability_down", "probability_neutral", "probability_up"]],
                labels=range(len(CLASS_NAMES)),
            )
        )
    if len(emitted):
        result["emitted_signal_accuracy"] = float(
            emitted["correct_emitted_signal"].astype(bool).mean()
        )
    return result


def main() -> None:
    frames = [
        evaluate_prediction_file(path)
        for path in sorted(FORWARD_DIR.glob("*_predictions.csv"))
    ]
    rows = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(ROW_REPORT_PATH, index=False, encoding="utf-8-sig")
    summary = summarize(rows)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
