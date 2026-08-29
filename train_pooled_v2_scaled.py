"""Pooled V2 experiment with per-stock, backward-looking chip scaling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train import REPORT_DIR
from train_pooled_v2 import (
    STOCK_IDS,
    _backtest_comparison,
    _classification_comparison,
    _fit,
    _load_data,
    _retrospective_split,
    _run_backtests,
    _score,
    _walk_forward_split,
)
from diagnose_v1 import VALIDATION_YEARS


MODEL_PATH = Path("models/xgboost_v2_pooled_scaled_retrospective.json")
DETAIL_PATH = REPORT_DIR / "pooled_v2_scaled_classification.csv"
SUMMARY_PATH = REPORT_DIR / "pooled_v2_scaled_classification_summary.csv"
CLASSIFICATION_COMPARISON_PATH = (
    REPORT_DIR / "pooled_v2_scaled_vs_independent_classification.csv"
)
BACKTEST_PATH = REPORT_DIR / "pooled_v2_scaled_backtest.csv"
BACKTEST_COMPARISON_PATH = (
    REPORT_DIR / "pooled_v2_scaled_vs_independent_backtest.csv"
)
DECISION_PATH = REPORT_DIR / "pooled_v2_scaled_decision.json"
PREDICTION_PATH = Path("data/processed/pooled_v2_scaled_predictions.csv")

DAILY_FLOW_COLUMNS = [
    "foreign_net", "trust_net", "dealer_net", "institutional_total_net",
    "margin_change", "short_change",
]
THREE_DAY_FLOW_COLUMNS = ["foreign_3d", "trust_3d", "dealer_3d"]
FIVE_DAY_FLOW_COLUMNS = ["foreign_5d", "trust_5d", "dealer_5d"]
BALANCE_COLUMNS = ["margin_balance", "short_balance"]
SCALED_CHIP_COLUMNS = [
    f"{column}_scaled"
    for column in (
        DAILY_FLOW_COLUMNS
        + THREE_DAY_FLOW_COLUMNS
        + FIVE_DAY_FLOW_COLUMNS
        + BALANCE_COLUMNS
    )
]
SCALE_FREE_TECHNICAL_COLUMNS = [
    "return_1d", "return_5d_past", "ma5_ratio", "ma20_ratio", "rsi_14",
    "macd_ratio", "macd_signal_ratio", "macd_hist_ratio", "atr_ratio",
    "volatility_20", "volume_ratio_20",
]
FEATURES = SCALE_FREE_TECHNICAL_COLUMNS + SCALED_CHIP_COLUMNS


def _scale_stock(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    volume = pd.to_numeric(group["Trading_Volume"], errors="coerce").replace(
        0, np.nan,
    )
    volume_3d = volume.rolling(3, min_periods=3).sum()
    volume_5d = volume.rolling(5, min_periods=5).sum()
    average_volume_20 = volume.rolling(20, min_periods=20).mean()
    close = pd.to_numeric(group["close"], errors="coerce").replace(0, np.nan)
    group["macd_ratio"] = group["macd"] / close
    group["macd_signal_ratio"] = group["macd_signal"] / close
    group["macd_hist_ratio"] = group["macd_hist"] / close
    for column in DAILY_FLOW_COLUMNS:
        group[f"{column}_scaled"] = group[column] / volume
    for column in THREE_DAY_FLOW_COLUMNS:
        group[f"{column}_scaled"] = group[column] / volume_3d
    for column in FIVE_DAY_FLOW_COLUMNS:
        group[f"{column}_scaled"] = group[column] / volume_5d
    for column in BALANCE_COLUMNS:
        group[f"{column}_scaled"] = group[column] / average_volume_20
    return group


def _scale_data(data: pd.DataFrame) -> pd.DataFrame:
    scaled = pd.concat(
        [_scale_stock(group) for _, group in data.groupby("stock_id")],
        ignore_index=True,
    )
    scaled[FEATURES] = scaled[FEATURES].replace([np.inf, -np.inf], np.nan)
    return scaled


def run_scaled_experiment() -> dict:
    data, prices = _load_data()
    data = _scale_data(data)
    classification_rows = []
    prediction_frames = []
    for year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(data, year)
        model = _fit(train, validation, FEATURES)
        rows, predictions = _score(
            model,
            validation,
            "walk_forward",
            year,
            FEATURES,
            "v2_pooled_scaled",
        )
        classification_rows.extend(rows)
        prediction_frames.append(predictions)

    train, validation, test = _retrospective_split(data)
    retrospective_model = _fit(train, validation, FEATURES)
    rows, predictions = _score(
        retrospective_model,
        test,
        "external_test",
        None,
        FEATURES,
        "v2_pooled_scaled",
    )
    classification_rows.extend(rows)
    prediction_frames.append(predictions)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    retrospective_model.save_model(MODEL_PATH)

    classification = pd.DataFrame(classification_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = (
        classification.groupby("evaluation", as_index=False)
        .agg(
            observations=("stock_id", "count"),
            stocks=("stock_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_mlogloss=("mlogloss", "mean"),
        )
    )
    classification_comparison = _classification_comparison(classification)
    backtests = _run_backtests(
        predictions, prices, strategy="v2_pooled_scaled",
    )
    backtest_comparison = _backtest_comparison(backtests)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    classification.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    classification_comparison.to_csv(
        CLASSIFICATION_COMPARISON_PATH, index=False, encoding="utf-8-sig",
    )
    backtests.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    backtest_comparison.to_csv(
        BACKTEST_COMPARISON_PATH, index=False, encoding="utf-8-sig",
    )
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    predictions.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")

    test_classification = classification_comparison[
        classification_comparison["evaluation"] == "external_test"
    ]
    test_backtest = backtest_comparison[
        backtest_comparison["split"] == "external_test"
    ]
    decision = {
        "version": "v2_pooled_scaled",
        "status": "rejected_retrospective",
        "stocks": list(STOCK_IDS),
        "feature_count": len(FEATURES),
        "scaling": {
            "price_level_indicators": "divide MACD family by same-day close; use ATR ratio",
            "daily_flows": "divide by same-day volume",
            "rolling_flows": "divide by matching rolling volume sum",
            "balances": "divide by trailing 20-day average volume",
            "future_data_used": False,
        },
        "warning": "2025-2026 was previously viewed; fresh forward data is required for promotion",
        "external_test": {
            "classification_balanced_accuracy_wins": int(
                (test_classification["delta_balanced_accuracy"] > 0).sum()
            ),
            "classification_macro_f1_wins": int(
                (test_classification["delta_macro_f1"] > 0).sum()
            ),
            "backtest_return_wins": int(
                (test_backtest["delta_net_total_return"] > 0).sum()
            ),
            "stocks": len(STOCK_IDS),
            "mean_delta_balanced_accuracy": float(
                test_classification["delta_balanced_accuracy"].mean()
            ),
            "mean_delta_macro_f1": float(
                test_classification["delta_macro_f1"].mean()
            ),
            "mean_delta_net_return": float(
                test_backtest["delta_net_total_return"].mean()
            ),
        },
        "model_path": MODEL_PATH.as_posix(),
    }
    DECISION_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    result = run_scaled_experiment()
    print(json.dumps(result, indent=2, ensure_ascii=False))
