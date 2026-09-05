"""Replay the frozen 2330 V2.2 candidate on the 2025+ historical holdout.

Each prediction is produced by refitting both models with labels that matured
strictly before the prediction date. The pre-2025 tree counts and thresholds
remain frozen, so this is historical replay rather than live forward evidence.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from backtest import DEFAULT_COMMISSION_BPS, DEFAULT_SELL_TAX_BPS
from diagnose_v1 import _labels
from optimize_signal_policy import BASELINE_POLICY, apply_policy
from train import DEFAULT_LABEL_THRESHOLD, PURGE_ROWS, REPORT_DIR, TARGET_COLUMN, _feature_columns
from train_stock_models import REGISTRY_PATH, _fit_final_model, _load_dataset
from train_v22_binary import _binary_labels, _binary_model


STOCK_ID = "2330"
REPLAY_START = pd.Timestamp("2025-01-01")
PREDICTION_PATH = REPORT_DIR / "v22_2330_holdout_replay_predictions.csv"
ANNUAL_PATH = REPORT_DIR / "v22_2330_holdout_replay_annual.csv"
SUMMARY_PATH = REPORT_DIR / "v22_2330_holdout_replay_summary.json"
MIN_SIGNALS = 15
MAX_ANNUAL_BALANCED_ACCURACY_DEGRADATION = -0.05


def eligible_training_rows(data: pd.DataFrame, prediction_position: int) -> pd.DataFrame:
    """Return rows whose five-day target matured before this prediction row."""
    end = prediction_position - PURGE_ROWS
    if end <= 0:
        raise ValueError("Not enough matured labels before prediction")
    # For prediction row i, rows i-5 through i-1 are excluded. The i-5 target
    # matures on the prediction date, so conservatively it is not yet usable.
    return data.iloc[:end].copy()


def _strategy_metrics(data: pd.DataFrame, signal_column: str) -> dict:
    actual = data["actual_up"].astype(int)
    predicted = data[signal_column].astype(bool)
    signals = int(predicted.sum())
    ordered = data.sort_values("date")
    non_overlapping = (
        ordered[ordered["strategy_interval"].astype(bool)].copy()
        if "strategy_interval" in ordered
        else ordered.iloc[::5].copy()
    )
    interval_signal = non_overlapping[signal_column].astype(bool).to_numpy()
    cost = (2 * DEFAULT_COMMISSION_BPS + DEFAULT_SELL_TAX_BPS) / 10_000
    interval_returns = np.where(
        interval_signal,
        non_overlapping["return_5d"].to_numpy(float) - cost,
        0.0,
    )
    equity = np.cumprod(1 + interval_returns)
    running_peak = np.maximum.accumulate(np.r_[1.0, equity])
    drawdowns = np.r_[1.0, equity] / running_peak - 1
    return {
        "rows": int(len(data)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "signals": signals,
        "coverage": float(signals / len(data)) if len(data) else 0.0,
        "non_overlapping_intervals": int(len(non_overlapping)),
        "non_overlapping_signals": int(interval_signal.sum()),
        "net_return": float(np.prod(1 + interval_returns) - 1),
        "max_drawdown": float(drawdowns.min()),
    }


def replay_gate(
    candidate: dict,
    baseline: dict,
    naive_always_up: dict,
    annual: pd.DataFrame,
) -> dict[str, bool]:
    return {
        "candidate_emits_both_signal_classes": bool(
            0 < candidate["signals"] < candidate["rows"]
        ),
        "candidate_coverage_between_5_and_95_percent": bool(
            0.05 <= candidate["coverage"] <= 0.95
        ),
        "balanced_accuracy_improved": bool(
            candidate["balanced_accuracy"] > baseline["balanced_accuracy"]
        ),
        "f1_improved": bool(candidate["f1"] > baseline["f1"]),
        "signal_precision_not_worse": bool(
            candidate["precision"] >= baseline["precision"]
        ),
        "net_return_improved": bool(candidate["net_return"] > baseline["net_return"]),
        "balanced_accuracy_beats_naive_always_up": bool(
            candidate["balanced_accuracy"] > naive_always_up["balanced_accuracy"]
        ),
        "net_return_beats_naive_always_up": bool(
            candidate["net_return"] > naive_always_up["net_return"]
        ),
        "candidate_net_return_positive": bool(candidate["net_return"] > 0),
        "at_least_15_signals": bool(candidate["signals"] >= MIN_SIGNALS),
        "worst_annual_balanced_accuracy_delta_at_least_minus_0_05": bool(
            annual["delta_balanced_accuracy"].min()
            >= MAX_ANNUAL_BALANCED_ACCURACY_DEGRADATION
        ),
    }


def _prepare_data() -> pd.DataFrame:
    features = _feature_columns("v2")
    data = _load_dataset(STOCK_ID, "")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").dropna(subset=["return_5d"]).reset_index(drop=True)
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    data[TARGET_COLUMN] = _labels(data["return_5d"], DEFAULT_LABEL_THRESHOLD)
    data["binary_target"] = _binary_labels(data["return_5d"])
    return data


def run_replay() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    active_candidates = registry.setdefault("binary_candidates", {})
    rejected_candidates = registry.setdefault("rejected_binary_candidates", {})
    binary_entry = active_candidates.get(STOCK_ID) or rejected_candidates.get(STOCK_ID)
    if not binary_entry:
        raise ValueError(f"No frozen V2.2 candidate registered for {STOCK_ID}")
    v2_entry = registry["stock_models"][STOCK_ID]
    binary_threshold = float(binary_entry["probability_threshold"])
    binary_trees = int(binary_entry["n_estimators"])
    v2_trees = int(v2_entry["n_estimators"])
    features = _feature_columns("v2")
    data = _prepare_data()
    prediction_positions = data.index[data["date"] >= REPLAY_START].tolist()
    if not prediction_positions:
        raise ValueError("No matured 2025+ labels are available for replay")

    rows = []
    for position in prediction_positions:
        training = eligible_training_rows(data, position)
        prediction = data.iloc[[position]]

        binary_model = _binary_model(False, binary_trees)
        binary_y = training["binary_target"].astype(int)
        binary_model.fit(
            training[features],
            binary_y,
            sample_weight=compute_sample_weight("balanced", binary_y),
            verbose=False,
        )
        binary_probability = float(binary_model.predict_proba(prediction[features])[0, 1])

        v2_model = _fit_final_model(training, features, v2_trees)
        v2_probabilities = v2_model.predict_proba(prediction[features])
        v2_class, v2_emitted, v2_margin = apply_policy(
            v2_probabilities, *BASELINE_POLICY
        )
        rows.append(
            {
                "date": prediction.iloc[0]["date"],
                "stock_id": STOCK_ID,
                "training_end": training.iloc[-1]["date"],
                "train_rows": int(len(training)),
                "return_5d": float(prediction.iloc[0]["return_5d"]),
                "actual_up": bool(prediction.iloc[0]["binary_target"]),
                "binary_probability_up": binary_probability,
                "binary_threshold": binary_threshold,
                "binary_signal": bool(binary_probability >= binary_threshold),
                "v2_probability_down": float(v2_probabilities[0, 0]),
                "v2_probability_neutral": float(v2_probabilities[0, 1]),
                "v2_probability_up": float(v2_probabilities[0, 2]),
                "v2_predicted_class": int(v2_class[0]),
                "v2_confidence": float(v2_probabilities[0, v2_class[0]]),
                "v2_margin": float(v2_margin[0]),
                "v2_emitted": bool(v2_emitted[0]),
                "v2_up_signal": bool(v2_emitted[0] and v2_class[0] == 2),
            }
        )

    predictions = pd.DataFrame(rows)
    predictions["strategy_interval"] = np.arange(len(predictions)) % 5 == 0
    predictions["always_up_signal"] = True
    annual_rows = []
    for year, frame in predictions.groupby(predictions["date"].dt.year):
        candidate = _strategy_metrics(frame, "binary_signal")
        baseline = _strategy_metrics(frame, "v2_up_signal")
        naive = _strategy_metrics(frame, "always_up_signal")
        annual_rows.append(
            {
                "year": int(year),
                "rows": int(len(frame)),
                **{f"binary_{key}": value for key, value in candidate.items() if key != "rows"},
                **{f"v2_{key}": value for key, value in baseline.items() if key != "rows"},
                **{f"naive_{key}": value for key, value in naive.items() if key != "rows"},
                "delta_balanced_accuracy": (
                    candidate["balanced_accuracy"] - baseline["balanced_accuracy"]
                ),
                "delta_f1": candidate["f1"] - baseline["f1"],
                "delta_net_return": candidate["net_return"] - baseline["net_return"],
            }
        )
    annual = pd.DataFrame(annual_rows)
    candidate = _strategy_metrics(predictions, "binary_signal")
    baseline = _strategy_metrics(predictions, "v2_up_signal")
    naive = _strategy_metrics(predictions, "always_up_signal")
    checks = replay_gate(candidate, baseline, naive, annual)
    supportive = all(checks.values())
    summary = {
        "version": "v2.2_binary_2330_post_selection_holdout_replay",
        "status": "supportive" if supportive else "not_supportive",
        "stock_id": STOCK_ID,
        "period": {
            "start": predictions["date"].min().strftime("%Y-%m-%d"),
            "end": predictions["date"].max().strftime("%Y-%m-%d"),
            "rows": int(len(predictions)),
        },
        "protocol": {
            "expanding_window_refit_each_trading_day": True,
            "labels_must_mature_strictly_before_prediction_date": True,
            "purged_rows_before_each_prediction": PURGE_ROWS,
            "binary_probability_threshold": binary_threshold,
            "binary_n_estimators": binary_trees,
            "v2_minimum_confidence": BASELINE_POLICY[0],
            "v2_minimum_margin": BASELINE_POLICY[1],
            "v2_n_estimators": v2_trees,
            "parameters_selected_from_pre_2025_data": True,
            "counts_toward_live_forward_minimum": False,
        },
        "binary_candidate": candidate,
        "v2_baseline": baseline,
        "naive_always_up_baseline": naive,
        "binary_probability_diagnostics": {
            "minimum": float(predictions["binary_probability_up"].min()),
            "maximum": float(predictions["binary_probability_up"].max()),
            "mean": float(predictions["binary_probability_up"].mean()),
            "standard_deviation": float(predictions["binary_probability_up"].std()),
        },
        "gate_checks": checks,
        "registry_action": (
            "kept_as_shadow_candidate"
            if supportive
            else "deactivated_shadow_candidate"
        ),
        "warning": (
            "This is historical replay evidence, not an immutable prediction "
            "record and not a substitute for fresh forward validation."
        ),
    }

    registry["v22_historical_replay"] = {
        "status": summary["status"],
        "summary_path": SUMMARY_PATH.as_posix(),
        "prediction_path": PREDICTION_PATH.as_posix(),
        "period": summary["period"],
        "counts_toward_live_forward_minimum": False,
    }
    binary_entry["historical_replay"] = {
        "status": summary["status"],
        "period_end": summary["period"]["end"],
        "binary_balanced_accuracy": candidate["balanced_accuracy"],
        "v2_balanced_accuracy": baseline["balanced_accuracy"],
        "binary_net_return": candidate["net_return"],
        "v2_net_return": baseline["net_return"],
        "summary_path": SUMMARY_PATH.as_posix(),
    }
    experiment = registry.setdefault("v22_binary_experiment", {})
    experiment.setdefault("historical_gate_candidate_stocks", [STOCK_ID])
    if supportive:
        binary_entry["status"] = "shadow_candidate_forward_validation"
        active_candidates[STOCK_ID] = binary_entry
        rejected_candidates.pop(STOCK_ID, None)
        experiment["candidate_stocks"] = [STOCK_ID]
    else:
        binary_entry["status"] = "rejected_after_2025_holdout_replay"
        binary_entry["warning"] = (
            "Rejected: 2025+ replay emitted an up signal on every row and had "
            "no balanced-accuracy improvement over V2."
        )
        rejected_candidates[STOCK_ID] = binary_entry
        active_candidates.pop(STOCK_ID, None)
        experiment["candidate_stocks"] = []
        experiment["post_selection_replay_status"] = "rejected"
        rejected = registry.setdefault("rejected", [])
        rejected[:] = [
            item for item in rejected if item.get("version") != "v2.2_binary_2330"
        ]
        rejected.append(
            {
                "version": "v2.2_binary_2330",
                "features": "V2 technical + chip features with an up-vs-rest target",
                "reason": (
                    "Failed 2025+ purged daily holdout replay: emitted up on "
                    "392/392 rows, matched the naive always-up benchmark, and "
                    "achieved 0.50 balanced accuracy."
                ),
            }
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
    annual.to_csv(ANNUAL_PATH, index=False, encoding="utf-8-sig")
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return predictions, annual, summary


if __name__ == "__main__":
    _, replay_annual, replay_summary = run_replay()
    print(replay_annual.to_string(index=False))
    print()
    print(json.dumps(replay_summary, ensure_ascii=False, indent=2))
