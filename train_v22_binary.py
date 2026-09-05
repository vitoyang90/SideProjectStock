"""Test an actionable up-vs-not-up target against the active three-class V2."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from backtest import DEFAULT_COMMISSION_BPS, DEFAULT_SELL_TAX_BPS
from diagnose_v1 import VALIDATION_YEARS, _walk_forward_split
from optimize_signal_policy import apply_policy
from train import DEFAULT_LABEL_THRESHOLD, REPORT_DIR, _feature_columns
from train_stock_models import REGISTRY_PATH, STOCK_IDS, _load_dataset, select_tree_count


BASELINE_PREDICTION_PATH = Path("data/processed/signal_policy_walk_forward_predictions.csv")
PREDICTION_PATH = Path("data/processed/v22_binary_walk_forward_predictions.csv")
FOLD_PATH = REPORT_DIR / "v22_binary_folds.csv"
NESTED_PATH = REPORT_DIR / "v22_binary_nested.csv"
DECISION_PATH = REPORT_DIR / "v22_binary_decision.json"
MODEL_DIR = Path("models/v22_binary_candidates")
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
OUTER_YEARS = (2022, 2023, 2024)
MIN_INNER_LONG_SIGNALS = 10
REQUIRED_WINS = 2
MAX_SINGLE_YEAR_DEGRADATION = -0.05


def _binary_model(early_stopping: bool = True, n_estimators: int = 1000) -> XGBClassifier:
    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = 50
    return XGBClassifier(**parameters)


def _binary_labels(returns: pd.Series) -> pd.Series:
    return returns.gt(DEFAULT_LABEL_THRESHOLD).astype("int8")


def _net_return(data: pd.DataFrame, long_signal: np.ndarray) -> float:
    intervals = data.sort_values("date").iloc[::5].copy()
    selected_signal = np.asarray(long_signal)[::5]
    cost = (2 * DEFAULT_COMMISSION_BPS + DEFAULT_SELL_TAX_BPS) / 10_000
    returns = np.where(
        selected_signal, intervals["return_5d"].to_numpy(float) - cost, 0.0
    )
    return float(np.prod(1 + returns) - 1)


def binary_metrics(
    data: pd.DataFrame, probability_up: np.ndarray, threshold: float
) -> dict:
    actual = data["binary_target"].to_numpy(int)
    predicted = probability_up >= threshold
    return {
        "threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "long_signals": int(predicted.sum()),
        "net_return": _net_return(data, predicted),
    }


def baseline_metrics(data: pd.DataFrame) -> dict:
    probabilities = data[
        ["probability_down", "probability_neutral", "probability_up"]
    ].to_numpy(float)
    predicted_class, emitted, _ = apply_policy(probabilities, 0.40, 0.05)
    predicted_up = emitted & (predicted_class == 2)
    actual = data["binary_target"].to_numpy(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted_up)),
        "f1": float(f1_score(actual, predicted_up, zero_division=0)),
        "precision": float(precision_score(actual, predicted_up, zero_division=0)),
        "recall": float(recall_score(actual, predicted_up, zero_division=0)),
        "long_signals": int(predicted_up.sum()),
        "net_return": _net_return(data, predicted_up),
    }


def select_threshold(predictions: pd.DataFrame) -> float:
    rows = []
    for threshold in THRESHOLDS:
        metrics = binary_metrics(
            predictions, predictions["binary_probability_up"].to_numpy(float), threshold
        )
        if metrics["long_signals"] >= MIN_INNER_LONG_SIGNALS:
            metrics["selection_score"] = (
                metrics["balanced_accuracy"]
                + metrics["f1"]
                + 0.10 * metrics["precision"]
                + 0.02 * float(np.clip(metrics["net_return"], -1, 1))
            )
            rows.append(metrics)
    if not rows:
        return 0.50
    return float(
        pd.DataFrame(rows)
        .sort_values(
            ["selection_score", "balanced_accuracy", "f1", "net_return"],
            ascending=False,
        )
        .iloc[0]["threshold"]
    )


def promotion_checks(nested: pd.DataFrame) -> dict[str, bool]:
    return {
        "mean_balanced_accuracy_improved": bool(
            nested["delta_balanced_accuracy"].mean() > 0
        ),
        "mean_f1_improved": bool(nested["delta_f1"].mean() > 0),
        "balanced_accuracy_wins_at_least_2_of_3": bool(
            nested["delta_balanced_accuracy"].gt(0).sum() >= REQUIRED_WINS
        ),
        "f1_wins_at_least_2_of_3": bool(
            nested["delta_f1"].gt(0).sum() >= REQUIRED_WINS
        ),
        "worst_balanced_accuracy_delta_at_least_minus_0_05": bool(
            nested["delta_balanced_accuracy"].min() >= MAX_SINGLE_YEAR_DEGRADATION
        ),
        "worst_f1_delta_at_least_minus_0_05": bool(
            nested["delta_f1"].min() >= MAX_SINGLE_YEAR_DEGRADATION
        ),
        "net_return_improved": bool(
            np.prod(1 + nested["binary_net_return"])
            > np.prod(1 + nested["baseline_net_return"])
        ),
    }


def _build_predictions(stock_id: str) -> tuple[pd.DataFrame, list[int]]:
    features = _feature_columns("v2")
    data = _load_dataset(stock_id, "")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data[data["date"] < "2025-01-01"].sort_values("date").dropna(subset=["return_5d"]).copy()
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    data["binary_target"] = _binary_labels(data["return_5d"])
    frames = []
    iterations = []
    for year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(data, year)
        model = _binary_model()
        y_train = train["binary_target"].astype(int)
        y_validation = validation["binary_target"].astype(int)
        model.fit(
            train[features],
            y_train,
            sample_weight=compute_sample_weight("balanced", y_train),
            eval_set=[(validation[features], y_validation)],
            sample_weight_eval_set=[compute_sample_weight("balanced", y_validation)],
            verbose=False,
        )
        probability = model.predict_proba(validation[features])[:, 1]
        frame = validation[["date", "stock_id", "return_5d", "binary_target"]].copy()
        frame["validation_year"] = year
        frame["binary_probability_up"] = probability
        frames.append(frame)
        iterations.append(int(model.best_iteration))
    return pd.concat(frames, ignore_index=True), iterations


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline = pd.read_csv(BASELINE_PREDICTION_PATH, dtype={"stock_id": str})
    baseline["date"] = pd.to_datetime(baseline["date"], errors="raise")
    predictions_by_stock = {}
    iteration_by_stock = {}
    for stock_id in STOCK_IDS:
        predictions, iterations = _build_predictions(stock_id)
        baseline_stock = baseline[baseline["stock_id"].eq(stock_id)][
            [
                "date",
                "probability_down",
                "probability_neutral",
                "probability_up",
            ]
        ]
        predictions = predictions.merge(
            baseline_stock, on="date", how="left", validate="one_to_one"
        )
        predictions_by_stock[stock_id] = predictions
        iteration_by_stock[stock_id] = iterations

    fold_rows = []
    nested_rows = []
    final_thresholds = {}
    for stock_id, predictions in predictions_by_stock.items():
        for year in VALIDATION_YEARS:
            year_data = predictions[predictions["validation_year"] == year]
            for threshold in THRESHOLDS:
                fold_rows.append(
                    {
                        "stock_id": stock_id,
                        "validation_year": year,
                        **binary_metrics(
                            year_data,
                            year_data["binary_probability_up"].to_numpy(float),
                            threshold,
                        ),
                    }
                )
        for outer_year in OUTER_YEARS:
            inner = predictions[predictions["validation_year"] < outer_year]
            outer = predictions[predictions["validation_year"] == outer_year]
            threshold = select_threshold(inner)
            candidate = binary_metrics(
                outer, outer["binary_probability_up"].to_numpy(float), threshold
            )
            active = baseline_metrics(outer)
            nested_rows.append(
                {
                    "stock_id": stock_id,
                    "outer_year": outer_year,
                    "inner_years": ",".join(
                        str(year) for year in VALIDATION_YEARS if year < outer_year
                    ),
                    "selected_threshold": threshold,
                    "binary_balanced_accuracy": candidate["balanced_accuracy"],
                    "baseline_balanced_accuracy": active["balanced_accuracy"],
                    "delta_balanced_accuracy": candidate["balanced_accuracy"]
                    - active["balanced_accuracy"],
                    "binary_f1": candidate["f1"],
                    "baseline_f1": active["f1"],
                    "delta_f1": candidate["f1"] - active["f1"],
                    "binary_precision": candidate["precision"],
                    "baseline_precision": active["precision"],
                    "binary_long_signals": candidate["long_signals"],
                    "baseline_long_signals": active["long_signals"],
                    "binary_net_return": candidate["net_return"],
                    "baseline_net_return": active["net_return"],
                }
            )
        final_thresholds[stock_id] = select_threshold(predictions)

    folds = pd.DataFrame(fold_rows)
    nested = pd.DataFrame(nested_rows)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    decisions = {}
    binary_candidates = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features = _feature_columns("v2")
    for stock_id in STOCK_IDS:
        stock_nested = nested[nested["stock_id"] == stock_id]
        checks = promotion_checks(stock_nested)
        passes = all(checks.values())
        nested_net_return = float(
            np.prod(1 + stock_nested["binary_net_return"]) - 1
        )
        baseline_net_return = float(
            np.prod(1 + stock_nested["baseline_net_return"]) - 1
        )
        data = _load_dataset(stock_id, "")
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        data = data.sort_values("date").dropna(subset=["return_5d"]).copy()
        data["binary_target"] = _binary_labels(data["return_5d"])
        n_estimators = select_tree_count(iteration_by_stock[stock_id])
        model = _binary_model(False, n_estimators)
        y = data["binary_target"].astype(int)
        model.fit(
            data[features], y, sample_weight=compute_sample_weight("balanced", y), verbose=False
        )
        model_path = MODEL_DIR / f"{stock_id}_v22_binary.json"
        model.save_model(model_path)
        decisions[stock_id] = {
            "passes_gate": passes,
            "status": "candidate_forward_only" if passes else "rejected",
            "selected_probability_threshold": final_thresholds[stock_id],
            "n_estimators": n_estimators,
            "model_path": model_path.as_posix(),
            "gate_checks": checks,
            "nested_mean_delta_balanced_accuracy": float(
                stock_nested["delta_balanced_accuracy"].mean()
            ),
            "nested_mean_delta_f1": float(stock_nested["delta_f1"].mean()),
            "nested_mean_balanced_accuracy": float(
                stock_nested["binary_balanced_accuracy"].mean()
            ),
            "baseline_mean_balanced_accuracy": float(
                stock_nested["baseline_balanced_accuracy"].mean()
            ),
            "nested_mean_f1": float(stock_nested["binary_f1"].mean()),
            "baseline_mean_f1": float(stock_nested["baseline_f1"].mean()),
            "nested_net_return": nested_net_return,
            "baseline_net_return": baseline_net_return,
        }
        if passes:
            binary_candidates[stock_id] = {
                "version": "v2.2_binary",
                "artifact_id": (
                    f"{stock_id}_v22_binary_forward_"
                    f"{date.today().isoformat().replace('-', '')}"
                ),
                "status": "shadow_candidate_forward_validation",
                "model_family": "xgboost_binary",
                "model_path": model_path.as_posix(),
                "probability_threshold": final_thresholds[stock_id],
                "target": "future_5d_return_gt_1_5pct",
                "n_estimators": n_estimators,
                "trained_on": date.today().isoformat(),
                "training_end": data["date"].max().strftime("%Y-%m-%d"),
                "historical_nested_net_return": nested_net_return,
                "historical_baseline_net_return": baseline_net_return,
                "warning": "Shadow only; at least 60 matured forward records are required.",
            }
    decision = {
        "version": "v2.2_binary_up_vs_rest",
        "status": "completed",
        "target": "future five-day return greater than +1.5%",
        "baseline": "active V2 three-class up signal with fixed abstention policy",
        "outer_years": list(OUTER_YEARS),
        "selection_uses_only_prior_years": True,
        "test_period_used_for_selection": False,
        "stock_decisions": decisions,
        "warning": "Passing artifacts are candidates for fresh forward comparison, not champion models.",
    }
    registry["v22_binary_experiment"] = {
        "status": "completed",
        "decision_path": DECISION_PATH.as_posix(),
        "candidate_stocks": [
            stock_id for stock_id, result in decisions.items() if result["passes_gate"]
        ],
    }
    registry["binary_candidates"] = binary_candidates
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")
    nested.to_csv(NESTED_PATH, index=False, encoding="utf-8-sig")
    combined = pd.concat(predictions_by_stock.values(), ignore_index=True)
    combined.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
    DECISION_PATH.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return folds, nested, decision


if __name__ == "__main__":
    _, result_nested, result_decision = run_experiment()
    print(result_nested.to_string(index=False))
    print()
    print(json.dumps(result_decision, ensure_ascii=False, indent=2))
