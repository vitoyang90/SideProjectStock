"""Test whether market context adds value to the retained V2 champion."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from backtest import (
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SELL_TAX_BPS,
    PRICE_PATH,
    _build_intervals,
    _simulate,
)
from diagnose_v1 import VALIDATION_YEARS, _build_model, _labels, _walk_forward_split
from train import (
    CLASS_NAMES,
    DATA_PATH,
    DEFAULT_LABEL_THRESHOLD,
    FEATURE_GROUPS,
    REPORT_DIR,
    TARGET_COLUMN,
    _class_counts,
    _feature_columns,
    _sample_weights,
)


REGISTRY_PATH = Path("models/registry.json")
PREDICTION_PATH = Path("data/processed/v2_market_walk_forward_predictions.csv")
FOLD_PATH = REPORT_DIR / "v2_market_walk_forward_folds.csv"
SUMMARY_PATH = REPORT_DIR / "v2_market_walk_forward_summary.csv"
BACKTEST_PATH = REPORT_DIR / "v2_market_backtest.csv"
BACKTEST_YEARLY_PATH = REPORT_DIR / "v2_market_backtest_yearly.csv"
DECISION_PATH = REPORT_DIR / "v2_market_decision.json"

BASELINE = "v2"
CANDIDATE = "v2_market"
CLASS_WEIGHTING = "balanced"
REQUIRED_FOLD_WINS = 3
MAX_ALLOWED_FOLD_DEGRADATION = -0.03


def _fit_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[object, np.ndarray, np.ndarray]:
    model = _build_model()
    y_train = train[TARGET_COLUMN].astype(int)
    y_validation = validation[TARGET_COLUMN].astype(int)
    model.fit(
        train[features],
        y_train,
        sample_weight=_sample_weights(y_train, CLASS_WEIGHTING),
        eval_set=[(validation[features], y_validation)],
        sample_weight_eval_set=[
            _sample_weights(y_validation, CLASS_WEIGHTING)
        ],
        verbose=False,
    )
    return (
        model,
        model.predict(validation[features]).astype(int),
        model.predict_proba(validation[features]),
    )


def _summarize(folds: pd.DataFrame) -> pd.DataFrame:
    summary = (
        folds.groupby("version", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            feature_count=("feature_count", "first"),
            mean_best_iteration=("best_iteration", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_mlogloss=("mlogloss", "mean"),
        )
        .sort_values("version")
    )
    baseline = summary.loc[summary["version"] == BASELINE].iloc[0]
    summary["delta_balanced_accuracy_vs_v2"] = (
        summary["mean_balanced_accuracy"] - baseline["mean_balanced_accuracy"]
    )
    summary["delta_macro_f1_vs_v2"] = (
        summary["mean_macro_f1"] - baseline["mean_macro_f1"]
    )
    summary["delta_mlogloss_vs_v2"] = (
        summary["mean_mlogloss"] - baseline["mean_mlogloss"]
    )
    return summary


def _run_backtest(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(PRICE_PATH)
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices = prices.sort_values("date").reset_index(drop=True)
    metric_rows = []
    curve_frames = []
    for version in (BASELINE, CANDIDATE):
        candidate = predictions[predictions["version"] == version].copy()
        intervals = _build_intervals(candidate, prices)
        metrics, curve, _ = _simulate(
            intervals,
            "walk_forward_2021_2024",
            version,
            DEFAULT_COMMISSION_BPS,
            DEFAULT_SELL_TAX_BPS,
        )
        metric_rows.append(metrics)
        curve_frames.append(curve)

    baseline_predictions = predictions[predictions["version"] == BASELINE].copy()
    intervals = _build_intervals(baseline_predictions, prices, always_long=True)
    metrics, curve, _ = _simulate(
        intervals,
        "walk_forward_2021_2024",
        "buy_hold",
        DEFAULT_COMMISSION_BPS,
        DEFAULT_SELL_TAX_BPS,
    )
    metric_rows.append(metrics)
    curve_frames.append(curve)

    metrics_frame = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    curves["date"] = pd.to_datetime(curves["date"])
    curves["year"] = curves["date"].dt.year
    yearly = (
        curves.groupby(["strategy", "year"], as_index=False)
        .agg(
            intervals=("date", "count"),
            gross_return=(
                "period_gross_return", lambda s: float((1 + s).prod() - 1)
            ),
            net_return=(
                "period_net_return", lambda s: float((1 + s).prod() - 1)
            ),
        )
    )
    return metrics_frame, yearly


def evaluate_gate(
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    backtest: pd.DataFrame,
) -> tuple[dict[str, bool], pd.DataFrame]:
    pivot = folds.pivot(
        index="validation_year",
        columns="version",
        values=["balanced_accuracy", "macro_f1"],
    )
    deltas = pd.DataFrame({
        "validation_year": pivot.index,
        "delta_balanced_accuracy": (
            pivot["balanced_accuracy"][CANDIDATE]
            - pivot["balanced_accuracy"][BASELINE]
        ).to_numpy(),
        "delta_macro_f1": (
            pivot["macro_f1"][CANDIDATE]
            - pivot["macro_f1"][BASELINE]
        ).to_numpy(),
    })
    candidate = summary.loc[summary["version"] == CANDIDATE].iloc[0]
    returns = backtest.set_index("strategy")["net_total_return"]
    checks = {
        "mean_balanced_accuracy_improved": bool(
            candidate["delta_balanced_accuracy_vs_v2"] > 0
        ),
        "mean_macro_f1_improved": bool(candidate["delta_macro_f1_vs_v2"] > 0),
        "balanced_accuracy_wins_at_least_3_of_4": bool(
            deltas["delta_balanced_accuracy"].gt(0).sum() >= REQUIRED_FOLD_WINS
        ),
        "macro_f1_wins_at_least_3_of_4": bool(
            deltas["delta_macro_f1"].gt(0).sum() >= REQUIRED_FOLD_WINS
        ),
        "worst_balanced_accuracy_delta_at_least_minus_0_03": bool(
            deltas["delta_balanced_accuracy"].min()
            >= MAX_ALLOWED_FOLD_DEGRADATION
        ),
        "worst_macro_f1_delta_at_least_minus_0_03": bool(
            deltas["delta_macro_f1"].min() >= MAX_ALLOWED_FOLD_DEGRADATION
        ),
        "net_return_improved_after_costs": bool(
            returns[CANDIDATE] > returns[BASELINE]
        ),
    }
    return checks, deltas


def _update_registry(decision: dict) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["rejected"] = [
        item for item in registry.get("rejected", [])
        if item.get("version") != CANDIDATE
    ]
    if (registry.get("candidate") or {}).get("version") == CANDIDATE:
        registry["candidate"] = None

    if decision["passes_gate"]:
        if registry.get("candidate") not in (None, {}):
            raise ValueError("A different model candidate already exists in registry")
        registry["candidate"] = {
            "version": CANDIDATE,
            "status": "candidate_requiring_fresh_forward_validation",
            "features": "V2 technical + chip + market context",
            "reason": "Passed the predeclared 2021-2024 walk-forward gate",
        }
    else:
        failed = [name for name, passed in decision["gate_checks"].items() if not passed]
        registry["rejected"].append({
            "version": CANDIDATE,
            "features": "V2 technical + chip + market context",
            "reason": "Failed historical walk-forward gate: " + ", ".join(failed),
        })
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    features_by_version = {
        BASELINE: _feature_columns("v2"),
        CANDIDATE: _feature_columns("v2") + FEATURE_GROUPS["market"],
    }
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    required = {
        "date", "stock_id", "return_5d",
        *features_by_version[CANDIDATE],
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    data = data.sort_values("date").dropna(subset=["return_5d"])
    data = data[data["date"] < "2025-01-01"].copy()
    data[TARGET_COLUMN] = _labels(data["return_5d"], DEFAULT_LABEL_THRESHOLD)
    all_features = features_by_version[CANDIDATE]
    data[all_features] = data[all_features].replace([np.inf, -np.inf], np.nan)

    fold_rows = []
    prediction_frames = []
    for validation_year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(data, validation_year)
        for version, features in features_by_version.items():
            model, predicted, probabilities = _fit_fold(train, validation, features)
            labels = validation[TARGET_COLUMN].astype(int)
            counts = _class_counts(pd.Series(predicted))
            fold_rows.append({
                "version": version,
                "validation_year": validation_year,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "feature_count": len(features),
                "best_iteration": int(model.best_iteration),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, predicted)
                ),
                "macro_f1": float(f1_score(labels, predicted, average="macro")),
                "mlogloss": float(
                    log_loss(labels, probabilities, labels=range(len(CLASS_NAMES)))
                ),
                "predicted_down": counts["down"],
                "predicted_neutral": counts["neutral"],
                "predicted_up": counts["up"],
            })
            frame = validation[["date", "stock_id", TARGET_COLUMN]].copy()
            frame["version"] = version
            frame["validation_year"] = validation_year
            frame["predicted_class"] = predicted
            frame["probability_down"] = probabilities[:, 0]
            frame["probability_neutral"] = probabilities[:, 1]
            frame["probability_up"] = probabilities[:, 2]
            prediction_frames.append(frame)

    folds = pd.DataFrame(fold_rows)
    summary = _summarize(folds)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    backtest, yearly = _run_backtest(predictions)
    gate_checks, fold_deltas = evaluate_gate(folds, summary, backtest)
    passes_gate = all(gate_checks.values())
    candidate_summary = summary.loc[summary["version"] == CANDIDATE].iloc[0]
    returns = backtest.set_index("strategy")["net_total_return"]
    decision = {
        "version": CANDIDATE,
        "status": (
            "candidate_requiring_fresh_forward_validation"
            if passes_gate else "rejected_historical_walk_forward"
        ),
        "passes_gate": passes_gate,
        "comparison": "V2 versus V2 plus market context",
        "validation_years": list(VALIDATION_YEARS),
        "test_period_used_for_selection": False,
        "market_features": FEATURE_GROUPS["market"],
        "gate_policy": {
            "required_fold_wins": REQUIRED_FOLD_WINS,
            "folds": len(VALIDATION_YEARS),
            "maximum_allowed_single_fold_degradation": abs(
                MAX_ALLOWED_FOLD_DEGRADATION
            ),
            "all_checks_must_pass": True,
        },
        "gate_checks": gate_checks,
        "candidate_delta_vs_v2": {
            "mean_balanced_accuracy": float(
                candidate_summary["delta_balanced_accuracy_vs_v2"]
            ),
            "mean_macro_f1": float(candidate_summary["delta_macro_f1_vs_v2"]),
            "mean_mlogloss": float(candidate_summary["delta_mlogloss_vs_v2"]),
            "net_total_return": float(returns[CANDIDATE] - returns[BASELINE]),
        },
        "fold_deltas": fold_deltas.to_dict(orient="records"),
        "warning": (
            "Historical research only. A passing result requires fresh forward "
            "data before promotion to champion."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    yearly.to_csv(BACKTEST_YEARLY_PATH, index=False, encoding="utf-8-sig")
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")
    predictions.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
    DECISION_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _update_registry(decision)
    return summary, folds, backtest, decision


if __name__ == "__main__":
    result_summary, _, result_backtest, result_decision = run_comparison()
    print(result_summary.to_string(index=False))
    print()
    print(result_backtest[[
        "strategy", "net_total_return", "net_annualized_sharpe",
        "net_max_drawdown", "one_way_turnover",
    ]].to_string(index=False))
    print()
    print(json.dumps(result_decision, indent=2, ensure_ascii=False))
