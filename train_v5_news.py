"""Compare V2 with frozen V5 news features on matched walk-forward data."""

import hashlib
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
from diagnose_v1 import _build_model, _labels, _walk_forward_split
from train import (
    CLASS_NAMES,
    DEFAULT_LABEL_THRESHOLD,
    REPORT_DIR,
    TARGET_COLUMN,
    _class_counts,
    _feature_columns,
    _sample_weights,
)


DATA_PATH = Path("data/processed/2330_dataset_v5_ready.csv")
SCHEMA_PATH = REPORT_DIR / "news_feature_schema_v5.json"
AUDIT_PATH = REPORT_DIR / "news_audit_summary.json"
REGISTRY_PATH = Path("models/registry.json")
PREDICTION_PATH = Path("data/processed/v5_news_walk_forward_predictions.csv")
FOLD_PATH = REPORT_DIR / "v5_news_walk_forward_folds.csv"
SUMMARY_PATH = REPORT_DIR / "v5_news_walk_forward_summary.csv"
BACKTEST_PATH = REPORT_DIR / "v5_news_backtest.csv"
BACKTEST_YEARLY_PATH = REPORT_DIR / "v5_news_backtest_yearly.csv"
DECISION_PATH = REPORT_DIR / "v5_news_decision.json"

VALIDATION_YEARS = (2022, 2023, 2024)
CLASS_WEIGHTING = "balanced"
MAX_ALLOWED_FOLD_DEGRADATION = -0.03
REQUIRED_FOLD_WINS = 2


def _verify_schema(schema: dict) -> None:
    expected_hash = schema.get("schema_sha256")
    canonical = {key: value for key, value in schema.items() if key != "schema_sha256"}
    actual_hash = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("Frozen V5 news schema hash does not match its contents")
    if schema.get("status") != "frozen_for_historical_walk_forward":
        raise ValueError("V5 news schema is not frozen for historical walk-forward")


def _load_inputs() -> tuple[pd.DataFrame, dict, list[str]]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    _verify_schema(schema)
    if not audit["decision"]["ready_to_build_v5_walk_forward"]:
        raise ValueError("News audit did not approve the V5 walk-forward comparison")

    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    news_features = list(schema["active_features"])
    required = {
        "date", "stock_id", "return_5d", *_feature_columns("v2"), *news_features,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"V5-ready dataset is missing columns: {sorted(missing)}")

    model_start = pd.Timestamp(schema["recommended_model_start"])
    data = data[
        (data["date"] >= model_start)
        & (data["date"] < pd.Timestamp("2025-01-01"))
    ].copy()
    data = data.sort_values("date").dropna(subset=["return_5d"])
    if data[news_features].isna().any().any():
        missing_counts = data[news_features].isna().sum()
        missing_counts = missing_counts[missing_counts.gt(0)].to_dict()
        raise ValueError(
            f"Frozen news features contain missing values after model start: {missing_counts}"
        )
    data[TARGET_COLUMN] = _labels(data["return_5d"], DEFAULT_LABEL_THRESHOLD)
    return data, schema, news_features


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


def _classification_summary(folds: pd.DataFrame) -> pd.DataFrame:
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
    baseline = summary.loc[summary["version"] == "v2_matched"].iloc[0]
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
    for version in ("v2_matched", "v5_news"):
        candidate = predictions[predictions["version"] == version].copy()
        intervals = _build_intervals(candidate, prices)
        metrics, curve, _ = _simulate(
            intervals,
            "walk_forward_2022_2024",
            version,
            DEFAULT_COMMISSION_BPS,
            DEFAULT_SELL_TAX_BPS,
        )
        metric_rows.append(metrics)
        curve_frames.append(curve)

    benchmark_predictions = predictions[
        predictions["version"] == "v2_matched"
    ].copy()
    benchmark_intervals = _build_intervals(
        benchmark_predictions, prices, always_long=True,
    )
    metrics, curve, _ = _simulate(
        benchmark_intervals,
        "walk_forward_2022_2024",
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
            pivot["balanced_accuracy"]["v5_news"]
            - pivot["balanced_accuracy"]["v2_matched"]
        ).to_numpy(),
        "delta_macro_f1": (
            pivot["macro_f1"]["v5_news"]
            - pivot["macro_f1"]["v2_matched"]
        ).to_numpy(),
    })
    v5_summary = summary.loc[summary["version"] == "v5_news"].iloc[0]
    returns = backtest.set_index("strategy")["net_total_return"]
    checks = {
        "mean_balanced_accuracy_improved": bool(
            v5_summary["delta_balanced_accuracy_vs_v2"] > 0
        ),
        "mean_macro_f1_improved": bool(v5_summary["delta_macro_f1_vs_v2"] > 0),
        "balanced_accuracy_wins_at_least_2_of_3": bool(
            deltas["delta_balanced_accuracy"].gt(0).sum() >= REQUIRED_FOLD_WINS
        ),
        "macro_f1_wins_at_least_2_of_3": bool(
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
            returns["v5_news"] > returns["v2_matched"]
        ),
    }
    return checks, deltas


def _update_registry(decision: dict) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["rejected"] = [
        item for item in registry.get("rejected", [])
        if item.get("version") != "v5_news"
    ]
    if (registry.get("candidate") or {}).get("version") == "v5_news":
        registry["candidate"] = None

    if decision["passes_gate"]:
        if registry.get("candidate") not in (None, {}):
            raise ValueError("A different model candidate already exists in registry")
        registry["candidate"] = {
            "version": "v5_news",
            "status": "candidate_requiring_fresh_forward_validation",
            "features": "V2 technical + chip + frozen news activity",
            "reason": "Passed the predeclared 2022-2024 matched-history gate",
        }
    else:
        failed = [name for name, passed in decision["gate_checks"].items() if not passed]
        registry["rejected"].append({
            "version": "v5_news",
            "features": "V2 technical + chip + frozen news activity",
            "reason": "Failed historical walk-forward gate: " + ", ".join(failed),
        })
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    data, schema, news_features = _load_inputs()
    versions = {
        "v2_matched": _feature_columns("v2"),
        "v5_news": _feature_columns("v2") + news_features,
    }
    all_features = sorted({feature for values in versions.values() for feature in values})
    data[all_features] = data[all_features].replace([np.inf, -np.inf], np.nan)
    fold_rows = []
    prediction_frames = []

    for validation_year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(data, validation_year)
        for version, features in versions.items():
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
            predictions = validation[
                ["date", "stock_id", TARGET_COLUMN]
            ].copy()
            predictions["version"] = version
            predictions["validation_year"] = validation_year
            predictions["predicted_class"] = predicted
            predictions["probability_down"] = probabilities[:, 0]
            predictions["probability_neutral"] = probabilities[:, 1]
            predictions["probability_up"] = probabilities[:, 2]
            prediction_frames.append(predictions)

    folds = pd.DataFrame(fold_rows)
    summary = _classification_summary(folds)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    backtest, yearly = _run_backtest(predictions)
    gate_checks, fold_deltas = evaluate_gate(folds, summary, backtest)
    passes_gate = all(gate_checks.values())
    v5_summary = summary.loc[summary["version"] == "v5_news"].iloc[0]
    backtest_returns = backtest.set_index("strategy")["net_total_return"]
    decision = {
        "version": "v5_news",
        "status": (
            "candidate_requiring_fresh_forward_validation"
            if passes_gate else "rejected_historical_walk_forward"
        ),
        "passes_gate": passes_gate,
        "comparison": "V2 versus V2 plus frozen news activity features",
        "matched_model_start": schema["recommended_model_start"],
        "validation_years": list(VALIDATION_YEARS),
        "test_period_used_for_selection": False,
        "schema_version": schema["schema_version"],
        "schema_sha256": schema["schema_sha256"],
        "news_features": news_features,
        "gate_policy": {
            "required_fold_wins": REQUIRED_FOLD_WINS,
            "folds": len(VALIDATION_YEARS),
            "maximum_allowed_single_fold_degradation": abs(
                MAX_ALLOWED_FOLD_DEGRADATION
            ),
            "all_checks_must_pass": True,
        },
        "gate_checks": gate_checks,
        "v5_delta_vs_v2": {
            "mean_balanced_accuracy": float(
                v5_summary["delta_balanced_accuracy_vs_v2"]
            ),
            "mean_macro_f1": float(v5_summary["delta_macro_f1_vs_v2"]),
            "mean_mlogloss": float(v5_summary["delta_mlogloss_vs_v2"]),
            "net_total_return": float(
                backtest_returns["v5_news"] - backtest_returns["v2_matched"]
            ),
        },
        "fold_deltas": fold_deltas.to_dict(orient="records"),
        "warning": (
            "Historical research only. Even a passing result requires fresh "
            "forward data before promotion to champion."
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
