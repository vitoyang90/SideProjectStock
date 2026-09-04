"""Compare V2 with frozen V6 disclosure-text features across five stocks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from backtest import DEFAULT_COMMISSION_BPS, DEFAULT_SELL_TAX_BPS
from collect_disclosures import SUPPORTED_STOCKS
from diagnose_v1 import _build_model, _labels, _walk_forward_split
from train import (
    CLASS_NAMES,
    DEFAULT_LABEL_THRESHOLD,
    REPORT_DIR,
    TARGET_COLUMN,
    _feature_columns,
    _sample_weights,
)


DATA_ROOT = Path("data/processed/disclosures")
SCHEMA_PATH = REPORT_DIR / "v6_disclosure_feature_schema.json"
READINESS_PATH = REPORT_DIR / "v6_disclosure_data_readiness.json"
REGISTRY_PATH = Path("models/registry.json")
FOLD_PATH = REPORT_DIR / "v6_disclosure_walk_forward_folds.csv"
SUMMARY_PATH = REPORT_DIR / "v6_disclosure_walk_forward_summary.csv"
BACKTEST_PATH = REPORT_DIR / "v6_disclosure_backtest.csv"
DECISION_PATH = REPORT_DIR / "v6_disclosure_decision.json"
PREDICTION_PATH = DATA_ROOT / "v6_walk_forward_predictions.csv"
VALIDATION_YEARS = (2021, 2022, 2023, 2024)
CLASS_WEIGHTING = "balanced"
REQUIRED_STOCK_WINS = 3
MAX_ALLOWED_STOCK_DEGRADATION = -0.03


def _verify_schema(schema: dict) -> None:
    expected = schema.get("sha256")
    canonical = {key: value for key, value in schema.items() if key != "sha256"}
    actual = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if not expected or expected != actual:
        raise ValueError("Frozen V6 disclosure schema hash does not match")


def _load_stock(stock_id: str, disclosure_features: list[str]) -> pd.DataFrame:
    path = DATA_ROOT / f"{stock_id}_dataset_v6_ready.csv"
    data = pd.read_csv(path, dtype={"stock_id": str})
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    required = {
        "date",
        "stock_id",
        "return_5d",
        *_feature_columns("v2"),
        *disclosure_features,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{stock_id} V6 dataset is missing: {sorted(missing)}")
    data = data[data["date"] < "2025-01-01"].sort_values("date").copy()
    data = data.dropna(subset=["return_5d"])
    data[TARGET_COLUMN] = _labels(data["return_5d"], DEFAULT_LABEL_THRESHOLD)
    return data


def _fit_fold(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str]
) -> tuple[object, np.ndarray, np.ndarray]:
    model = _build_model()
    y_train = train[TARGET_COLUMN].astype(int)
    y_validation = validation[TARGET_COLUMN].astype(int)
    model.fit(
        train[features],
        y_train,
        sample_weight=_sample_weights(y_train, CLASS_WEIGHTING),
        eval_set=[(validation[features], y_validation)],
        sample_weight_eval_set=[_sample_weights(y_validation, CLASS_WEIGHTING)],
        verbose=False,
    )
    return (
        model,
        model.predict(validation[features]).astype(int),
        model.predict_proba(validation[features]),
    )


def _net_return(predictions: pd.DataFrame) -> dict:
    selected = predictions.sort_values("date").iloc[::5].copy()
    selected["gross_period_return"] = np.where(
        selected["predicted_class"].eq(2), selected["return_5d"], 0.0
    )
    round_trip_cost = (
        2 * DEFAULT_COMMISSION_BPS + DEFAULT_SELL_TAX_BPS
    ) / 10_000
    selected["net_period_return"] = selected["gross_period_return"] - np.where(
        selected["predicted_class"].eq(2), round_trip_cost, 0.0
    )
    return {
        "intervals": int(len(selected)),
        "long_intervals": int(selected["predicted_class"].eq(2).sum()),
        "gross_total_return": float((1 + selected["gross_period_return"]).prod() - 1),
        "net_total_return": float((1 + selected["net_period_return"]).prod() - 1),
    }


def evaluate_gate(summary: pd.DataFrame, backtest: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    metrics = summary.pivot(index="stock_id", columns="version")
    deltas = pd.DataFrame(
        {
            "stock_id": metrics.index,
            "delta_balanced_accuracy": (
                metrics["mean_balanced_accuracy"]["v6_disclosure"]
                - metrics["mean_balanced_accuracy"]["v2_matched"]
            ).to_numpy(),
            "delta_macro_f1": (
                metrics["mean_macro_f1"]["v6_disclosure"]
                - metrics["mean_macro_f1"]["v2_matched"]
            ).to_numpy(),
        }
    )
    returns = backtest.pivot(index="stock_id", columns="version", values="net_total_return")
    deltas["delta_net_return"] = (
        returns["v6_disclosure"] - returns["v2_matched"]
    ).reindex(deltas["stock_id"]).to_numpy()
    checks = {
        "mean_balanced_accuracy_improved": bool(deltas["delta_balanced_accuracy"].mean() > 0),
        "mean_macro_f1_improved": bool(deltas["delta_macro_f1"].mean() > 0),
        "balanced_accuracy_wins_at_least_3_of_5_stocks": bool(
            deltas["delta_balanced_accuracy"].gt(0).sum() >= REQUIRED_STOCK_WINS
        ),
        "macro_f1_wins_at_least_3_of_5_stocks": bool(
            deltas["delta_macro_f1"].gt(0).sum() >= REQUIRED_STOCK_WINS
        ),
        "worst_stock_balanced_accuracy_delta_at_least_minus_0_03": bool(
            deltas["delta_balanced_accuracy"].min() >= MAX_ALLOWED_STOCK_DEGRADATION
        ),
        "worst_stock_macro_f1_delta_at_least_minus_0_03": bool(
            deltas["delta_macro_f1"].min() >= MAX_ALLOWED_STOCK_DEGRADATION
        ),
        "net_return_wins_at_least_3_of_5_stocks": bool(
            deltas["delta_net_return"].gt(0).sum() >= REQUIRED_STOCK_WINS
        ),
        "median_net_return_improved": bool(deltas["delta_net_return"].median() > 0),
    }
    return checks, deltas


def _update_registry(decision: dict) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["rejected"] = [
        item for item in registry.get("rejected", [])
        if item.get("version") != "v6_disclosure_text"
    ]
    if decision["passes_gate"]:
        if registry.get("candidate") not in (None, {}):
            raise ValueError("A different model candidate already exists in registry")
        registry["candidate"] = {
            "version": "v6_disclosure_text",
            "status": "candidate_requiring_fresh_forward_validation",
            "features": "V2 + frozen official disclosure text counts",
            "reason": "Passed the predeclared five-stock historical gate",
        }
    else:
        failed = [name for name, passed in decision["gate_checks"].items() if not passed]
        registry["rejected"].append(
            {
                "version": "v6_disclosure_text",
                "features": "V2 + frozen official disclosure text counts",
                "reason": "Failed historical five-stock gate: " + ", ".join(failed),
            }
        )
    registry["v6_disclosure_experiment"] = {
        "status": decision["status"],
        "decision_path": DECISION_PATH.as_posix(),
    }
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    _verify_schema(schema)
    if not readiness.get("ready_for_training"):
        raise ValueError("V6 disclosure data-readiness gate did not pass")
    disclosure_features = list(schema["feature_columns"])
    versions = {
        "v2_matched": _feature_columns("v2"),
        "v6_disclosure": _feature_columns("v2") + disclosure_features,
    }
    fold_rows = []
    prediction_parts = []
    for stock_id in SUPPORTED_STOCKS:
        data = _load_stock(stock_id, disclosure_features)
        all_features = sorted({item for group in versions.values() for item in group})
        data[all_features] = data[all_features].replace([np.inf, -np.inf], np.nan)
        for year in VALIDATION_YEARS:
            train, validation = _walk_forward_split(data, year)
            for version, features in versions.items():
                model, predicted, probabilities = _fit_fold(train, validation, features)
                truth = validation[TARGET_COLUMN].astype(int)
                fold_rows.append(
                    {
                        "stock_id": stock_id,
                        "version": version,
                        "validation_year": year,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "feature_count": len(features),
                        "best_iteration": int(model.best_iteration),
                        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                        "macro_f1": float(f1_score(truth, predicted, average="macro")),
                        "mlogloss": float(log_loss(truth, probabilities, labels=range(len(CLASS_NAMES)))),
                    }
                )
                prediction = validation[["date", "stock_id", "return_5d", TARGET_COLUMN]].copy()
                prediction["version"] = version
                prediction["validation_year"] = year
                prediction["predicted_class"] = predicted
                prediction_parts.append(prediction)

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["stock_id", "version"], as_index=False)
        .agg(
            folds=("validation_year", "count"),
            feature_count=("feature_count", "first"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_mlogloss=("mlogloss", "mean"),
        )
    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    backtest_rows = []
    for (stock_id, version), group in predictions.groupby(["stock_id", "version"]):
        backtest_rows.append({"stock_id": stock_id, "version": version, **_net_return(group)})
    backtest = pd.DataFrame(backtest_rows)
    checks, stock_deltas = evaluate_gate(summary, backtest)
    passes = all(checks.values())
    decision = {
        "version": "v6_disclosure_text",
        "status": "candidate_requiring_fresh_forward_validation" if passes else "rejected_historical_walk_forward",
        "passes_gate": passes,
        "comparison": "V2 versus V2 plus 13 frozen official disclosure-text features",
        "stocks": list(SUPPORTED_STOCKS),
        "validation_years": list(VALIDATION_YEARS),
        "test_period_used_for_selection": False,
        "schema_sha256": schema["sha256"],
        "gate_policy": {
            "required_stock_wins": REQUIRED_STOCK_WINS,
            "maximum_allowed_single_stock_degradation": abs(MAX_ALLOWED_STOCK_DEGRADATION),
            "all_checks_must_pass": True,
        },
        "gate_checks": checks,
        "stock_deltas": stock_deltas.to_dict(orient="records"),
        "warning": "Historical research only; a passing model still requires fresh forward validation.",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    predictions.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
    DECISION_PATH.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _update_registry(decision)
    return summary, folds, backtest, decision


if __name__ == "__main__":
    result_summary, _, result_backtest, result_decision = run_comparison()
    print(result_summary.to_string(index=False))
    print()
    print(result_backtest.to_string(index=False))
    print()
    print(json.dumps(result_decision, ensure_ascii=False, indent=2))
