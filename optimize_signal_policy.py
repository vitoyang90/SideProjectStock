"""Nested optimization of abstention thresholds for stock-specific V2 models."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from backtest import DEFAULT_COMMISSION_BPS, DEFAULT_SELL_TAX_BPS
from diagnose_v1 import VALIDATION_YEARS, _build_model, _labels, _walk_forward_split
from train import DEFAULT_LABEL_THRESHOLD, REPORT_DIR, TARGET_COLUMN, _feature_columns, _sample_weights
from train_stock_models import REGISTRY_PATH, STOCK_IDS, _load_dataset


POLICIES = tuple(
    (confidence, margin)
    for confidence, margin in itertools.product(
        (0.34, 0.36, 0.38, 0.40, 0.42, 0.45, 0.50),
        (0.00, 0.02, 0.05, 0.08, 0.10),
    )
)
BASELINE_POLICY = (0.40, 0.05)
OUTER_YEARS = (2022, 2023, 2024)
MIN_INNER_COVERAGE = 0.05
MIN_INNER_CALLS = 20
MIN_OUTER_TOTAL_CALLS = 40
REQUIRED_ACCURACY_WINS = 2
MIN_ACCURACY_IMPROVEMENT = 0.01
PREDICTION_PATH = Path("data/processed/signal_policy_walk_forward_predictions.csv")
GRID_PATH = REPORT_DIR / "signal_policy_grid.csv"
NESTED_PATH = REPORT_DIR / "signal_policy_nested.csv"
DECISION_PATH = REPORT_DIR / "signal_policy_decision.json"


def apply_policy(
    probabilities: np.ndarray,
    minimum_confidence: float,
    minimum_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(probabilities, axis=1)
    predicted = order[:, -1].astype(int)
    confidence = probabilities[np.arange(len(probabilities)), predicted]
    margin = confidence - probabilities[np.arange(len(probabilities)), order[:, -2]]
    emitted = (confidence >= minimum_confidence) & (margin >= minimum_margin)
    return predicted, emitted, margin


def policy_metrics(predictions: pd.DataFrame, policy: tuple[float, float]) -> dict:
    probabilities = predictions[
        ["probability_down", "probability_neutral", "probability_up"]
    ].to_numpy(float)
    predicted, emitted, _ = apply_policy(probabilities, *policy)
    actual = predictions[TARGET_COLUMN].to_numpy(int)
    calls = int(emitted.sum())
    selective_accuracy = (
        float(accuracy_score(actual[emitted], predicted[emitted])) if calls else None
    )

    intervals = predictions.sort_values("date").iloc[::5].copy()
    interval_probabilities = intervals[
        ["probability_down", "probability_neutral", "probability_up"]
    ].to_numpy(float)
    interval_predicted, interval_emitted, _ = apply_policy(
        interval_probabilities, *policy
    )
    long_signal = interval_emitted & (interval_predicted == 2)
    cost = (2 * DEFAULT_COMMISSION_BPS + DEFAULT_SELL_TAX_BPS) / 10_000
    period_return = np.where(long_signal, intervals["return_5d"].to_numpy(float) - cost, 0.0)
    return {
        "minimum_confidence": policy[0],
        "minimum_margin": policy[1],
        "rows": int(len(predictions)),
        "calls": calls,
        "coverage": float(calls / len(predictions)) if len(predictions) else 0.0,
        "selective_accuracy": selective_accuracy,
        "long_intervals": int(long_signal.sum()),
        "net_return": float(np.prod(1 + period_return) - 1),
    }


def select_policy(predictions: pd.DataFrame) -> tuple[float, float]:
    rows = []
    for policy in POLICIES:
        metrics = policy_metrics(predictions, policy)
        if (
            metrics["calls"] >= MIN_INNER_CALLS
            and metrics["coverage"] >= MIN_INNER_COVERAGE
            and metrics["selective_accuracy"] is not None
        ):
            metrics["selection_score"] = (
                metrics["selective_accuracy"]
                + 0.10 * min(metrics["coverage"], 0.50)
                + 0.02 * float(np.clip(metrics["net_return"], -1, 1))
            )
            rows.append(metrics)
    if not rows:
        return BASELINE_POLICY
    ranking = pd.DataFrame(rows).sort_values(
        ["selection_score", "selective_accuracy", "net_return", "coverage"],
        ascending=False,
    )
    winner = ranking.iloc[0]
    return float(winner["minimum_confidence"]), float(winner["minimum_margin"])


def promotion_checks(nested: pd.DataFrame) -> dict[str, bool]:
    candidate_calls = int(nested["selected_calls"].sum())
    candidate_correct = float(
        (nested["selected_accuracy"] * nested["selected_calls"]).sum()
    )
    baseline_calls = int(nested["baseline_calls"].sum())
    baseline_correct = float(
        (nested["baseline_accuracy"] * nested["baseline_calls"]).sum()
    )
    candidate_accuracy = candidate_correct / candidate_calls if candidate_calls else 0.0
    baseline_accuracy = baseline_correct / baseline_calls if baseline_calls else 0.0
    selected_net = float(np.prod(1 + nested["selected_net_return"]) - 1)
    baseline_net = float(np.prod(1 + nested["baseline_net_return"]) - 1)
    return {
        "selective_accuracy_improves_by_at_least_1pp": bool(
            candidate_accuracy >= baseline_accuracy + MIN_ACCURACY_IMPROVEMENT
        ),
        "accuracy_wins_at_least_2_of_3_years": bool(
            nested["delta_accuracy"].gt(0).sum() >= REQUIRED_ACCURACY_WINS
        ),
        "at_least_40_emitted_signals": bool(candidate_calls >= MIN_OUTER_TOTAL_CALLS),
        "coverage_at_least_5_percent": bool(
            candidate_calls / nested["validation_rows"].sum() >= MIN_INNER_COVERAGE
        ),
        "net_return_not_worse": bool(selected_net >= baseline_net),
    }


def _walk_forward_predictions(stock_id: str) -> pd.DataFrame:
    data = _load_dataset(stock_id, "")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data[data["date"] < "2025-01-01"].sort_values("date").dropna(subset=["return_5d"]).copy()
    features = _feature_columns("v2")
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    data[TARGET_COLUMN] = _labels(data["return_5d"], DEFAULT_LABEL_THRESHOLD)
    frames = []
    for year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(data, year)
        y_train = train[TARGET_COLUMN].astype(int)
        y_validation = validation[TARGET_COLUMN].astype(int)
        model = _build_model()
        model.fit(
            train[features],
            y_train,
            sample_weight=_sample_weights(y_train, "balanced"),
            eval_set=[(validation[features], y_validation)],
            sample_weight_eval_set=[_sample_weights(y_validation, "balanced")],
            verbose=False,
        )
        probabilities = model.predict_proba(validation[features])
        frame = validation[["date", "stock_id", "return_5d", TARGET_COLUMN]].copy()
        frame["validation_year"] = year
        frame["probability_down"] = probabilities[:, 0]
        frame["probability_neutral"] = probabilities[:, 1]
        frame["probability_up"] = probabilities[:, 2]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_optimization() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    prediction_parts = [_walk_forward_predictions(stock_id) for stock_id in STOCK_IDS]
    predictions = pd.concat(prediction_parts, ignore_index=True)
    grid_rows = []
    nested_rows = []
    final_policies = {}
    for stock_id in STOCK_IDS:
        stock = predictions[predictions["stock_id"].astype(str).eq(stock_id)].copy()
        for policy in POLICIES:
            grid_rows.append(
                {"stock_id": stock_id, **policy_metrics(stock, policy)}
            )
        for outer_year in OUTER_YEARS:
            inner = stock[stock["validation_year"] < outer_year]
            outer = stock[stock["validation_year"] == outer_year]
            selected_policy = select_policy(inner)
            selected = policy_metrics(outer, selected_policy)
            baseline = policy_metrics(outer, BASELINE_POLICY)
            selected_accuracy = selected["selective_accuracy"] or 0.0
            baseline_accuracy = baseline["selective_accuracy"] or 0.0
            nested_rows.append(
                {
                    "stock_id": stock_id,
                    "outer_year": outer_year,
                    "inner_years": ",".join(
                        str(year) for year in VALIDATION_YEARS if year < outer_year
                    ),
                    "selected_minimum_confidence": selected_policy[0],
                    "selected_minimum_margin": selected_policy[1],
                    "validation_rows": len(outer),
                    "selected_calls": selected["calls"],
                    "selected_coverage": selected["coverage"],
                    "selected_accuracy": selected_accuracy,
                    "selected_net_return": selected["net_return"],
                    "baseline_calls": baseline["calls"],
                    "baseline_coverage": baseline["coverage"],
                    "baseline_accuracy": baseline_accuracy,
                    "baseline_net_return": baseline["net_return"],
                    "delta_accuracy": selected_accuracy - baseline_accuracy,
                }
            )
        final_policies[stock_id] = select_policy(stock)

    grid = pd.DataFrame(grid_rows)
    nested = pd.DataFrame(nested_rows)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    decisions = {}
    for stock_id in STOCK_IDS:
        stock_nested = nested[nested["stock_id"] == stock_id]
        checks = promotion_checks(stock_nested)
        passes = all(checks.values()) and registry["stock_models"][stock_id].get(
            "direction_signal_eligible", False
        )
        final_policy = final_policies[stock_id]
        decisions[stock_id] = {
            "passes_gate": passes,
            "status": "active_research_policy" if passes else "rejected_keep_fixed_policy",
            "selected_minimum_confidence": final_policy[0],
            "selected_minimum_margin": final_policy[1],
            "gate_checks": checks,
            "model_direction_eligible": registry["stock_models"][stock_id].get(
                "direction_signal_eligible", False
            ),
        }
        if passes:
            registry["stock_models"][stock_id]["abstention_policy"] = {
                "version": "nested_selective_v1",
                "minimum_confidence": final_policy[0],
                "minimum_top_probability_margin": final_policy[1],
                "status": "research_active_forward_validation_required",
            }
    decision = {
        "version": "nested_selective_signal_policy_v1",
        "status": "completed",
        "baseline_policy": {
            "minimum_confidence": BASELINE_POLICY[0],
            "minimum_top_probability_margin": BASELINE_POLICY[1],
        },
        "outer_years": list(OUTER_YEARS),
        "selection_uses_only_prior_years": True,
        "promotion_policy": {
            "minimum_accuracy_improvement": MIN_ACCURACY_IMPROVEMENT,
            "required_accuracy_wins": REQUIRED_ACCURACY_WINS,
            "minimum_total_calls": MIN_OUTER_TOTAL_CALLS,
            "minimum_coverage": MIN_INNER_COVERAGE,
            "net_return_must_not_worsen": True,
            "model_must_already_be_direction_eligible": True,
        },
        "stock_decisions": decisions,
        "warning": "Thresholds remain research-only until post-build forward outcomes mature.",
    }
    registry["signal_policy_experiment"] = {
        "status": "completed",
        "decision_path": DECISION_PATH.as_posix(),
        "active_stocks": [
            stock_id for stock_id, result in decisions.items() if result["passes_gate"]
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_PATH, index=False, encoding="utf-8-sig")
    nested.to_csv(NESTED_PATH, index=False, encoding="utf-8-sig")
    predictions.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
    DECISION_PATH.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return grid, nested, decision


if __name__ == "__main__":
    _, result_nested, result_decision = run_optimization()
    print(result_nested.to_string(index=False))
    print()
    print(json.dumps(result_decision, ensure_ascii=False, indent=2))
