"""Nested V2.1 search across label policies, XGBoost, and LightGBM."""

import json
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

from backtest import (
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SELL_TAX_BPS,
    _build_intervals,
    _simulate,
)
from diagnose_v1 import VALIDATION_YEARS, _walk_forward_split
from train import (
    CLASS_NAMES,
    DEFAULT_LABEL_THRESHOLD,
    REPORT_DIR,
    TARGET_COLUMN,
    _class_counts,
    _feature_columns,
    _sample_weights,
)
from train_stock_models import (
    CLASS_WEIGHTING,
    MODEL_DIR,
    REGISTRY_PATH,
    STOCK_IDS,
    _load_dataset,
    select_tree_count,
)


LABEL_POLICIES = ("fixed", "volatility", "train_quantile")
VOLATILITY_MULTIPLIER = 0.75
VOLATILITY_FLOOR = 0.0075
QUANTILE_LEVEL = 1 / 3
OUTER_YEARS = (2022, 2023, 2024)
MAX_ALLOWED_FOLD_DEGRADATION = -0.03
REQUIRED_OUTER_WINS = 2

MODEL_CONFIGS = (
    {
        "name": "xgb_baseline",
        "family": "xgboost",
        "params": {"max_depth": 3, "min_child_weight": 5, "reg_lambda": 1.0},
    },
    {
        "name": "xgb_shallow",
        "family": "xgboost",
        "params": {"max_depth": 2, "min_child_weight": 10, "reg_lambda": 2.0},
    },
    {
        "name": "xgb_conservative",
        "family": "xgboost",
        "params": {"max_depth": 2, "min_child_weight": 20, "reg_lambda": 5.0},
    },
    {
        "name": "lgb_small",
        "family": "lightgbm",
        "params": {
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 30,
            "reg_lambda": 2.0,
        },
    },
    {
        "name": "lgb_medium",
        "family": "lightgbm",
        "params": {
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
        },
    },
)

BASELINE_CANDIDATE = "fixed__xgb_baseline"
CANDIDATE_FOLD_PATH = REPORT_DIR / "v21_candidate_folds.csv"
LABEL_DIAGNOSTIC_PATH = REPORT_DIR / "v21_label_diagnostics.csv"
NESTED_SELECTION_PATH = REPORT_DIR / "v21_nested_selections.csv"
NESTED_PREDICTION_PATH = Path("data/processed/v21_nested_predictions.csv")
DECISION_PATH = REPORT_DIR / "v21_decision.json"
CANDIDATE_MODEL_DIR = Path("models/v21_candidates")


def _candidate_id(policy: str, config: dict) -> str:
    return f"{policy}__{config['name']}"


def _thresholds(
    frame: pd.DataFrame,
    policy: str,
    quantile_threshold: float | None = None,
) -> pd.Series:
    if policy == "fixed":
        return pd.Series(DEFAULT_LABEL_THRESHOLD, index=frame.index, dtype=float)
    if policy == "volatility":
        dynamic = (
            frame["volatility_20"].astype(float)
            * np.sqrt(5)
            * VOLATILITY_MULTIPLIER
        )
        return dynamic.clip(lower=VOLATILITY_FLOOR).fillna(DEFAULT_LABEL_THRESHOLD)
    if policy == "train_quantile":
        if quantile_threshold is None:
            raise ValueError("train_quantile requires a training-derived threshold")
        return pd.Series(quantile_threshold, index=frame.index, dtype=float)
    raise ValueError(f"Unknown label policy: {policy}")


def _policy_labels(
    frame: pd.DataFrame,
    policy: str,
    quantile_threshold: float | None = None,
) -> tuple[np.ndarray, pd.Series]:
    thresholds = _thresholds(frame, policy, quantile_threshold)
    labels = np.select(
        [
            frame["return_5d"] < -thresholds,
            frame["return_5d"] > thresholds,
        ],
        [0, 2],
        default=1,
    ).astype("int8")
    return labels, thresholds


def _build_model(config: dict):
    if config["family"] == "xgboost":
        is_baseline = config["name"] == "xgb_baseline"
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=1000 if is_baseline else 600,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50 if is_baseline else 40,
            **config["params"],
        )
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=600,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **config["params"],
    )


def _fit_candidate(
    config: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    policy: str,
) -> tuple[object, np.ndarray, np.ndarray, dict, dict]:
    quantile_threshold = None
    if policy == "train_quantile":
        quantile_threshold = float(
            train["return_5d"].abs().quantile(QUANTILE_LEVEL)
        )
    y_train, train_thresholds = _policy_labels(
        train, policy, quantile_threshold,
    )
    y_validation_fit, validation_thresholds = _policy_labels(
        validation, policy, quantile_threshold,
    )
    model = _build_model(config)
    if config["family"] == "xgboost":
        fit_kwargs = {
            "sample_weight": _sample_weights(
                pd.Series(y_train), CLASS_WEIGHTING,
            ),
            "eval_set": [(validation[features], y_validation_fit)],
            "sample_weight_eval_set": [
                _sample_weights(pd.Series(y_validation_fit), CLASS_WEIGHTING)
            ],
            "verbose": False,
        }
    else:
        fit_kwargs = {
            "sample_weight": _sample_weights(
                pd.Series(y_train), CLASS_WEIGHTING,
            ),
            "eval_X": validation[features],
            "eval_y": y_validation_fit,
            "eval_metric": "multi_logloss",
            "callbacks": [lgb.early_stopping(40, verbose=False)],
        }
    model.fit(train[features], y_train, **fit_kwargs)
    predicted = model.predict(validation[features]).astype(int)
    probabilities = model.predict_proba(validation[features])
    best_iteration = (
        int(model.best_iteration)
        if config["family"] == "xgboost"
        else max(int(model.best_iteration_) - 1, 0)
    )
    label_info = {
        "quantile_threshold": quantile_threshold,
        "train_threshold_mean": float(train_thresholds.mean()),
        "validation_threshold_mean": float(validation_thresholds.mean()),
        "train_class_counts": _class_counts(pd.Series(y_train)),
        "validation_fit_class_counts": _class_counts(
            pd.Series(y_validation_fit)
        ),
    }
    return model, predicted, probabilities, label_info, {
        "best_iteration": best_iteration,
    }


def _candidate_backtest_return(
    validation: pd.DataFrame,
    predicted: np.ndarray,
    prices: pd.DataFrame,
    validation_year: int,
) -> float:
    predictions = validation[["date"]].copy()
    predictions["validation_year"] = validation_year
    predictions["predicted_class"] = predicted
    intervals = _build_intervals(predictions, prices)
    metrics, _, _ = _simulate(
        intervals,
        f"validation_{validation_year}",
        "candidate",
        DEFAULT_COMMISSION_BPS,
        DEFAULT_SELL_TAX_BPS,
    )
    return float(metrics["net_total_return"])


def _selection_table(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(
            ["candidate_id", "label_policy", "model_name", "family"],
            as_index=False,
        )
        .agg(
            years=("validation_year", "count"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_mlogloss=("mlogloss", "mean"),
        )
    )
    summary[["std_balanced_accuracy", "std_macro_f1"]] = summary[
        ["std_balanced_accuracy", "std_macro_f1"]
    ].fillna(0.0)
    summary["selection_score"] = (
        summary["mean_balanced_accuracy"]
        + summary["mean_macro_f1"]
        - 0.25
        * (
            summary["std_balanced_accuracy"]
            + summary["std_macro_f1"]
        )
    )
    return summary.sort_values(
        ["selection_score", "mean_mlogloss", "candidate_id"],
        ascending=[False, True, True],
    )


def _promotion_checks(nested: pd.DataFrame) -> dict[str, bool]:
    selected_return = float((1 + nested["selected_net_return"]).prod() - 1)
    baseline_return = float((1 + nested["baseline_net_return"]).prod() - 1)
    return {
        "mean_balanced_accuracy_improved": bool(
            nested["delta_balanced_accuracy"].mean() > 0
        ),
        "mean_macro_f1_improved": bool(nested["delta_macro_f1"].mean() > 0),
        "balanced_accuracy_wins_at_least_2_of_3": bool(
            nested["delta_balanced_accuracy"].gt(0).sum() >= REQUIRED_OUTER_WINS
        ),
        "macro_f1_wins_at_least_2_of_3": bool(
            nested["delta_macro_f1"].gt(0).sum() >= REQUIRED_OUTER_WINS
        ),
        "worst_balanced_accuracy_delta_at_least_minus_0_03": bool(
            nested["delta_balanced_accuracy"].min()
            >= MAX_ALLOWED_FOLD_DEGRADATION
        ),
        "worst_macro_f1_delta_at_least_minus_0_03": bool(
            nested["delta_macro_f1"].min() >= MAX_ALLOWED_FOLD_DEGRADATION
        ),
        "net_return_improved_after_costs": bool(
            selected_return > baseline_return
        ),
    }


def _fit_final(
    data: pd.DataFrame,
    features: list[str],
    policy: str,
    config: dict,
    n_estimators: int,
):
    quantile_threshold = None
    if policy == "train_quantile":
        quantile_threshold = float(
            data["return_5d"].abs().quantile(QUANTILE_LEVEL)
        )
    labels, _ = _policy_labels(data, policy, quantile_threshold)
    if config["family"] == "xgboost":
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=n_estimators,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            **config["params"],
        )
    else:
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=n_estimators,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **config["params"],
        )
    label_series = pd.Series(labels)
    model.fit(
        data[features],
        labels,
        sample_weight=_sample_weights(label_series, CLASS_WEIGHTING),
    )
    return model, quantile_threshold


def _save_model(model, family: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if family == "xgboost":
        model.save_model(path)
    else:
        model.booster_.save_model(str(path))


def run_v21_search() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features = _feature_columns("v2")
    candidate_rows = []
    prediction_frames = {}
    label_rows = []
    datasets = {}

    for stock_id in STOCK_IDS:
        data = _load_dataset(stock_id, "")
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        data = data.sort_values("date").dropna(subset=["return_5d"]).copy()
        data[features] = data[features].replace([np.inf, -np.inf], np.nan)
        data[TARGET_COLUMN] = np.select(
            [
                data["return_5d"] < -DEFAULT_LABEL_THRESHOLD,
                data["return_5d"] > DEFAULT_LABEL_THRESHOLD,
            ],
            [0, 2],
            default=1,
        ).astype("int8")
        datasets[stock_id] = data
        pretest = data[data["date"] < "2025-01-01"].copy()
        prices = data[["date", "open"]].drop_duplicates("date").reset_index(drop=True)

        for validation_year in VALIDATION_YEARS:
            train, validation = _walk_forward_split(pretest, validation_year)
            canonical_labels = validation[TARGET_COLUMN].astype(int)
            for policy in LABEL_POLICIES:
                quantile_threshold = (
                    float(train["return_5d"].abs().quantile(QUANTILE_LEVEL))
                    if policy == "train_quantile" else None
                )
                train_labels, train_thresholds = _policy_labels(
                    train, policy, quantile_threshold,
                )
                validation_labels, validation_thresholds = _policy_labels(
                    validation, policy, quantile_threshold,
                )
                label_rows.append({
                    "stock_id": stock_id,
                    "validation_year": validation_year,
                    "label_policy": policy,
                    "train_threshold_mean": float(train_thresholds.mean()),
                    "validation_threshold_mean": float(
                        validation_thresholds.mean()
                    ),
                    "train_down": int((train_labels == 0).sum()),
                    "train_neutral": int((train_labels == 1).sum()),
                    "train_up": int((train_labels == 2).sum()),
                    "validation_down": int((validation_labels == 0).sum()),
                    "validation_neutral": int((validation_labels == 1).sum()),
                    "validation_up": int((validation_labels == 2).sum()),
                })

                for config in MODEL_CONFIGS:
                    candidate_id = _candidate_id(policy, config)
                    model, predicted, probabilities, label_info, fit_info = (
                        _fit_candidate(
                            config, train, validation, features, policy,
                        )
                    )
                    counts = _class_counts(pd.Series(predicted))
                    net_return = _candidate_backtest_return(
                        validation, predicted, prices, validation_year,
                    )
                    candidate_rows.append({
                        "stock_id": stock_id,
                        "validation_year": validation_year,
                        "candidate_id": candidate_id,
                        "label_policy": policy,
                        "model_name": config["name"],
                        "family": config["family"],
                        "best_iteration": fit_info["best_iteration"],
                        "balanced_accuracy": float(
                            balanced_accuracy_score(canonical_labels, predicted)
                        ),
                        "macro_f1": float(
                            f1_score(canonical_labels, predicted, average="macro")
                        ),
                        "mlogloss": float(
                            log_loss(
                                canonical_labels,
                                probabilities,
                                labels=range(len(CLASS_NAMES)),
                            )
                        ),
                        "net_return": net_return,
                        "predicted_down": counts["down"],
                        "predicted_neutral": counts["neutral"],
                        "predicted_up": counts["up"],
                        "train_threshold_mean": label_info[
                            "train_threshold_mean"
                        ],
                    })
                    predictions = validation[[
                        "date", "stock_id", TARGET_COLUMN,
                    ]].copy()
                    predictions["validation_year"] = validation_year
                    predictions["candidate_id"] = candidate_id
                    predictions["predicted_class"] = predicted
                    predictions["probability_down"] = probabilities[:, 0]
                    predictions["probability_neutral"] = probabilities[:, 1]
                    predictions["probability_up"] = probabilities[:, 2]
                    prediction_frames[(stock_id, validation_year, candidate_id)] = (
                        predictions
                    )

    folds = pd.DataFrame(candidate_rows)
    nested_rows = []
    nested_predictions = []
    final_selections = {}
    configs_by_name = {config["name"]: config for config in MODEL_CONFIGS}

    for stock_id in STOCK_IDS:
        stock_folds = folds[folds["stock_id"] == stock_id]
        for outer_year in OUTER_YEARS:
            inner = stock_folds[stock_folds["validation_year"] < outer_year]
            selected = _selection_table(inner).iloc[0]
            outer = stock_folds[
                (stock_folds["validation_year"] == outer_year)
                & (stock_folds["candidate_id"] == selected["candidate_id"])
            ].iloc[0]
            baseline = stock_folds[
                (stock_folds["validation_year"] == outer_year)
                & (stock_folds["candidate_id"] == BASELINE_CANDIDATE)
            ].iloc[0]
            nested_rows.append({
                "stock_id": stock_id,
                "outer_year": outer_year,
                "inner_years": ",".join(
                    str(year) for year in VALIDATION_YEARS if year < outer_year
                ),
                "selected_candidate": selected["candidate_id"],
                "selected_label_policy": selected["label_policy"],
                "selected_model_name": selected["model_name"],
                "selected_family": selected["family"],
                "selection_score": float(selected["selection_score"]),
                "selected_balanced_accuracy": float(outer["balanced_accuracy"]),
                "baseline_balanced_accuracy": float(
                    baseline["balanced_accuracy"]
                ),
                "delta_balanced_accuracy": float(
                    outer["balanced_accuracy"] - baseline["balanced_accuracy"]
                ),
                "selected_macro_f1": float(outer["macro_f1"]),
                "baseline_macro_f1": float(baseline["macro_f1"]),
                "delta_macro_f1": float(
                    outer["macro_f1"] - baseline["macro_f1"]
                ),
                "selected_net_return": float(outer["net_return"]),
                "baseline_net_return": float(baseline["net_return"]),
            })
            selected_predictions = prediction_frames[
                (stock_id, outer_year, selected["candidate_id"])
            ].copy()
            selected_predictions["role"] = "nested_selected"
            nested_predictions.append(selected_predictions)

        all_years = _selection_table(stock_folds).iloc[0]
        final_selections[stock_id] = all_years.to_dict()

    nested = pd.DataFrame(nested_rows)
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    active_models = registry.get("stock_models", {}).copy()
    decisions = {}
    build_date = date.today().isoformat()

    for stock_id in STOCK_IDS:
        stock_nested = nested[nested["stock_id"] == stock_id]
        checks = _promotion_checks(stock_nested)
        passes_gate = all(checks.values())
        selection = final_selections[stock_id]
        selected_id = selection["candidate_id"]
        selected_folds = folds[
            (folds["stock_id"] == stock_id)
            & (folds["candidate_id"] == selected_id)
        ]
        n_estimators = select_tree_count(
            selected_folds["best_iteration"].astype(int).tolist()
        )
        config = configs_by_name[selection["model_name"]]
        candidate_path = CANDIDATE_MODEL_DIR / (
            f"{stock_id}_v21_candidate.json"
            if config["family"] == "xgboost"
            else f"{stock_id}_v21_candidate.txt"
        )
        model, quantile_threshold = _fit_final(
            datasets[stock_id],
            features,
            selection["label_policy"],
            config,
            n_estimators,
        )
        _save_model(model, config["family"], candidate_path)

        selected_return = float(
            (1 + stock_nested["selected_net_return"]).prod() - 1
        )
        baseline_return = float(
            (1 + stock_nested["baseline_net_return"]).prod() - 1
        )
        decisions[stock_id] = {
            "passes_gate": passes_gate,
            "status": "promoted_for_fresh_forward" if passes_gate else "rejected",
            "selected_candidate": selected_id,
            "family": config["family"],
            "label_policy": selection["label_policy"],
            "quantile_threshold_if_applicable": quantile_threshold,
            "n_estimators": n_estimators,
            "candidate_model_path": candidate_path.as_posix(),
            "gate_checks": checks,
            "nested_mean_delta_balanced_accuracy": float(
                stock_nested["delta_balanced_accuracy"].mean()
            ),
            "nested_mean_delta_macro_f1": float(
                stock_nested["delta_macro_f1"].mean()
            ),
            "nested_net_return": selected_return,
            "baseline_net_return": baseline_return,
        }
        if passes_gate:
            current = active_models[stock_id]
            active_models[stock_id] = {
                **current,
                "version": "v2.1",
                "artifact_id": (
                    f"{stock_id}_v21_forward_{build_date.replace('-', '')}"
                ),
                "status": "v21_candidate_fresh_forward",
                "model_path": candidate_path.as_posix(),
                "model_family": config["family"],
                "label_policy": selection["label_policy"],
                "n_estimators": n_estimators,
                "trained_on": build_date,
                "direction_signal_eligible": True,
                "warning": "V2.1 passed nested history; fresh forward proof pending.",
            }

    registry["stock_models"] = active_models
    registry["v21_experiment"] = {
        "status": "completed",
        "decision_path": DECISION_PATH.as_posix(),
        "promoted_stocks": [
            stock_id for stock_id, result in decisions.items()
            if result["passes_gate"]
        ],
    }
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    decision = {
        "version": "v2.1_nested_search",
        "status": "completed",
        "canonical_evaluation_target": "fixed symmetric +/-1.5% five-day return",
        "candidate_label_policies": list(LABEL_POLICIES),
        "candidate_models": [config["name"] for config in MODEL_CONFIGS],
        "selection_score": (
            "mean balanced accuracy + mean macro F1 - 0.25 * their std sum"
        ),
        "outer_years": list(OUTER_YEARS),
        "test_period_used_for_selection": False,
        "promotion_policy": {
            "required_metric_wins": REQUIRED_OUTER_WINS,
            "outer_folds": len(OUTER_YEARS),
            "maximum_allowed_single_fold_degradation": abs(
                MAX_ALLOWED_FOLD_DEGRADATION
            ),
            "net_return_must_improve": True,
            "all_checks_must_pass": True,
        },
        "stock_decisions": decisions,
        "warning": (
            "Historical nested validation can select candidates, but only data "
            "after the build date can supply fresh promotion evidence."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_FOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(CANDIDATE_FOLD_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(
        LABEL_DIAGNOSTIC_PATH, index=False, encoding="utf-8-sig",
    )
    nested.to_csv(NESTED_SELECTION_PATH, index=False, encoding="utf-8-sig")
    combined_predictions = pd.concat(nested_predictions, ignore_index=True)
    combined_predictions["date"] = combined_predictions["date"].dt.strftime(
        "%Y-%m-%d"
    )
    NESTED_PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined_predictions.to_csv(
        NESTED_PREDICTION_PATH, index=False, encoding="utf-8-sig",
    )
    DECISION_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return folds, nested, decision


if __name__ == "__main__":
    result_folds, result_nested, result_decision = run_v21_search()
    print(result_nested.to_string(index=False))
    print()
    print(json.dumps(result_decision, indent=2, ensure_ascii=False))
