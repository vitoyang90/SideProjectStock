"""Compare V1-V3 cumulative features with fixed walk-forward settings."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from diagnose_v1 import VALIDATION_YEARS, _build_model, _labels, _walk_forward_split
from train import (
    CLASS_NAMES,
    DATA_PATH,
    DEFAULT_LABEL_THRESHOLD,
    REPORT_DIR,
    TARGET_COLUMN,
    _class_counts,
    _feature_columns,
    _sample_weights,
)


VERSIONS = ("v1", "v2", "v3")
CLASS_WEIGHTING = "balanced"
SUMMARY_PATH = REPORT_DIR / "xgboost_walk_forward_version_comparison.csv"
FOLD_PATH = REPORT_DIR / "xgboost_walk_forward_version_folds.csv"
PREDICTION_PATH = Path("data/processed/xgboost_walk_forward_predictions.csv")


def run_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").dropna(subset=["return_5d"]).copy()
    data = data[data["date"] < "2025-01-01"].copy()
    data[TARGET_COLUMN] = _labels(
        data["return_5d"], DEFAULT_LABEL_THRESHOLD,
    )
    rows = []
    prediction_rows = []

    for version in VERSIONS:
        features = _feature_columns(version)
        candidate = data.copy()
        candidate[features] = candidate[features].replace(
            [np.inf, -np.inf], np.nan,
        )
        for validation_year in VALIDATION_YEARS:
            train, validation = _walk_forward_split(
                candidate, validation_year,
            )
            x_train = train[features]
            y_train = train[TARGET_COLUMN].astype(int)
            x_validation = validation[features]
            y_validation = validation[TARGET_COLUMN].astype(int)
            model = _build_model()
            model.fit(
                x_train,
                y_train,
                sample_weight=_sample_weights(y_train, CLASS_WEIGHTING),
                eval_set=[(x_validation, y_validation)],
                sample_weight_eval_set=[
                    _sample_weights(y_validation, CLASS_WEIGHTING)
                ],
                verbose=False,
            )
            predicted = model.predict(x_validation).astype(int)
            probabilities = model.predict_proba(x_validation)
            counts = _class_counts(pd.Series(predicted))
            fold_predictions = validation[
                ["date", "stock_id", TARGET_COLUMN]
            ].copy()
            fold_predictions["version"] = version
            fold_predictions["validation_year"] = validation_year
            fold_predictions["predicted_class"] = predicted
            fold_predictions["probability_down"] = probabilities[:, 0]
            fold_predictions["probability_neutral"] = probabilities[:, 1]
            fold_predictions["probability_up"] = probabilities[:, 2]
            prediction_rows.append(fold_predictions)
            rows.append(
                {
                    "version": version,
                    "validation_year": validation_year,
                    "feature_count": len(features),
                    "label_threshold": DEFAULT_LABEL_THRESHOLD,
                    "class_weighting": CLASS_WEIGHTING,
                    "best_iteration": int(model.best_iteration),
                    "validation_mlogloss": float(
                        log_loss(
                            y_validation,
                            probabilities,
                            labels=range(len(CLASS_NAMES)),
                        )
                    ),
                    "validation_balanced_accuracy": float(
                        balanced_accuracy_score(y_validation, predicted)
                    ),
                    "validation_macro_f1": float(
                        f1_score(y_validation, predicted, average="macro")
                    ),
                    "predicted_down": counts["down"],
                    "predicted_neutral": counts["neutral"],
                    "predicted_up": counts["up"],
                }
            )

    folds = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")
    summary = (
        folds.groupby("version", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            feature_count=("feature_count", "first"),
            mean_best_iteration=("best_iteration", "mean"),
            zero_iteration_folds=(
                "best_iteration", lambda s: int((s == 0).sum())
            ),
            mean_validation_mlogloss=("validation_mlogloss", "mean"),
            mean_balanced_accuracy=(
                "validation_balanced_accuracy", "mean"
            ),
            std_balanced_accuracy=(
                "validation_balanced_accuracy", "std"
            ),
            mean_macro_f1=("validation_macro_f1", "mean"),
            std_macro_f1=("validation_macro_f1", "std"),
        )
        .sort_values("version")
    )
    v1 = summary.loc[summary["version"] == "v1"].iloc[0]
    summary["delta_balanced_accuracy_vs_v1"] = (
        summary["mean_balanced_accuracy"] - v1["mean_balanced_accuracy"]
    )
    summary["delta_macro_f1_vs_v1"] = (
        summary["mean_macro_f1"] - v1["mean_macro_f1"]
    )
    summary["delta_mlogloss_vs_v1"] = (
        summary["mean_validation_mlogloss"]
        - v1["mean_validation_mlogloss"]
    )

    Path(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    folds.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        PREDICTION_PATH, index=False, encoding="utf-8-sig",
    )
    return summary, folds


if __name__ == "__main__":
    comparison, _ = run_comparison()
    print(comparison.to_string(index=False))
