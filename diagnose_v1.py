"""Walk-forward diagnosis for V1 label thresholds using pre-test data only."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

from train import (
    CLASS_NAMES,
    DATA_PATH,
    PURGE_ROWS,
    REPORT_DIR,
    TARGET_COLUMN,
    _class_counts,
    _feature_columns,
    _sample_weights,
)


THRESHOLDS = (0.01, 0.015, 0.02, 0.025, 0.03)
CLASS_WEIGHTINGS = ("none", "balanced")
VALIDATION_YEARS = (2021, 2022, 2023, 2024)
SUMMARY_PATH = REPORT_DIR / "xgboost_v1_threshold_diagnostics.csv"
FOLD_PATH = REPORT_DIR / "xgboost_v1_walk_forward_folds.csv"


def _labels(returns: pd.Series, threshold: float) -> np.ndarray:
    return np.select(
        [returns < -threshold, returns > threshold],
        [0, 2],
        default=1,
    ).astype("int8")


def _build_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )


def _walk_forward_split(
    data: pd.DataFrame, validation_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_start = pd.Timestamp(validation_year, 1, 1)
    validation_end = pd.Timestamp(validation_year + 1, 1, 1)
    train = data[data["date"] < validation_start].copy()
    validation = data[
        (data["date"] >= validation_start)
        & (data["date"] < validation_end)
    ].copy()
    if len(train) <= PURGE_ROWS or len(validation) <= PURGE_ROWS:
        raise ValueError(f"Not enough data for validation year {validation_year}")
    # Purge both boundaries: train labels must not overlap validation, and
    # validation labels must remain inside their stated calendar year.
    return (
        train.iloc[:-PURGE_ROWS].copy(),
        validation.iloc[:-PURGE_ROWS].copy(),
    )


def run_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _feature_columns("v1")
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").dropna(subset=["return_5d"]).copy()
    # Keep the 2025+ test period completely outside threshold selection.
    data = data[data["date"] < "2025-01-01"].copy()
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    rows = []

    for threshold in THRESHOLDS:
        candidate = data.copy()
        candidate[TARGET_COLUMN] = _labels(candidate["return_5d"], threshold)
        for validation_year in VALIDATION_YEARS:
            train, validation = _walk_forward_split(
                candidate, validation_year,
            )
            x_train = train[features]
            y_train = train[TARGET_COLUMN].astype(int)
            x_validation = validation[features]
            y_validation = validation[TARGET_COLUMN].astype(int)

            for class_weighting in CLASS_WEIGHTINGS:
                model = _build_model()
                model.fit(
                    x_train,
                    y_train,
                    sample_weight=_sample_weights(y_train, class_weighting),
                    eval_set=[(x_validation, y_validation)],
                    sample_weight_eval_set=[
                        _sample_weights(y_validation, class_weighting)
                    ],
                    verbose=False,
                )
                predicted = model.predict(x_validation).astype(int)
                probabilities = model.predict_proba(x_validation)
                actual_counts = _class_counts(y_validation)
                prediction_counts = _class_counts(pd.Series(predicted))
                rows.append(
                    {
                        "threshold": threshold,
                        "class_weighting": class_weighting,
                        "validation_year": validation_year,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
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
                            f1_score(
                                y_validation, predicted, average="macro",
                            )
                        ),
                        "actual_down": actual_counts["down"],
                        "actual_neutral": actual_counts["neutral"],
                        "actual_up": actual_counts["up"],
                        "predicted_down": prediction_counts["down"],
                        "predicted_neutral": prediction_counts["neutral"],
                        "predicted_up": prediction_counts["up"],
                    }
                )

    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["threshold", "class_weighting"], as_index=False)
        .agg(
            folds=("validation_year", "count"),
            mean_best_iteration=("best_iteration", "mean"),
            zero_iteration_folds=("best_iteration", lambda s: int((s == 0).sum())),
            mean_validation_mlogloss=("validation_mlogloss", "mean"),
            mean_balanced_accuracy=("validation_balanced_accuracy", "mean"),
            std_balanced_accuracy=("validation_balanced_accuracy", "std"),
            mean_macro_f1=("validation_macro_f1", "mean"),
            std_macro_f1=("validation_macro_f1", "std"),
        )
        .sort_values(
            ["mean_macro_f1", "mean_balanced_accuracy"], ascending=False,
        )
    )
    Path(SUMMARY_PATH).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    folds.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")
    return summary, folds


if __name__ == "__main__":
    diagnostics, _ = run_diagnostics()
    print(diagnostics.to_string(index=False))
