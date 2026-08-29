"""Train the V2 + PB candidate without evaluating on the viewed test set."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from train import (
    CLASS_NAMES,
    DATA_PATH,
    DEFAULT_LABEL_THRESHOLD,
    PURGE_ROWS,
    TARGET_COLUMN,
    _class_counts,
    _feature_columns,
)


ABLATION_FOLDS_PATH = Path(
    "reports/xgboost_v3_fundamental_ablation_folds.csv"
)
MODEL_PATH = Path("models/xgboost_v3_lite_candidate.json")
REPORT_PATH = Path("reports/xgboost_v3_lite_candidate.json")
TRAINING_CUTOFF = pd.Timestamp("2025-01-01")
EXTERNAL_VALIDATION_AFTER = "2026-08-24"
FEATURES = _feature_columns("v2") + ["pb"]


def _selected_tree_count() -> tuple[int, list[int]]:
    folds = pd.read_csv(ABLATION_FOLDS_PATH)
    iterations = folds.loc[
        folds["candidate"] == "plus_pb", "best_iteration"
    ].astype(int)
    if len(iterations) != 4:
        raise ValueError("Expected four plus_pb walk-forward folds")
    # best_iteration is zero-based. The median is less sensitive to one fold
    # training much longer than the others.
    return int(np.median(iterations)) + 1, iterations.tolist()


def train_candidate() -> dict:
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES) | {"date", "return_5d"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data[
        (data["date"] < TRAINING_CUTOFF) & data["return_5d"].notna()
    ].sort_values("date").copy()
    # Prevent the final pre-cutoff labels from using post-cutoff prices.
    data = data.iloc[:-PURGE_ROWS].copy()
    if data.empty or data["date"].max() >= TRAINING_CUTOFF:
        raise RuntimeError("Candidate training cutoff was not enforced")
    data[TARGET_COLUMN] = np.select(
        [
            data["return_5d"] < -DEFAULT_LABEL_THRESHOLD,
            data["return_5d"] > DEFAULT_LABEL_THRESHOLD,
        ],
        [0, 2],
        default=1,
    ).astype("int8")
    data[FEATURES] = data[FEATURES].replace([np.inf, -np.inf], np.nan)
    tree_count, fold_iterations = _selected_tree_count()

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=tree_count,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    labels = data[TARGET_COLUMN].astype(int)
    model.fit(
        data[FEATURES],
        labels,
        sample_weight=compute_sample_weight("balanced", labels),
        verbose=False,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    report = {
        "version": "v3_lite",
        "status": "rejected_after_external_validation",
        "base_version": "v2",
        "added_features": ["pb"],
        "features": FEATURES,
        "label_threshold": DEFAULT_LABEL_THRESHOLD,
        "class_weighting": "balanced",
        "training": {
            "rows": len(data),
            "start": data["date"].min().strftime("%Y-%m-%d"),
            "end": data["date"].max().strftime("%Y-%m-%d"),
            "class_counts": _class_counts(labels),
            "cutoff_exclusive": TRAINING_CUTOFF.strftime("%Y-%m-%d"),
            "boundary_purge_rows": PURGE_ROWS,
        },
        "tree_selection": {
            "source": "four pre-test walk-forward folds",
            "fold_best_iterations_zero_based": fold_iterations,
            "method": "median best iteration plus one",
            "n_estimators": tree_count,
        },
        "test_policy": {
            "evaluated_on_2025_2026_viewed_test": False,
            "external_validation_after": EXTERNAL_VALIDATION_AFTER,
            "external_stock_validation": "completed_and_failed",
        },
        "model_path": MODEL_PATH.as_posix(),
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return report


if __name__ == "__main__":
    result = train_candidate()
    print(json.dumps(result, indent=2, ensure_ascii=False))
