"""Ablate V3 fundamental features using pre-test walk-forward folds."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from diagnose_v1 import VALIDATION_YEARS, _build_model, _labels, _walk_forward_split
from train import (
    CLASS_NAMES,
    DATA_PATH,
    DEFAULT_LABEL_THRESHOLD,
    FEATURE_GROUPS,
    REPORT_DIR,
    TARGET_COLUMN,
    _feature_columns,
    _sample_weights,
)


V2_FEATURES = _feature_columns("v2")
CANDIDATES = {
    "v2_baseline": [],
    "plus_revenue_mom": ["revenue_mom"],
    "plus_revenue_yoy": ["revenue_yoy"],
    "plus_pe": ["pe"],
    "plus_pb": ["pb"],
    "plus_dividend_yield": ["dividend_yield"],
    "plus_revenue_group": ["revenue_mom", "revenue_yoy"],
    "plus_valuation_group": ["pe", "pb", "dividend_yield"],
    "v3_all_fundamental": FEATURE_GROUPS["fundamental"],
}
SUMMARY_PATH = REPORT_DIR / "xgboost_v3_fundamental_ablation.csv"
FOLD_PATH = REPORT_DIR / "xgboost_v3_fundamental_ablation_folds.csv"


def run_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").dropna(subset=["return_5d"]).copy()
    data = data[data["date"] < "2025-01-01"].copy()
    data[TARGET_COLUMN] = _labels(
        data["return_5d"], DEFAULT_LABEL_THRESHOLD,
    )
    rows = []

    for candidate_name, added_features in CANDIDATES.items():
        features = V2_FEATURES + added_features
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
                sample_weight=_sample_weights(y_train, "balanced"),
                eval_set=[(x_validation, y_validation)],
                sample_weight_eval_set=[
                    _sample_weights(y_validation, "balanced")
                ],
                verbose=False,
            )
            predicted = model.predict(x_validation).astype(int)
            probabilities = model.predict_proba(x_validation)
            rows.append(
                {
                    "candidate": candidate_name,
                    "added_features": "+".join(added_features) or "none",
                    "feature_count": len(features),
                    "validation_year": validation_year,
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
                }
            )

    folds = pd.DataFrame(rows)
    baseline = (
        folds[folds["candidate"] == "v2_baseline"]
        .set_index("validation_year")
    )
    folds["delta_balanced_accuracy_vs_v2"] = folds.apply(
        lambda row: row["validation_balanced_accuracy"]
        - baseline.loc[row["validation_year"], "validation_balanced_accuracy"],
        axis=1,
    )
    folds["delta_macro_f1_vs_v2"] = folds.apply(
        lambda row: row["validation_macro_f1"]
        - baseline.loc[row["validation_year"], "validation_macro_f1"],
        axis=1,
    )
    folds["delta_mlogloss_vs_v2"] = folds.apply(
        lambda row: row["validation_mlogloss"]
        - baseline.loc[row["validation_year"], "validation_mlogloss"],
        axis=1,
    )

    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            added_features=("added_features", "first"),
            feature_count=("feature_count", "first"),
            mean_best_iteration=("best_iteration", "mean"),
            zero_iteration_folds=(
                "best_iteration", lambda s: int((s == 0).sum())
            ),
            mean_mlogloss=("validation_mlogloss", "mean"),
            mean_balanced_accuracy=(
                "validation_balanced_accuracy", "mean"
            ),
            mean_macro_f1=("validation_macro_f1", "mean"),
            delta_balanced_accuracy_vs_v2=(
                "delta_balanced_accuracy_vs_v2", "mean"
            ),
            delta_macro_f1_vs_v2=("delta_macro_f1_vs_v2", "mean"),
            delta_mlogloss_vs_v2=("delta_mlogloss_vs_v2", "mean"),
            balanced_accuracy_win_folds=(
                "delta_balanced_accuracy_vs_v2", lambda s: int((s > 0).sum())
            ),
            macro_f1_win_folds=(
                "delta_macro_f1_vs_v2", lambda s: int((s > 0).sum())
            ),
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
    result, _ = run_ablation()
    print(result.to_string(index=False))
