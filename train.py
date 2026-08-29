import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/2330_dataset.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
PREDICTION_DIR = Path("data/processed")
COMPARISON_PATH = REPORT_DIR / "xgboost_version_comparison.csv"
TARGET_COLUMN = "target_3class"
CLASS_NAMES = ["down", "neutral", "up"]
PURGE_ROWS = 5
DEFAULT_LABEL_THRESHOLD = 0.015

FEATURE_GROUPS = {
    "technical": [
        "return_1d", "return_5d_past", "ma5_ratio", "ma20_ratio",
        "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14",
        "atr_ratio", "volatility_20", "volume_ratio_20",
    ],
    "chip": [
        "foreign_net", "foreign_3d", "foreign_5d", "trust_net",
        "trust_3d", "trust_5d", "dealer_net", "dealer_3d", "dealer_5d",
        "institutional_total_net", "margin_balance", "margin_change",
        "short_balance", "short_change",
    ],
    "fundamental": [
        "revenue_mom", "revenue_yoy", "pe", "pb", "dividend_yield",
    ],
    "market": [
        "taiex_return_1d", "taiex_return_5d", "spy_return_1d",
        "spy_return_5d", "qqq_return_1d", "qqq_return_5d",
        "soxx_return_1d", "soxx_return_5d", "usd_twd",
        "usd_twd_change_1d", "us10y_yield", "us10y_change_1d",
    ],
}

VERSION_GROUPS = {
    "v1": ["technical"],
    "v2": ["technical", "chip"],
    "v3": ["technical", "chip", "fundamental"],
    "v4": ["technical", "chip", "fundamental", "market"],
}

LEAKAGE_COLUMNS = {
    "future_close_5d", "return_5d", "label", TARGET_COLUMN,
}


def _feature_columns(version: str) -> list[str]:
    if version not in VERSION_GROUPS:
        raise ValueError(f"Unknown feature version: {version}")
    return [
        feature
        for group in VERSION_GROUPS[version]
        for feature in FEATURE_GROUPS[group]
    ]


def _class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().reindex(range(3), fill_value=0)
    return {
        CLASS_NAMES[class_id]: int(counts.loc[class_id])
        for class_id in range(3)
    }


def _split_by_time(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = data[data["date"] < "2023-01-01"].copy()
    validation = data[
        (data["date"] >= "2023-01-01")
        & (data["date"] < "2025-01-01")
    ].copy()
    test = data[data["date"] >= "2025-01-01"].copy()
    if len(train) <= PURGE_ROWS or len(validation) <= PURGE_ROWS or test.empty:
        raise ValueError("Not enough rows for the configured time split")

    # The target looks five trading days ahead, so purge split boundaries.
    return train.iloc[:-PURGE_ROWS].copy(), validation.iloc[:-PURGE_ROWS].copy(), test


def _sample_weights(labels: pd.Series, class_weighting: str) -> np.ndarray:
    if class_weighting == "balanced":
        return compute_sample_weight(class_weight="balanced", y=labels)
    return np.ones(len(labels), dtype=float)


def _split_summary(frame: pd.DataFrame, labels: pd.Series) -> dict:
    return {
        "rows": len(frame),
        "start": frame["date"].min().strftime("%Y-%m-%d"),
        "end": frame["date"].max().strftime("%Y-%m-%d"),
        "class_counts": _class_counts(labels),
    }


def train_model(
    version: str = "v1",
    class_weighting: str = "balanced",
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> dict:
    if label_threshold <= 0:
        raise ValueError("label_threshold must be greater than zero")
    feature_columns = _feature_columns(version)
    if LEAKAGE_COLUMNS.intersection(feature_columns):
        raise RuntimeError("Leakage columns must not appear in model features")

    data = pd.read_csv(DATA_PATH)
    required = set(feature_columns) | {"date", "stock_id", "return_5d"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").dropna(subset=["return_5d"]).copy()
    data[TARGET_COLUMN] = np.select(
        [
            data["return_5d"] < -label_threshold,
            data["return_5d"] > label_threshold,
        ],
        [0, 2],
        default=1,
    ).astype("int8")
    data[feature_columns] = data[feature_columns].replace(
        [np.inf, -np.inf], np.nan,
    )
    train, validation, test = _split_by_time(data)
    x_train, y_train = train[feature_columns], train[TARGET_COLUMN].astype(int)
    x_validation = validation[feature_columns]
    y_validation = validation[TARGET_COLUMN].astype(int)
    x_test, y_test = test[feature_columns], test[TARGET_COLUMN].astype(int)

    model = XGBClassifier(
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

    predicted = model.predict(x_test).astype(int)
    probabilities = model.predict_proba(x_test)
    majority_class = int(y_train.mode().iloc[0])
    baseline_predictions = np.full(len(y_test), majority_class)
    prior = (
        y_train.value_counts(normalize=True)
        .reindex(range(3), fill_value=0)
        .to_numpy()
    )
    baseline_probabilities = np.tile(prior, (len(y_test), 1))

    balanced_accuracy = float(balanced_accuracy_score(y_test, predicted))
    macro_f1 = float(f1_score(y_test, predicted, average="macro"))
    baseline_balanced_accuracy = float(
        balanced_accuracy_score(y_test, baseline_predictions)
    )
    baseline_macro_f1 = float(
        f1_score(
            y_test, baseline_predictions, average="macro", zero_division=0,
        )
    )

    metrics = {
        "version": version,
        "feature_groups": VERSION_GROUPS[version],
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "class_weighting": class_weighting,
        "label_threshold": label_threshold,
        "purge_rows": PURGE_ROWS,
        "best_iteration": int(model.best_iteration),
        "best_validation_mlogloss": float(model.best_score),
        "splits": {
            "train": _split_summary(train, y_train),
            "validation": _split_summary(validation, y_validation),
            "test": _split_summary(test, y_test),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "mlogloss": float(
                log_loss(y_test, probabilities, labels=[0, 1, 2])
            ),
            "prediction_counts": _class_counts(pd.Series(predicted)),
            "confusion_matrix": confusion_matrix(
                y_test, predicted, labels=[0, 1, 2],
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predicted,
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                output_dict=True,
                zero_division=0,
            ),
        },
        "majority_baseline": {
            "class": CLASS_NAMES[majority_class],
            "accuracy": float(accuracy_score(y_test, baseline_predictions)),
            "balanced_accuracy": baseline_balanced_accuracy,
            "macro_f1": baseline_macro_f1,
        },
        "delta_vs_majority_baseline": {
            "balanced_accuracy": balanced_accuracy - baseline_balanced_accuracy,
            "macro_f1": macro_f1 - baseline_macro_f1,
        },
        "training_prior_probability_baseline": {
            "probabilities": {
                CLASS_NAMES[class_id]: float(prior[class_id])
                for class_id in range(3)
            },
            "mlogloss": float(
                log_loss(y_test, baseline_probabilities, labels=[0, 1, 2])
            ),
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / f"xgboost_{version}.json")
    (REPORT_DIR / f"xgboost_{version}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        REPORT_DIR / f"xgboost_{version}_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions = test[
        ["date", "stock_id", "return_5d", TARGET_COLUMN]
    ].copy()
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")
    predictions = predictions.rename(columns={TARGET_COLUMN: "actual_class"})
    predictions["predicted_class"] = predicted
    names = dict(enumerate(CLASS_NAMES))
    predictions["actual_label"] = predictions["actual_class"].map(names)
    predictions["predicted_label"] = predictions["predicted_class"].map(names)
    predictions["probability_down"] = probabilities[:, 0]
    predictions["probability_neutral"] = probabilities[:, 1]
    predictions["probability_up"] = probabilities[:, 2]
    predictions.to_csv(
        PREDICTION_DIR / f"2330_predictions_{version}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics


def _comparison_row(metrics: dict) -> dict:
    test = metrics["test"]
    return {
        "version": metrics["version"],
        "feature_groups": "+".join(metrics["feature_groups"]),
        "feature_count": metrics["feature_count"],
        "label_threshold": metrics["label_threshold"],
        "best_iteration": metrics["best_iteration"],
        "accuracy": test["accuracy"],
        "balanced_accuracy": test["balanced_accuracy"],
        "macro_f1": test["macro_f1"],
        "mlogloss": test["mlogloss"],
        "predicted_down": test["prediction_counts"]["down"],
        "predicted_neutral": test["prediction_counts"]["neutral"],
        "predicted_up": test["prediction_counts"]["up"],
        "baseline_balanced_accuracy": metrics["majority_baseline"][
            "balanced_accuracy"
        ],
        "delta_balanced_accuracy": metrics[
            "delta_vs_majority_baseline"
        ]["balanced_accuracy"],
        "delta_macro_f1": metrics["delta_vs_majority_baseline"]["macro_f1"],
    }


def run_experiments(
    versions: list[str],
    class_weighting: str = "balanced",
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    for version in versions:
        print(f"Training {version.upper()} ({'+'.join(VERSION_GROUPS[version])})")
        rows.append(
            _comparison_row(
                train_model(version, class_weighting, label_threshold)
            )
        )
    comparison = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")
    return comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cumulative V1-V4 XGBoost baselines."
    )
    version_group = parser.add_mutually_exclusive_group()
    version_group.add_argument(
        "--version",
        choices=[*VERSION_GROUPS, "all"],
        help="Feature version to train; defaults to V1 technical baseline.",
    )
    version_group.add_argument(
        "--through",
        choices=VERSION_GROUPS,
        help="Train cumulative versions from V1 through this version.",
    )
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="balanced",
        help="Balanced weights reduce single-class prediction collapse.",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=DEFAULT_LABEL_THRESHOLD,
        help="Symmetric five-day return threshold; defaults to 0.015.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    selected_version = args.version or "v1"
    if args.through:
        all_versions = list(VERSION_GROUPS)
        versions = all_versions[: all_versions.index(args.through) + 1]
    elif selected_version == "all":
        versions = list(VERSION_GROUPS)
    else:
        versions = [selected_version]
    summary = run_experiments(
        versions,
        args.class_weighting,
        args.label_threshold,
    )
    print(summary.to_string(index=False))
