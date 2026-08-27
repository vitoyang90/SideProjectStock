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
from xgboost import XGBClassifier


DATA_PATH = Path("data/processed/2330_dataset.csv")
MODEL_PATH = Path("models/xgboost_3class.json")
METRICS_PATH = Path("reports/xgboost_metrics.json")
IMPORTANCE_PATH = Path("reports/xgboost_feature_importance.csv")
PREDICTIONS_PATH = Path("data/processed/2330_predictions.csv")

TARGET_COLUMN = "target_3class"
CLASS_NAMES = ["down", "neutral", "up"]
PURGE_ROWS = 5

FEATURE_COLUMNS = [
    # Technical
    "return_1d",
    "return_5d_past",
    "ma5_ratio",
    "ma20_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "atr_ratio",
    "volatility_20",
    "volume_ratio_20",
    # Chip
    "foreign_net",
    "foreign_3d",
    "foreign_5d",
    "trust_net",
    "trust_3d",
    "trust_5d",
    "dealer_net",
    "dealer_3d",
    "dealer_5d",
    "institutional_total_net",
    "margin_balance",
    "margin_change",
    "short_balance",
    "short_change",
    # Fundamental
    "revenue_mom",
    "revenue_yoy",
    "pe",
    "pb",
    "dividend_yield",
    # Market and macro
    "taiex_return_1d",
    "taiex_return_5d",
    "spy_return_1d",
    "spy_return_5d",
    "qqq_return_1d",
    "qqq_return_5d",
    "soxx_return_1d",
    "soxx_return_5d",
    "usd_twd",
    "usd_twd_change_1d",
    "us10y_yield",
    "us10y_change_1d",
]

LEAKAGE_COLUMNS = {
    "future_close_5d",
    "return_5d",
    "label",
    "target_3class",
}


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

    # Remove the final five observations from train and validation because
    # their forward-looking labels overlap the following period.
    train = train.iloc[:-PURGE_ROWS].copy()
    validation = validation.iloc[:-PURGE_ROWS].copy()

    return train, validation, test


def train_model() -> dict:
    if LEAKAGE_COLUMNS.intersection(FEATURE_COLUMNS):
        raise RuntimeError("Leakage columns must not appear in FEATURE_COLUMNS")

    data = pd.read_csv(DATA_PATH)
    required_columns = set(FEATURE_COLUMNS) | {
        "date",
        "stock_id",
        "return_5d",
        TARGET_COLUMN,
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").dropna(subset=[TARGET_COLUMN]).copy()
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    train, validation, test = _split_by_time(data)

    x_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(int)
    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN].astype(int)
    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN].astype(int)

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
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )

    predicted = model.predict(x_test).astype(int)
    probabilities = model.predict_proba(x_test)

    majority_class = int(y_train.mode().iloc[0])
    baseline_predictions = np.full(len(y_test), majority_class)
    train_class_probabilities = (
        y_train.value_counts(normalize=True)
        .reindex(range(3), fill_value=0)
        .to_numpy()
    )
    baseline_probabilities = np.tile(
        train_class_probabilities,
        (len(y_test), 1),
    )

    report = classification_report(
        y_test,
        predicted,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "feature_count": len(FEATURE_COLUMNS),
        "class_weighting": "none",
        "purge_rows": PURGE_ROWS,
        "best_iteration": int(model.best_iteration),
        "best_validation_mlogloss": float(model.best_score),
        "splits": {
            "train": {
                "rows": len(train),
                "start": train["date"].min().strftime("%Y-%m-%d"),
                "end": train["date"].max().strftime("%Y-%m-%d"),
                "class_counts": _class_counts(y_train),
            },
            "validation": {
                "rows": len(validation),
                "start": validation["date"].min().strftime("%Y-%m-%d"),
                "end": validation["date"].max().strftime("%Y-%m-%d"),
                "class_counts": _class_counts(y_validation),
            },
            "test": {
                "rows": len(test),
                "start": test["date"].min().strftime("%Y-%m-%d"),
                "end": test["date"].max().strftime("%Y-%m-%d"),
                "class_counts": _class_counts(y_test),
            },
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predicted)
            ),
            "macro_f1": float(
                f1_score(y_test, predicted, average="macro")
            ),
            "mlogloss": float(
                log_loss(y_test, probabilities, labels=[0, 1, 2])
            ),
            "confusion_matrix": confusion_matrix(
                y_test,
                predicted,
                labels=[0, 1, 2],
            ).tolist(),
            "classification_report": report,
        },
        "majority_baseline": {
            "class": CLASS_NAMES[majority_class],
            "accuracy": float(
                accuracy_score(y_test, baseline_predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, baseline_predictions)
            ),
            "macro_f1": float(
                f1_score(
                    y_test,
                    baseline_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "training_prior_probability_baseline": {
            "probabilities": {
                CLASS_NAMES[class_id]: float(
                    train_class_probabilities[class_id]
                )
                for class_id in range(3)
            },
            "mlogloss": float(
                log_loss(
                    y_test,
                    baseline_probabilities,
                    labels=[0, 1, 2],
                )
            ),
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    predictions = test[
        ["date", "stock_id", "return_5d", TARGET_COLUMN]
    ].copy()
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")
    predictions = predictions.rename(columns={TARGET_COLUMN: "actual_class"})
    predictions["predicted_class"] = predicted
    predictions["actual_label"] = predictions["actual_class"].map(
        dict(enumerate(CLASS_NAMES))
    )
    predictions["predicted_label"] = predictions["predicted_class"].map(
        dict(enumerate(CLASS_NAMES))
    )
    predictions["probability_down"] = probabilities[:, 0]
    predictions["probability_neutral"] = probabilities[:, 1]
    predictions["probability_up"] = probabilities[:, 2]
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

    return metrics


if __name__ == "__main__":
    results = train_model()
    print(json.dumps(results, indent=2, ensure_ascii=False))
