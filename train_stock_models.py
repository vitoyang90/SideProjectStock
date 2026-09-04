"""Train stock-specific V2 artifacts for fresh forward evaluation."""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier

from collectors.chip import (
    fetch_institutional_investors,
    fetch_margin_purchase_short_sale,
)
from collectors.price import fetch_stock_price
from diagnose_v1 import VALIDATION_YEARS, _build_model, _labels, _walk_forward_split
from features.chip import (
    build_institutional_features,
    build_margin_features,
    merge_chip_features,
)
from features.technical import add_technical_features
from train import (
    CLASS_NAMES,
    DEFAULT_LABEL_THRESHOLD,
    REPORT_DIR,
    TARGET_COLUMN,
    _class_counts,
    _feature_columns,
    _sample_weights,
)


STOCK_IDS = ("2330", "2454", "2317", "2308", "2881")
START_DATE = "2018-01-01"
END_DATE = date.today().isoformat()
MODEL_DIR = Path("models/stocks")
STOCK_DATA_DIR = Path("data/stock_models")
EXTERNAL_DATA_DIR = Path("data/external_validation")
REGISTRY_PATH = Path("models/registry.json")
FOLD_PATH = REPORT_DIR / "stock_models_walk_forward_folds.csv"
SUMMARY_PATH = REPORT_DIR / "stock_models_summary.csv"
DECISION_PATH = REPORT_DIR / "stock_models_decision.json"
CLASS_WEIGHTING = "balanced"


def _valid_price_rows(price: pd.DataFrame) -> pd.DataFrame:
    numeric = ["open", "max", "min", "close", "Trading_Volume"]
    price = price.copy()
    price[numeric] = price[numeric].apply(pd.to_numeric, errors="coerce")
    return price[
        price[["open", "max", "min", "close"]].gt(0).all(axis=1)
        & price["Trading_Volume"].gt(0)
    ].copy()


def _build_dataset(
    price: pd.DataFrame,
    institutional: pd.DataFrame,
    margin: pd.DataFrame,
) -> pd.DataFrame:
    dataset = merge_chip_features(
        _valid_price_rows(price),
        build_institutional_features(institutional),
        build_margin_features(margin),
    )
    dataset = add_technical_features(dataset)
    dataset["future_close_5d"] = dataset.groupby("stock_id")["close"].shift(-5)
    dataset["return_5d"] = dataset["future_close_5d"] / dataset["close"] - 1
    return dataset.dropna(subset=["return_5d"]).reset_index(drop=True)


def _fetch_missing_stock(stock_id: str, token: str) -> pd.DataFrame:
    frames = {
        "price": fetch_stock_price(stock_id, START_DATE, END_DATE, token),
        "institutional": fetch_institutional_investors(
            stock_id, START_DATE, END_DATE, token,
        ),
        "margin": fetch_margin_purchase_short_sale(
            stock_id, START_DATE, END_DATE, token,
        ),
    }
    empty = [name for name, frame in frames.items() if frame.empty]
    if empty:
        raise ValueError(f"FinMind returned empty datasets for {stock_id}: {empty}")
    STOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(
            STOCK_DATA_DIR / f"{stock_id}_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    dataset = _build_dataset(
        frames["price"], frames["institutional"], frames["margin"],
    )
    dataset.to_csv(
        STOCK_DATA_DIR / f"{stock_id}_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return dataset


def _load_dataset(stock_id: str, token: str) -> pd.DataFrame:
    candidates = [
        Path("data/processed") / f"{stock_id}_dataset.csv",
        EXTERNAL_DATA_DIR / f"{stock_id}_dataset.csv",
        STOCK_DATA_DIR / f"{stock_id}_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return _fetch_missing_stock(stock_id, token)


def select_tree_count(best_iterations: list[int]) -> int:
    if not best_iterations:
        raise ValueError("At least one walk-forward best iteration is required")
    return max(int(np.median(best_iterations)) + 1, 1)


def _fit_final_model(
    data: pd.DataFrame,
    features: list[str],
    n_estimators: int,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=n_estimators,
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
        data[features],
        labels,
        sample_weight=_sample_weights(labels, CLASS_WEIGHTING),
        verbose=False,
    )
    return model


def _walk_forward_stock(
    stock_id: str,
    data: pd.DataFrame,
    features: list[str],
) -> tuple[list[dict], list[int]]:
    pretest = data[data["date"] < "2025-01-01"].copy()
    rows = []
    best_iterations = []
    for year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(pretest, year)
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
        predicted = model.predict(validation[features]).astype(int)
        probabilities = model.predict_proba(validation[features])
        majority_class = int(y_train.mode().iloc[0])
        majority = np.full(len(y_validation), majority_class)
        counts = _class_counts(pd.Series(predicted))
        best_iterations.append(int(model.best_iteration))
        rows.append({
            "stock_id": stock_id,
            "validation_year": year,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "best_iteration": int(model.best_iteration),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_validation, predicted)
            ),
            "macro_f1": float(f1_score(y_validation, predicted, average="macro")),
            "mlogloss": float(
                log_loss(y_validation, probabilities, labels=range(len(CLASS_NAMES)))
            ),
            "majority_balanced_accuracy": float(
                balanced_accuracy_score(y_validation, majority)
            ),
            "majority_macro_f1": float(
                f1_score(y_validation, majority, average="macro", zero_division=0)
            ),
            "predicted_down": counts["down"],
            "predicted_neutral": counts["neutral"],
            "predicted_up": counts["up"],
        })
    return rows, best_iterations


def _summary(folds: pd.DataFrame) -> pd.DataFrame:
    folds = folds.copy()
    folds["delta_balanced_accuracy_vs_majority"] = (
        folds["balanced_accuracy"] - folds["majority_balanced_accuracy"]
    )
    folds["delta_macro_f1_vs_majority"] = (
        folds["macro_f1"] - folds["majority_macro_f1"]
    )
    return (
        folds.groupby("stock_id", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_mlogloss=("mlogloss", "mean"),
            mean_delta_balanced_accuracy_vs_majority=(
                "delta_balanced_accuracy_vs_majority", "mean"
            ),
            mean_delta_macro_f1_vs_majority=(
                "delta_macro_f1_vs_majority", "mean"
            ),
            balanced_accuracy_winning_folds=(
                "delta_balanced_accuracy_vs_majority", lambda s: int(s.gt(0).sum())
            ),
            macro_f1_winning_folds=(
                "delta_macro_f1_vs_majority", lambda s: int(s.gt(0).sum())
            ),
        )
        .sort_values("stock_id")
    )


def _update_registry(model_entries: dict[str, dict]) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["stock_models"] = model_entries
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_training() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    token = os.environ.get("FINMIND_TOKEN", "")
    features = _feature_columns("v2")
    fold_rows = []
    prepared = {}
    selected_trees = {}

    for stock_id in STOCK_IDS:
        data = _load_dataset(stock_id, token)
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        missing = set(features + ["date", "stock_id", "return_5d"]).difference(
            data.columns
        )
        if missing:
            raise ValueError(f"{stock_id} dataset is missing: {sorted(missing)}")
        data = data.sort_values("date").dropna(subset=["return_5d"]).copy()
        data[features] = data[features].replace([np.inf, -np.inf], np.nan)
        data[TARGET_COLUMN] = _labels(
            data["return_5d"], DEFAULT_LABEL_THRESHOLD,
        )
        rows, iterations = _walk_forward_stock(stock_id, data, features)
        fold_rows.extend(rows)
        prepared[stock_id] = data
        selected_trees[stock_id] = select_tree_count(iterations)

    folds = pd.DataFrame(fold_rows)
    summary = _summary(folds)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_entries = {}
    trained_on = date.today().isoformat()
    for stock_id, data in prepared.items():
        model = _fit_final_model(data, features, selected_trees[stock_id])
        model_path = MODEL_DIR / f"{stock_id}_v2_forward.json"
        model.save_model(model_path)
        stock_summary = summary.loc[summary["stock_id"] == stock_id].iloc[0]
        direction_signal_eligible = bool(
            stock_summary["mean_balanced_accuracy"] > (1 / 3)
            and stock_summary["balanced_accuracy_winning_folds"] >= 3
        )
        model_entries[stock_id] = {
            "version": "v2",
            "artifact_id": f"{stock_id}_v2_forward_{trained_on.replace('-', '')}",
            "status": (
                "research_champion_forward_refit"
                if stock_id == "2330" else "research_baseline_forward_refit"
            ),
            "model_path": model_path.as_posix(),
            "feature_count": len(features),
            "n_estimators": selected_trees[stock_id],
            "training_start": data["date"].min().strftime("%Y-%m-%d"),
            "training_end": data["date"].max().strftime("%Y-%m-%d"),
            "trained_on": trained_on,
            "walk_forward_mean_balanced_accuracy": float(
                stock_summary["mean_balanced_accuracy"]
            ),
            "walk_forward_mean_macro_f1": float(stock_summary["mean_macro_f1"]),
            "balanced_accuracy_winning_folds": int(
                stock_summary["balanced_accuracy_winning_folds"]
            ),
            "direction_signal_eligible": direction_signal_eligible,
            "warning": "Fresh forward evidence is required; not investment-ready.",
        }

    decision = {
        "version": "stock_specific_v2_forward_models",
        "status": "research_forward_evaluation",
        "stocks": list(STOCK_IDS),
        "feature_version": "v2",
        "validation_years": list(VALIDATION_YEARS),
        "tree_selection": "median zero-based best iteration plus one",
        "test_period_used_for_hyperparameter_selection": False,
        "models": model_entries,
        "warning": (
            "Models were refit on all labels available at build time and are only "
            "eligible for predictions after their trained_on date."
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    DECISION_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _update_registry(model_entries)
    return summary, folds, decision


if __name__ == "__main__":
    result_summary, _, result_decision = run_training()
    print(result_summary.to_string(index=False))
    print()
    print(json.dumps(result_decision, indent=2, ensure_ascii=False))
