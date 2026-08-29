"""Train and compare a six-stock pooled V2 XGBoost model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

from backtest import (
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SELL_TAX_BPS,
    _build_intervals,
    _simulate,
)
from diagnose_v1 import VALIDATION_YEARS, _build_model, _labels
from train import (
    CLASS_NAMES,
    DEFAULT_LABEL_THRESHOLD,
    PURGE_ROWS,
    REPORT_DIR,
    TARGET_COLUMN,
    _feature_columns,
    _sample_weights,
)


STOCK_IDS = ("2330", "2317", "2454", "2303", "2881", "1301")
FEATURES = _feature_columns("v2")
MODEL_PATH = Path("models/xgboost_v2_pooled_retrospective.json")
DETAIL_PATH = REPORT_DIR / "pooled_v2_classification.csv"
SUMMARY_PATH = REPORT_DIR / "pooled_v2_classification_summary.csv"
CLASSIFICATION_COMPARISON_PATH = (
    REPORT_DIR / "pooled_v2_vs_independent_classification.csv"
)
BACKTEST_PATH = REPORT_DIR / "pooled_v2_backtest.csv"
BACKTEST_COMPARISON_PATH = REPORT_DIR / "pooled_v2_vs_independent_backtest.csv"
DECISION_PATH = REPORT_DIR / "pooled_v2_decision.json"
PREDICTION_PATH = Path("data/processed/pooled_v2_predictions.csv")


def _load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    datasets = []
    prices = {}
    for stock_id in STOCK_IDS:
        if stock_id == "2330":
            dataset_path = Path("data/processed/2330_dataset.csv")
            price_path = Path("data/raw/2330_price.csv")
        else:
            dataset_path = Path(
                f"data/external_validation/{stock_id}_dataset.csv"
            )
            price_path = Path(
                f"data/external_validation/{stock_id}_price.csv"
            )
        dataset = pd.read_csv(dataset_path)
        dataset["stock_id"] = dataset["stock_id"].astype(str)
        dataset["date"] = pd.to_datetime(dataset["date"], errors="raise")
        dataset = dataset.dropna(subset=["return_5d"]).copy()
        dataset[TARGET_COLUMN] = _labels(
            dataset["return_5d"], DEFAULT_LABEL_THRESHOLD,
        )
        dataset[FEATURES] = dataset[FEATURES].replace(
            [np.inf, -np.inf], np.nan,
        )
        datasets.append(dataset)

        price = pd.read_csv(price_path)
        price["stock_id"] = price["stock_id"].astype(str)
        price["date"] = pd.to_datetime(price["date"], errors="raise")
        prices[stock_id] = price.sort_values("date").reset_index(drop=True)
    return pd.concat(datasets, ignore_index=True), prices


def _purge_tail(frame: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _, group in frame.groupby("stock_id", sort=False):
        group = group.sort_values("date")
        if len(group) <= PURGE_ROWS:
            raise ValueError("Not enough rows to purge a stock boundary")
        groups.append(group.iloc[:-PURGE_ROWS])
    return pd.concat(groups, ignore_index=True)


def _walk_forward_split(
    data: pd.DataFrame, year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(year, 1, 1)
    end = pd.Timestamp(year + 1, 1, 1)
    train = _purge_tail(data[data["date"] < start].copy())
    validation = _purge_tail(
        data[(data["date"] >= start) & (data["date"] < end)].copy()
    )
    return train, validation


def _retrospective_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _purge_tail(data[data["date"] < "2023-01-01"].copy())
    validation = _purge_tail(
        data[
            (data["date"] >= "2023-01-01")
            & (data["date"] < "2025-01-01")
        ].copy()
    )
    test = data[data["date"] >= "2025-01-01"].copy()
    return train, validation, test


def _fit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str] = FEATURES,
):
    model = _build_model()
    y_train = train[TARGET_COLUMN].astype(int)
    y_validation = validation[TARGET_COLUMN].astype(int)
    model.fit(
        train[features],
        y_train,
        sample_weight=_sample_weights(y_train, "balanced"),
        eval_set=[(validation[features], y_validation)],
        sample_weight_eval_set=[
            _sample_weights(y_validation, "balanced")
        ],
        verbose=False,
    )
    return model


def _score(
    model,
    frame: pd.DataFrame,
    evaluation: str,
    validation_year: int | None,
    features: list[str] = FEATURES,
    version: str = "v2_pooled",
) -> tuple[list[dict], pd.DataFrame]:
    predicted = model.predict(frame[features]).astype(int)
    probabilities = model.predict_proba(frame[features])
    scored = frame[["date", "stock_id", TARGET_COLUMN]].copy()
    scored["version"] = version
    scored["evaluation"] = evaluation
    scored["validation_year"] = validation_year
    scored["predicted_class"] = predicted
    scored["probability_down"] = probabilities[:, 0]
    scored["probability_neutral"] = probabilities[:, 1]
    scored["probability_up"] = probabilities[:, 2]
    rows = []
    for stock_id, group in scored.groupby("stock_id"):
        indices = group.index
        labels = frame.loc[indices, TARGET_COLUMN].astype(int)
        group_probabilities = probabilities[frame.index.get_indexer(indices)]
        group_predictions = predicted[frame.index.get_indexer(indices)]
        rows.append(
            {
                "stock_id": stock_id,
                "evaluation": evaluation,
                "validation_year": validation_year,
                "rows": len(group),
                "best_iteration": int(model.best_iteration),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, group_predictions)
                ),
                "macro_f1": float(
                    f1_score(labels, group_predictions, average="macro")
                ),
                "mlogloss": float(
                    log_loss(
                        labels,
                        group_probabilities,
                        labels=range(len(CLASS_NAMES)),
                    )
                ),
            }
        )
    return rows, scored


def _independent_classification() -> pd.DataFrame:
    external = pd.read_csv("reports/external_validation_classification.csv")
    external = external[external["version"] == "v2"].copy()
    external["stock_id"] = external["stock_id"].astype(str)
    external = external[
        [
            "stock_id", "evaluation", "validation_year",
            "balanced_accuracy", "macro_f1", "mlogloss",
        ]
    ]

    folds = pd.read_csv("reports/xgboost_walk_forward_version_folds.csv")
    folds = folds[folds["version"] == "v2"].copy()
    folds["stock_id"] = "2330"
    folds["evaluation"] = "walk_forward"
    folds = folds.rename(
        columns={
            "validation_balanced_accuracy": "balanced_accuracy",
            "validation_macro_f1": "macro_f1",
            "validation_mlogloss": "mlogloss",
        }
    )
    folds = folds[
        [
            "stock_id", "evaluation", "validation_year",
            "balanced_accuracy", "macro_f1", "mlogloss",
        ]
    ]

    metrics = json.loads(
        Path("reports/xgboost_v2_metrics.json").read_text(encoding="utf-8")
    )
    test = pd.DataFrame(
        [
            {
                "stock_id": "2330",
                "evaluation": "external_test",
                "validation_year": np.nan,
                "balanced_accuracy": metrics["test"]["balanced_accuracy"],
                "macro_f1": metrics["test"]["macro_f1"],
                "mlogloss": metrics["test"]["mlogloss"],
            }
        ]
    )
    return pd.concat([external, folds, test], ignore_index=True)


def _classification_comparison(pooled: pd.DataFrame) -> pd.DataFrame:
    independent = _independent_classification()
    for frame in (pooled, independent):
        frame["fold_key"] = frame["validation_year"].fillna(-1).astype(int)
    comparison = pooled.merge(
        independent,
        on=["stock_id", "evaluation", "fold_key"],
        suffixes=("_pooled", "_independent"),
        validate="one_to_one",
    )
    for metric in ("balanced_accuracy", "macro_f1", "mlogloss"):
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_pooled"]
            - comparison[f"{metric}_independent"]
        )
    return comparison


def _run_backtests(
    predictions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    strategy: str = "v2_pooled",
) -> pd.DataFrame:
    rows = []
    for evaluation in ("walk_forward", "external_test"):
        evaluation_predictions = predictions[
            predictions["evaluation"] == evaluation
        ]
        for stock_id in STOCK_IDS:
            stock_predictions = evaluation_predictions[
                evaluation_predictions["stock_id"] == stock_id
            ].copy()
            if evaluation == "external_test":
                stock_predictions = stock_predictions.drop(
                    columns="validation_year"
                )
            intervals = _build_intervals(
                stock_predictions, prices[stock_id],
            )
            metrics, _, _ = _simulate(
                intervals,
                evaluation,
                strategy,
                DEFAULT_COMMISSION_BPS,
                DEFAULT_SELL_TAX_BPS,
            )
            metrics["stock_id"] = stock_id
            rows.append(metrics)
    return pd.DataFrame(rows)


def _independent_backtests() -> pd.DataFrame:
    external = pd.read_csv("reports/external_validation_backtest.csv")
    external = external[external["strategy"] == "v2"].copy()
    external["stock_id"] = external["stock_id"].astype(str)
    local = pd.read_csv("reports/backtest_version_metrics.csv")
    local = local[local["strategy"] == "v2"].copy()
    local["stock_id"] = "2330"
    local["split"] = local["split"].replace({"test": "external_test"})
    return pd.concat([external, local], ignore_index=True)


def _backtest_comparison(pooled: pd.DataFrame) -> pd.DataFrame:
    independent = _independent_backtests()
    columns = [
        "stock_id", "split", "net_total_return", "net_annualized_sharpe",
        "net_max_drawdown", "exposure", "one_way_turnover",
    ]
    comparison = pooled[columns].merge(
        independent[columns],
        on=["stock_id", "split"],
        suffixes=("_pooled", "_independent"),
        validate="one_to_one",
    )
    for metric in (
        "net_total_return", "net_annualized_sharpe", "net_max_drawdown",
    ):
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_pooled"]
            - comparison[f"{metric}_independent"]
        )
    return comparison


def run_pooled_experiment() -> dict:
    data, prices = _load_data()
    classification_rows = []
    prediction_frames = []
    for year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(data, year)
        model = _fit(train, validation)
        rows, predictions = _score(
            model, validation, "walk_forward", year,
        )
        classification_rows.extend(rows)
        prediction_frames.append(predictions)

    train, validation, test = _retrospective_split(data)
    retrospective_model = _fit(train, validation)
    rows, predictions = _score(
        retrospective_model, test, "external_test", None,
    )
    classification_rows.extend(rows)
    prediction_frames.append(predictions)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    retrospective_model.save_model(MODEL_PATH)

    classification = pd.DataFrame(classification_rows)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = (
        classification.groupby("evaluation", as_index=False)
        .agg(
            observations=("stock_id", "count"),
            stocks=("stock_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_mlogloss=("mlogloss", "mean"),
        )
    )
    classification_comparison = _classification_comparison(classification)
    pooled_backtests = _run_backtests(all_predictions, prices)
    backtest_comparison = _backtest_comparison(pooled_backtests)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    classification.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    classification_comparison.to_csv(
        CLASSIFICATION_COMPARISON_PATH, index=False, encoding="utf-8-sig",
    )
    pooled_backtests.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    backtest_comparison.to_csv(
        BACKTEST_COMPARISON_PATH, index=False, encoding="utf-8-sig",
    )
    all_predictions["date"] = pd.to_datetime(
        all_predictions["date"]
    ).dt.strftime("%Y-%m-%d")
    all_predictions.to_csv(
        PREDICTION_PATH, index=False, encoding="utf-8-sig",
    )

    test_classification = classification_comparison[
        classification_comparison["evaluation"] == "external_test"
    ]
    test_backtest = backtest_comparison[
        backtest_comparison["split"] == "external_test"
    ]
    decision = {
        "version": "v2_pooled",
        "status": "rejected_retrospective",
        "stocks": list(STOCK_IDS),
        "features": FEATURES,
        "warning": "2025-2026 was previously viewed; fresh forward data is required for promotion",
        "external_test": {
            "classification_balanced_accuracy_wins": int(
                (test_classification["delta_balanced_accuracy"] > 0).sum()
            ),
            "classification_macro_f1_wins": int(
                (test_classification["delta_macro_f1"] > 0).sum()
            ),
            "backtest_return_wins": int(
                (test_backtest["delta_net_total_return"] > 0).sum()
            ),
            "stocks": len(STOCK_IDS),
            "mean_delta_balanced_accuracy": float(
                test_classification["delta_balanced_accuracy"].mean()
            ),
            "mean_delta_macro_f1": float(
                test_classification["delta_macro_f1"].mean()
            ),
            "mean_delta_net_return": float(
                test_backtest["delta_net_total_return"].mean()
            ),
        },
        "model_path": MODEL_PATH.as_posix(),
    }
    DECISION_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    result = run_pooled_experiment()
    print(json.dumps(result, indent=2, ensure_ascii=False))
