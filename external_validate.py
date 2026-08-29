"""External multi-stock validation for V2 versus the V3-lite PB candidate."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from collectors.chip import (
    fetch_institutional_investors,
    fetch_margin_purchase_short_sale,
)
from collectors.fundamental import fetch_valuation
from collectors.price import fetch_stock_price
from diagnose_v1 import VALIDATION_YEARS, _build_model, _labels, _walk_forward_split
from features.chip import (
    build_institutional_features,
    build_margin_features,
    merge_chip_features,
)
from features.fundamental import build_valuation_features
from features.technical import add_technical_features
from train import (
    CLASS_NAMES,
    DEFAULT_LABEL_THRESHOLD,
    REPORT_DIR,
    TARGET_COLUMN,
    _feature_columns,
    _sample_weights,
    _split_by_time,
)


# Fixed before downloading results to reduce selection bias.
STOCK_IDS = ("2317", "2454", "2303", "2881", "1301")
START_DATE = "2018-01-01"
END_DATE = "2026-08-24"
CACHE_DIR = Path("data/external_validation")
CLASSIFICATION_DETAIL_PATH = REPORT_DIR / "external_validation_classification.csv"
CLASSIFICATION_SUMMARY_PATH = (
    REPORT_DIR / "external_validation_classification_summary.csv"
)
BACKTEST_DETAIL_PATH = REPORT_DIR / "external_validation_backtest.csv"
BACKTEST_SUMMARY_PATH = REPORT_DIR / "external_validation_backtest_summary.csv"
PREDICTION_PATH = Path("data/processed/external_validation_predictions.csv")
VERSIONS = {
    "v2": _feature_columns("v2"),
    "v3_lite": _feature_columns("v2") + ["pb"],
}


def _cached_fetch(
    stock_id: str,
    name: str,
    fetcher,
    token: str,
) -> pd.DataFrame:
    path = CACHE_DIR / f"{stock_id}_{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    frame = fetcher(
        stock_id=stock_id,
        start_date=START_DATE,
        end_date=END_DATE,
        token=token,
    )
    if frame.empty:
        raise ValueError(f"FinMind returned no {name} data for {stock_id}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def _build_stock_dataset(stock_id: str, token: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    price = _cached_fetch(stock_id, "price", fetch_stock_price, token)
    price = price.copy()
    price_columns = ["open", "max", "min", "close", "Trading_Volume"]
    price[price_columns] = price[price_columns].apply(
        pd.to_numeric, errors="coerce",
    )
    # FinMind may include zero-filled non-trading placeholders. They are not
    # executable observations and would create infinite forward returns.
    price = price[
        price[["open", "max", "min", "close"]].gt(0).all(axis=1)
        & price["Trading_Volume"].gt(0)
    ].copy()
    institutional = _cached_fetch(
        stock_id, "institutional", fetch_institutional_investors, token,
    )
    margin = _cached_fetch(
        stock_id, "margin", fetch_margin_purchase_short_sale, token,
    )
    valuation = _cached_fetch(
        stock_id, "valuation", fetch_valuation, token,
    )
    dataset = merge_chip_features(
        price,
        build_institutional_features(institutional),
        build_margin_features(margin),
    )
    dataset = dataset.merge(
        build_valuation_features(valuation)[["date", "stock_id", "pb"]],
        on=["date", "stock_id"],
        how="left",
        validate="one_to_one",
    )
    dataset = add_technical_features(dataset)
    dataset["future_close_5d"] = dataset.groupby("stock_id")["close"].shift(-5)
    dataset["return_5d"] = (
        dataset["future_close_5d"] / dataset["close"] - 1
    )
    dataset = dataset.dropna(subset=["future_close_5d"]).reset_index(drop=True)
    path = CACHE_DIR / f"{stock_id}_dataset.csv"
    dataset.to_csv(path, index=False, encoding="utf-8-sig")
    price = price.copy()
    price["date"] = pd.to_datetime(price["date"], errors="raise")
    return dataset, price.sort_values("date").reset_index(drop=True)


def _classification_row(
    stock_id: str,
    version: str,
    evaluation: str,
    validation_year: int | None,
    model,
    x_data: pd.DataFrame,
    labels: pd.Series,
) -> tuple[dict, np.ndarray, np.ndarray]:
    predicted = model.predict(x_data).astype(int)
    probabilities = model.predict_proba(x_data)
    return (
        {
            "stock_id": stock_id,
            "version": version,
            "evaluation": evaluation,
            "validation_year": validation_year,
            "rows": len(labels),
            "best_iteration": int(model.best_iteration),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predicted)
            ),
            "macro_f1": float(f1_score(labels, predicted, average="macro")),
            "mlogloss": float(
                log_loss(labels, probabilities, labels=range(len(CLASS_NAMES)))
            ),
        },
        predicted,
        probabilities,
    )


def _fit_and_score(
    stock_id: str,
    version: str,
    evaluation: str,
    validation_year: int | None,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    score: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    features = VERSIONS[version]
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
    score_labels = score[TARGET_COLUMN].astype(int)
    row, predicted, probabilities = _classification_row(
        stock_id,
        version,
        evaluation,
        validation_year,
        model,
        score[features],
        score_labels,
    )
    predictions = score[["date", "stock_id", TARGET_COLUMN]].copy()
    predictions["version"] = version
    predictions["evaluation"] = evaluation
    predictions["validation_year"] = validation_year
    predictions["predicted_class"] = predicted
    predictions["probability_down"] = probabilities[:, 0]
    predictions["probability_neutral"] = probabilities[:, 1]
    predictions["probability_up"] = probabilities[:, 2]
    return row, predictions


def _evaluate_stock(
    stock_id: str,
    dataset: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="raise")
    dataset[TARGET_COLUMN] = _labels(
        dataset["return_5d"], DEFAULT_LABEL_THRESHOLD,
    )
    all_features = sorted({f for values in VERSIONS.values() for f in values})
    dataset[all_features] = dataset[all_features].replace(
        [np.inf, -np.inf], np.nan,
    )
    classification_rows = []
    prediction_frames = []

    pretest = dataset[dataset["date"] < "2025-01-01"].copy()
    for year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(pretest, year)
        for version in VERSIONS:
            row, predictions = _fit_and_score(
                stock_id,
                version,
                "walk_forward",
                year,
                train,
                validation,
                validation,
            )
            classification_rows.append(row)
            prediction_frames.append(predictions)

    train, validation, test = _split_by_time(dataset)
    for version in VERSIONS:
        row, predictions = _fit_and_score(
            stock_id,
            version,
            "external_test",
            None,
            train,
            validation,
            test,
        )
        classification_rows.append(row)
        prediction_frames.append(predictions)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    backtest_rows = []
    for evaluation in ("walk_forward", "external_test"):
        evaluation_predictions = all_predictions[
            all_predictions["evaluation"] == evaluation
        ]
        for version in VERSIONS:
            version_predictions = evaluation_predictions[
                evaluation_predictions["version"] == version
            ].copy()
            if evaluation == "external_test":
                version_predictions = version_predictions.drop(
                    columns="validation_year"
                )
            intervals = _build_intervals(version_predictions, prices)
            metrics, _, _ = _simulate(
                intervals,
                evaluation,
                version,
                DEFAULT_COMMISSION_BPS,
                DEFAULT_SELL_TAX_BPS,
            )
            metrics["stock_id"] = stock_id
            backtest_rows.append(metrics)

        benchmark_predictions = evaluation_predictions[
            evaluation_predictions["version"] == "v2"
        ].copy()
        if evaluation == "external_test":
            benchmark_predictions = benchmark_predictions.drop(
                columns="validation_year"
            )
        intervals = _build_intervals(
            benchmark_predictions, prices, always_long=True,
        )
        metrics, _, _ = _simulate(
            intervals,
            evaluation,
            "buy_hold",
            DEFAULT_COMMISSION_BPS,
            DEFAULT_SELL_TAX_BPS,
        )
        metrics["stock_id"] = stock_id
        backtest_rows.append(metrics)
    return classification_rows, prediction_frames, backtest_rows


def _classification_summary(details: pd.DataFrame) -> pd.DataFrame:
    summary = (
        details.groupby(["evaluation", "version"], as_index=False)
        .agg(
            observations=("stock_id", "count"),
            stocks=("stock_id", "nunique"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            median_balanced_accuracy=("balanced_accuracy", "median"),
            mean_macro_f1=("macro_f1", "mean"),
            median_macro_f1=("macro_f1", "median"),
            mean_mlogloss=("mlogloss", "mean"),
        )
    )
    return summary


def _backtest_summary(details: pd.DataFrame) -> pd.DataFrame:
    return (
        details.groupby(["split", "strategy"], as_index=False)
        .agg(
            stocks=("stock_id", "nunique"),
            mean_net_return=("net_total_return", "mean"),
            median_net_return=("net_total_return", "median"),
            profitable_stocks=(
                "net_total_return", lambda s: int((s > 0).sum())
            ),
            mean_sharpe=("net_annualized_sharpe", "mean"),
            mean_max_drawdown=("net_max_drawdown", "mean"),
        )
    )


def run_external_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    token = os.getenv("FINMIND_TOKEN", "")
    datasets = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_build_stock_dataset, stock_id, token): stock_id
            for stock_id in STOCK_IDS
        }
        for future in as_completed(futures):
            stock_id = futures[future]
            datasets[stock_id] = future.result()
            print(f"Prepared external dataset for {stock_id}")

    classification_rows = []
    prediction_frames = []
    backtest_rows = []
    for stock_id in STOCK_IDS:
        dataset, prices = datasets[stock_id]
        rows, predictions, stock_backtests = _evaluate_stock(
            stock_id, dataset, prices,
        )
        classification_rows.extend(rows)
        prediction_frames.extend(predictions)
        backtest_rows.extend(stock_backtests)
        print(f"Evaluated V2 and V3-lite for {stock_id}")

    classification = pd.DataFrame(classification_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    backtests = pd.DataFrame(backtest_rows)
    classification_summary = _classification_summary(classification)
    backtest_summary = _backtest_summary(backtests)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    classification.to_csv(
        CLASSIFICATION_DETAIL_PATH, index=False, encoding="utf-8-sig",
    )
    classification_summary.to_csv(
        CLASSIFICATION_SUMMARY_PATH, index=False, encoding="utf-8-sig",
    )
    backtests.to_csv(BACKTEST_DETAIL_PATH, index=False, encoding="utf-8-sig")
    backtest_summary.to_csv(
        BACKTEST_SUMMARY_PATH, index=False, encoding="utf-8-sig",
    )
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    predictions.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
    return classification_summary, backtest_summary


if __name__ == "__main__":
    classification_result, backtest_result = run_external_validation()
    print("\nClassification summary")
    print(classification_result.to_string(index=False))
    print("\nBacktest summary")
    print(backtest_result.to_string(index=False))
