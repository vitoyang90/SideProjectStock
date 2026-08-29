"""Pooled five-day return regression and cross-sectional ranking experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from backtest import (
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SELL_TAX_BPS,
    HOLDING_DAYS,
    TRADING_DAYS_PER_YEAR,
)
from diagnose_v1 import VALIDATION_YEARS
from train_pooled_v2 import STOCK_IDS, _load_data, _retrospective_split, _walk_forward_split
from train_pooled_v2_scaled import FEATURES, _scale_data


MODEL_PATH = Path("models/xgboost_return_ranker_retrospective.json")
METRICS_PATH = Path("reports/return_ranker_metrics.csv")
PREDICTION_PATH = Path("data/processed/return_ranker_predictions.csv")
BACKTEST_PATH = Path("reports/return_ranker_backtest.csv")
INTERVAL_PATH = Path("reports/return_ranker_intervals.csv")
DECISION_PATH = Path("reports/return_ranker_decision.json")


def _build_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
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


def _fit(train: pd.DataFrame, validation: pd.DataFrame) -> XGBRegressor:
    model = _build_model()
    model.fit(
        train[FEATURES],
        train["return_5d"],
        eval_set=[(validation[FEATURES], validation["return_5d"])],
        verbose=False,
    )
    return model


def _rank_statistics(predictions: pd.DataFrame) -> dict:
    rank_ics = []
    top_returns = []
    bottom_returns = []
    for _, group in predictions.groupby("date"):
        if len(group) < 4:
            continue
        predicted_rank = group["predicted_return_5d"].rank()
        actual_rank = group["actual_return_5d"].rank()
        rank_ic = predicted_rank.corr(actual_rank)
        if pd.notna(rank_ic):
            rank_ics.append(float(rank_ic))
        ordered = group.sort_values("predicted_return_5d")
        bottom_returns.append(float(ordered.iloc[0]["actual_return_5d"]))
        top_returns.append(float(ordered.iloc[-1]["actual_return_5d"]))
    rank_ics_series = pd.Series(rank_ics, dtype=float)
    top = pd.Series(top_returns, dtype=float)
    bottom = pd.Series(bottom_returns, dtype=float)
    return {
        "rank_dates": len(rank_ics_series),
        "mean_rank_ic": float(rank_ics_series.mean()),
        "median_rank_ic": float(rank_ics_series.median()),
        "positive_rank_ic_rate": float((rank_ics_series > 0).mean()),
        "mean_top_rank_actual_return": float(top.mean()),
        "top_rank_positive_return_rate": float((top > 0).mean()),
        "mean_top_minus_bottom_return": float((top - bottom).mean()),
    }


def _score(
    model: XGBRegressor,
    frame: pd.DataFrame,
    evaluation: str,
    validation_year: int | None,
) -> tuple[dict, pd.DataFrame]:
    predicted = model.predict(frame[FEATURES])
    predictions = frame[["date", "stock_id", "return_5d"]].copy()
    predictions = predictions.rename(
        columns={"return_5d": "actual_return_5d"}
    )
    predictions["predicted_return_5d"] = predicted
    predictions["evaluation"] = evaluation
    predictions["validation_year"] = validation_year
    actual = predictions["actual_return_5d"]
    metrics = {
        "evaluation": evaluation,
        "validation_year": validation_year,
        "rows": len(predictions),
        "stocks": predictions["stock_id"].nunique(),
        "best_iteration": int(model.best_iteration),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "sign_accuracy": float(
            (np.sign(predicted) == np.sign(actual.to_numpy())).mean()
        ),
        **_rank_statistics(predictions),
    }
    return metrics, predictions


def _price_lookup(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    lookups = {}
    for stock_id, frame in prices.items():
        indexed = frame.copy()
        indexed["date"] = pd.to_datetime(indexed["date"])
        lookups[stock_id] = indexed.set_index("date").sort_index()
    return lookups


def _ranking_intervals(
    predictions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    equal_weight: bool = False,
) -> pd.DataFrame:
    price_tables = _price_lookup(prices)
    calendar = price_tables["2330"].index.tolist()
    calendar_index = {date: index for index, date in enumerate(calendar)}
    segments = (
        predictions.groupby("validation_year", dropna=False)
        if "validation_year" in predictions.columns
        else [(None, predictions)]
    )
    rows = []
    for segment_number, (_, segment) in enumerate(segments):
        common_dates = (
            segment.groupby("date")["stock_id"].nunique()
            .loc[lambda counts: counts == len(STOCK_IDS)]
            .index.sort_values()
        )
        for signal_number, signal_date in enumerate(common_dates[::HOLDING_DAYS]):
            position = calendar_index.get(signal_date)
            if position is None or position + HOLDING_DAYS + 1 >= len(calendar):
                continue
            entry_date = calendar[position + 1]
            exit_date = calendar[position + HOLDING_DAYS + 1]
            cross_section = segment[segment["date"] == signal_date]
            if equal_weight:
                selected_stock = "equal_weight_6"
                asset_returns = []
                for stock_id in STOCK_IDS:
                    table = price_tables[stock_id]
                    if entry_date not in table.index or exit_date not in table.index:
                        raise RuntimeError("Missing aligned external price")
                    asset_returns.append(
                        table.loc[exit_date, "open"]
                        / table.loc[entry_date, "open"]
                        - 1
                    )
                asset_return = float(np.mean(asset_returns))
            else:
                winner = cross_section.loc[
                    cross_section["predicted_return_5d"].idxmax()
                ]
                selected_stock = (
                    str(winner["stock_id"])
                    if winner["predicted_return_5d"] > 0 else "cash"
                )
                if selected_stock == "cash":
                    asset_return = 0.0
                else:
                    table = price_tables[selected_stock]
                    asset_return = float(
                        table.loc[exit_date, "open"]
                        / table.loc[entry_date, "open"]
                        - 1
                    )
            rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "selected_stock": selected_stock,
                    "asset_return": asset_return,
                    "new_segment": signal_number == 0,
                    "segment_number": segment_number,
                }
            )
    intervals = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
    if intervals.empty:
        raise ValueError("No complete ranking intervals")
    return intervals


def _max_drawdown(wealth: pd.Series) -> float:
    values = pd.concat([pd.Series([1.0]), wealth], ignore_index=True)
    return float((values / values.cummax() - 1).min())


def _simulate_ranking(
    intervals: pd.DataFrame,
    evaluation: str,
    strategy: str,
) -> tuple[dict, pd.DataFrame]:
    commission = DEFAULT_COMMISSION_BPS / 10_000
    sell_cost = (
        DEFAULT_COMMISSION_BPS + DEFAULT_SELL_TAX_BPS
    ) / 10_000
    net_wealth = 1.0
    gross_wealth = 1.0
    previous_stock = "cash"
    buys = 0
    sells = 0
    curve_rows = []

    for interval in intervals.itertuples(index=False):
        starting_net = net_wealth
        starting_gross = gross_wealth
        if interval.new_segment and curve_rows and previous_stock != "cash":
            net_wealth *= 1 - sell_cost
            sells += 1
            previous_stock = "cash"
            prior_net = (
                1.0 if len(curve_rows) == 1 else curve_rows[-2]["net_equity"]
            )
            curve_rows[-1]["net_equity"] = net_wealth
            curve_rows[-1]["period_net_return"] = net_wealth / prior_net - 1
            starting_net = net_wealth

        if previous_stock != interval.selected_stock:
            if previous_stock != "cash":
                net_wealth *= 1 - sell_cost
                sells += 1
            if interval.selected_stock != "cash":
                net_wealth *= 1 - commission
                buys += 1
        gross_wealth *= 1 + interval.asset_return
        net_wealth *= 1 + interval.asset_return
        curve_rows.append(
            {
                "date": interval.exit_date,
                "gross_equity": gross_wealth,
                "net_equity": net_wealth,
                "period_gross_return": gross_wealth / starting_gross - 1,
                "period_net_return": net_wealth / starting_net - 1,
            }
        )
        previous_stock = interval.selected_stock

    if previous_stock != "cash":
        net_wealth *= 1 - sell_cost
        sells += 1
        prior_net = 1.0 if len(curve_rows) == 1 else curve_rows[-2]["net_equity"]
        curve_rows[-1]["net_equity"] = net_wealth
        curve_rows[-1]["period_net_return"] = net_wealth / prior_net - 1

    curve = pd.DataFrame(curve_rows)
    start = intervals["entry_date"].iloc[0]
    end = intervals["exit_date"].iloc[-1]
    years = (end - start).days / 365.25
    period_returns = curve["period_net_return"]
    standard_deviation = period_returns.std(ddof=1)
    sharpe = (
        float(
            period_returns.mean()
            / standard_deviation
            * np.sqrt(TRADING_DAYS_PER_YEAR / HOLDING_DAYS)
        )
        if standard_deviation > 0 else 0.0
    )
    net_return = net_wealth - 1
    gross_return = gross_wealth - 1
    metrics = {
        "evaluation": evaluation,
        "strategy": strategy,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "intervals": len(intervals),
        "gross_total_return": gross_return,
        "net_total_return": net_return,
        "gross_cagr": (
            float((1 + gross_return) ** (1 / years) - 1)
            if years > 0 else 0.0
        ),
        "net_cagr": (
            float((1 + net_return) ** (1 / years) - 1) if years > 0 else 0.0
        ),
        "net_annualized_sharpe": sharpe,
        "net_max_drawdown": _max_drawdown(curve["net_equity"]),
        "buys": buys,
        "sells": sells,
        "one_way_turnover": buys + sells,
    }
    return metrics, curve


def _run_backtests(
    predictions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    interval_frames = []
    for evaluation in ("walk_forward", "external_test"):
        evaluation_predictions = predictions[
            predictions["evaluation"] == evaluation
        ].copy()
        if evaluation == "external_test":
            evaluation_predictions = evaluation_predictions.drop(
                columns="validation_year"
            )
        for strategy, equal_weight in (
            ("predicted_top1", False),
            ("equal_weight_6", True),
        ):
            intervals = _ranking_intervals(
                evaluation_predictions, prices, equal_weight=equal_weight,
            )
            result, _ = _simulate_ranking(intervals, evaluation, strategy)
            metrics.append(result)
            intervals["evaluation"] = evaluation
            intervals["strategy"] = strategy
            interval_frames.append(intervals)
    return pd.DataFrame(metrics), pd.concat(interval_frames, ignore_index=True)


def run_return_ranker() -> dict:
    data, prices = _load_data()
    nonfinite_target_rows = int((~np.isfinite(data["return_5d"])).sum())
    data = data[np.isfinite(data["return_5d"])].copy()
    data = _scale_data(data)
    metric_rows = []
    prediction_frames = []
    for year in VALIDATION_YEARS:
        train, validation = _walk_forward_split(data, year)
        model = _fit(train, validation)
        metrics, predictions = _score(
            model, validation, "walk_forward", year,
        )
        metric_rows.append(metrics)
        prediction_frames.append(predictions)

    train, validation, test = _retrospective_split(data)
    retrospective_model = _fit(train, validation)
    metrics, predictions = _score(
        retrospective_model, test, "external_test", None,
    )
    metric_rows.append(metrics)
    prediction_frames.append(predictions)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    retrospective_model.save_model(MODEL_PATH)

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    backtests, intervals = _run_backtests(predictions_frame, prices)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    predictions_frame["date"] = pd.to_datetime(
        predictions_frame["date"]
    ).dt.strftime("%Y-%m-%d")
    predictions_frame.to_csv(
        PREDICTION_PATH, index=False, encoding="utf-8-sig",
    )
    backtests.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    intervals.to_csv(INTERVAL_PATH, index=False, encoding="utf-8-sig")

    walk = metrics_frame[metrics_frame["evaluation"] == "walk_forward"]
    test_metrics = metrics_frame[
        metrics_frame["evaluation"] == "external_test"
    ].iloc[0]
    test_backtests = backtests.set_index(["evaluation", "strategy"])
    strategy = test_backtests.loc[("external_test", "predicted_top1")]
    benchmark = test_backtests.loc[("external_test", "equal_weight_6")]
    decision = {
        "version": "return_ranker",
        "status": "research_signal_only",
        "target": "continuous future five-day return",
        "stocks": list(STOCK_IDS),
        "feature_count": len(FEATURES),
        "dropped_nonfinite_target_rows": nonfinite_target_rows,
        "walk_forward": {
            "mean_rank_ic": float(walk["mean_rank_ic"].mean()),
            "positive_rank_ic_folds": int((walk["mean_rank_ic"] > 0).sum()),
            "folds": len(walk),
            "mean_top_minus_bottom_return": float(
                walk["mean_top_minus_bottom_return"].mean()
            ),
            "top1_gross_return": float(
                test_backtests.loc[
                    ("walk_forward", "predicted_top1"),
                    "gross_total_return",
                ]
            ),
            "top1_net_return": float(
                test_backtests.loc[
                    ("walk_forward", "predicted_top1"),
                    "net_total_return",
                ]
            ),
        },
        "external_test": {
            "rank_ic": float(test_metrics["mean_rank_ic"]),
            "positive_rank_ic_rate": float(
                test_metrics["positive_rank_ic_rate"]
            ),
            "top1_net_return": float(strategy["net_total_return"]),
            "top1_gross_return": float(strategy["gross_total_return"]),
            "equal_weight_net_return": float(benchmark["net_total_return"]),
            "top1_sharpe": float(strategy["net_annualized_sharpe"]),
            "equal_weight_sharpe": float(
                benchmark["net_annualized_sharpe"]
            ),
        },
        "warning": "Retrospective only because the 2025-2026 period was previously viewed",
        "model_path": MODEL_PATH.as_posix(),
    }
    DECISION_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    result = run_return_ranker()
    print(json.dumps(result, indent=2, ensure_ascii=False))
