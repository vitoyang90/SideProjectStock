"""Nested walk-forward validation for low-turnover ranking rules."""

import json
import math
from itertools import product
from pathlib import Path

import pandas as pd

from optimize_rank_strategy import (
    MAX_DRAWDOWN_DEGRADATION,
    MIN_ALLOWED_MAX_DRAWDOWN,
    MINIMUM_RETURNS,
    PREDICTION_PATH,
    RANK_BUFFERS,
    SWITCH_EDGES,
    _build_intervals,
)
from train_pooled_v2 import _load_data
from train_return_ranker import _simulate_ranking


SELECTION_PATH = Path("reports/rank_strategy_nested_selections.csv")
METRICS_PATH = Path("reports/rank_strategy_nested_metrics.json")
INTERVAL_PATH = Path("data/processed/rank_strategy_nested_intervals.csv")
EVALUATION_YEARS = (2022, 2023, 2024)
BASELINE_ONE_WAY_TURNOVER_PER_YEAR = 52


def _simulate_subset(
    intervals: pd.DataFrame,
    years: list[int],
) -> tuple[dict, int]:
    subset = intervals[intervals["validation_year"].isin(years)].copy()
    subset = subset.sort_values("entry_date").reset_index(drop=True)
    metrics, _ = _simulate_ranking(
        subset,
        "nested_training",
        "low_turnover_rank",
    )
    profitable_years = 0
    for year in years:
        yearly = subset[subset["validation_year"] == year].copy()
        yearly = yearly.reset_index(drop=True)
        yearly["new_segment"] = False
        yearly.loc[0, "new_segment"] = True
        yearly_metrics, _ = _simulate_ranking(
            yearly,
            f"nested_training_{year}",
            "low_turnover_rank",
        )
        profitable_years += yearly_metrics["net_total_return"] > 0
    return metrics, int(profitable_years)


def _cash_intervals(template: pd.DataFrame) -> pd.DataFrame:
    cash = template.copy()
    cash["selected_stock"] = "cash"
    cash["asset_return"] = 0.0
    return cash


def run_nested_validation() -> dict:
    predictions = pd.read_csv(PREDICTION_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise")
    predictions = predictions[
        predictions["evaluation"] == "walk_forward"
    ].copy()
    if predictions["date"].max() >= pd.Timestamp("2025-01-01"):
        raise RuntimeError("Nested validation must not use the test period")
    _, prices = _load_data()

    interval_cache = {}
    for parameters in product(RANK_BUFFERS, SWITCH_EDGES, MINIMUM_RETURNS):
        interval_cache[parameters] = _build_intervals(
            predictions,
            prices,
            rank_buffer=parameters[0],
            switch_edge=parameters[1],
            minimum_return=parameters[2],
        )

    selection_rows = []
    evaluation_intervals = []
    for evaluation_year in EVALUATION_YEARS:
        training_years = list(range(2021, evaluation_year))
        required_profitable_years = math.ceil(0.75 * len(training_years))
        maximum_turnover = (
            BASELINE_ONE_WAY_TURNOVER_PER_YEAR
            * len(training_years)
            // 2
        )
        candidates = []
        for parameters, intervals in interval_cache.items():
            metrics, profitable_years = _simulate_subset(
                intervals, training_years,
            )
            passes = (
                metrics["one_way_turnover"] <= maximum_turnover
                and metrics["net_total_return"] > 0
                and profitable_years >= required_profitable_years
                and metrics["net_max_drawdown"] >= MIN_ALLOWED_MAX_DRAWDOWN
            )
            if passes:
                candidates.append(
                    {
                        "parameters": parameters,
                        "training_net_return": metrics["net_total_return"],
                        "training_sharpe": metrics["net_annualized_sharpe"],
                        "training_max_drawdown": metrics["net_max_drawdown"],
                        "training_turnover": metrics["one_way_turnover"],
                        "training_profitable_years": profitable_years,
                    }
                )

        if candidates:
            selected = sorted(
                candidates,
                key=lambda row: (
                    row["training_sharpe"],
                    row["training_net_return"],
                    -row["training_turnover"],
                ),
                reverse=True,
            )[0]
            parameters = selected["parameters"]
            yearly_intervals = interval_cache[parameters]
            yearly_intervals = yearly_intervals[
                yearly_intervals["validation_year"] == evaluation_year
            ].copy()
            selection_status = "rule_selected"
        else:
            parameters = None
            template = next(iter(interval_cache.values()))
            yearly_intervals = template[
                template["validation_year"] == evaluation_year
            ].copy()
            yearly_intervals = _cash_intervals(yearly_intervals)
            selected = {
                "training_net_return": 0.0,
                "training_sharpe": 0.0,
                "training_max_drawdown": 0.0,
                "training_turnover": 0,
                "training_profitable_years": 0,
            }
            selection_status = "cash_no_rule_passed"

        yearly_intervals["nested_evaluation_year"] = evaluation_year
        evaluation_intervals.append(yearly_intervals)
        selection_rows.append(
            {
                "evaluation_year": evaluation_year,
                "training_years": "+".join(map(str, training_years)),
                "eligible_rules": len(candidates),
                "selection_status": selection_status,
                "rank_buffer": parameters[0] if parameters else None,
                "switch_edge": parameters[1] if parameters else None,
                "minimum_return": parameters[2] if parameters else None,
                **{
                    key: value
                    for key, value in selected.items()
                    if key != "parameters"
                },
            }
        )

    nested_intervals = pd.concat(
        evaluation_intervals, ignore_index=True,
    ).sort_values("entry_date").reset_index(drop=True)
    nested_metrics, _ = _simulate_ranking(
        nested_intervals,
        "nested_walk_forward_2022_2024",
        "nested_low_turnover_rank",
    )
    yearly_results = []
    for year in EVALUATION_YEARS:
        yearly = nested_intervals[
            nested_intervals["validation_year"] == year
        ].copy().reset_index(drop=True)
        yearly["new_segment"] = False
        yearly.loc[0, "new_segment"] = True
        metrics, _ = _simulate_ranking(
            yearly,
            f"nested_{year}",
            "nested_low_turnover_rank",
        )
        yearly_results.append(
            {
                "year": year,
                "net_total_return": metrics["net_total_return"],
                "net_annualized_sharpe": metrics[
                    "net_annualized_sharpe"
                ],
                "net_max_drawdown": metrics["net_max_drawdown"],
                "one_way_turnover": metrics["one_way_turnover"],
            }
        )

    profitable_years = sum(
        result["net_total_return"] > 0 for result in yearly_results
    )
    passes_gate = (
        nested_metrics["net_total_return"] > 0
        and nested_metrics["one_way_turnover"]
        <= BASELINE_ONE_WAY_TURNOVER_PER_YEAR * len(EVALUATION_YEARS) // 2
        and nested_metrics["net_max_drawdown"] >= MIN_ALLOWED_MAX_DRAWDOWN
        and profitable_years >= 2
    )
    decision = {
        "version": "low_turnover_rank_strategy_nested",
        "status": (
            "nested_validation_passed"
            if passes_gate else "nested_validation_failed"
        ),
        "selection_policy": "Each year uses only prior walk-forward years; cash when no rule passes",
        "test_period_used": False,
        "evaluation_years": list(EVALUATION_YEARS),
        "maximum_drawdown_degradation": MAX_DRAWDOWN_DEGRADATION,
        "metrics": nested_metrics,
        "profitable_years": profitable_years,
        "yearly": yearly_results,
    }
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTERVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(
        SELECTION_PATH, index=False, encoding="utf-8-sig",
    )
    nested_intervals.to_csv(
        INTERVAL_PATH, index=False, encoding="utf-8-sig",
    )
    METRICS_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    result = run_nested_validation()
    print(json.dumps(result, indent=2, ensure_ascii=False))
