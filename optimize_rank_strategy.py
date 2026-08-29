"""Select a low-turnover rank strategy using walk-forward predictions only."""

import json
from itertools import product
from pathlib import Path

import pandas as pd

from backtest import HOLDING_DAYS
from train_pooled_v2 import STOCK_IDS, _load_data
from train_return_ranker import _simulate_ranking


PREDICTION_PATH = Path("data/processed/return_ranker_predictions.csv")
GRID_PATH = Path("reports/rank_strategy_grid.csv")
YEARLY_PATH = Path("reports/rank_strategy_yearly.csv")
DECISION_PATH = Path("reports/rank_strategy_decision.json")
SELECTED_INTERVAL_PATH = Path(
    "data/processed/rank_strategy_selected_intervals.csv"
)

RANK_BUFFERS = (1, 2, 3)
SWITCH_EDGES = (0.0, 0.0025, 0.005, 0.01)
MINIMUM_RETURNS = (0.0, 0.0025, 0.005)
BASELINE_ONE_WAY_TURNOVER = 208
MAX_ALLOWED_TURNOVER = BASELINE_ONE_WAY_TURNOVER // 2
BASELINE_MAX_DRAWDOWN = -0.4992224518386196
MAX_DRAWDOWN_DEGRADATION = 0.05
MIN_ALLOWED_MAX_DRAWDOWN = (
    BASELINE_MAX_DRAWDOWN - MAX_DRAWDOWN_DEGRADATION
)
MIN_PROFITABLE_YEARS = 3


def _price_tables(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result = {}
    for stock_id, frame in prices.items():
        table = frame.copy()
        table["date"] = pd.to_datetime(table["date"], errors="raise")
        result[stock_id] = table.set_index("date").sort_index()
    return result


def _select_stock(
    cross_section: pd.DataFrame,
    current_stock: str,
    rank_buffer: int,
    switch_edge: float,
    minimum_return: float,
) -> tuple[str, dict]:
    ordered = cross_section.sort_values(
        "predicted_return_5d", ascending=False,
    ).reset_index(drop=True)
    top = ordered.iloc[0]
    top_stock = str(top["stock_id"])
    top_return = float(top["predicted_return_5d"])
    diagnostics = {
        "top_stock": top_stock,
        "top_predicted_return": top_return,
        "current_rank": None,
        "current_predicted_return": None,
        "prediction_edge": None,
    }
    if current_stock == "cash":
        return (top_stock if top_return >= minimum_return else "cash"), diagnostics

    current_rows = ordered[ordered["stock_id"].astype(str) == current_stock]
    if current_rows.empty:
        return (top_stock if top_return >= minimum_return else "cash"), diagnostics
    current_index = int(current_rows.index[0])
    current_rank = current_index + 1
    current_return = float(current_rows.iloc[0]["predicted_return_5d"])
    edge = top_return - current_return
    diagnostics.update(
        {
            "current_rank": current_rank,
            "current_predicted_return": current_return,
            "prediction_edge": edge,
        }
    )

    if current_return >= minimum_return:
        if current_rank <= rank_buffer or edge < switch_edge:
            return current_stock, diagnostics
        return top_stock, diagnostics
    if top_return >= minimum_return and edge >= switch_edge:
        return top_stock, diagnostics
    return "cash", diagnostics


def _build_intervals(
    predictions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    rank_buffer: int,
    switch_edge: float,
    minimum_return: float,
) -> pd.DataFrame:
    tables = _price_tables(prices)
    calendar = tables["2330"].index.tolist()
    calendar_index = {date: index for index, date in enumerate(calendar)}
    rows = []

    for segment_number, (year, segment) in enumerate(
        predictions.groupby("validation_year", sort=True)
    ):
        current_stock = "cash"
        common_dates = (
            segment.groupby("date")["stock_id"].nunique()
            .loc[lambda counts: counts == len(STOCK_IDS)]
            .index.sort_values()
        )
        for signal_number, signal_date in enumerate(common_dates[::HOLDING_DAYS]):
            calendar_position = calendar_index.get(signal_date)
            if (
                calendar_position is None
                or calendar_position + HOLDING_DAYS + 1 >= len(calendar)
            ):
                continue
            entry_date = calendar[calendar_position + 1]
            exit_date = calendar[calendar_position + HOLDING_DAYS + 1]
            cross_section = segment[segment["date"] == signal_date]
            selected_stock, diagnostics = _select_stock(
                cross_section,
                current_stock,
                rank_buffer,
                switch_edge,
                minimum_return,
            )
            if selected_stock == "cash":
                asset_return = 0.0
            else:
                table = tables[selected_stock]
                asset_return = float(
                    table.loc[exit_date, "open"]
                    / table.loc[entry_date, "open"]
                    - 1
                )
            rows.append(
                {
                    "validation_year": int(year),
                    "segment_number": segment_number,
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "selected_stock": selected_stock,
                    "asset_return": asset_return,
                    "new_segment": signal_number == 0,
                    **diagnostics,
                }
            )
            current_stock = selected_stock
    intervals = pd.DataFrame(rows).sort_values("entry_date").reset_index(drop=True)
    if intervals.empty:
        raise ValueError("No complete low-turnover intervals")
    return intervals


def _evaluate_parameters(
    predictions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    rank_buffer: int,
    switch_edge: float,
    minimum_return: float,
) -> tuple[dict, list[dict], pd.DataFrame]:
    intervals = _build_intervals(
        predictions,
        prices,
        rank_buffer,
        switch_edge,
        minimum_return,
    )
    metrics, _ = _simulate_ranking(
        intervals,
        "walk_forward",
        "low_turnover_rank",
    )
    yearly_rows = []
    for year, yearly_intervals in intervals.groupby("validation_year"):
        yearly_intervals = yearly_intervals.copy().reset_index(drop=True)
        yearly_intervals["new_segment"] = False
        yearly_intervals.loc[0, "new_segment"] = True
        yearly_metrics, _ = _simulate_ranking(
            yearly_intervals,
            f"walk_forward_{year}",
            "low_turnover_rank",
        )
        yearly_rows.append(
            {
                "rank_buffer": rank_buffer,
                "switch_edge": switch_edge,
                "minimum_return": minimum_return,
                "year": int(year),
                "net_total_return": yearly_metrics["net_total_return"],
                "net_annualized_sharpe": yearly_metrics[
                    "net_annualized_sharpe"
                ],
                "one_way_turnover": yearly_metrics["one_way_turnover"],
            }
        )
    profitable_years = sum(row["net_total_return"] > 0 for row in yearly_rows)
    result = {
        "rank_buffer": rank_buffer,
        "switch_edge": switch_edge,
        "minimum_return": minimum_return,
        "gross_total_return": metrics["gross_total_return"],
        "net_total_return": metrics["net_total_return"],
        "net_cagr": metrics["net_cagr"],
        "net_annualized_sharpe": metrics["net_annualized_sharpe"],
        "net_max_drawdown": metrics["net_max_drawdown"],
        "buys": metrics["buys"],
        "sells": metrics["sells"],
        "one_way_turnover": metrics["one_way_turnover"],
        "profitable_years": profitable_years,
        "turnover_reduction": (
            1 - metrics["one_way_turnover"] / BASELINE_ONE_WAY_TURNOVER
        ),
    }
    result["passes_gate"] = (
        result["one_way_turnover"] <= MAX_ALLOWED_TURNOVER
        and result["net_total_return"] > 0
        and result["profitable_years"] >= MIN_PROFITABLE_YEARS
        and result["net_max_drawdown"] >= MIN_ALLOWED_MAX_DRAWDOWN
    )
    return result, yearly_rows, intervals


def run_optimization() -> dict:
    predictions = pd.read_csv(PREDICTION_PATH)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise")
    predictions = predictions[
        predictions["evaluation"] == "walk_forward"
    ].copy()
    if (
        predictions.empty
        or predictions["date"].max() >= pd.Timestamp("2025-01-01")
    ):
        raise RuntimeError("Optimization data must end before 2025")
    _, prices = _load_data()
    grid_rows = []
    yearly_rows = []
    interval_cache = {}

    for rank_buffer, switch_edge, minimum_return in product(
        RANK_BUFFERS, SWITCH_EDGES, MINIMUM_RETURNS,
    ):
        result, years, intervals = _evaluate_parameters(
            predictions,
            prices,
            rank_buffer,
            switch_edge,
            minimum_return,
        )
        key = (rank_buffer, switch_edge, minimum_return)
        grid_rows.append(result)
        yearly_rows.extend(years)
        interval_cache[key] = intervals

    grid = pd.DataFrame(grid_rows)
    yearly = pd.DataFrame(yearly_rows)
    eligible = grid[grid["passes_gate"]].copy()
    if eligible.empty:
        selected = grid.sort_values(
            ["net_annualized_sharpe", "one_way_turnover"],
            ascending=[False, True],
        ).iloc[0]
        status = "rejected_no_rule_passed_gate"
    else:
        selected = eligible.sort_values(
            ["net_annualized_sharpe", "net_total_return", "one_way_turnover"],
            ascending=[False, False, True],
        ).iloc[0]
        status = "in_sample_candidate_pending_nested_validation"

    selected_key = (
        int(selected["rank_buffer"]),
        float(selected["switch_edge"]),
        float(selected["minimum_return"]),
    )
    selected_intervals = interval_cache[selected_key]
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_INTERVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.sort_values(
        ["passes_gate", "net_annualized_sharpe"], ascending=False,
    ).to_csv(GRID_PATH, index=False, encoding="utf-8-sig")
    yearly.to_csv(YEARLY_PATH, index=False, encoding="utf-8-sig")
    selected_intervals.to_csv(
        SELECTED_INTERVAL_PATH, index=False, encoding="utf-8-sig",
    )

    decision = {
        "version": "low_turnover_rank_strategy",
        "status": status,
        "selection_period": "2021-01-01 through 2024-12-31 walk-forward only",
        "test_period_used_for_selection": False,
        "grid_size": len(grid),
        "gate": {
            "maximum_one_way_turnover": MAX_ALLOWED_TURNOVER,
            "minimum_profitable_years": MIN_PROFITABLE_YEARS,
            "net_total_return_must_be_positive": True,
            "minimum_allowed_max_drawdown": MIN_ALLOWED_MAX_DRAWDOWN,
            "maximum_drawdown_degradation": MAX_DRAWDOWN_DEGRADATION,
        },
        "passing_rules": len(eligible),
        "selected": {
            key: (
                bool(value) if key == "passes_gate"
                else int(value) if key in {
                    "rank_buffer", "buys", "sells", "one_way_turnover",
                    "profitable_years",
                }
                else float(value)
            )
            for key, value in selected.to_dict().items()
        },
    }
    DECISION_PATH.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision


if __name__ == "__main__":
    result = run_optimization()
    print(json.dumps(result, indent=2, ensure_ascii=False))
