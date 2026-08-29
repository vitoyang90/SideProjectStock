"""Leakage-aware five-day long-or-cash backtest for cumulative versions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRICE_PATH = Path("data/raw/2330_price.csv")
WALK_FORWARD_PREDICTION_PATH = Path(
    "data/processed/xgboost_walk_forward_predictions.csv"
)
TEST_PREDICTION_PATHS = {
    "v1": Path("data/processed/2330_predictions_v1.csv"),
    "v2": Path("data/processed/2330_predictions_v2.csv"),
    "v3": Path("data/processed/2330_predictions_v3.csv"),
}
VERSIONS = tuple(TEST_PREDICTION_PATHS)
METRICS_CSV_PATH = Path("reports/backtest_version_metrics.csv")
METRICS_JSON_PATH = Path("reports/backtest_version_metrics.json")
EQUITY_PATH = Path("data/processed/backtest_version_equity.csv")
TRADES_PATH = Path("reports/backtest_version_trades.csv")
YEARLY_PATH = Path("reports/backtest_version_yearly_returns.csv")

HOLDING_DAYS = 5
TRADING_DAYS_PER_YEAR = 252
DEFAULT_COMMISSION_BPS = 14.25
DEFAULT_SELL_TAX_BPS = 30.0


def _load_predictions() -> dict[str, pd.DataFrame]:
    walk_forward = pd.read_csv(WALK_FORWARD_PREDICTION_PATH)
    walk_forward["date"] = pd.to_datetime(walk_forward["date"])
    frames = {"walk_forward": walk_forward}

    test_frames = []
    for version, path in TEST_PREDICTION_PATHS.items():
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["version"] = version
        test_frames.append(frame)
    frames["test"] = pd.concat(test_frames, ignore_index=True)
    return frames


def _build_intervals(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    always_long: bool = False,
) -> pd.DataFrame:
    predictions = (
        predictions.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )
    # One signal every five trading days prevents overlapping holdings. Walk-
    # forward years are sampled separately because their purged tails are gaps.
    if "validation_year" in predictions.columns:
        signals = pd.concat(
            [
                group.iloc[::HOLDING_DAYS]
                for _, group in predictions.groupby(
                    "validation_year", sort=True
                )
            ],
            ignore_index=True,
        )
    else:
        signals = predictions.iloc[::HOLDING_DAYS].copy()
    price_index = {date: index for index, date in enumerate(prices["date"])}
    rows = []

    for signal in signals.itertuples(index=False):
        signal_index = price_index.get(signal.date)
        if signal_index is None or signal_index + HOLDING_DAYS + 1 >= len(prices):
            continue
        entry = prices.iloc[signal_index + 1]
        exit_row = prices.iloc[signal_index + HOLDING_DAYS + 1]
        if not signal.date < entry["date"] <= exit_row["date"]:
            raise RuntimeError("Invalid signal, entry, or exit date ordering")
        position = 1 if always_long else int(signal.predicted_class == 2)
        rows.append(
            {
                "signal_date": signal.date,
                "entry_date": entry["date"],
                "exit_date": exit_row["date"],
                "entry_price": float(entry["open"]),
                "exit_price": float(exit_row["open"]),
                "position": position,
                "predicted_class": (
                    2 if always_long else int(signal.predicted_class)
                ),
                "asset_return": float(
                    exit_row["open"] / entry["open"] - 1
                ),
            }
        )

    intervals = pd.DataFrame(rows)
    if intervals.empty:
        raise ValueError("No complete backtest intervals were available")
    intervals["new_segment"] = (
        intervals["entry_date"].ne(intervals["exit_date"].shift(1))
    )
    intervals.loc[intervals.index[0], "new_segment"] = True
    return intervals


def _max_drawdown(wealth: pd.Series) -> float:
    with_start = pd.concat([pd.Series([1.0]), wealth], ignore_index=True)
    return float((with_start / with_start.cummax() - 1).min())


def _cagr(total_return: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    years = (end - start).days / 365.25
    if years <= 0 or total_return <= -1:
        return float("nan")
    return float((1 + total_return) ** (1 / years) - 1)


def _sharpe(period_returns: pd.Series) -> float:
    standard_deviation = period_returns.std(ddof=1)
    if len(period_returns) < 2 or standard_deviation == 0:
        return 0.0
    periods_per_year = TRADING_DAYS_PER_YEAR / HOLDING_DAYS
    return float(
        period_returns.mean() / standard_deviation * np.sqrt(periods_per_year)
    )


def _simulate(
    intervals: pd.DataFrame,
    split: str,
    strategy: str,
    commission_bps: float,
    sell_tax_bps: float,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    commission = commission_bps / 10_000
    sell_cost = (commission_bps + sell_tax_bps) / 10_000
    gross_wealth = 1.0
    net_wealth = 1.0
    previous_position = 0
    buys = 0
    sells = 0
    cost_paid = 0.0
    curve_rows = []
    trade_rows = []

    for interval in intervals.itertuples(index=False):
        if interval.new_segment and curve_rows and previous_position == 1:
            charge = net_wealth * sell_cost
            net_wealth -= charge
            cost_paid += charge
            sells += 1
            prior_net = (
                1.0 if len(curve_rows) == 1 else curve_rows[-2]["net_equity"]
            )
            curve_rows[-1]["net_equity"] = net_wealth
            curve_rows[-1]["period_net_return"] = net_wealth / prior_net - 1
            previous_position = 0
        starting_gross = gross_wealth
        starting_net = net_wealth
        if previous_position == 0 and interval.position == 1:
            charge = net_wealth * commission
            net_wealth -= charge
            cost_paid += charge
            buys += 1
        elif previous_position == 1 and interval.position == 0:
            charge = net_wealth * sell_cost
            net_wealth -= charge
            cost_paid += charge
            sells += 1

        gross_wealth *= 1 + interval.position * interval.asset_return
        net_wealth *= 1 + interval.position * interval.asset_return
        curve_rows.append(
            {
                "split": split,
                "strategy": strategy,
                "date": interval.exit_date,
                "gross_equity": gross_wealth,
                "net_equity": net_wealth,
                "position": interval.position,
                "period_gross_return": gross_wealth / starting_gross - 1,
                "period_net_return": net_wealth / starting_net - 1,
            }
        )
        trade_rows.append(
            {
                "split": split,
                "strategy": strategy,
                **interval._asdict(),
            }
        )
        previous_position = interval.position

    if previous_position == 1:
        charge = net_wealth * sell_cost
        net_wealth -= charge
        cost_paid += charge
        sells += 1
        curve_rows[-1]["net_equity"] = net_wealth
        prior_net = 1.0 if len(curve_rows) == 1 else curve_rows[-2]["net_equity"]
        curve_rows[-1]["period_net_return"] = net_wealth / prior_net - 1

    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    start = intervals["entry_date"].iloc[0]
    end = intervals["exit_date"].iloc[-1]
    gross_total_return = gross_wealth - 1
    net_total_return = net_wealth - 1
    invested = intervals[intervals["position"] == 1]
    metrics = {
        "split": split,
        "strategy": strategy,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "intervals": len(intervals),
        "holding_days": HOLDING_DAYS,
        "commission_bps_per_side": commission_bps,
        "sell_tax_bps": sell_tax_bps,
        "gross_total_return": gross_total_return,
        "net_total_return": net_total_return,
        "gross_cagr": _cagr(gross_total_return, start, end),
        "net_cagr": _cagr(net_total_return, start, end),
        "net_annualized_sharpe": _sharpe(curve["period_net_return"]),
        "net_max_drawdown": _max_drawdown(curve["net_equity"]),
        "exposure": float(intervals["position"].mean()),
        "invested_interval_win_rate": (
            float((invested["asset_return"] > 0).mean())
            if not invested.empty else 0.0
        ),
        "buys": buys,
        "sells": sells,
        "one_way_turnover": buys + sells,
        "cost_paid_in_initial_capital_units": cost_paid,
    }
    return metrics, curve, trades


def run_backtest(
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    sell_tax_bps: float = DEFAULT_SELL_TAX_BPS,
) -> pd.DataFrame:
    if commission_bps < 0 or sell_tax_bps < 0:
        raise ValueError("Transaction costs cannot be negative")
    prices = pd.read_csv(PRICE_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)
    predictions_by_split = _load_predictions()
    metric_rows = []
    curves = []
    trades = []

    for split, predictions in predictions_by_split.items():
        for version in VERSIONS:
            version_predictions = predictions[
                predictions["version"] == version
            ].copy()
            intervals = _build_intervals(version_predictions, prices)
            metrics, curve, version_trades = _simulate(
                intervals,
                split,
                version,
                commission_bps,
                sell_tax_bps,
            )
            metric_rows.append(metrics)
            curves.append(curve)
            trades.append(version_trades)

        benchmark_intervals = _build_intervals(
            predictions[predictions["version"] == "v1"].copy(),
            prices,
            always_long=True,
        )
        metrics, curve, benchmark_trades = _simulate(
            benchmark_intervals,
            split,
            "buy_hold",
            commission_bps,
            sell_tax_bps,
        )
        metric_rows.append(metrics)
        curves.append(curve)
        trades.append(benchmark_trades)

    metrics_frame = pd.DataFrame(metric_rows)
    equity_frame = pd.concat(curves, ignore_index=True)
    trades_frame = pd.concat(trades, ignore_index=True)
    equity_frame["date"] = pd.to_datetime(equity_frame["date"])
    equity_frame["year"] = equity_frame["date"].dt.year
    yearly_frame = (
        equity_frame.groupby(["split", "strategy", "year"], as_index=False)
        .agg(
            intervals=("date", "count"),
            gross_return=(
                "period_gross_return", lambda s: float((1 + s).prod() - 1)
            ),
            net_return=(
                "period_net_return", lambda s: float((1 + s).prod() - 1)
            ),
        )
    )
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    EQUITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(METRICS_CSV_PATH, index=False, encoding="utf-8-sig")
    METRICS_JSON_PATH.write_text(
        json.dumps(
            metrics_frame.to_dict(orient="records"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    equity_frame.to_csv(EQUITY_PATH, index=False, encoding="utf-8-sig")
    trades_frame.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")
    yearly_frame.to_csv(YEARLY_PATH, index=False, encoding="utf-8-sig")
    return metrics_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest non-overlapping long-or-cash version signals."
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=DEFAULT_COMMISSION_BPS,
        help="Broker commission per buy or sell side.",
    )
    parser.add_argument(
        "--sell-tax-bps",
        type=float,
        default=DEFAULT_SELL_TAX_BPS,
        help="Additional stock transaction tax applied on sells.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_backtest(args.commission_bps, args.sell_tax_bps)
    columns = [
        "split", "strategy", "net_total_return", "net_cagr",
        "net_annualized_sharpe", "net_max_drawdown", "exposure",
        "one_way_turnover",
    ]
    print(results[columns].to_string(index=False))
