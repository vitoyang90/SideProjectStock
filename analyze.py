"""Generate a latest-data analysis report from the registered champion model."""

import argparse
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from collectors.chip import (
    fetch_institutional_investors,
    fetch_margin_purchase_short_sale,
)
from collectors.price import fetch_stock_price
from features.chip import (
    build_institutional_features,
    build_margin_features,
    merge_chip_features,
)
from features.technical import add_technical_features
from train import _feature_columns


RAW_DIR = Path("data/raw")
LIVE_DIR = Path("data/live")
DEFAULT_OUTPUT_DIR = Path("reports/analysis")
FORWARD_DIR = Path("data/forward")
REGISTRY_PATH = Path("models/registry.json")
CLASS_NAMES_ZH = {0: "下跌", 1: "盤整", 2: "上漲"}
FORECAST_HORIZON = "未來五個交易日"
MIN_PREDICTION_CONFIDENCE = 0.40
MIN_TOP_PROBABILITY_MARGIN = 0.05
MAX_FORWARD_RECORD_STALENESS_DAYS = 3


def _valid_price_rows(price: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "max", "min", "close", "Trading_Volume"]
    price = price.copy()
    price[required] = price[required].apply(pd.to_numeric, errors="coerce")
    return price[
        price[["open", "max", "min", "close"]].gt(0).all(axis=1)
        & price["Trading_Volume"].gt(0)
    ].copy()


def build_v2_features(
    price: pd.DataFrame,
    institutional: pd.DataFrame,
    margin: pd.DataFrame,
) -> pd.DataFrame:
    """Build inference features without creating or requiring future labels."""
    features = merge_chip_features(
        _valid_price_rows(price),
        build_institutional_features(institutional),
        build_margin_features(margin),
    )
    return add_technical_features(features)


def _cached_paths(stock_id: str) -> dict[str, Path]:
    for directory in (
        LIVE_DIR,
        RAW_DIR,
        Path("data/external_validation"),
        Path("data/stock_models"),
    ):
        paths = {
            name: directory / f"{stock_id}_{name}.csv"
            for name in ("price", "institutional", "margin")
        }
        if all(path.exists() for path in paths.values()):
            return paths
    return {
        name: LIVE_DIR / f"{stock_id}_{name}.csv"
        for name in ("price", "institutional", "margin")
    }


def load_cached_inputs(stock_id: str) -> tuple[dict[str, pd.DataFrame], str]:
    paths = _cached_paths(stock_id)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cached inputs; run with --refresh first: " + ", ".join(missing)
        )
    return ({name: pd.read_csv(path) for name, path in paths.items()}, "cache")


def fetch_latest_inputs(
    stock_id: str,
    end_date: str,
    token: str = "",
    history_days: int = 240,
) -> tuple[dict[str, pd.DataFrame], str]:
    end = date.fromisoformat(end_date)
    start_date = (end - timedelta(days=history_days)).isoformat()
    frames = {
        "price": fetch_stock_price(stock_id, start_date, end_date, token),
        "institutional": fetch_institutional_investors(
            stock_id, start_date, end_date, token,
        ),
        "margin": fetch_margin_purchase_short_sale(
            stock_id, start_date, end_date, token,
        ),
    }
    empty = [name for name, frame in frames.items() if frame.empty]
    if empty:
        raise ValueError(f"FinMind returned empty datasets: {empty}")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(
            LIVE_DIR / f"{stock_id}_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return frames, "FinMind"


def _view_from_score(score: int) -> str:
    if score >= 2:
        return "偏多"
    if score <= -2:
        return "偏空"
    return "中性"


def _prediction_label(probabilities: np.ndarray) -> tuple[int, str, bool, float]:
    order = np.argsort(probabilities)
    predicted_class = int(order[-1])
    margin = float(probabilities[order[-1]] - probabilities[order[-2]])
    abstained = bool(
        probabilities[predicted_class] < MIN_PREDICTION_CONFIDENCE
        or margin < MIN_TOP_PROBABILITY_MARGIN
    )
    trend = "不確定" if abstained else CLASS_NAMES_ZH[predicted_class]
    return predicted_class, trend, abstained, margin


def summarize_factors(row: pd.Series) -> dict:
    technical_score = 0
    technical = []
    if row["ma20_ratio"] > 0:
        technical_score += 1
        technical.append(f"收盤價高於 20 日均價 {row['ma20_ratio']:.2%}")
    else:
        technical_score -= 1
        technical.append(f"收盤價低於 20 日均價 {abs(row['ma20_ratio']):.2%}")
    if row["macd_hist"] > 0:
        technical_score += 1
        technical.append("MACD 柱狀體為正")
    else:
        technical_score -= 1
        technical.append("MACD 柱狀體為負")
    if row["return_5d_past"] > 0:
        technical_score += 1
        technical.append(f"近五日報酬為 {row['return_5d_past']:.2%}")
    else:
        technical_score -= 1
        technical.append(f"近五日報酬為 {row['return_5d_past']:.2%}")

    chip_score = 0
    chip = []
    if row["foreign_5d"] > 0:
        chip_score += 1
        chip.append(f"外資近五日淨買超 {row['foreign_5d']:,.0f}")
    else:
        chip_score -= 1
        chip.append(f"外資近五日淨賣超 {abs(row['foreign_5d']):,.0f}")
    if row["trust_5d"] > 0:
        chip_score += 1
        chip.append(f"投信近五日淨買超 {row['trust_5d']:,.0f}")
    else:
        chip_score -= 1
        chip.append(f"投信近五日淨賣超 {abs(row['trust_5d']):,.0f}")
    if row["institutional_total_net"] > 0:
        chip_score += 1
        chip.append(f"三大法人當日合計買超 {row['institutional_total_net']:,.0f}")
    else:
        chip_score -= 1
        chip.append(f"三大法人當日合計賣超 {abs(row['institutional_total_net']):,.0f}")

    risks = []
    if row["rsi_14"] >= 70:
        risks.append(f"RSI {row['rsi_14']:.1f}，短線可能過熱")
    elif row["rsi_14"] <= 30:
        risks.append(f"RSI {row['rsi_14']:.1f}，走勢偏弱且可能劇烈反彈")
    if row["atr_ratio"] >= 0.03:
        risks.append(f"ATR 約為股價的 {row['atr_ratio']:.2%}，波動偏高")
    if row["margin_change"] > 0 and row["return_5d_past"] > 0:
        risks.append("融資增加且近期股價上漲，需留意追價槓桿")
    if not risks:
        risks.append("模型信心與歷史回測不代表未來績效")

    return {
        "technical_view": _view_from_score(technical_score),
        "chip_view": _view_from_score(chip_score),
        "technical_observations": technical,
        "chip_observations": chip,
        "risks": risks,
    }


def build_report(
    stock_id: str,
    feature_frame: pd.DataFrame,
    model_path: Path,
    source: str,
    requested_as_of: str,
    model_info: dict | None = None,
) -> dict:
    model_features = _feature_columns("v2")
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[model_features] = frame[model_features].replace(
        [np.inf, -np.inf], np.nan,
    )
    complete = frame.dropna(subset=model_features).sort_values("date")
    if complete.empty:
        raise ValueError("No row has all V2 inference features")
    latest = complete.iloc[-1]

    model = XGBClassifier()
    model.load_model(model_path)
    model_input = complete.tail(1)[model_features].astype(float)
    probabilities = model.predict_proba(model_input)[0]
    model_info = model_info or {
        "version": "v2",
        "artifact_id": "2330_v2_legacy",
        "status": "research_champion",
    }
    predicted_class, trend, abstained, probability_margin = _prediction_label(
        probabilities
    )
    abstention_reasons = []
    if probabilities[predicted_class] < MIN_PREDICTION_CONFIDENCE:
        abstention_reasons.append("最高類別機率低於 40%")
    if probability_margin < MIN_TOP_PROBABILITY_MARGIN:
        abstention_reasons.append("前兩名類別機率差低於 5%")
    if not model_info.get("direction_signal_eligible", True):
        abstained = True
        trend = "不確定"
        abstention_reasons.append("此股票模型未通過歷史方向訊號品質門檻")
    prediction_date = latest["date"].date()
    requested_date = date.fromisoformat(requested_as_of)
    stale_days = max((requested_date - prediction_date).days, 0)
    factors = summarize_factors(latest)
    return {
        "stock_id": stock_id,
        "prediction_date": prediction_date.isoformat(),
        "requested_as_of": requested_as_of,
        "data_source": source,
        "data_staleness_calendar_days": stale_days,
        "stale_data_warning": stale_days > 7,
        "model": {
            "version": model_info["version"],
            "artifact_id": model_info.get("artifact_id", model_info["version"]),
            "path": str(model_path),
            "target": FORECAST_HORIZON,
            "status": model_info["status"],
            "trained_on": model_info.get("trained_on"),
            "training_end": model_info.get("training_end"),
            "walk_forward_mean_balanced_accuracy": model_info.get(
                "walk_forward_mean_balanced_accuracy"
            ),
            "direction_signal_eligible": model_info.get(
                "direction_signal_eligible", True
            ),
        },
        "prediction": {
            "class_id": predicted_class,
            "raw_top_class": CLASS_NAMES_ZH[predicted_class],
            "trend": trend,
            "abstained": abstained,
            "abstention_reasons": abstention_reasons,
            "confidence": float(probabilities[predicted_class]),
            "top_probability_margin": probability_margin,
            "abstention_policy": {
                "minimum_confidence": MIN_PREDICTION_CONFIDENCE,
                "minimum_top_probability_margin": MIN_TOP_PROBABILITY_MARGIN,
            },
            "probabilities": {
                "down": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "up": float(probabilities[2]),
            },
        },
        "market_snapshot": {
            "close": float(latest["close"]),
            "rsi_14": float(latest["rsi_14"]),
            "return_5d_past": float(latest["return_5d_past"]),
            "volume_ratio_20": float(latest["volume_ratio_20"]),
        },
        "factor_summary": factors,
        "limitations": [
            "因素摘要是規則式觀察，不是模型的因果解釋或 SHAP 歸因。",
            "此模型尚未通過建置日之後的全新 forward 驗證。",
            "內容僅供研究，不構成投資建議。",
        ],
    }


def render_markdown(report: dict) -> str:
    prediction = report["prediction"]
    probabilities = prediction["probabilities"]
    factors = report["factor_summary"]
    lines = [
        f"# {report['stock_id']} 股票分析",
        "",
        f"- 資料日期：{report['prediction_date']}",
        f"- 預測區間：{report['model']['target']}",
        f"- 趨勢判斷：{prediction['trend']}",
        f"- 模型信心：{prediction['confidence']:.1%}",
        (
            "- 類別機率："
            f"上漲 {probabilities['up']:.1%}／"
            f"盤整 {probabilities['neutral']:.1%}／"
            f"下跌 {probabilities['down']:.1%}"
        ),
        f"- 技術面：{factors['technical_view']}",
        f"- 籌碼面：{factors['chip_view']}",
    ]
    if prediction.get("abstained"):
        lines.append(
            "- 判斷說明：" + "；".join(prediction["abstention_reasons"])
        )
    if report["stale_data_warning"]:
        lines.extend([
            "",
            (
                f"> 資料已落後請求日期 {report['data_staleness_calendar_days']} 天；"
                "請先使用 `--refresh` 更新後再解讀。"
            ),
        ])
    lines.extend(["", "## 目前觀察", ""])
    for observation in (
        factors["technical_observations"] + factors["chip_observations"]
    ):
        lines.append(f"- {observation}")
    lines.extend(["", "## 風險與限制", ""])
    for risk in factors["risks"] + report["limitations"]:
        lines.append(f"- {risk}")
    return "\n".join(lines) + "\n"


def record_forward_prediction(report: dict, path: Path | None = None) -> Path:
    """Append one immutable daily prediction for later forward evaluation."""
    if (
        report["data_staleness_calendar_days"]
        > MAX_FORWARD_RECORD_STALENESS_DAYS
    ):
        raise ValueError(
            "Refusing to record a stale forward prediction: "
            f"{report['data_staleness_calendar_days']} calendar days old"
        )
    path = path or FORWARD_DIR / f"{report['stock_id']}_predictions.csv"
    prediction = report["prediction"]
    probabilities = prediction["probabilities"]
    row = {
        "prediction_date": report["prediction_date"],
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stock_id": report["stock_id"],
        "model_version": report["model"]["artifact_id"],
        "close": report["market_snapshot"]["close"],
        "raw_top_class": prediction["raw_top_class"],
        "trend": prediction["trend"],
        "abstained": prediction["abstained"],
        "confidence": prediction["confidence"],
        "probability_down": probabilities["down"],
        "probability_neutral": probabilities["neutral"],
        "probability_up": probabilities["up"],
    }
    if path.exists():
        existing = pd.read_csv(path)
        duplicate = (
            existing["prediction_date"].astype(str).eq(row["prediction_date"])
            & existing["model_version"].eq(row["model_version"])
        )
        if duplicate.any():
            return path
        frame = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def analyze(
    stock_id: str,
    refresh: bool = False,
    as_of: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict, Path, Path]:
    as_of = as_of or date.today().isoformat()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    stock_models = registry.get("stock_models", {})
    if stock_id in stock_models:
        model_info = stock_models[stock_id]
    elif stock_id == "2330":
        champion = registry["champion"]
        model_info = {
            **champion,
            "artifact_id": "2330_v2_legacy",
        }
    else:
        available = sorted(stock_models) or ["2330"]
        raise ValueError(
            f"No stock-specific model for {stock_id}; currently supported: {available}"
        )
    if model_info["version"] != "v2":
        raise ValueError(f"Unsupported stock model version: {model_info['version']}")
    model_path = Path(model_info["model_path"])
    if refresh:
        frames, source = fetch_latest_inputs(
            stock_id,
            as_of,
            os.environ.get("FINMIND_TOKEN", ""),
        )
    else:
        frames, source = load_cached_inputs(stock_id)
    features = build_v2_features(
        frames["price"], frames["institutional"], frames["margin"],
    )
    report = build_report(
        stock_id, features, model_path, source, as_of, model_info,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stock_id}_latest.json"
    markdown_path = output_dir / f"{stock_id}_latest.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report, json_path, markdown_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a latest analysis from the registered champion.",
    )
    parser.add_argument("--stock-id", default="2330")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch the latest price and chip data from FinMind before analysis.",
    )
    parser.add_argument(
        "--as-of",
        help="Requested data end date in YYYY-MM-DD; defaults to today.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record one immutable prediction for later forward evaluation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result, json_output, markdown_output = analyze(
        args.stock_id,
        args.refresh,
        args.as_of,
        args.output_dir,
    )
    print(render_markdown(result))
    print(f"JSON: {json_output}")
    print(f"Markdown: {markdown_output}")
    if args.record:
        print(f"Forward record: {record_forward_prediction(result)}")
