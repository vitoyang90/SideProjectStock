"""Run and record binary V2.2 candidates without replacing the V2 champion."""

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from analyze import (
    DEFAULT_OUTPUT_DIR,
    FORWARD_DIR,
    REGISTRY_PATH,
    build_v2_features,
    load_cached_inputs,
)
from train import _feature_columns


def build_shadow_report(
    stock_id: str,
    feature_frame: pd.DataFrame,
    candidate: dict,
    requested_as_of: str,
) -> dict:
    features = _feature_columns("v2")
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan)
    complete = frame.dropna(subset=features).sort_values("date")
    if complete.empty:
        raise ValueError("No row has all V2 inference features")
    latest = complete.iloc[-1]
    model = XGBClassifier()
    model.load_model(candidate["model_path"])
    probability_up = float(model.predict_proba(complete.tail(1)[features])[:, 1][0])
    threshold = float(candidate["probability_threshold"])
    signal = probability_up >= threshold
    prediction_date = latest["date"].date()
    return {
        "stock_id": stock_id,
        "prediction_date": prediction_date.isoformat(),
        "requested_as_of": requested_as_of,
        "data_staleness_calendar_days": max(
            (date.fromisoformat(requested_as_of) - prediction_date).days, 0
        ),
        "model": candidate,
        "prediction": {
            "probability_up": probability_up,
            "probability_not_up": 1 - probability_up,
            "threshold": threshold,
            "signal": "上漲" if signal else "不出訊號",
            "emitted": signal,
        },
        "market_snapshot": {"close": float(latest["close"])},
        "limitations": [
            "這是 shadow candidate，不會取代目前 V2 趨勢判斷。",
            "只有建置後成熟的 forward 結果可決定是否升級。",
            "內容僅供研究，不構成投資建議。",
        ],
    }


def record_shadow_prediction(report: dict, path: Path | None = None) -> Path:
    path = path or FORWARD_DIR / f"{report['stock_id']}_v22_binary_predictions.csv"
    row = {
        "prediction_date": report["prediction_date"],
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stock_id": report["stock_id"],
        "model_version": report["model"]["artifact_id"],
        "close": report["market_snapshot"]["close"],
        "probability_up": report["prediction"]["probability_up"],
        "probability_threshold": report["prediction"]["threshold"],
        "signal_emitted": report["prediction"]["emitted"],
    }
    if path.exists():
        existing = pd.read_csv(path, dtype={"stock_id": str})
        duplicate = (
            existing["prediction_date"].astype(str).eq(row["prediction_date"])
            & existing["model_version"].astype(str).eq(row["model_version"])
        )
        if duplicate.any():
            return path
        output = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        output = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def render_markdown(report: dict) -> str:
    prediction = report["prediction"]
    return "\n".join(
        [
            f"# {report['stock_id']} V2.2 Binary Shadow",
            "",
            f"- 資料日期：{report['prediction_date']}",
            f"- 上漲機率：{prediction['probability_up']:.1%}",
            f"- 出場門檻：{prediction['threshold']:.1%}",
            f"- Shadow 訊號：{prediction['signal']}",
            "",
            "> 此結果只做 forward 比較，不取代 V2，也不是投資建議。",
            "",
        ]
    )


def analyze_shadow(
    stock_id: str = "2330",
    as_of: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict, Path, Path]:
    as_of = as_of or date.today().isoformat()
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    candidate = registry.get("binary_candidates", {}).get(stock_id)
    if not candidate:
        raise ValueError(f"No V2.2 binary shadow candidate for {stock_id}")
    frames, _ = load_cached_inputs(stock_id)
    features = build_v2_features(
        frames["price"], frames["institutional"], frames["margin"]
    )
    report = build_shadow_report(stock_id, features, candidate, as_of)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stock_id}_v22_binary_shadow.json"
    markdown_path = output_dir / f"{stock_id}_v22_binary_shadow.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report, json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a V2.2 binary shadow candidate.")
    parser.add_argument("--stock-id", default="2330")
    parser.add_argument("--as-of")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    report, json_path, markdown_path = analyze_shadow(args.stock_id, args.as_of)
    print(render_markdown(report))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    if args.record:
        print(f"Forward record: {record_shadow_prediction(report)}")


if __name__ == "__main__":
    main()
