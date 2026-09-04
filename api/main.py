"""FastAPI wrapper around the stock-specific analysis pipeline."""

import json
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from analyze import REGISTRY_PATH, analyze, record_forward_prediction


app = FastAPI(
    title="台股多因子研究 API",
    description="Stock-specific V2 research predictions with explicit abstention.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _stock_models() -> dict[str, dict]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return registry.get("stock_models", {})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "stock-research-api",
        "supported_stocks": sorted(_stock_models()),
    }


@app.get("/api/stocks")
def stocks() -> dict:
    items = []
    for stock_id, model in sorted(_stock_models().items()):
        items.append({
            "stock_id": stock_id,
            "model_version": model["version"],
            "status": model["status"],
            "trained_on": model["trained_on"],
            "training_end": model["training_end"],
            "walk_forward_mean_balanced_accuracy": model[
                "walk_forward_mean_balanced_accuracy"
            ],
            "walk_forward_mean_macro_f1": model[
                "walk_forward_mean_macro_f1"
            ],
            "direction_signal_eligible": model["direction_signal_eligible"],
        })
    return {"stocks": items}


@app.post("/api/analyze/{stock_id}")
def analyze_stock(
    stock_id: str,
    refresh: bool = Query(False),
    record: bool = Query(False),
    as_of: date | None = Query(None),
) -> dict:
    try:
        report, _, _ = analyze(
            stock_id=stock_id,
            refresh=refresh,
            as_of=(as_of or date.today()).isoformat(),
        )
        if record:
            record_forward_prediction(report)
        return report
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
