import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from build_news_features import write_news_outputs
from collectors.news import fetch_stock_news


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumably backfill V5 news from newest to oldest."
    )
    parser.add_argument("--stock-id", default="2330")
    parser.add_argument("--target-start-date", default="2020-04-01")
    parser.add_argument(
        "--target-end-date",
        help="Defaults to the final date in the current model dataset.",
    )
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=1,
        help="Chunks per run; use 0 to continue until the target is complete.",
    )
    parser.add_argument("--market-close-time", default="13:30")
    parser.add_argument("--request-delay-seconds", type=float, default=0.2)
    return parser.parse_args()


def _cache_path(cache_root: Path, item: pd.Timestamp) -> Path:
    return cache_root / f"{item:%Y-%m-%d}.csv"


def find_contiguous_cache_start(
    cache_root: Path,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
) -> pd.Timestamp | None:
    """Return the oldest date in the gap-free cached suffix ending at target_end."""
    item = target_end
    if not _cache_path(cache_root, item).exists():
        return None
    while item > target_start and _cache_path(
        cache_root, item - pd.Timedelta(days=1)
    ).exists():
        item -= pd.Timedelta(days=1)
    return item


def next_backfill_chunk(
    cache_root: Path,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    chunk_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be greater than zero")
    contiguous_start = find_contiguous_cache_start(
        cache_root, target_start, target_end
    )
    chunk_end = (
        target_end
        if contiguous_start is None
        else contiguous_start - pd.Timedelta(days=1)
    )
    if chunk_end < target_start:
        return None
    chunk_start = max(
        target_start,
        chunk_end - pd.Timedelta(days=chunk_days - 1),
    )
    return chunk_start, chunk_end


def _write_progress(path: Path, payload: dict) -> None:
    payload = {
        **payload,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _dataset_end(stock_id: str) -> pd.Timestamp:
    dataset_path = Path("data/processed") / f"{stock_id}_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing model dataset: {dataset_path}")
    dataset = pd.read_csv(dataset_path, usecols=["date"])
    return pd.to_datetime(dataset["date"], errors="raise").max().normalize()


def main() -> None:
    args = parse_args()
    if args.max_chunks < 0:
        raise ValueError("max_chunks must be zero or greater")
    target_start = pd.Timestamp(args.target_start_date).normalize()
    target_end = (
        pd.Timestamp(args.target_end_date).normalize()
        if args.target_end_date
        else _dataset_end(args.stock_id)
    )
    if target_end < target_start:
        raise ValueError("target_end_date must not be earlier than target_start_date")

    cache_root = Path("data/raw/news") / str(args.stock_id)
    cache_root.mkdir(parents=True, exist_ok=True)
    progress_path = (
        Path("data/processed") / f"{args.stock_id}_news_backfill_progress.json"
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    completed_chunks: list[dict[str, str]] = []
    chunks_run = 0

    while args.max_chunks == 0 or chunks_run < args.max_chunks:
        chunk = next_backfill_chunk(
            cache_root,
            target_start,
            target_end,
            args.chunk_days,
        )
        if chunk is None:
            break
        chunk_start, chunk_end = chunk
        try:
            fetch_stock_news(
                stock_id=args.stock_id,
                start_date=chunk_start.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d"),
                token=os.environ.get("FINMIND_TOKEN", ""),
                request_delay_seconds=args.request_delay_seconds,
                max_request_days=args.chunk_days,
            )
        except Exception as error:
            contiguous_start = find_contiguous_cache_start(
                cache_root, target_start, target_end
            )
            _write_progress(
                progress_path,
                {
                    "status": "interrupted",
                    "stock_id": str(args.stock_id),
                    "target_start": target_start.strftime("%Y-%m-%d"),
                    "target_end": target_end.strftime("%Y-%m-%d"),
                    "contiguous_cache_start": (
                        contiguous_start.strftime("%Y-%m-%d")
                        if contiguous_start is not None
                        else None
                    ),
                    "failed_chunk_start": chunk_start.strftime("%Y-%m-%d"),
                    "failed_chunk_end": chunk_end.strftime("%Y-%m-%d"),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        completed_chunks.append(
            {
                "start": chunk_start.strftime("%Y-%m-%d"),
                "end": chunk_end.strftime("%Y-%m-%d"),
            }
        )
        chunks_run += 1
        contiguous_start = find_contiguous_cache_start(
            cache_root, target_start, target_end
        )
        _write_progress(
            progress_path,
            {
                "status": "complete" if contiguous_start == target_start else "running",
                "stock_id": str(args.stock_id),
                "target_start": target_start.strftime("%Y-%m-%d"),
                "target_end": target_end.strftime("%Y-%m-%d"),
                "contiguous_cache_start": contiguous_start.strftime("%Y-%m-%d"),
                "chunks_completed_this_run": completed_chunks,
            },
        )

    contiguous_start = find_contiguous_cache_start(
        cache_root, target_start, target_end
    )
    if contiguous_start is None:
        raise RuntimeError("No gap-free cached news range is available")
    consolidated = fetch_stock_news(
        stock_id=args.stock_id,
        start_date=contiguous_start.strftime("%Y-%m-%d"),
        end_date=target_end.strftime("%Y-%m-%d"),
        max_request_days=0,
    )
    quality = write_news_outputs(
        stock_id=args.stock_id,
        start_date=contiguous_start.strftime("%Y-%m-%d"),
        end_date=target_end.strftime("%Y-%m-%d"),
        news=consolidated,
        market_close_time=args.market_close_time,
        merge_dataset=True,
    )
    status = "complete" if contiguous_start == target_start else "paused"
    next_date = (
        None
        if status == "complete"
        else (contiguous_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )
    _write_progress(
        progress_path,
        {
            "status": status,
            "stock_id": str(args.stock_id),
            "target_start": target_start.strftime("%Y-%m-%d"),
            "target_end": target_end.strftime("%Y-%m-%d"),
            "contiguous_cache_start": contiguous_start.strftime("%Y-%m-%d"),
            "next_missing_date": next_date,
            "chunks_completed_this_run": completed_chunks,
            "quality": quality,
        },
    )
    print(f"Backfill status: {status}")
    print(f"Gap-free cache: {contiguous_start:%Y-%m-%d} to {target_end:%Y-%m-%d}")
    print(f"Progress saved to {progress_path}")


if __name__ == "__main__":
    main()
