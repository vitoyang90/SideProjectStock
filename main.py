from pathlib import Path

from collectors.price import fetch_stock_price


stock_id = "2330"

df = fetch_stock_price(
    stock_id=stock_id,
    start_date="2018-01-01",
    end_date="2026-08-24"
)

print(df.head())
print(df.tail())
print(df.shape)

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{stock_id}_price.csv"
df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved to {output_path}")
