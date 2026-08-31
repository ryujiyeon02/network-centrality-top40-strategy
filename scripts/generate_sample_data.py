"""Generate synthetic examples for the private market-data schemas.

These values document file layout only and are not used in published results.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "sample_data"
CODES = ["A000001", "A000002", "A000003"]
NAMES = ["SAMPLE_A", "SAMPLE_B", "SAMPLE_C"]


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / name).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dates = [date(2025, 1, 2) + timedelta(days=i) for i in range(5)]
    prices = []
    total_returns = []
    market_caps = []
    trading_values = []
    trading_values_60 = []
    factors = []

    for i, current in enumerate(dates):
        prices.append({"Date": current.isoformat(), **{code: 100 + 10 * j + i for j, code in enumerate(CODES)}})
        total_returns.append(
            {"코드명": current.isoformat(), **{name: 100 + 10 * j + i * 1.1 for j, name in enumerate(NAMES)}}
        )
        market_caps.append(
            {
                "코드명": current.isoformat(),
                **{name: 1_000_000_000 + 200_000_000 * j + 10_000_000 * i for j, name in enumerate(NAMES)},
            }
        )
        trading_values.append(
            {"Date": current.isoformat(), **{code: 100_000_000 + 10_000_000 * j for j, code in enumerate(CODES)}}
        )
        trading_values_60.append(
            {"Date": current.isoformat(), **{code: 95_000_000 + 10_000_000 * j for j, code in enumerate(CODES)}}
        )
        factors.append(
            {
                "Date": current.isoformat(),
                "KOSPI": 2_500 + i * 5,
                "HML": 100 + i * 0.2,
                "SMB": 100 + i * 0.1,
                "MOM": 100 + i * 0.3,
                "CD91": 3.0,
            }
        )

    write_csv("prices_sample.csv", ["Date", *CODES], prices)
    write_csv("total_return_sample.csv", ["코드명", *NAMES], total_returns)
    write_csv("market_cap_sample.csv", ["코드명", *NAMES], market_caps)
    write_csv("trading_value_sample.csv", ["Date", *CODES], trading_values)
    write_csv("trading_value_60_sample.csv", ["Date", *CODES], trading_values_60)
    write_csv("factors_sample.csv", ["Date", "KOSPI", "HML", "SMB", "MOM", "CD91"], factors)


if __name__ == "__main__":
    main()
