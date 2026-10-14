"""A2 + A3 - Daily open/close/volume for every US stock, used for:
  - labels: the official open and close are the NYSE auction prices for NYSE-listed stocks
  - scaling: 20-day average daily volume (ADV), so imbalance can be read relative to normal

Fetch (one REST call per trading day, skips days already saved):
    uv run python -m noi_tool.daily_bars --start 2026-08-01 --end 2026-10-14
"""

import argparse
from datetime import date, timedelta

import pandas as pd
from massive import RESTClient

from shared.config import DATA_DIR, api_key

DAILY_DIR = DATA_DIR / "daily"
ADV_WINDOW = 20


def fetch(start: date, end: date):
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    client = RESTClient(api_key())
    day = start
    while day <= end:
        path = DAILY_DIR / f"{day}.csv"
        if day.weekday() < 5 and not path.exists():
            bars = client.get_grouped_daily_aggs(day.isoformat(), adjusted=True)
            rows = [
                {"symbol": b.ticker, "open": b.open, "close": b.close, "volume": b.volume}
                for b in bars or []
            ]
            if rows:  # market holidays come back empty
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"{day}: {len(rows):,} stocks")
            else:
                print(f"{day}: no data (holiday?)")
        day += timedelta(days=1)


def load() -> pd.DataFrame:
    """All saved days, with ADV and previous close computed from PRIOR days only (no peeking ahead)."""
    files = sorted(DAILY_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No daily bars in {DAILY_DIR}. Run: uv run python -m noi_tool.daily_bars --start ... --end ...")
    df = pd.concat([pd.read_csv(f).assign(date=f.stem) for f in files], ignore_index=True)
    df = df.sort_values(["symbol", "date"])
    g = df.groupby("symbol")
    df["adv20"] = g["volume"].transform(lambda v: v.shift(1).rolling(ADV_WINDOW, min_periods=10).mean())
    df["prev_close"] = g["close"].shift(1)
    return df.reset_index(drop=True)


def latest_reference() -> pd.DataFrame:
    """For live use: per symbol, ADV over the last 20 saved days and the most recent close."""
    df = load().sort_values(["symbol", "date"])
    last = df.groupby("symbol").tail(ADV_WINDOW).groupby("symbol")
    return pd.DataFrame({"adv20": last["volume"].mean(), "prev_close": last["close"].last()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", default=date.today(), type=date.fromisoformat)
    args = parser.parse_args()
    fetch(args.start, args.end)


if __name__ == "__main__":
    main()
