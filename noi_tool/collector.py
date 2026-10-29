"""A1 - Stream NYSE order imbalance (NOI) events and log them to a daily CSV.

Run before the open (~9:00 ET) and leave it running through the close (~4:05 ET):
    uv run python -m noi_tool.collector
    uv run python -m noi_tool.collector --ticker JPM BAC GS
    uv run python -m noi_tool.collector --alerts            (needs daily bars; uses model if trained)
"""

import argparse
import csv

from massive import WebSocketClient
from massive.websocket.models import Feed, Imbalance, Market

from shared.config import DATA_DIR, api_key, ns_to_eastern

NOI_DIR = DATA_DIR / "noi"
COLUMNS = [
    "ts_ns", "time_et", "symbol", "auction_type", "auction_time",
    "imbalance", "paired", "indicative_price", "exchange_id", "sequence",
]


class DailyCsvWriter:
    """Appends rows to data/noi/YYYY-MM-DD.csv, switching files when the ET date changes."""

    def __init__(self):
        NOI_DIR.mkdir(parents=True, exist_ok=True)
        self.date = None
        self.file = None
        self.writer = None

    def write(self, row: dict, date: str):
        if date != self.date:
            self.close()
            path = NOI_DIR / f"{date}.csv"
            is_new = not path.exists()
            self.file = open(path, "a", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=COLUMNS)
            if is_new:
                self.writer.writeheader()
            self.date = date
        self.writer.writerow(row)

    def flush(self):
        if self.file:
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", nargs="+", help="Symbols to watch (default: all NYSE)")
    parser.add_argument("--alerts", action="store_true", help="Fire live alerts (BIG / FLIP / DRIFT)")
    parser.add_argument("--big-pct-adv", type=float, default=0.01, help="BIG alert threshold as fraction of ADV")
    args = parser.parse_args()

    engine = None
    if args.alerts:
        from noi_tool.alerts import AlertEngine
        from noi_tool.model import MODEL_PATH, Ridge
        try:
            from shared.market_reference import latest_reference as _ref_fn
            reference = _ref_fn()  # has sigma20 + sector
            reference_source = "shared.market_reference (free S&P 500 history, with volatility)"
        except SystemExit:
            from noi_tool.daily_bars import latest_reference as _ref_fn  # fallback: Massive REST daily bars
            reference = _ref_fn()
            reference_source = "noi_tool.daily_bars (Massive REST, no volatility)"
        model = Ridge.load(MODEL_PATH) if MODEL_PATH.exists() else None
        engine = AlertEngine(reference, model, big_pct_adv=args.big_pct_adv)
        print(f"Alerts on, reference: {reference_source} (model {'loaded' if model else 'not trained yet - score only'})")

    subs = [f"NOI.{t.upper()}" for t in args.ticker] if args.ticker else ["NOI.*"]
    client = WebSocketClient(api_key=api_key(), feed=Feed.RealTime, market=Market.Stocks, subscriptions=subs)
    out = DailyCsvWriter()
    count = 0

    def handler(msgs):
        nonlocal count
        for m in msgs:
            if not isinstance(m, Imbalance) or not m.time_stamp:
                continue
            ts = ns_to_eastern(m.time_stamp)
            row = {
                "ts_ns": m.time_stamp,
                "time_et": ts.strftime("%H:%M:%S.%f")[:-3],
                "symbol": m.symbol,
                "auction_type": m.auction_type,
                "auction_time": m.auction_time,
                "imbalance": m.imbalance_quantity or 0,
                "paired": m.paired_quantity or 0,
                "indicative_price": m.book_clearing_price,
                "exchange_id": m.exchange_id,
                "sequence": m.symbol_sequence,
            }
            date = ts.strftime("%Y-%m-%d")
            out.write(row, date)
            if engine:
                engine.check(row, date)
            count += 1
        out.flush()
        print(f"\rlogged {count:,} NOI events", end="", flush=True)

    print(f"Collecting {', '.join(subs)} -> {NOI_DIR}  (Ctrl+C to stop)")
    try:
        client.run(handle_msg=handler)
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        print(f"\nStopped. {count:,} events logged.")


if __name__ == "__main__":
    main()
