"""Live NOI reader - streams order imbalance events and prints a plain-English read
of each one: direction, size (vs normal volume if known), and convergence trend.

No training, no price prediction - this mirrors the tutorial's own noi_monitor.py,
with the signal-score idea (direction x magnitude x convergence) layered on top.

    uv run python -m noi_tool.reader
    uv run python -m noi_tool.reader --ticker JPM BAC GS
    uv run python -m noi_tool.reader --min-imbalance 5000
"""

import argparse
from collections import defaultdict, deque

from massive import WebSocketClient
from massive.websocket.models import Feed, Imbalance, Market

from noi_tool.signal import signal_score
from shared.config import api_key, ns_to_eastern

AUCTION_LABELS = {"M": "Open", "C": "Close", "H": "Halt"}
LOOKBACK_SEC = 5 * 60  # convergence window

# Fallback size buckets when ADV isn't available: imbalance / paired shares
SIZE_BUCKETS = [(1.5, "very large"), (0.6, "large"), (0.2, "moderate")]


class ImbalanceReader:
    def __init__(self, adv_lookup: dict[str, float] | None = None):
        self.adv = adv_lookup or {}
        self.history: dict[tuple, deque] = defaultdict(deque)

    def direction(self, imbalance: int) -> str:
        if imbalance > 0:
            return "BUY"
        if imbalance < 0:
            return "SELL"
        return "FLAT"

    def convergence(self, key: tuple, ts_ns: int, imbalance: int) -> str:
        hist = self.history[key]
        hist.append((ts_ns, imbalance))
        cutoff = ts_ns - LOOKBACK_SEC * 1_000_000_000
        while len(hist) > 1 and hist[1][0] <= cutoff:
            hist.popleft()
        baseline = hist[0][1]
        change = (abs(imbalance) - abs(baseline)) / (abs(baseline) + 1)
        if change < -0.05:
            return "shrinking"
        if change > 0.05:
            return "growing"
        return "steady"

    def magnitude(self, symbol: str, imbalance: int, paired: int) -> str:
        adv = self.adv.get(symbol)
        if adv:
            pct = abs(imbalance) / adv
            if pct >= 0.02:
                return f"very large ({pct:.1%} of ADV)"
            if pct >= 0.005:
                return f"large ({pct:.1%} of ADV)"
            if pct >= 0.001:
                return f"moderate ({pct:.1%} of ADV)"
            return f"small ({pct:.1%} of ADV)"
        ratio = abs(imbalance) / max(paired, 1)
        for cutoff, label in SIZE_BUCKETS:
            if ratio >= cutoff:
                return f"{label} (imb/paired {ratio:.2f})"
        return f"small (imb/paired {ratio:.2f})"

    def read(self, row: dict) -> str:
        """Run all three checks on one event and return one printable line."""
        symbol, imbalance, paired = row["symbol"], row["imbalance"], row["paired"]
        key = (symbol, row["auction_type"])
        direction = self.direction(imbalance)
        trend = self.convergence(key, row["ts_ns"], imbalance)
        size = self.magnitude(symbol, imbalance, paired)
        adv = self.adv.get(symbol)
        score = signal_score(imbalance, adv, 0.0) if adv else None  # score needs ADV; skip if unknown

        auction = AUCTION_LABELS.get(row["auction_type"], row["auction_type"])
        price = f"${row['indicative_price']:.2f}" if row["indicative_price"] else "n/a"
        score_txt = f"  score {float(score):+.2f}" if score is not None else ""
        return (
            f"{row['time_et'][:8]}  {symbol:6} {auction:5}  {direction:4}  "
            f"{imbalance:>+10,}  size={size:<28}  trend={trend:<9}  price={price}{score_txt}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", nargs="+", help="Symbols to watch (default: all NYSE)")
    parser.add_argument("--min-imbalance", type=int, default=0, help="Skip events smaller than this many shares")
    args = parser.parse_args()

    adv_lookup = {}
    try:
        from noi_tool.daily_bars import latest_reference
        adv_lookup = latest_reference()["adv20"].to_dict()
        print(f"ADV known for {len(adv_lookup)} symbols (scaled sizing on)")
    except SystemExit:
        print("No daily bars yet - sizing falls back to imbalance/paired ratio")

    reader = ImbalanceReader(adv_lookup)
    subs = [f"NOI.{t.upper()}" for t in args.ticker] if args.ticker else ["NOI.*"]
    client = WebSocketClient(api_key=api_key(), feed=Feed.RealTime, market=Market.Stocks, subscriptions=subs)

    def handler(msgs):
        for m in msgs:
            if not isinstance(m, Imbalance) or not m.time_stamp:
                continue
            imbalance = m.imbalance_quantity or 0
            if abs(imbalance) < args.min_imbalance:
                continue
            ts = ns_to_eastern(m.time_stamp)
            row = {
                "ts_ns": m.time_stamp, "time_et": ts.strftime("%H:%M:%S.%f")[:-3],
                "symbol": m.symbol, "auction_type": m.auction_type,
                "imbalance": imbalance, "paired": m.paired_quantity or 0,
                "indicative_price": m.book_clearing_price,
            }
            print(reader.read(row), flush=True)

    print(f"Reading {', '.join(subs)}  (Ctrl+C to stop)")
    try:
        client.run(handle_msg=handler)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
