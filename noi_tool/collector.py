"""A1 - Stream NYSE order imbalance (NOI) events and log them to a daily CSV.

Run before the open (~9:00 ET) and leave it running through the close (~4:05 ET):
    uv run python -m noi_tool.collector
    uv run python -m noi_tool.collector --ticker JPM BAC GS
    uv run python -m noi_tool.collector --alerts            (needs daily bars; uses model if trained)
"""

import argparse
import csv
import json
import os
import atexit
import asyncio
import ssl
import certifi
from websockets.asyncio.client import connect
from datetime import datetime
from shared.config import EASTERN

from massive import WebSocketClient
from massive.websocket.models import Feed, Market

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
            is_new = not path.exists() or path.stat().st_size == 0
            self.file = open(path, "a", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=COLUMNS)
            if is_new:
                self.writer.writeheader()
            self.date = date
        self.writer.writerow(row)
        self.file.flush()

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

    # An OS lock prevents two collectors appending to the same data directory.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(DATA_DIR / "collector.lock", "a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise SystemExit("A collector is already running for this data directory.")
    atexit.register(lock.close)

    engine = None
    if args.alerts:
        from noi_tool.alerts import AlertEngine
        try:
            from shared.market_reference import latest_reference as _ref_fn
            reference = _ref_fn()  # has sigma20 + sector
            reference_source = "shared.market_reference (free S&P 500 history, with volatility)"
        except SystemExit:
            from noi_tool.daily_bars import latest_reference as _ref_fn  # fallback: Massive REST daily bars
            reference = _ref_fn()
            reference_source = "noi_tool.daily_bars (Massive REST, no volatility)"
        engine = AlertEngine(reference, None, big_pct_adv=args.big_pct_adv)
        print(f"Alerts on, reference: {reference_source} (analyst rules only)")

    subs = [f"NOI.{t.upper()}" for t in args.ticker] if args.ticker else ["NOI.*"]
    out = DailyCsvWriter()
    count = 0

    last_state = "connecting"

    def status(state):
        nonlocal last_state
        last_state = state
        path = DATA_DIR / "collector_status.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"status": state, "events": count,
                                  "updated": datetime.now(EASTERN).isoformat(), "pid": os.getpid()}))
        tmp.replace(path)

    def handler(msgs):
        nonlocal count
        previous_count = count
        for m in json.loads(msgs):
            if m.get("ev") == "status":
                state = m.get("status", "unknown")
                status(state)
                print(f"Feed status: {state}", flush=True)
                continue
            if m.get("ev") != "NOI" or not m.get("t"):
                continue
            ts = ns_to_eastern(m["t"])
            row = {
                "ts_ns": m.get("t"),
                "time_et": ts.strftime("%H:%M:%S.%f")[:-3],
                "symbol": m.get("T"),
                "auction_type": m.get("a"),
                "auction_time": m.get("at"),
                "imbalance": m.get("o"),
                "paired": m.get("p"),
                "indicative_price": m.get("b"),
                "exchange_id": m.get("x"),
                "sequence": m.get("i"),
            }
            date = ts.strftime("%Y-%m-%d")
            out.write(row, date)
            if engine and row["imbalance"] is not None and row["paired"] is not None:
                engine.check(row, date)
            count += 1
        if count != previous_count:
            status("receiving")
        out.flush()
        print(f"\rlogged {count:,} NOI events", end="", flush=True)

    async def stream():
        context = ssl.create_default_context(cafile=certifi.where())
        key = api_key()
        # max_connections means Massive hasn't released our last session yet (it
        # takes a few seconds) - back off and retry instead of giving up, since the
        # slot almost always frees up on its own. auth_failed/not_authorized stay
        # fatal (wrong key, not a timing issue).
        max_connections_retries = 0
        while True:
            try:
                status("connecting")
                async with connect("wss://socket.massive.com/stocks", ssl=context, open_timeout=20) as ws:
                    greeting = json.loads(await ws.recv())
                    if any(m.get("status") == "max_connections" for m in greeting):
                        max_connections_retries += 1
                        status("max_connections")
                        wait = min(10 * max_connections_retries, 120)
                        print(f"Massive connection limit reached; retrying in {wait}s.", flush=True)
                        await asyncio.sleep(wait)
                        continue
                    await ws.send(json.dumps({"action": "auth", "params": key}))
                    auth = json.loads(await ws.recv())
                    if not any(m.get("status") == "auth_success" for m in auth):
                        if any(m.get("status") == "max_connections" for m in auth):
                            max_connections_retries += 1
                            status("max_connections")
                            wait = min(10 * max_connections_retries, 120)
                            print(f"Massive connection limit reached; retrying in {wait}s.", flush=True)
                            await asyncio.sleep(wait)
                            continue
                        status("auth_failed")
                        print("Feed authentication did not succeed; see dashboard status.", flush=True)
                        return
                    max_connections_retries = 0
                    status("authenticated")
                    await ws.send(json.dumps({"action": "subscribe", "params": ",".join(subs)}))
                    while True:
                        try:
                            payload = await asyncio.wait_for(ws.recv(), timeout=20)
                        except asyncio.TimeoutError:
                            status(last_state)  # heartbeat while idle, preserving actual subscription state
                            continue
                        messages = json.loads(payload)
                        rejected = next((m.get("status") for m in messages if m.get("status") in {"max_connections", "auth_failed", "not_authorized", "error"}), None)
                        if rejected == "max_connections":
                            max_connections_retries += 1
                            status("max_connections")
                            wait = min(10 * max_connections_retries, 120)
                            print(f"Feed dropped (max_connections); retrying in {wait}s.", flush=True)
                            await asyncio.sleep(wait)
                            break
                        if rejected:
                            status(rejected)
                            print(f"Feed rejected connection/subscription: {rejected}", flush=True)
                            return
                        handler(payload)
            except Exception as exc:
                # Never log exception text that might contain credentials.
                status("reconnecting")
                print(f"Feed interrupted ({type(exc).__name__}); retrying in 10 seconds.", flush=True)
                await asyncio.sleep(10)

    print(f"Collecting {', '.join(subs)} -> {NOI_DIR}  (Ctrl+C to stop)")
    status("connecting")
    try:
        asyncio.run(stream())
    except KeyboardInterrupt:
        pass
    finally:
        if last_state not in {"max_connections", "auth_failed", "not_authorized", "error"}:
            status("stopped")
        out.close()
        print(f"\nStopped. {count:,} events logged.")



if __name__ == "__main__":
    main()
