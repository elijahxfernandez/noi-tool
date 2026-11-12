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
import time
import traceback
from datetime import datetime

from massive import WebSocketClient
from massive.websocket.models import Feed, Market

from shared.config import DATA_DIR, EASTERN, api_key, ns_to_eastern

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

    def status(state):
        # collector_status.json lives in a OneDrive-synced folder, which
        # intermittently locks the file mid-sync (WinError 5 / Access is
        # denied on the atomic replace - seen in production on 2026-11-11).
        # This file is cosmetic dashboard status, not data: skip this one
        # update rather than letting it crash (and restart) the whole feed.
        try:
            path = DATA_DIR / "collector_status.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"status": state, "events": count,
                                      "updated": datetime.now(EASTERN).isoformat(), "pid": os.getpid()}))
            tmp.replace(path)
        except OSError as exc:
            print(f"Status file update skipped ({type(exc).__name__}): {exc}", flush=True)

    def handler(msgs):
        # One malformed/unexpected message must not take down the whole feed -
        # an uncaught exception here would propagate out of client.run() and
        # kill the entire process (which is what happened on 2026-11-09: 18
        # minutes of collection, then silence, with no log to explain why).
        nonlocal count
        previous_count = count
        try:
            parsed = json.loads(msgs)
        except (ValueError, TypeError) as exc:
            print(f"Skipped unparseable message batch ({type(exc).__name__}): {exc}", flush=True)
            return
        for m in parsed:
            try:
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
            except Exception as exc:
                print(f"Skipped one malformed message ({type(exc).__name__}): {exc}", flush=True)
        if count != previous_count:
            status("receiving")
        out.flush()
        print(f"\rlogged {count:,} NOI events", end="", flush=True)

    # Documented client (see massive/websocket/__init__.py: WebSocketClient.connect):
    # auths, subscribes, and reconnects itself on a dropped connection using the
    # websockets library's own backoff - that backoff is exactly what a
    # max_connections rejection needs (Massive takes a few seconds to release the
    # previous session), so there's no reconnect logic to write by hand here.
    # max_reconnects=None means "keep retrying forever" instead of giving up after 5.
    client = WebSocketClient(
        api_key=api_key(),
        feed=Feed.RealTime,
        market=Market.Stocks,
        subscriptions=subs,
        raw=True,  # hand handler() the raw JSON text, same shape it already parses
        max_reconnects=None,
    )

    print(f"Collecting {', '.join(subs)} -> {NOI_DIR}  (Ctrl+C to stop)")
    status("connecting")
    # client.run() is documented to reconnect forever on a dropped connection
    # (max_reconnects=None above) - but if something still escapes that (an SDK-
    # internal exception, a hard network failure, etc.), restart the whole feed
    # instead of exiting. A short backoff avoids a tight crash-loop; the scheduled
    # task's own execution-time limit is still the outer stop for the day.
    try:
        while True:
            try:
                client.run(handler)
                break  # a clean return (e.g. after KeyboardInterrupt re-raise below) ends the loop
            except KeyboardInterrupt:
                raise
            except Exception:
                status("reconnecting")
                print(f"\nFeed crashed, restarting in 15s:\n{traceback.format_exc()}", flush=True)
                time.sleep(15)
    except KeyboardInterrupt:
        pass
    finally:
        status("stopped")
        out.close()
        print(f"\nStopped. {count:,} events logged.")

    # --- Old hand-rolled raw-websockets version (replaced above with the
    # documented massive.WebSocketClient) - kept for reference only, not used. ---
    #
    # async def stream():
    #     context = ssl.create_default_context(cafile=certifi.where())
    #     key = api_key()
    #     while True:
    #         try:
    #             status("connecting")
    #             async with connect("wss://socket.massive.com/stocks", ssl=context, open_timeout=20) as ws:
    #                 greeting = json.loads(await ws.recv())
    #                 if any(m.get("status") == "max_connections" for m in greeting):
    #                     status("max_connections")
    #                     print("Massive connection limit reached. Stop other streams and restart.", flush=True)
    #                     return
    #                 await ws.send(json.dumps({"action": "auth", "params": key}))
    #                 auth = json.loads(await ws.recv())
    #                 if not any(m.get("status") == "auth_success" for m in auth):
    #                     status("max_connections" if any(m.get("status") == "max_connections" for m in auth) else "auth_failed")
    #                     print("Feed authentication did not succeed; see dashboard status.", flush=True)
    #                     return
    #                 status("authenticated")
    #                 await ws.send(json.dumps({"action": "subscribe", "params": ",".join(subs)}))
    #                 while True:
    #                     try:
    #                         payload = await asyncio.wait_for(ws.recv(), timeout=20)
    #                     except asyncio.TimeoutError:
    #                         status(last_state)
    #                         continue
    #                     messages = json.loads(payload)
    #                     rejected = next((m.get("status") for m in messages if m.get("status") in {"max_connections", "auth_failed", "not_authorized", "error"}), None)
    #                     if rejected:
    #                         status(rejected)
    #                         print(f"Feed rejected connection/subscription: {rejected}", flush=True)
    #                         return
    #                     handler(payload)
    #         except Exception as exc:
    #             status("reconnecting")
    #             print(f"Feed interrupted ({type(exc).__name__}); retrying in 10 seconds.", flush=True)
    #             await asyncio.sleep(10)
    #
    # asyncio.run(stream())



if __name__ == "__main__":
    main()
