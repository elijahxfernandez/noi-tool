"""Local NOI dashboard. Run: python -m noi_tool.dashboard --collect"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from noi_tool.collector import NOI_DIR
from noi_tool.signal import expected_price_impact_bps, size_bucket
from shared.config import EASTERN, REPORTS_DIR
from shared.market_reference import latest_reference

BUCKETS = ["small", "moderate", "large", "very large"]


def analyze(events, reference):
    """Deterministic analyst rules; missing inputs remain unknown."""
    if events.empty:
        return []
    events = events.sort_values("ts_ns").copy()
    events["symbol"] = events["symbol"].str.upper().str.replace("-", ".", regex=False)
    output = []
    for (date, symbol, auction), group in events.groupby(["date", "symbol", "auction_type"]):
        row = group.iloc[-1].to_dict()
        price, imb = row.get("indicative_price"), row.get("imbalance")
        if pd.isna(price) or price <= 0:
            continue
        group = group.dropna(subset=["imbalance"])
        valid_imb = pd.notna(imb) and np.isfinite(imb)
        cutoff = row["ts_ns"] - 300 * 1_000_000_000
        before = group[group.ts_ns <= cutoff]
        base_row = before.iloc[-1] if not before.empty else group.iloc[0] if not group.empty else None
        base = base_row["imbalance"] if base_row is not None else np.nan
        change = float(np.clip((abs(imb) - abs(base)) / (abs(base) + 1), -1, 1)) if valid_imb else 0.0
        item = dict(date=date, symbol=symbol, auction=auction, time=row["time_et"],
                    imbalance=float(imb) if valid_imb else None, paired=row.get("paired"), indicative=float(price),
                    pressure="UNKNOWN" if not valid_imb else "BUY" if imb > 0 else "SELL" if imb < 0 else "FLAT",
                    trend="unknown" if not valid_imb else "insufficient history" if len(group) < 2 else "growing" if change > .05 else "shrinking" if change < -.05 else "unchanged",
                    side_flip=bool(valid_imb and len(group) > 1 and group.iloc[-2].imbalance != 0 and imb != 0 and np.sign(group.iloc[-2].imbalance) != np.sign(imb)),
                    price_change_bps=float((price / base_row.indicative_price - 1) * 1e4) if base_row is not None and pd.notna(base_row.indicative_price) and base_row.indicative_price > 0 and len(group) > 1 else None,
                    bucket="unavailable", adv_pct=None, impact_bps=None, reference_date=None,
                    estimated_final=float(price), basis="Exchange indicative auction estimate",
                    assessment="Awaiting movement history", sigma_pct=None, score=None)
        if symbol in reference.index:
            ref = reference.loc[symbol]
            adv, prev, sigma = ref.adv20, ref.prev_close, ref.sigma20
            item["reference_date"] = ref.get("reference_date")
            if valid_imb and np.isfinite(adv) and adv > 0 and np.isfinite(prev) and prev > 0:
                item["bucket"] = BUCKETS[int(size_bucket(imb, adv))]
                item["adv_pct"] = float(abs(imb) / adv * 100)
                if np.isfinite(sigma) and sigma > 0:
                    item["impact_bps"] = float(expected_price_impact_bps(imb, adv, sigma))
                    item["sigma_pct"] = float(sigma * 100)
                # Bucket and convergence influence a readable pressure score, not a fabricated dollar target.
                item["score"] = float(np.sign(imb) * (int(size_bucket(imb, adv)) + 1) * np.clip(1 + change, .5, 1.5))
        delta = item["price_change_bps"]
        if item["side_flip"]:
            item["assessment"] = "Pressure side flipped; reassess"
        elif valid_imb and imb == 0:
            item["assessment"] = "Balanced auction interest"
        elif delta is not None:
            agrees = np.sign(delta) == np.sign(imb)
            if item["trend"] == "growing" and agrees:
                item["assessment"] = "Building upward pressure" if imb > 0 else "Building downward pressure"
            elif item["trend"] == "shrinking":
                item["assessment"] = "Imbalance easing; watch indicative price"
            elif delta != 0 and not agrees:
                item["assessment"] = "Price and imbalance disagree"
            else:
                item["assessment"] = "Stable or mixed auction pressure"
        output.append(item)
    return output



def with_historical_watchlist(rows, reference):
    """Historical context is visible even without NOI; never invent auction fields."""
    seen = {r["symbol"] for r in rows}
    for row in rows:
        row["data_status"] = "captured NOI"
        if row["symbol"] in reference.index:
            r = reference.loc[row["symbol"]]
            row.update(historical_close=r.prev_close, historical_adv=r.adv20,
                       historical_sigma_pct=r.sigma20 * 100)
    for symbol, ref in reference.iterrows():
        if symbol in seen:
            continue
        rows.append(dict(symbol=symbol, date=None, auction="", time=None,
                         data_status="historical only", pressure="WAITING", imbalance=None,
                         paired=None, indicative=None, estimated_final=None, adv_pct=None,
                         bucket="awaiting NOI", trend="awaiting NOI", side_flip=False,
                         price_change_bps=None, impact_bps=None, score=None,
                         sigma_pct=ref.sigma20 * 100, reference_date=ref.get("reference_date"),
                         historical_close=ref.prev_close, historical_adv=ref.adv20,
                         historical_sigma_pct=ref.sigma20 * 100,
                         assessment="Historical context loaded. Waiting for a live auction update."))
    return sorted(rows, key=lambda r: (r["data_status"] == "historical only", r["symbol"]))

def captured_events():
    files = sorted(NOI_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    # Latest captured day; full files are retained for research and training.
    for path in files[-1:]:
        try:
            frames.append(pd.read_csv(path, on_bad_lines="skip").assign(date=path.stem))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def feed_status():
    try:
        return json.loads((NOI_DIR.parent / "collector_status.json").read_text())
    except (OSError, ValueError):
        return {"status": "unknown", "events": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--collect", action="store_true", help="Start an owned collector; do not run another collector simultaneously")
    args = parser.parse_args()
    try:
        reference = latest_reference()
    except SystemExit:
        reference = pd.DataFrame()
    collector = None

    def snapshot():
        rows = with_historical_watchlist(analyze(captured_events(), reference), reference)
        # JSON must remain valid even when paired shares are missing in captured data.
        rows = json.loads(pd.DataFrame(rows).to_json(orient="records")) if rows else []
        dates = reference["reference_date"] if "reference_date" in reference else pd.Series(dtype=str)
        return dict(rows=rows, method="Rules: direction, ADV buckets, five-minute movement", reference_date=dates.max() if len(dates) else None,
                    collector="running" if collector and collector.poll() is None else "external or stopped",
                    feed=feed_status(),
                    updated=datetime.now(EASTERN).isoformat())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?")[0]
            if route == "/":
                body = Path(__file__).with_name("dashboard.html").read_bytes()
                mime = "text/html; charset=utf-8"
            elif route == "/api/state":
                body = json.dumps(snapshot(), allow_nan=False).encode()
                mime = "application/json"
            elif route == "/export.csv":
                body = pd.DataFrame(snapshot()["rows"]).to_csv(index=False).encode()
                mime = "text/csv"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log = None
    try:
        if args.collect:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            log = open(REPORTS_DIR / "collector.log", "a", buffering=1)
            collector = subprocess.Popen([sys.executable, "-u", "-m", "noi_tool.collector"], stdout=log, stderr=log)
        print(f"NOI dashboard: http://127.0.0.1:{args.port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if collector:
            collector.terminate()
            collector.wait(timeout=10)
        if log:
            log.close()

if __name__ == "__main__":
    main()
