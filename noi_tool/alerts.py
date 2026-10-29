"""A6 - Live alerts on imbalance events. Used by the collector with --alerts.

Fires when:
  BIG    |imbalance| passes a % of the stock's average daily volume (once per symbol/auction/day)
  FLIP   a closing imbalance changes side in the final minutes
  DRIFT  the indicative price sits far from the previous close

Each alert carries the signal score and, if a trained model exists, the predicted gap
between the indicative price and the final auction print.
Set SLACK_WEBHOOK_URL in .env to also post alerts to Slack.
"""

import json
import os
import urllib.request
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from noi_tool.dataset import FEATURES, LOOKBACK_MIN
from noi_tool.signal import event_features, expected_price_impact_bps, signal_score

AUCTION_LABELS = {"M": "Open", "C": "Close", "H": "Halt"}


class AlertEngine:
    def __init__(self, reference, model=None, big_pct_adv=0.01, drift_bps=300, flip_after="15:55"):
        self.ref = reference  # DataFrame indexed by symbol: adv20, prev_close, optionally sigma20
        self.has_sigma = "sigma20" in reference.columns
        self.model = model
        self.big_pct_adv = big_pct_adv
        self.drift_bps = drift_bps
        self.flip_after = flip_after
        self.history = defaultdict(deque)  # (symbol, auction) -> deque[(ts_ns, imbalance)]
        self.fired = set()
        self.webhook = os.getenv("SLACK_WEBHOOK_URL")

    def _imb_change(self, key, ts_ns, imb):
        """Returns (convergence vs ~5 min ago, the previous event's imbalance)."""
        hist = self.history[key]
        previous = hist[-1][1] if hist else None
        hist.append((ts_ns, imb))
        cutoff = ts_ns - LOOKBACK_MIN * 60 * 1_000_000_000
        while len(hist) > 1 and hist[1][0] <= cutoff:
            hist.popleft()
        before = hist[0][1]  # latest event at/before the cutoff, or earliest in window
        return float(np.clip((abs(imb) - abs(before)) / (abs(before) + 1), -1, 1)), previous

    def check(self, row: dict, date: str) -> list[str]:
        sym, auction, imb, price = row["symbol"], row["auction_type"], row["imbalance"], row["indicative_price"]
        if sym not in self.ref.index or not price:
            return []
        adv, prev_close = self.ref.at[sym, "adv20"], self.ref.at[sym, "prev_close"]
        sigma = self.ref.at[sym, "sigma20"] if self.has_sigma else None
        change, previous = self._imb_change((sym, auction), row["ts_ns"], imb)
        score = float(signal_score(imb, adv, change))
        label = AUCTION_LABELS.get(auction, auction)
        alerts = []

        if abs(imb) / adv >= self.big_pct_adv and (key := (date, sym, auction, "big")) not in self.fired:
            self.fired.add(key)
            alerts.append(f"BIG   {sym} {label}: {'BUY' if imb > 0 else 'SELL'} {abs(imb):,} = {abs(imb) / adv:.1%} of ADV")
        if auction == "C" and row["time_et"] >= self.flip_after and previous and imb and np.sign(previous) != np.sign(imb):
            alerts.append(f"FLIP  {sym} Close: imbalance flipped {previous:+,} -> {imb:+,}")
        drift = (price / prev_close - 1) * 1e4
        if abs(drift) >= self.drift_bps and (key := (date, sym, auction, "drift")) not in self.fired:
            self.fired.add(key)
            alerts.append(f"DRIFT {sym} {label}: indicative ${price:.2f} is {drift:+.0f} bps vs prev close")

        if not alerts:
            return []
        tail = f"  | score {score:+.2f}"
        if sigma is not None and pd.notna(sigma) and sigma > 0:
            impact = expected_price_impact_bps(imb, adv, sigma)
            tail += f" | expected move ~{impact:+.0f} bps (sqrt-impact model, ${price * (1 + impact / 1e4):.2f})"
        if self.model is not None and row.get("auction_time"):
            feats = event_features(row, adv, prev_close, change, sigma)
            pred = float(self.model.predict(np.array([[feats[f] for f in FEATURES]]))[0])
            tail += f" | model: final print {pred:+.1f} bps vs indicative (~${price * (1 + pred / 1e4):.2f})"
        alerts = [a + tail for a in alerts]
        self._notify(alerts)
        return alerts

    def _notify(self, alerts):
        for a in alerts:
            print(f"\n[ALERT] {a}", flush=True)
        if self.webhook:
            body = json.dumps({"text": "\n".join(alerts)}).encode()
            req = urllib.request.Request(self.webhook, body, {"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=5)
            except OSError as e:
                print(f"\n(Slack post failed: {e})")
