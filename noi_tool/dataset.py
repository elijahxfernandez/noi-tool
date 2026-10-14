"""A4 - Build the training set: one row per (date, symbol, auction, checkpoint).

Each row is a snapshot of the imbalance at a checkpoint before the auction, plus the
actual auction price as the label:

    target_bps = (final auction price - indicative price at checkpoint) / indicative * 10,000

i.e. "how far did the final print land from NYSE's own estimate?"  A model has to beat
predicting 0 (trusting NYSE's indicative price) to be worth anything.

    uv run python -m noi_tool.dataset
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from noi_tool.daily_bars import load as load_daily
from noi_tool.signal import signal_score
from shared.config import DATA_DIR

NOI_DIR = DATA_DIR / "noi"
OUT_PATH = DATA_DIR / "training.csv"

# auction code -> (checkpoints ET, auction time, daily-bar column holding the auction price)
AUCTIONS = {
    "C": (["15:51", "15:55", "15:58"], "16:00", "close"),  # NYSE publishes close imbalances from 15:50
    "M": (["09:20", "09:25", "09:28"], "09:30", "open"),
}
LOOKBACK_MIN = 5  # convergence window

FEATURES = [
    "side", "imb_adv", "imb_paired", "imb_change",
    "ind_vs_prev_bps", "minutes_left", "log_adv", "score",
]


def load_noi() -> pd.DataFrame:
    files = sorted(NOI_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No NOI files in {NOI_DIR}. Run the collector first.")
    df = pd.concat([pd.read_csv(f).assign(date=f.stem) for f in files], ignore_index=True)
    return df.dropna(subset=["indicative_price"]).query("indicative_price > 0")


def _minus(hhmm: str, minutes: int) -> str:
    return (datetime.strptime(hhmm, "%H:%M") - timedelta(minutes=minutes)).strftime("%H:%M")


def _snapshot(noi: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Latest event per (date, symbol) at or before the cutoff time."""
    upto = noi[noi["time_et"] <= cutoff + ":00.000"]
    return upto.sort_values("ts_ns").groupby(["date", "symbol"]).last().reset_index()


def _lookback(noi: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Imbalance as of `start`; if the feed hadn't started yet, the earliest event before `end`."""
    asof = _snapshot(noi, start)
    window = noi[(noi["time_et"] > start + ":00.000") & (noi["time_et"] <= end + ":00.000")]
    first = window.sort_values("ts_ns").groupby(["date", "symbol"]).first().reset_index()
    both = pd.concat([asof, first[~first.set_index(["date", "symbol"]).index.isin(asof.set_index(["date", "symbol"]).index)]])
    return both[["date", "symbol", "imbalance"]]


def build(noi: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code, (checkpoints, auction_time, price_col) in AUCTIONS.items():
        events = noi[noi["auction_type"] == code]
        if events.empty:
            continue
        for cp in checkpoints:
            now = _snapshot(events, cp)
            before = _lookback(events, _minus(cp, LOOKBACK_MIN), cp)
            snap = now.merge(before, on=["date", "symbol"], how="left", suffixes=("", "_before"))
            snap["auction"] = code
            snap["checkpoint"] = cp
            snap["minutes_left"] = (
                datetime.strptime(auction_time, "%H:%M") - datetime.strptime(cp, "%H:%M")
            ).seconds / 60
            snap = snap.merge(
                daily[["date", "symbol", price_col, "adv20", "prev_close"]].rename(columns={price_col: "final_price"}),
                on=["date", "symbol"], how="inner",
            )
            rows.append(snap)

    if not rows:
        raise SystemExit("No checkpoint snapshots matched daily bars. Check that dates overlap.")
    df = pd.concat(rows, ignore_index=True).dropna(subset=["adv20", "prev_close", "final_price"])

    prev_abs = df["imbalance_before"].abs()
    df["side"] = np.sign(df["imbalance"])
    df["imb_adv"] = df["imbalance"] / df["adv20"]
    df["imb_paired"] = df["imbalance"] / (df["paired"] + df["imbalance"].abs()).replace(0, np.nan)
    df["imb_change"] = ((df["imbalance"].abs() - prev_abs) / (prev_abs + 1)).fillna(0).clip(-1, 1)
    df["ind_vs_prev_bps"] = (df["indicative_price"] / df["prev_close"] - 1) * 1e4
    df["log_adv"] = np.log10(df["adv20"])
    df["score"] = signal_score(df["imbalance"], df["adv20"], df["imb_change"])
    df["target_bps"] = (df["final_price"] / df["indicative_price"] - 1) * 1e4

    # Drop bad prints / data errors: a >10% gap between indicative and final is not a signal
    df = df[df["target_bps"].abs() < 1000].dropna(subset=FEATURES)
    keep = ["date", "symbol", "auction", "checkpoint", "imbalance", "paired",
            "indicative_price", "final_price", *FEATURES, "target_bps"]
    return df[keep].sort_values(["date", "auction", "checkpoint", "symbol"]).reset_index(drop=True)


def main():
    df = build(load_noi(), load_daily())
    df.to_csv(OUT_PATH, index=False)
    print(f"{len(df):,} rows across {df['date'].nunique()} days, {df['symbol'].nunique()} symbols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
