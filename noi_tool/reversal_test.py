"""Part B - does a large closing imbalance predict next-day reversal/follow-through?

This is the empirical companion to signal.expected_price_impact_bps (Part A). Part A
needs no NOI history - it estimates today's move from a formula. This needs real
captured closing-auction imbalances, joined to each stock's ACTUAL next-day return:

    Next-Day Return[s, d+1] = alpha + beta * (Closing Imbalance[s, d] / ADV[s])

beta < 0  -> reversal (today's imbalance partially gives back tomorrow - the
             literature's usual finding, attributed to liquidity providers
             unwinding the inventory they absorbed)
beta > 0  -> follow-through instead

This is a hypothesis test, not a deployed model: it needs many (symbol, day) pairs
across different market conditions before beta means anything. Right now (collector
just started) there is close to no real data, so build() will say so rather than
fit noise and report a number.

    uv run python -m noi_tool.reversal_test
"""

import numpy as np
import pandas as pd

from noi_tool.dataset import NOI_DIR
from shared.market_reference import load_combined

MIN_ROWS = 60  # a rough floor - below this, beta is noise, not a finding


def closing_imbalances() -> pd.DataFrame:
    """Latest closing-auction imbalance per (date, symbol), from everything the
    collector has captured so far."""
    files = sorted(NOI_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["date", "symbol", "imbalance"])
    df = pd.concat([pd.read_csv(f).assign(date=f.stem) for f in files], ignore_index=True)
    closing = df[df["auction_type"] == "C"].dropna(subset=["imbalance"])
    if closing.empty:
        return pd.DataFrame(columns=["date", "symbol", "imbalance"])
    return closing.sort_values("ts_ns").groupby(["date", "symbol"]).last().reset_index()[["date", "symbol", "imbalance"]]


def build_dataset() -> pd.DataFrame:
    """Join each captured closing imbalance to that stock's ADV (for scaling) and
    its ACTUAL next trading day's return (both from market_reference's price
    history). One row per (symbol, day) we've actually captured."""
    imb = closing_imbalances()
    if imb.empty:
        return imb.assign(adv20=[], next_day_return_pct=[])

    bars = load_combined()[["symbol", "date", "close", "adv20"]].sort_values(["symbol", "date"])
    bars["next_close"] = bars.groupby("symbol")["close"].shift(-1)
    bars["next_day_return_pct"] = (bars["next_close"] / bars["close"] - 1) * 100

    out = imb.merge(bars[["symbol", "date", "adv20", "next_day_return_pct"]], on=["symbol", "date"], how="inner")
    out["imb_pct_adv"] = out["imbalance"] / out["adv20"]
    return out.dropna(subset=["next_day_return_pct"])


def fit_beta(df: pd.DataFrame) -> dict:
    """Simple OLS: next_day_return_pct = alpha + beta * imb_pct_adv, solved from
    the normal equations directly (same from-the-math approach as the Ridge model)."""
    x, y = df["imb_pct_adv"].to_numpy(float), df["next_day_return_pct"].to_numpy(float)
    X = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ [alpha, beta]
    se_beta = np.sqrt(np.sum(resid**2) / (len(y) - 2) / np.sum((x - x.mean()) ** 2))
    return {"alpha": float(alpha), "beta": float(beta), "se_beta": float(se_beta),
            "t_stat": float(beta / se_beta) if se_beta > 0 else float("nan"), "n": len(y)}


def main():
    df = build_dataset()
    print(f"Captured closing-imbalance rows joinable to a real next-day return: {len(df)}")
    if len(df) < MIN_ROWS:
        print(f"Below the {MIN_ROWS}-row floor - not enough real data yet for beta to mean anything.")
        print("Keep the collector running day over day; re-run this once more days are captured.")
        return
    result = fit_beta(df)
    direction = "reversal" if result["beta"] < 0 else "follow-through"
    print(f"beta = {result['beta']:+.4f} ({direction}), t-stat {result['t_stat']:+.2f}, n={result['n']}")
    print("A |t-stat| below ~2 means this isn't distinguishable from no effect yet.")


if __name__ == "__main__":
    main()
