"""Free historical reference data, shared by both tools: S&P 500 constituents with
sector/industry (from Wikipedia) and daily OHLCV history (from Yahoo Finance via
yfinance). Downloaded once and committed to the repo under reference_data/ - this is
a static snapshot for backtesting/context, not meant to be re-pulled every day.

    uv run python -m shared.market_reference --refresh-constituents
    uv run python -m shared.market_reference --history --years 2
    uv run python -m shared.market_reference --history --symbols JPM BAC GS --years 2
"""

import argparse
import time

import pandas as pd
import yfinance as yf

from shared.config import ROOT

REF_DIR = ROOT / "reference_data"
CONSTITUENTS_PATH = REF_DIR / "sp500_constituents.csv"
DAILY_DIR = REF_DIR / "daily"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) market_reference.py (personal research script)"}


def fetch_constituents() -> pd.DataFrame:
    df = pd.read_html(WIKI_URL, storage_options=HEADERS)[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
    df.columns = ["symbol", "name", "sector", "sub_industry"]
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (yfinance/Massive style)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONSTITUENTS_PATH, index=False)
    print(f"{len(df)} S&P 500 constituents -> {CONSTITUENTS_PATH}")
    return df


def fetch_history(symbols: list[str], years: int = 2, pause: float = 0.3) -> tuple[int, list[str]]:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for i, sym in enumerate(symbols, 1):
        try:
            hist = yf.Ticker(sym).history(period=f"{years}y", auto_adjust=False)
            if hist.empty:
                raise ValueError("no data returned")
            hist = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
            hist.columns = ["date", "open", "high", "low", "close", "volume"]
            hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
            hist = hist.dropna(subset=["close"]).query("volume > 0")  # drop today's in-progress/empty session
            hist[["open", "high", "low", "close"]] = hist[["open", "high", "low", "close"]].round(2)
            hist.to_csv(DAILY_DIR / f"{sym}.csv", index=False)
            ok += 1
        except Exception:
            failed.append(sym)
        if i % 25 == 0 or i == len(symbols):
            print(f"  {i}/{len(symbols)} done ({ok} ok, {len(failed)} failed)")
        time.sleep(pause)
    print(f"History: {ok} succeeded, {len(failed)} failed -> {DAILY_DIR}")
    if failed:
        print("Failed:", ", ".join(failed))
    return ok, failed


def load_combined(adv_window: int = 20, vol_window: int = 20) -> pd.DataFrame:
    """All cached symbol histories, one row per (symbol, date), with ADV, daily
    volatility (sigma), previous close, and sector/sub-industry joined in. Same shape
    as noi_tool.daily_bars.load() so either source can feed the same downstream code.

    sigma = trailing std dev of daily % returns - the stock's own typical day-to-day
    swing size, used by signal.expected_price_impact() alongside ADV (see its
    docstring for the square-root price-impact model this feeds)."""
    if not DAILY_DIR.exists() or not any(DAILY_DIR.iterdir()):
        raise SystemExit(f"No cached history in {DAILY_DIR}. Run with --history first.")
    df = pd.concat(
        [pd.read_csv(f).assign(symbol=f.stem) for f in DAILY_DIR.glob("*.csv")],
        ignore_index=True,
    ).sort_values(["symbol", "date"])
    g = df.groupby("symbol")
    df["adv20"] = g["volume"].transform(lambda v: v.shift(1).rolling(adv_window, min_periods=10).mean())
    df["prev_close"] = g["close"].shift(1)
    daily_return = g["close"].pct_change()
    df["sigma20"] = daily_return.groupby(df["symbol"]).transform(
        lambda r: r.shift(1).rolling(vol_window, min_periods=10).std()
    )
    if CONSTITUENTS_PATH.exists():
        sectors = pd.read_csv(CONSTITUENTS_PATH)[["symbol", "sector", "sub_industry"]]
        df = df.merge(sectors, on="symbol", how="left")
    return df.reset_index(drop=True)


def latest_reference(adv_window: int = 20, vol_window: int = 20) -> pd.DataFrame:
    """For live use: per symbol, the most recent ADV, sigma, previous close and
    sector. Same shape/role as noi_tool.daily_bars.latest_reference()."""
    df = load_combined(adv_window, vol_window).sort_values(["symbol", "date"])
    last = df.groupby("symbol").tail(1).set_index("symbol")
    return last[["adv20", "sigma20", "prev_close", "sector", "sub_industry"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh-constituents", action="store_true", help="Re-pull the S&P 500 list even if cached")
    parser.add_argument("--history", action="store_true", help="Pull daily OHLCV history")
    parser.add_argument("--symbols", nargs="+", help="Limit history pull to these symbols (default: all S&P 500)")
    parser.add_argument("--years", type=int, default=2)
    args = parser.parse_args()

    constituents = fetch_constituents() if args.refresh_constituents or not CONSTITUENTS_PATH.exists() \
        else pd.read_csv(CONSTITUENTS_PATH)

    if args.history:
        symbols = args.symbols or constituents["symbol"].tolist()
        fetch_history(symbols, years=args.years)


if __name__ == "__main__":
    main()
