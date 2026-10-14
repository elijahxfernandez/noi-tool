"""A7 - Report charts (PNG) for the auction pressure dashboard.

    uv run python -m noi_tool.reports pressure   [--date 2026-10-14] [--top 15]
    uv run python -m noi_tool.reports trajectory --symbol JPM [--date 2026-10-14]
    uv run python -m noi_tool.reports scorecard

  pressure    which stocks had the biggest closing imbalance vs normal volume (buy vs sell)
  trajectory  one stock: indicative price path and imbalance into the close, vs the final print
  scorecard   model vs "trust NYSE" baseline on held-out days, plus predicted vs actual
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from noi_tool.daily_bars import load as load_daily  # noqa: E402
from noi_tool.dataset import FEATURES, NOI_DIR, OUT_PATH as TRAINING_PATH  # noqa: E402
from noi_tool.model import Ridge, split_by_date  # noqa: E402
from shared.config import REPORTS_DIR  # noqa: E402

# Reference palette (dataviz skill): diverging blue/red poles, recessive chrome
BUY, SELL, NEUTRAL = "#2a78d6", "#e34948", "#898781"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e8e7e3", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": NEUTRAL, "ytick.color": NEUTRAL,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "font.size": 10, "axes.axisbelow": True,
})


def _noi_file(date):
    path = NOI_DIR / f"{date}.csv" if date else max(NOI_DIR.glob("*.csv"), default=None)
    if not path or not path.exists():
        raise SystemExit(f"No NOI file ({path}). Run the collector first.")
    return path


def _save(fig, name):
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def pressure(date=None, top=15):
    path = _noi_file(date)
    noi = pd.read_csv(path).query("auction_type == 'C'")
    last = noi.sort_values("ts_ns").groupby("symbol").last()
    daily = load_daily()
    adv = daily[daily["date"] < path.stem].sort_values("date").groupby("symbol").tail(20).groupby("symbol")["volume"].mean()
    last = last.join(adv.rename("adv"), how="inner")
    last["pct_adv"] = last["imbalance"] / last["adv"] * 100
    pick = last.reindex(last["pct_adv"].abs().sort_values(ascending=False).index).head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(pick) + 1.5))
    colors = [BUY if v > 0 else SELL for v in pick["pct_adv"]]
    ax.barh(pick.index, pick["pct_adv"], color=colors, height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=NEUTRAL, linewidth=1)
    for y, v in enumerate(pick["pct_adv"]):
        ax.text(v, y, f" {'BUY' if v > 0 else 'SELL'} {abs(v):.1f}% ", va="center",
                ha="left" if v > 0 else "right", color=INK_2, fontsize=9)
    ax.set_xlabel("Closing imbalance as % of 20-day average daily volume")
    ax.set_title(f"Closing auction pressure - {path.stem}", loc="left")
    ax.grid(axis="y", visible=False)
    lim = pick["pct_adv"].abs().max() * 1.35
    ax.set_xlim(-lim, lim)
    _save(fig, f"pressure_{path.stem}.png")


def trajectory(symbol, date=None):
    path = _noi_file(date)
    ev = pd.read_csv(path).query("symbol == @symbol and auction_type == 'C'").sort_values("ts_ns")
    if ev.empty:
        raise SystemExit(f"No closing imbalance events for {symbol} on {path.stem}.")
    t = pd.to_datetime(path.stem + " " + ev["time_et"])
    final = load_daily().query("symbol == @symbol and date == @path.stem")["close"]

    # Two stacked panels sharing time - never two y-scales on one chart
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    top.plot(t, ev["indicative_price"], color=BUY, linewidth=2, label="Indicative price")
    if not final.empty:
        top.axhline(final.iloc[0], color=INK_2, linewidth=1.2, linestyle="--")
        top.text(t.iloc[0], final.iloc[0], f" Final close ${final.iloc[0]:.2f}", va="bottom", color=INK_2, fontsize=9)
    top.set_ylabel("Price ($)")
    top.set_title(f"{symbol} closing auction - {path.stem}", loc="left")

    colors = [BUY if v > 0 else SELL for v in ev["imbalance"]]
    bot.bar(t, ev["imbalance"], width=pd.Timedelta(seconds=5), color=colors)
    bot.axhline(0, color=NEUTRAL, linewidth=1)
    bot.set_ylabel("Imbalance (shares)\nbuy + / sell -")
    bot.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bot.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    _save(fig, f"trajectory_{symbol}_{path.stem}.png")


def scorecard():
    results_path = REPORTS_DIR / "noi_ridge_results.json"
    if not results_path.exists():
        raise SystemExit("Train the model first: uv run python -m noi_tool.model")
    res = json.loads(results_path.read_text())
    df = pd.read_csv(TRAINING_PATH)
    train, test = split_by_date(df, res["test_days"] / (res["train_days"] + res["test_days"]))
    m = Ridge(res["lambda"]).fit(train[FEATURES].to_numpy(float), train["target_bps"].to_numpy(float))
    pred, actual = m.predict(test[FEATURES].to_numpy(float)), test["target_bps"].to_numpy(float)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 3]})
    names = ["Trust NYSE\n(baseline)", "Ridge model"]
    maes = [res["baseline_trust_nyse"]["mae_bps"], res["ridge_closed_form"]["mae_bps"]]
    left.bar(names, maes, color=[NEUTRAL, BUY], width=0.55)
    for x, v in enumerate(maes):
        left.text(x, v, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=10)
    left.set_ylabel("Avg error, bps (lower is better)")
    left.set_title("Error on held-out days", loc="left")
    left.grid(axis="x", visible=False)

    lim = np.percentile(np.abs(np.concatenate([pred, actual])), 99)
    right.scatter(pred, actual, s=10, color=BUY, alpha=0.35, linewidths=0)
    right.plot([-lim, lim], [-lim, lim], color=NEUTRAL, linewidth=1, linestyle="--")
    right.set_xlim(-lim, lim)
    right.set_ylim(-lim, lim)
    right.set_xlabel("Predicted gap (bps)")
    right.set_ylabel("Actual gap: final vs indicative (bps)")
    hit = res["ridge_closed_form"]["direction_hit"]
    right.set_title(f"Predicted vs actual - direction right {hit:.0%}", loc="left")
    fig.suptitle(f"Model scorecard - {res['test_days']} test days, {res['test_rows']:,} snapshots", x=0.01, ha="left", color=INK_2)
    _save(fig, "scorecard.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chart", choices=["pressure", "trajectory", "scorecard"])
    parser.add_argument("--date")
    parser.add_argument("--symbol")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    if args.chart == "pressure":
        pressure(args.date, args.top)
    elif args.chart == "trajectory":
        if not args.symbol:
            parser.error("trajectory needs --symbol")
        trajectory(args.symbol.upper(), args.date)
    else:
        scorecard()


if __name__ == "__main__":
    main()
