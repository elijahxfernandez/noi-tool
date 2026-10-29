"""AI explainer - turns the auction signal + model prediction into a plain-English read
tailored to an investor profile, using a small local LLM through Ollama.

The regression model makes the price prediction; the LLM only explains it and lays out
options and trade-offs for the profile. It does not make the call.

Setup (once):  install Ollama from ollama.com, then:  ollama pull llama3.2:3b
Run:
    uv run python -m noi_tool.explainer --symbol JPM               (asks the questionnaire)
    uv run python -m noi_tool.explainer --symbol JPM --profile     (reuse saved answers)
    uv run python -m noi_tool.explainer --symbol JPM --date 2026-10-14 --model qwen3:4b
"""

import argparse
import json
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from noi_tool.dataset import FEATURES, LOOKBACK_MIN, NOI_DIR
from noi_tool.model import MODEL_PATH, Ridge
from noi_tool.signal import event_features
from shared.config import DATA_DIR


def _load_reference():
    """Prefer the free S&P 500 history (has sigma20/volatility + sector); fall back
    to Massive's REST daily bars (ADV + previous close only) if that's unavailable."""
    try:
        from shared.market_reference import latest_reference
        return latest_reference()
    except SystemExit:
        from noi_tool.daily_bars import latest_reference
        return latest_reference()

PROFILE_PATH = DATA_DIR / "investor_profile.json"
OLLAMA_URL = "http://localhost:11434/api/chat"

QUESTIONS = [
    ("horizon", "Time horizon?", ["same day", "days to weeks", "months to years"]),
    ("risk", "Risk comfort?", ["low", "medium", "high"]),
    ("position", "Current position in this stock?", ["none", "long (own shares)", "short"]),
    ("options", "Options use?", ["none", "protective puts", "covered calls", "both"]),
]

SYSTEM_PROMPT = """You explain NYSE auction order-imbalance signals to an investor in plain English.
Rules:
- Use ONLY the numbers given. Do not invent prices, news, or history.
- expected_move_bps (if present) comes from a market-impact formula (size and this stock's normal volatility),
  not a trained model - describe it as a theory-based estimate, not a prediction or a certainty.
- model_predicted_gap_bps (if present) comes from a regression model trained on past auctions; describe it
  as a separate, statistically-fit estimate, not a certainty. Don't present the two as the same kind of number.
- Explain what the imbalance direction, size vs. normal volume, and convergence mean for THIS stock right now.
- Tailor it to the investor profile, especially the time horizon: an auction signal mostly affects today's
  closing/opening print and part of the move often reverses the next day, so for long horizons frame it as
  entry/exit price timing, not a trend call.
- Lay out 2-3 options with the trade-offs of each (including doing nothing). If the investor uses options,
  explain how the predicted move relates to their position (e.g. a protective put gaining or losing value).
- Do not tell them what to do. End with one line: "This is an educational read of the data, not investment advice."
- Keep it under 250 words."""


def ask_profile() -> dict:
    profile = {}
    for key, question, choices in QUESTIONS:
        print(f"\n{question}")
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        while True:
            pick = input("> ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(choices):
                profile[key] = choices[int(pick) - 1]
                break
    profile["notes"] = input("\nAnything else (e.g. goal, cost basis)? Enter to skip\n> ").strip()
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))
    return profile


def snapshot(symbol: str, date: str | None) -> tuple[dict, float]:
    """Latest NOI event for the symbol on that day, plus its 5-minute convergence."""
    path = NOI_DIR / f"{date}.csv" if date else max(NOI_DIR.glob("*.csv"), default=None)
    if not path or not path.exists():
        raise SystemExit(f"No NOI file found ({path}). Run the collector first.")
    ev = pd.read_csv(path).query("symbol == @symbol").dropna(subset=["indicative_price"]).sort_values("ts_ns")
    if ev.empty:
        raise SystemExit(f"No imbalance events for {symbol} in {path.name}.")
    last = ev.iloc[-1].to_dict()
    same = ev[ev["auction_type"] == last["auction_type"]]
    cutoff = last["ts_ns"] - LOOKBACK_MIN * 60 * 1_000_000_000
    earlier = same[same["ts_ns"] <= cutoff]
    before = (earlier.iloc[-1] if not earlier.empty else same.iloc[0])["imbalance"]
    change = float(np.clip((abs(last["imbalance"]) - abs(before)) / (abs(before) + 1), -1, 1))
    last["date"] = path.stem
    return last, change


def build_facts(symbol: str, row: dict, change: float) -> dict:
    ref = _load_reference()
    if symbol not in ref.index:
        raise SystemExit(f"No daily bars for {symbol}. Fetch daily bars first.")
    adv, prev_close = ref.at[symbol, "adv20"], ref.at[symbol, "prev_close"]
    sigma = ref.at[symbol, "sigma20"] if "sigma20" in ref.columns else None
    feats = event_features(row, adv, prev_close, change, sigma)
    price = row["indicative_price"]
    facts = {
        "symbol": symbol, "date": row["date"], "time_et": row["time_et"],
        "auction": {"M": "opening", "C": "closing", "H": "halt"}.get(row["auction_type"], row["auction_type"]),
        "imbalance_shares": int(row["imbalance"]), "side": "buy" if row["imbalance"] > 0 else "sell",
        "imbalance_pct_of_avg_daily_volume": round(abs(feats["imb_adv"]) * 100, 2),
        "paired_shares": int(row["paired"]),
        "convergence": "growing (pressure building)" if change > 0.05 else "shrinking (settling)" if change < -0.05 else "steady",
        "indicative_price": round(price, 2), "previous_close": round(prev_close, 2),
        "signal_score": round(feats["score"], 2),
    }
    if "sector" in ref.columns and pd.notna(ref.at[symbol, "sector"]):
        facts["sector"] = ref.at[symbol, "sector"]
    if "expected_move_bps" in feats:
        facts["expected_move_bps"] = round(feats["expected_move_bps"], 1)
        facts["expected_move_note"] = "theory-based estimate (square-root price-impact model), not a guarantee"
    if MODEL_PATH.exists():
        pred_bps = float(Ridge.load(MODEL_PATH).predict(np.array([[feats[f] for f in FEATURES]]))[0])
        facts["model_predicted_gap_bps"] = round(pred_bps, 1)
        facts["model_predicted_final_price"] = round(price * (1 + pred_bps / 1e4), 2)
    return facts


def ask_ollama(model: str, facts: dict, profile: dict) -> str:
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Signal data:\n{json.dumps(facts, indent=2)}\n\nInvestor profile:\n{json.dumps(profile, indent=2)}"},
        ],
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", help="YYYY-MM-DD (default: latest NOI file)")
    parser.add_argument("--profile", action="store_true", help="Reuse saved questionnaire answers")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    profile = json.loads(PROFILE_PATH.read_text()) if args.profile and PROFILE_PATH.exists() else ask_profile()
    row, change = snapshot(symbol, args.date)
    facts = build_facts(symbol, row, change)

    print("\n--- Signal ---")
    for k, v in facts.items():
        print(f"  {k:36} {v}")
    print("\n--- Explanation ---")
    try:
        print(ask_ollama(args.model, facts, profile))
    except urllib.error.URLError:
        print(f"Ollama isn't reachable at {OLLAMA_URL}. Install it from ollama.com, run `ollama pull {args.model}`, and try again.")


if __name__ == "__main__":
    main()
