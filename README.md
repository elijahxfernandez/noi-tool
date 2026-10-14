# Massive Analyst Lab

Two analyst-training tools built on [Massive](https://massive.com) market data.

1. **NYSE Order Imbalance (NOI) tool:** tracks buy/sell pressure before NYSE auctions. Uses historical data and a regression model to estimate where the final auction price lands based on that pressure.
2. **Off-Exchange Flow tool:** tracks dark pool trading against exchange trading and whether it comes before or after price moves. *(coming next)*

## Setup

```bash
uv sync
cp .env.example .env      # add your MASSIVE_API_KEY
uv run pytest -q
```

Your Massive plan needs NYSE order imbalance (NOI) access and REST daily aggregates.

## NOI tool: run order

| Step | Command | When |
|---|---|---|
| 1. Collect imbalances | `uv run python -m noi_tool.collector` (add `--alerts` once you have daily bars) | Every trading day, ~9:00 to 16:05 ET |
| 2. Daily bars (labels + ADV) | `uv run python -m noi_tool.daily_bars --start 2026-08-01` | After each close (first run: ~30 days back) |
| 3. Build training set | `uv run python -m noi_tool.dataset` | After a few days of collection |
| 4. Train + score model | `uv run python -m noi_tool.model` | Same |
| 5. Charts | `uv run python -m noi_tool.reports pressure` / `trajectory --symbol JPM` / `scorecard` | Any time |
| 6. AI explainer | `uv run python -m noi_tool.explainer --symbol JPM` | Needs Ollama (`ollama pull llama3.2:3b`) |

Outputs: `data/` (CSVs, saved model) and `reports/` (charts, results JSON). Both are gitignored.

## How the NOI model works

- **Target:** `(final auction price - NYSE indicative price) / indicative`, in bps. The model has to beat the baseline of trusting NYSE's own estimate (predicting 0).
- **Features:** side, imbalance / 20-day ADV, imbalance / paired shares, 5-minute convergence, indicative vs previous close, minutes to auction, log ADV, signal score.
- **Signal score:** direction x magnitude (capped imbalance/ADV) x convergence.
- **Model:** Ridge regression written from the math, solved both in closed form `w = (Z'Z + n*lambda*I)^-1 Z'y` and by gradient descent (they must agree). Lambda is tuned on later training days; testing is walk-forward by date.
- **Explainer:** the regression makes the prediction; a local LLM (Ollama) explains it for an investor profile and lays out options and trade-offs.

## Try it on sample data

Point the tools at a separate folder so sample data never mixes with real data:

```bash
export LAB_DATA_DIR=.sample/data LAB_REPORTS_DIR=.sample/reports
```
