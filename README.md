# NYSE Auction Imbalance Monitor

A local analyst dashboard using Massive's live NYSE order-imbalance feed and
historical daily bars. The primary workflow is deterministic: pressure, relative
size, movement, and the exchange's indicative auction price. No regression,
machine-learning service, or Ollama is required to run it.

## Start locally

1. Install Python 3.11+ and uv; run `uv sync` in this folder.
2. Copy `.env.example` to `.env` and set `MASSIVE_API_KEY`. Your plan needs NOI access.
3. Run `uv run python -m noi_tool.dashboard --collect`, or double-click `start-noi.cmd`.
4. Open http://127.0.0.1:8765. Stop with Ctrl+C in the server terminal.

The dashboard refreshes every five seconds and exports the same analyzed snapshot
as CSV. Select a row for the relative-size chart and historical context. With no
captured events it shows the historical-symbol watchlist, with cached close, ADV,
and volatility. Auction fields remain unavailable until real events arrive. Old captures keep
their dates. The collector records all NOI events in `data/noi/YYYY-MM-DD.csv` and
continues while this process and computer are running. It does not restart after
reboot. The dashboard terminates its owned collector when stopped normally.
A collector lock prevents simultaneous writers to the same data directory.
If you already run the collector separately, omit `--collect`.

## Historical connection

The sibling `market-charts/reference_data` is detected automatically. Set
`MARKET_CHARTS_REFERENCE_DIR` in `.env` to override it. Otherwise the repository's
`reference_data` is used. This connection is read-only; raw files are not merged
or modified. Yahoo-style BRK-B symbols become BRK.B for matching.

ADV is the mean volume of the latest 20 completed cached trading sessions; daily
volatility is the standard deviation of 20 close-to-close returns. These are fixed
historical context, and do not require collecting 20 new NOI sessions. The supplied
snapshot ends 2026-10-22. Older references still produce outputs; the date and a
warning explain their age. They do not describe today's volatility with certainty.
Unknown symbols are collected and shown without invented historical context.

## Frozen rules (version 1)

- Pressure: positive imbalance = BUY, negative = SELL, zero = FLAT, missing = UNKNOWN.
- Size: absolute imbalance / historical ADV. Small below 0.1%; moderate from 0.1%
  to below 0.5%; large from 0.5% to below 2%; very large from 2% upward.
- Movement: compare absolute imbalance to the last observation at/before five
  minutes earlier, or the first available observation in a shorter window.
  Fractional change is `(abs(now)-abs(before))/(abs(before)+1)`; above +5% is
  growing, below -5% shrinking, otherwise unchanged. A first event has insufficient
  history. Side flips are flagged independently against the previous valid event.
- Pressure score: sign x bucket rank (1..4) x clipped convergence multiplier
  `clip(1+change, 0.5, 1.5)`. This summarizes pressure, not dollars or probability.
- Price movement: indicative-price change over that same observation window, in bps.
  Growing imbalance and an aligned price move support building directional pressure;
  conflicting moves are flagged. Shrinking imbalance indicates easing pressure.
- Final-price estimate: NYSE's current indicative auction estimate. No arbitrary
  bucket-to-dollar adjustment is applied. This estimate changes with live messages.
- Volatility impact scale: signed `sigma * sqrt(abs(imbalance)/ADV) * 10000`.
  Context only: it is uncalibrated and is not added to the indicative price.

These bucket boundaries and trend tolerances are our analytical conventions,
not exchange standards. ADV differs from average closing-auction volume. Final
paired shares in a pre-auction message are not guaranteed executed auction volume.

## Reproducible checks and research

`uv run pytest -q` checks boundaries, missing values, side changes, price movement,
historical calculations and research joins. Test fixtures stay separate from live
captures. `uv run python -m noi_tool.reversal_test` exports
`reports/reversal_join.csv`, retaining pending historical labels. Next-day returns
use the next observed historical bar; missing sessions and corporate actions need
review. A current price history is required to label newly captured days, not to
run the live monitor.

The older `dataset`, `model`, `reports`, `explainer`, and `daily_bars` modules remain
optional research utilities. The regression pipeline is not used by the dashboard.
Daily OHLC endpoints provide proxies, not verified exchange auction executions.
Exploratory OLS standard errors are not clustered by date/symbol. No predictive
performance is claimed without real outcome tests.

## Data and privacy

API keys stay in `.env` on the Python server. The dashboard binds only to localhost
and uses no external browser scripts. `.env`, data, reports, and model files are
ignored by Git. Local alerts do not send external messages. A fresh GitHub checkout
includes cached reference history, but never your credentials or captured events.

Reference: https://massive.com/blog/build-a-nyse-order-imbalance-tracker-with-massives-websocket-api

## Connection limits

A `max_connections` feed status means another stream is using the account allowance.
Stop older collectors/dashboard terminals with Ctrl+C before starting one instance.
The collector stops on connection-limit or authorization rejection; it does not
reconnect endlessly or turn a rejection into a success status. Ordinary network
interruptions retry after ten seconds. Idle connections publish a status heartbeat.
Changing the dashboard port does not increase the Massive connection allowance.
