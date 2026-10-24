# Reference data

Free historical data, committed as a static snapshot (not meant to be re-pulled daily).

- **`sp500_constituents.csv`** — S&P 500 symbol, company name, GICS sector and sub-industry. Source: Wikipedia's [List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
- **`daily/<SYMBOL>.csv`** — 5 years of daily open/high/low/close/volume per symbol. Source: Yahoo Finance via `yfinance`.

Pulled: 2026-10-23 · 503 symbols · 2021-10-25 to 2026-10-22

## Refresh

```bash
uv run python -m shared.market_reference --refresh-constituents --history --years 5
```

## Use

```python
from shared.market_reference import load_combined
df = load_combined()  # one row per (symbol, date): OHLCV, adv20, prev_close, sector, sub_industry
```

Same shape as `noi_tool.daily_bars.load()`, so either source works with the same downstream code. This is free, unofficial data (Yahoo Finance via yfinance) — fine for personal research/backtesting, not for redistribution as a data product.
