import numpy as np
import pandas as pd
from shared import market_reference as ref


def test_live_reference_uses_latest_completed_close_and_volume(tmp_path, monkeypatch):
    dates = pd.bdate_range("2025-01-01", periods=30)
    bars = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": np.arange(30) + 100., "volume": np.arange(30) + 1000.})
    bars.to_csv(tmp_path / "BRK-B.csv", index=False)
    monkeypatch.setattr(ref, "DAILY_DIR", tmp_path)
    monkeypatch.setattr(ref, "CONSTITUENTS_PATH", tmp_path / "missing.csv")
    historical = ref.load_combined()
    assert historical.iloc[-1]["prev_close"] == 128.
    assert historical.iloc[-1]["adv20"] == bars.volume.iloc[-21:-1].mean()
    live = ref.latest_reference().loc["BRK.B"]
    assert live.prev_close == 129.
    assert live.adv20 == bars.volume.tail(20).mean()
    assert np.isfinite(live.sigma20)


def test_research_join_preserves_pending_labels(monkeypatch):
    from noi_tool import reversal_test as research
    monkeypatch.setattr(research, "closing_imbalances", lambda: pd.DataFrame({"symbol": ["JPM", "JPM"], "date": ["2026-10-23", "2026-10-26"], "imbalance": [1000, -2000]}))
    monkeypatch.setattr(research, "load_combined", lambda: pd.DataFrame({"symbol": ["JPM", "JPM"], "date": ["2026-10-23", "2026-10-26"], "close": [100., 102.], "adv20": [100000., 100000.]}))
    result = research.build_dataset()
    assert np.isclose(result.iloc[0].next_day_return_pct, 2.)
    assert result.iloc[0].label_status == "ready"
    assert result.iloc[1].label_status == "pending_history"
