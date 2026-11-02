import numpy as np
import pandas as pd
from noi_tool.dashboard import analyze
from noi_tool.signal import size_bucket


def reference():
    return pd.DataFrame({"adv20":[1000000.], "prev_close":[99.], "sigma20":[.02], "reference_date":["2026-10-22"]}, index=["JPM"])


def events(values):
    return pd.DataFrame([dict(date="2026-11-02", symbol="JPM", auction_type="C", ts_ns=i*300_000_000_000+1, time_et="15:55:00", imbalance=v, paired=100000, indicative_price=100.+i) for i,v in enumerate(values)])


def test_bucket_boundaries():
    assert list(size_bucket(np.array([999,1000,4999,5000,19999,20000]), 1000000)) == [0,1,1,2,2,3]


def test_pressure_growth_price_and_export_fields():
    row=analyze(events([10000,20000]),reference())[0]
    assert row["pressure"] == "BUY"
    assert row["bucket"] == "very large"
    assert row["trend"] == "growing"
    assert row["assessment"] == "Building upward pressure"
    assert row["estimated_final"] == 101.
    assert row["score"] == 6.
    assert np.isclose(row["price_change_bps"],100.)


def test_flip_zero_missing_and_no_reference():
    assert analyze(events([1000,-1000]),reference())[0]["side_flip"]
    assert analyze(events([0]),reference())[0]["pressure"] == "FLAT"
    row=analyze(events([None]),reference())[0]
    assert row["pressure"] == "UNKNOWN" and row["bucket"] == "unavailable"
    assert analyze(events([2000]),pd.DataFrame())[0]["bucket"] == "unavailable"
    assert analyze(pd.DataFrame(),reference()) == []


def test_historical_watchlist_has_real_context_without_fabricated_noi():
    from noi_tool.dashboard import with_historical_watchlist
    rows=with_historical_watchlist([],reference())
    assert rows[0]["symbol"] == "JPM"
    assert rows[0]["historical_close"] == 99.
    assert rows[0]["historical_adv"] == 1000000.
    assert rows[0]["imbalance"] is None
    assert rows[0]["estimated_final"] is None
    assert rows[0]["data_status"] == "historical only"
    captured=analyze(events([100]),reference())
    assert len(with_historical_watchlist(captured,reference())) == 1
