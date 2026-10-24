import numpy as np
import pandas as pd

from noi_tool.dataset import FEATURES, build
from noi_tool.model import Ridge
from noi_tool.signal import signal_score


def _toy_regression(n=500, p=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * rng.uniform(1, 50, size=p)
    y = X @ rng.normal(size=p) + 3 + rng.normal(scale=2, size=n)
    return X, y


def test_closed_form_matches_independent_solver():
    # Ridge == ordinary least squares on data augmented with sqrt(n*lam)*I rows
    X, y = _toy_regression()
    lam, n, p = 0.1, len(y), X.shape[1]
    ours = Ridge(lam).fit(X, y)
    Z = (X - X.mean(0)) / X.std(0)
    A = np.vstack([Z, np.sqrt(n * lam) * np.eye(p)])
    b = np.concatenate([y - y.mean(), np.zeros(p)])
    w_ref = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(ours.w, w_ref, atol=1e-8)
    assert np.isclose(ours.b, y.mean())


def test_gradient_descent_converges_to_closed_form():
    X, y = _toy_regression()
    closed = Ridge(0.05).fit(X, y, method="closed")
    gd = Ridge(0.05).fit(X, y, method="gd")
    assert np.max(np.abs(closed.w - gd.w)) < 1e-4
    assert gd.loss_history[-1] <= gd.loss_history[0]


def test_signal_score_direction_and_convergence():
    s = signal_score(np.array([10_000, -10_000, 10_000]), np.array([1e6] * 3), np.array([0.0, 0.0, 0.5]))
    assert s[0] > 0 and s[1] < 0
    assert s[2] > s[0]  # growing imbalance = stronger signal
    assert abs(signal_score(1e9, 1e6, 0.0)) == 1.0  # capped


def test_build_dataset_labels_and_features():
    noi = pd.DataFrame([
        # JPM close auction: buy imbalance shrinking from 50k to 20k between 15:50 and 15:55
        {"ts_ns": 1, "time_et": "15:50:30.000", "symbol": "JPM", "auction_type": "C", "imbalance": 50_000, "paired": 200_000, "indicative_price": 200.0},
        {"ts_ns": 2, "time_et": "15:54:59.000", "symbol": "JPM", "auction_type": "C", "imbalance": 20_000, "paired": 300_000, "indicative_price": 201.0},
        {"ts_ns": 3, "time_et": "15:56:00.000", "symbol": "JPM", "auction_type": "C", "imbalance": 99_999, "paired": 300_000, "indicative_price": 999.0},  # after checkpoint, must be ignored
    ]).assign(date="2026-10-14")
    daily = pd.DataFrame([{"date": "2026-10-14", "symbol": "JPM", "open": 198.0, "close": 202.0, "adv20": 10_000_000, "prev_close": 199.0}])

    df = build(noi, daily)
    row = df[df["checkpoint"] == "15:55"].iloc[0]
    assert row["indicative_price"] == 201.0  # no look-ahead to the 15:56 event
    assert np.isclose(row["target_bps"], (202 / 201 - 1) * 1e4)
    assert np.isclose(row["imb_adv"], 0.002)
    assert row["imb_change"] < 0  # shrinking
    assert not df[FEATURES].isna().any().any()


def _event(t, imb, price=100.0, ts=None):
    hh, mm, ss = map(int, t.split(":"))
    return {"ts_ns": ts or (hh * 3600 + mm * 60 + ss) * 1_000_000_000, "time_et": t + ".000", "symbol": "JPM",
            "auction_type": "C", "auction_time": 1600, "imbalance": imb, "paired": 100_000, "indicative_price": price}


def test_alerts_big_flip_drift_and_model():
    from noi_tool.alerts import AlertEngine

    ref = pd.DataFrame({"adv20": [1_000_000], "prev_close": [100.0]}, index=["JPM"])
    X, y = _toy_regression(p=len(FEATURES))
    engine = AlertEngine(ref, model=Ridge(1.0).fit(X, y))
    engine.webhook = None

    assert engine.check(_event("15:50:10", 5_000), "d") == []                  # 0.5% of ADV: quiet
    big = engine.check(_event("15:52:00", 20_000), "d")                        # 2% of ADV
    assert len(big) == 1 and big[0].startswith("BIG") and "model:" in big[0]
    assert engine.check(_event("15:53:00", 25_000), "d") == []                 # BIG fires once
    flip = engine.check(_event("15:56:00", -8_000), "d")
    assert any(a.startswith("FLIP") for a in flip)
    drift = engine.check(_event("15:57:00", -8_000, price=104.0), "d")          # +400 bps vs prev close
    assert any(a.startswith("DRIFT") for a in drift)


def test_reader_direction_convergence_magnitude():
    from noi_tool.reader import ImbalanceReader

    reader = ImbalanceReader(adv_lookup={"JPM": 1_000_000})

    def ev(t_sec, imb, paired=50_000, symbol="JPM"):
        return {"ts_ns": t_sec * 1_000_000_000, "time_et": "15:5X:XX.000", "symbol": symbol,
                "auction_type": "C", "imbalance": imb, "paired": paired, "indicative_price": 100.0}

    assert reader.direction(5_000) == "BUY"
    assert reader.direction(-5_000) == "SELL"
    assert reader.direction(0) == "FLAT"

    # first event: nothing to compare to yet -> baseline is itself -> "steady"
    line1 = reader.read(ev(0, 10_000))
    assert "trend=steady" in line1 and "BUY" in line1

    # imbalance grows within the 5-min window -> "growing"
    line2 = reader.read(ev(60, 20_000))
    assert "trend=growing" in line2

    # imbalance shrinks back down, still within window of the first event -> "shrinking"
    line3 = reader.read(ev(120, 5_000))
    assert "trend=shrinking" in line3

    # magnitude: 20,000 / 1,000,000 ADV = 2% -> "very large"
    assert "very large" in line2

    # a symbol with no ADV on record falls back to imbalance/paired ratio
    reader2 = ImbalanceReader(adv_lookup={})
    line = reader2.read(ev(0, 40_000, paired=20_000, symbol="ZZZ"))
    assert "imb/paired" in line
