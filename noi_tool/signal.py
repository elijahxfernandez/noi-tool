"""Signal score = direction x magnitude x convergence.

  direction    +1 buy-side imbalance, -1 sell-side
  magnitude    |imbalance| / ADV, capped so one extreme name can't dominate, scaled 0..1
  convergence  >1 if the imbalance grew over the last few minutes (pressure building),
               <1 if it shrank (contra-side liquidity arriving, price settling)

Positive = buy pressure, negative = sell pressure, bigger = stronger.
"""

import numpy as np

MAGNITUDE_CAP = 0.05  # 5% of a day's normal volume is already a very large auction imbalance


def signal_score(imbalance, adv20, imb_change):
    direction = np.sign(imbalance)
    magnitude = np.minimum(np.abs(imbalance) / adv20, MAGNITUDE_CAP) / MAGNITUDE_CAP
    convergence = np.clip(1 + imb_change, 0.5, 1.5)
    return direction * magnitude * convergence


# Uncalibrated scale factor; auction imbalances are not executed metaorders.
IMPACT_Y = 1.0


def expected_price_impact_bps(imbalance, adv20, sigma20, y: float = IMPACT_Y):
    """Theory-grounded expected move, in basis points, from the square-root market-
    impact model: impact = Y * sigma * sqrt(|imbalance| / ADV), signed by direction.

        imbalance  signed order imbalance (shares)
        adv20      20-day average daily volume (shares)      - how big ADV-wise
        sigma20    20-day std dev of daily % returns          - this stock's normal swing size
        y          calibration constant (see IMPACT_Y)

    This is an uncalibrated impact scale, not a validated price forecast.
    Auction imbalance is not executed order size; empirical validation is required.
    """
    direction = np.sign(imbalance)
    participation = np.abs(imbalance) / adv20
    return direction * y * sigma20 * np.sqrt(participation) * 1e4


def event_features(row: dict, adv20: float, prev_close: float, imb_change: float, sigma20: float | None = None) -> dict:
    """Model features for one live NOI event (same definitions as dataset.build).
    Pass sigma20 (from shared.market_reference) to also get expected_move_bps."""
    imb, price = row["imbalance"], row["indicative_price"]
    at = int(row["auction_time"])
    hh, mm = map(int, row["time_et"][:5].split(":"))
    feats = {
        "side": float(np.sign(imb)),
        "size_bucket": float(size_bucket(imb, adv20)),
        "imb_adv": imb / adv20,
        "imb_paired": imb / max(row["paired"] + abs(imb), 1),
        "imb_change": imb_change,
        "ind_vs_prev_bps": (price / prev_close - 1) * 1e4,
        "minutes_left": (at // 100 * 60 + at % 100) - (hh * 60 + mm),
        "log_adv": float(np.log10(adv20)),
        "score": float(signal_score(imb, adv20, imb_change)),
    }
    if sigma20 and sigma20 > 0:
        feats["expected_move_bps"] = float(expected_price_impact_bps(imb, adv20, sigma20))
    return feats


# Frozen ADV boundaries: small <0.1%, moderate <0.5%, large <2%, very large >=2%.
def size_bucket(imbalance, adv20):
    return np.searchsorted([0.001, 0.005, 0.02], np.abs(imbalance) / adv20, side="right")
