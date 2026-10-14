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


def event_features(row: dict, adv20: float, prev_close: float, imb_change: float) -> dict:
    """Model features for one live NOI event (same definitions as dataset.build)."""
    imb, price = row["imbalance"], row["indicative_price"]
    at = int(row["auction_time"])
    hh, mm = map(int, row["time_et"][:5].split(":"))
    return {
        "side": float(np.sign(imb)),
        "imb_adv": imb / adv20,
        "imb_paired": imb / max(row["paired"] + abs(imb), 1),
        "imb_change": imb_change,
        "ind_vs_prev_bps": (price / prev_close - 1) * 1e4,
        "minutes_left": (at // 100 * 60 + at % 100) - (hh * 60 + mm),
        "log_adv": float(np.log10(adv20)),
        "score": float(signal_score(imb, adv20, imb_change)),
    }
