"""Filter MA20 oscillation on sessions when the two models disagree."""

from collections import deque
from math import isfinite
from statistics import fmean

from step15_trend_vote import TREND_WINDOW, trend_vote_decisions

HYSTERESIS = "hysteresis"
CONFIRMATION = "trend_confirmation"
BAND = 0.005
CONFIRM_SESSIONS = 2
BOUNDARY_TOLERANCE = 1e-12

PLAN = {
    "reference": "Step 15 trend vote using the last 20 closes including the signal date",
    "on_agreement": "Honor the two models' common cash or full position",
    "on_disagreement": "Half position if the chosen filtered trend is active; otherwise cash",
    "initial_state": "Inactive before the first available MA20; warm using all provided closes before the first signal",
    "hysteresis": {
        "band": BAND,
        "activation": "Close / MA20 - 1 strictly exceeds +0.5%",
        "deactivation": "Close / MA20 - 1 strictly below -0.5%",
        "otherwise": "Keep the previous trend state, including at a boundary",
        "relative_gap_tolerance_at_boundaries": BOUNDARY_TOLERANCE,
    },
    "trend_confirmation": {
        "consecutive_sessions": CONFIRM_SESSIONS,
        "activation": "Two consecutive closes strictly above their respective MA20",
        "deactivation": "Two consecutive closes at or below their respective MA20",
        "otherwise": "Keep the previous trend state",
    },
    "execution": "Decide using the signal-day close, trade at the next open only when the target changes",
}


def sideways_decisions(rows, base, direct):
    """Return causal target records with warmed trend states for both filters."""
    validated = trend_vote_decisions(rows, base, direct)
    signals = {signal["asof_date"]: signal for signal in validated}
    if len(signals) != len(validated):
        raise ValueError("Sideways filters require one signal per session")
    last_asof = validated[-1]["asof_date"]

    closes = deque(maxlen=TREND_WINDOW)
    band_active = confirmed_active = False
    above_streak = at_or_below_streak = 0
    outputs = {HYSTERESIS: [], CONFIRMATION: []}
    for row in rows:
        day = str(row["date"])
        if day > last_asof:
            break
        close = float(row["close"])
        if not isfinite(close) or close <= 0:
            raise ValueError(f"Invalid trend close on {day}")
        closes.append(close)
        gap = None
        if len(closes) == TREND_WINDOW:
            mean = fmean(closes)
            gap = close / mean - 1.0
            if gap > BAND + BOUNDARY_TOLERANCE:
                band_active = True
            elif gap < -BAND - BOUNDARY_TOLERANCE:
                band_active = False

            if close > mean:
                above_streak += 1
                at_or_below_streak = 0
                if above_streak >= CONFIRM_SESSIONS:
                    confirmed_active = True
            else:
                above_streak = 0
                at_or_below_streak += 1
                if at_or_below_streak >= CONFIRM_SESSIONS:
                    confirmed_active = False

        if day not in signals:
            continue
        signal = signals[day]
        raw = signal["raw_consensus_weight"]
        for name, active in ((HYSTERESIS, band_active), (CONFIRMATION, confirmed_active)):
            outputs[name].append({
                **signal,
                "model": name,
                "strategy_weight": (0.5 if active else 0.0) if raw == 0.5 else raw,
                "raw_trend_vote_weight": signal["strategy_weight"],
                "filtered_trend_holds": active,
                "trend_gap": gap,
                "above_ma20_streak": above_streak,
                "at_or_below_ma20_streak": at_or_below_streak,
            })
    return outputs
