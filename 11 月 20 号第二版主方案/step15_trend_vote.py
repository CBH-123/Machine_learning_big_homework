"""Use the 20-session trend to gate exposure when the two models disagree."""

from math import isfinite
from statistics import fmean

from step13_adaptive_expert import TREND_WINDOW
from step14_consensus import PLAN as CONSENSUS_PLAN, consensus_decisions


PLAN = {
    **CONSENSUS_PLAN,
    "sources": [*CONSENSUS_PLAN["sources"], f"{TREND_WINDOW}-session close trend"],
    "target_weights": {
        "both_models_hold": 1.0,
        "both_models_cash": 0.0,
        "models_disagree_above_mean": 0.5,
        "models_disagree_at_or_below_mean": 0.0,
    },
    "trend_rule": f"The signal day's close is strictly above the mean of the latest {TREND_WINDOW} closes",
    "insufficient_trend_history": "Treat trend as inactive",
}


def trend_vote_decisions(rows, base, direct):
    """Apply the trend gate using the signal day's close and preceding closes."""
    dates = [str(row["date"]) for row in rows]
    if any(a >= b for a, b in zip(dates, dates[1:])):
        raise ValueError("Trend vote requires strictly ordered trading dates")
    row_indices = {day: i for i, day in enumerate(dates)}
    decisions = consensus_decisions(base, direct)
    for decision in decisions:
        asof, target = decision["asof_date"], decision["target_date"]
        index = row_indices.get(asof)
        if index is None or (index + 1 < len(dates) and dates[index + 1] != target):
            raise ValueError(f"Trend vote requires an aligned prior session: {asof}, {target}")
        ready = index + 1 >= TREND_WINDOW
        trend_average = None
        trend_holds = False
        if ready:
            closes = [float(row["close"]) for row in rows[index - TREND_WINDOW + 1:index + 1]]
            if not all(isfinite(close) and close > 0 for close in closes):
                raise ValueError(f"Invalid trend close on or before {asof}")
            trend_average = fmean(closes)
            trend_holds = closes[-1] > trend_average
        raw_target = decision["strategy_weight"]
        decision.update(
            model="trend_vote",
            raw_consensus_weight=raw_target,
            strategy_weight=0.0 if raw_target == 0.5 and not trend_holds else raw_target,
            trend_average=trend_average,
            trend_window_ready=ready,
            trend_holds=trend_holds,
        )
    return decisions
