"""Use 20- and 60-session trends to size exposure when the models disagree."""

from math import isfinite
from statistics import fmean

from step15_trend_vote import TREND_WINDOW, trend_vote_decisions


LONG_TREND_WINDOW = 60
DUAL_HORIZON_GATE = "dual_horizon_gate"
ALIGNED_TREND_WEIGHT = "aligned_trend_weight"
PLAN = {
    "sources": ["original TCN buffer", "direct utility model", "20- and 60-session close trends"],
    "trend_windows": [TREND_WINDOW, LONG_TREND_WINDOW],
    "trend_rule": "Signal-day close must strictly exceed the trailing mean including that close",
    "insufficient_history": "The unavailable trend is inactive",
    "models_agree": "Retain their common zero or full target",
    "models_disagree": {
        DUAL_HORIZON_GATE: "Half if at least one trend is positive, otherwise zero",
        ALIGNED_TREND_WEIGHT: "Zero, half, or full according to the number of positive trends",
    },
    "execution": "Trade at the next open only when the target changes",
}


def trend_refinement_decisions(rows, base, direct):
    """Return aligned causal targets and signal-day trend diagnostics for both rules."""
    validated = trend_vote_decisions(rows, base, direct)
    row_indices = {str(row["date"]): i for i, row in enumerate(rows)}
    outputs = {DUAL_HORIZON_GATE: [], ALIGNED_TREND_WEIGHT: []}
    for signal in validated:
        index = row_indices[signal["asof_date"]]
        long_ready = index + 1 >= LONG_TREND_WINDOW
        long_average = None
        long_holds = False
        if long_ready:
            start = index + 1 - LONG_TREND_WINDOW
            closes = [float(row["close"]) for row in rows[start:index + 1]]
            if not all(isfinite(close) and close > 0 for close in closes):
                raise ValueError(f"Invalid trend close on or before {signal['asof_date']}")
            long_average = fmean(closes)
            long_holds = closes[-1] > long_average
        raw_target = signal["raw_consensus_weight"]
        trend_count = int(signal["trend_holds"]) + int(long_holds)
        targets = {
            DUAL_HORIZON_GATE: 0.5 if trend_count else 0.0,
            ALIGNED_TREND_WEIGHT: trend_count / 2,
        }
        diagnostics = {
            "asof_date": signal["asof_date"],
            "target_date": signal["target_date"],
            "fit_asof": signal["fit_asof"],
            "max_training_label_date": signal["max_training_label_date"],
            "raw_consensus_weight": raw_target,
            "tcn_holds": signal["tcn_holds"],
            "utility_holds": signal["utility_holds"],
            "trend20_average": signal["trend_average"],
            "trend20_window_ready": signal["trend_window_ready"],
            "trend20_holds": signal["trend_holds"],
            "trend60_average": long_average,
            "trend60_window_ready": long_ready,
            "trend60_holds": long_holds,
        }
        for name, target in targets.items():
            outputs[name].append({
                **diagnostics,
                "model": name,
                "strategy_weight": target if raw_target == 0.5 else raw_target,
            })
    return outputs
