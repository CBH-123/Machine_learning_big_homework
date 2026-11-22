"""Keep the current target through the first disagreement between the models."""

from step14_consensus import consensus_decisions


HALF_WEIGHT = 0.5
REQUIRED_DISAGREEMENTS = 2
PLAN = {
    "base_rule": "Step 14 consensus",
    "initial_target_weight": 0.0,
    "immediate_agreement_weights": [0.0, 1.0],
    "disagreement_weight": HALF_WEIGHT,
    "consecutive_disagreement_sessions": REQUIRED_DISAGREEMENTS,
    "disagreement_source_can_alternate": True,
    "streak_reset": "agreement or a missing session",
    "execution": "single account; rebalance only when the target changes; trade at the next open",
    "rebalance_on_weight_change_only": True,
    "uses_calendar_features": False,
}


def confirmed_decisions(base, direct):
    """Start in cash and move to half only after two consecutive disagreements.

    Agreement updates the target immediately and resets the streak. A first
    disagreement keeps the previous target, including an already-held half.
    """
    target = 0.0
    streak = 0
    previous_target_date = None
    decisions = []
    for signal in consensus_decisions(base, direct):
        if previous_target_date != signal["asof_date"]:
            streak = 0
        raw_weight = signal["strategy_weight"]
        if raw_weight != HALF_WEIGHT:
            streak = 0
            target = raw_weight
        else:
            streak += 1
            if streak >= REQUIRED_DISAGREEMENTS:
                target = HALF_WEIGHT
        decisions.append({
            **signal,
            "model": "confirmed_consensus",
            "strategy_weight": target,
            "raw_strategy_weight": raw_weight,
            "disagreement_streak": streak,
            "awaiting_confirmation": raw_weight == HALF_WEIGHT and target != HALF_WEIGHT,
        })
        previous_target_date = signal["target_date"]
    return decisions
