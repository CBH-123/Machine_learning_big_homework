"""用扩展历史比较动态选择策略和直接学习交易仓位。"""
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import math

import step2_models as m
import step3_robustness as s3
import step7_net_return as s7
from step8_auxiliary import cash_audit
import step9_breadth_reentry as s9
from step13_adaptive_expert import PLAN as ADAPTIVE_PLAN, adaptive_decisions
from step13_utility_policy import PLAN as UTILITY_PLAN, utility_decisions

OUT = m.ROOT / "outputs/step13"
EARLY_PREDICTIONS = m.ROOT / "outputs/step13_history/predictions.csv"
WARMUP = date(2018, 1, 1)
EARLY_END = date(2022, 12, 31)
HISTORY_START = date(2019, 1, 1)
HISTORY_END = m.START - timedelta(days=1)
FOLDS = {str(y): (date(y, 1, 1), min(date(y, 12, 31), HISTORY_END))
         for y in range(HISTORY_START.year, HISTORY_END.year + 1)}
BASELINE, TREND, ADAPTIVE, UTILITY = "original_tcn", "trend_only", "adaptive_expert", "utility_policy"
TOLERANCE = 1e-12
PLAN = {
    "baseline": "original close-return TCN and the established buffer rule",
    "adaptive": ADAPTIVE_PLAN,
    "direct_policy": UTILITY_PLAN,
    "history_start": str(HISTORY_START),
    "history_end": str(HISTORY_END),
    "candidate_requirements": "full history net return above original, no larger drawdown, beats original in 4 of 7 years",
    "fee_each_side": s7.FEE,
    "execution": "decide after close, execute next open, long-only, T+1, scheduled closing liquidation",
    "formal_period_previously_viewed": True,
}


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset(entries, first, last):
    return [r for r in entries if str(first) <= r["target_date"] <= str(last)]


def period_choices(original, adaptive, utility, first, last):
    original, adaptive, utility = (subset(records, first, last) for records in (original, adaptive, utility))
    target_dates = [r["target_date"] for r in original]
    if any([r["target_date"] for r in entries] != target_dates for entries in (adaptive, utility)):
        raise ValueError("All models must cover the same trading dates")
    return {
        BASELINE: s7.weighted_forecasts(original, "buffer"),
        TREND: [{**r, "strategy_weight": r["trend_target_weight"]} for r in adaptive],
        ADAPTIVE: adaptive,
        UTILITY: utility,
    }


def replay(rows, choices, fee=s7.FEE):
    return {name: s3.replay(rows, records, {"name": name, "kind": "forecast_weight"}, fee_rate=fee)
            for name, records in choices.items()}


def summarize(rows, choices, runs):
    metrics = {name: {**run["metrics"], **cash_audit(rows, run)} for name, run in runs.items()}
    chosen = [r["selected_expert"] for r in choices[ADAPTIVE]]
    metrics[ADAPTIVE]["expert_days"] = dict(Counter(chosen))
    metrics[ADAPTIVE]["expert_changes"] = sum(a != b for a, b in zip(chosen, chosen[1:]))
    return metrics


def evaluate(rows, original, adaptive, utility, first, last):
    selected = period_choices(original, adaptive, utility, first, last)
    return summarize(rows, selected, replay(rows, selected))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("plan.json", PLAN)
    raw, audit = m.read_prices(s9.RAW)
    history_rows = [r for r in raw if r["date"] < m.START]
    early = s7.load_forecasts(EARLY_PREDICTIONS, history_rows, WARMUP, EARLY_END)
    later = s7.load_forecasts(s9.HISTORY_FORECAST, history_rows, m.VALID_START, HISTORY_END)
    history_forecasts = early + later
    history_adaptive = adaptive_decisions(history_rows, history_forecasts, trade_from=HISTORY_START)
    history_utility = utility_decisions(history_rows, HISTORY_START, HISTORY_END)
    history = evaluate(history_rows, history_forecasts, history_adaptive, history_utility, HISTORY_START, HISTORY_END)
    folds = {name: evaluate(history_rows, history_forecasts, history_adaptive, history_utility, first, last)
             for name, (first, last) in FOLDS.items()}
    eligibility = {}
    for name in (ADAPTIVE, UTILITY):
        wins = sum(f[name]["total_return"] > f[BASELINE]["total_return"] + TOLERANCE for f in folds.values())
        eligibility[name] = {"annual_wins": wins, "passed": wins >= 4
                             and history[name]["total_return"] > history[BASELINE]["total_return"] + TOLERANCE
                             and history[name]["max_drawdown"] <= history[BASELINE]["max_drawdown"] + TOLERANCE}
    passing = [name for name, measure in eligibility.items() if measure["passed"]]
    selected = max(passing, key=lambda n: history[n]["total_return"], default=BASELINE)
    save("history_protocol.json", {"plan": PLAN, "selected": selected, "eligibility": eligibility,
                                   "history": history, "folds": folds, "index_sha256": audit["sha256"],
                                   "early_predictions_sha256": sha(EARLY_PREDICTIONS),
                                   "later_predictions_sha256": sha(s9.HISTORY_FORECAST),
                                   "methods_sha256": {name: sha(m.ROOT / name) for name in
                                                      ("step13_adaptive_expert.py", "step13_utility_policy.py")}})
    print(f"History selected {selected}: {eligibility}", flush=True)

    rows = [r for r in raw if r["date"] <= m.END]
    formal = s7.load_forecasts(s9.FORMAL_FORECAST, rows, m.START, m.END)
    formal_adaptive = adaptive_decisions(rows, history_forecasts + formal, trade_from=m.START)
    formal_utility = utility_decisions(rows, m.START, m.END)
    if formal_utility[0]["max_training_label_date"] >= str(m.START):
        raise ValueError("The first policy used a training label from the trial period")
    first_trial_day = next(r for r in rows if r["date"] == m.START)
    first_result = utility_decisions(history_rows + [{**history_rows[-1], "date": first_trial_day["date"]}],
                                     m.START, m.START)
    if first_result != formal_utility[:1]:
        raise ValueError("Unobserved first test price changed the direct policy")
    mid = len(formal) // 2
    prefix_rows = [r for r in rows if str(r["date"]) <= formal[mid - 1]["target_date"]]
    prefix = adaptive_decisions(prefix_rows, history_forecasts + formal[:mid], trade_from=m.START)
    if prefix != formal_adaptive[:mid]:
        raise ValueError("Future shadow results changed preceding trading decisions")
    choices = period_choices(formal, formal_adaptive, formal_utility, m.START, m.END)
    runs = replay(rows, choices)
    for name, run in runs.items():
        s7.verify_ledger(rows, run)
        m.write_csv(OUT / f"{name}_trades.csv", run["trades"], s7.TRADE_FIELDS)
        m.write_csv(OUT / f"{name}_daily.csv", run["curve"])
        data = choices[name]
        if not all(math.isfinite(float(r["strategy_weight"])) and r["strategy_weight"] in (0, 1) for r in data):
            raise ValueError(f"Invalid decision in {name}")
        m.write_csv(OUT / f"{name}_signals.csv", data, sorted({k for r in data for k in r}))
    metrics = summarize(rows, choices, runs)
    scenarios = []
    for fee in (0, 0.00025, s7.FEE, 0.001):
        model_results = runs if fee == s7.FEE else replay(rows, choices, fee)
        scenarios.extend(r["metrics"] for r in model_results.values())
    m.write_csv(OUT / "cost_sensitivity.csv", scenarios)
    save("formal_results.json", {"selected_in_history": selected, "formal": metrics,
                                 "previously_viewed_evaluation": True, "ledger_and_prefix_checks": "passed"})
    for period, values in (("history", history), *folds.items(), ("formal", metrics)):
        print(period, [(name, round(r["total_return"] * 100, 3), round(r["max_drawdown"] * 100, 3), r["order_count"])
                       for name, r in values.items()])


if __name__ == "__main__":
    main()
