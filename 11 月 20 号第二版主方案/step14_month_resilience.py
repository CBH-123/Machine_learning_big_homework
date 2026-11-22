"""比较收益是否集中在少数月份，并检验两种降低集中度的策略。"""
import json
import math

import step13_strategy_directions as s13
from step8_auxiliary import cash_audit
from step14_consensus import PLAN as CONSENSUS_PLAN, consensus_decisions, verify_consensus_ledger
from step14_month_audit import month_audit
from step14_robust_policy import PLAN as ROBUST_PLAN, robust_decisions

OUT = s13.m.ROOT / "outputs/step14"
BASE, DIRECT = s13.BASELINE, s13.UTILITY
ROBUST, CONSENSUS = "block_robust", "consensus"
PLAN = {
    "history_start": str(s13.HISTORY_START), "history_end": str(s13.HISTORY_END),
    "reference": BASE, "direct_reference": DIRECT,
    "strategies": {ROBUST: ROBUST_PLAN, CONSENSUS: CONSENSUS_PLAN},
    "diagnostics": "relative equity compounded with each month omitted; also omit the two strongest relative months",
    "hist_gate": s13.PLAN["candidate_requirements"],
    "fee_each_side": s13.s7.FEE,
    "previously_viewed_evaluation": True,
}


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare(rows, forecasts, direct, robust, first, last):
    forecasts, direct, robust = (s13.subset(records, first, last) for records in (forecasts, direct, robust))
    baseline = s13.s7.weighted_forecasts(forecasts, "buffer")
    consensus = consensus_decisions(baseline, direct)
    choices = {BASE: baseline, DIRECT: direct, ROBUST: robust, CONSENSUS: consensus}
    calendar = [d["target_date"] for d in baseline]
    if any([d["target_date"] for d in decisions] != calendar for decisions in choices.values()):
        raise ValueError("Comparison must use the same dates")
    runs = {name: s13.s3.replay(rows, decisions, {"name": name, "kind": "forecast_weight"},
                               rebalance_on_weight_change_only=(name == CONSENSUS))
            for name, decisions in choices.items()}
    return choices, runs


def summarize(rows, runs):
    return {name: {**run["metrics"], **cash_audit(rows, run)} for name, run in runs.items()}


def monthly_comparison(runs):
    results = {}
    for name, run in runs.items():
        if name == BASE:
            continue
        curve, reference = run["curve"], runs[BASE]["curve"]
        audit = month_audit(curve, reference)
        relative = run["metrics"]["final_equity"] / runs[BASE]["metrics"]["final_equity"] - 1
        if not math.isclose(audit["total_relative_equity_advantage"], relative, abs_tol=1e-11):
            raise ValueError("Monthly attribution failed to reconcile")
        strongest = audit["leave_one_month_out"][:2]
        audit["strongest_two_months"] = [r["month"] for r in strongest]
        audit["relative_equity_excluding_strongest_two_months"] = math.expm1(
            math.fsum(r["relative_log_growth"] for r in audit["leave_one_month_out"][2:]))
        results[name] = audit
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("plan.json", PLAN)
    raw, data = s13.m.read_prices(s13.s9.RAW)
    history_rows = [r for r in raw if r["date"] < s13.m.START]
    early = s13.s7.load_forecasts(s13.EARLY_PREDICTIONS, history_rows, s13.WARMUP, s13.EARLY_END)
    late = s13.s7.load_forecasts(s13.s9.HISTORY_FORECAST, history_rows, s13.m.VALID_START, s13.HISTORY_END)
    history_forecasts = early + late
    historical_direct = s13.utility_decisions(history_rows, s13.HISTORY_START, s13.HISTORY_END)
    historical_robust = robust_decisions(history_rows, s13.HISTORY_START, s13.HISTORY_END)
    _, historical_runs = compare(history_rows, history_forecasts, historical_direct, historical_robust,
                                 s13.HISTORY_START, s13.HISTORY_END)
    history = summarize(history_rows, historical_runs)
    history_months = monthly_comparison(historical_runs)
    folds = {}
    for label, (first, last) in s13.FOLDS.items():
        _, runs = compare(history_rows, history_forecasts, historical_direct, historical_robust, first, last)
        folds[label] = summarize(history_rows, runs)
    eligibility = {}
    tol = s13.TOLERANCE
    for name in (ROBUST, CONSENSUS):
        wins = sum(f[name]["total_return"] > f[BASE]["total_return"] + tol for f in folds.values())
        eligible = (wins >= 4 and history[name]["total_return"] > history[BASE]["total_return"] + tol
                    and history[name]["max_drawdown"] <= history[BASE]["max_drawdown"] + tol)
        eligibility[name] = {"wins": wins, "passed": eligible}
    selected = max((name for name, r in eligibility.items() if r["passed"]),
                   key=lambda name: history_months[name]["worst_leave_one_month_relative_equity_advantage"],
                   default=BASE)
    save("history_protocol.json", {"plan": PLAN, "history": history, "folds": folds, "selected": selected,
                                   "eligibility": eligibility, "index_sha256": data["sha256"],
                                   "history_monthly_audit": history_months,
                                   "scripts_sha256": {file.name: s13.sha(file) for file in
                                                      (s13.m.ROOT / "step14_robust_policy.py",
                                                       s13.m.ROOT / "step14_consensus.py")}})
    print("Historical selection:", selected, eligibility, flush=True)

    first, last = s13.m.START, s13.m.END
    rows = [r for r in raw if r["date"] <= last]
    original = s13.s7.load_forecasts(s13.s9.FORMAL_FORECAST, rows, first, last)
    direct = s13.utility_decisions(rows, first, last)
    robust = robust_decisions(rows, first, last)
    if robust[0]["max_training_label_date"] >= str(first):
        raise ValueError("The block model learned from the evaluation period on the first day")
    position = len(robust) // 2
    prefix_last = robust[position - 1]["target_date"]
    prefix = [r for r in rows if str(r["date"]) <= prefix_last]
    if robust_decisions(prefix, first, prefix_last) != robust[:position]:
        raise ValueError("Later prices changed earlier block-policy decisions")
    choices, runs = compare(rows, original, direct, robust, first, last)
    for name, run in runs.items():
        if name == CONSENSUS:
            verify_consensus_ledger(rows, choices[name], run)
        else:
            s13.s7.verify_ledger(rows, run)
        s13.m.write_csv(OUT / f"{name}_daily.csv", run["curve"])
        s13.m.write_csv(OUT / f"{name}_trades.csv", run["trades"], s13.s7.TRADE_FIELDS)
        s13.m.write_csv(OUT / f"{name}_signals.csv", choices[name])
    formal = summarize(rows, runs)
    audits = monthly_comparison(runs)
    s13.m.write_csv(OUT / "monthly_audit.csv", [{"strategy": name, **r} for name, a in audits.items() for r in a["months"]])
    save("formal_results.json", {"selected_in_history": selected, "formal": formal, "monthly_audit": audits,
                                 "previously_viewed_evaluation": True, "ledger_and_prefix_checks": "passed"})
    fields = ("total_return", "max_drawdown", "order_count", "longest_without_any_order")
    for period, metrics in (("history", history), ("formal", formal)):
        print(period, {name: [r[f] for f in fields] for name, r in metrics.items()})
    for name, a in audits.items():
        print(name, a["worst_leave_one_month"], a["worst_leave_one_month_relative_equity_advantage"],
              a["strongest_two_months"], a["relative_equity_excluding_strongest_two_months"])


if __name__ == "__main__":
    main()
