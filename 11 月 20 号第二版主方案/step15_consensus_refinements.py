"""比较延迟半仓与均线过滤对双模型仓位的影响。"""
from collections import Counter
import json
import math

import step13_strategy_directions as s13
import step14_month_resilience as s14
from step14_consensus import consensus_decisions, verify_consensus_ledger
from step15_confirmed_consensus import PLAN as CONFIRM_PLAN, confirmed_decisions
from step15_trend_vote import PLAN as TREND_PLAN, trend_vote_decisions

OUT = s13.m.ROOT / "outputs/step15"
BASE, CONSENSUS = s14.BASE, s14.CONSENSUS
CONFIRMED, TREND = "confirmed_consensus", "trend_vote"
FIXED_MONTHS = ("2026-03", "2026-04")
PLAN = {
    "baseline": BASE,
    "previous_consensus": CONSENSUS,
    "variants": {CONFIRMED: CONFIRM_PLAN, TREND: TREND_PLAN},
    "history_selection": s13.PLAN["candidate_requirements"],
    "comparison": ["all-period net return and drawdown", "exclude each single month",
                   "exclude same fixed March and April 2026 for all strategies", "exclude each strategy's top two months"],
    "fixed_month_choice": "the two biggest contributors to step 14 consensus relative to TCN",
    "fee_each_side": s13.s7.FEE,
    "previously_viewed_evaluation": True,
}


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def decisions_for(rows, forecasts, direct, first, last):
    baseline = s13.s7.weighted_forecasts(s13.subset(forecasts, first, last), "buffer")
    direct = s13.subset(direct, first, last)
    raw = consensus_decisions(baseline, direct)
    return {BASE: baseline, CONSENSUS: raw, CONFIRMED: confirmed_decisions(baseline, direct),
            TREND: trend_vote_decisions(rows, baseline, direct)}


def replay(rows, choices, fee=s13.s7.FEE):
    return {name: s13.s3.replay(rows, records, {"name": name, "kind": "forecast_weight"}, fee_rate=fee,
                               rebalance_on_weight_change_only=(name != BASE))
            for name, records in choices.items()}


def summarize(rows, choices, runs):
    measures = s14.summarize(rows, runs)
    prior = choices[CONSENSUS]
    for name, records in choices.items():
        weights = [r["strategy_weight"] for r in records]
        measures[name]["target_counts"] = dict(Counter(f"{w:.1f}" for w in weights))
        measures[name]["target_days_different_from_step14"] = sum(
            w != p["strategy_weight"] for w, p in zip(weights, prior, strict=True))
        measures[name]["traded_notional"] = math.fsum(t["price"] * t["units"] for t in runs[name]["trades"])
    return measures


def formal_months(runs):
    results = s14.monthly_comparison(runs)
    for audit in results.values():
        if not set(FIXED_MONTHS).issubset({r["month"] for r in audit["months"]}):
            raise ValueError("The fixed stress months are absent")
        audit["relative_equity_excluding_fixed_march_april"] = math.expm1(
            math.fsum(r["relative_log_growth"] for r in audit["months"] if r["month"] not in FIXED_MONTHS))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("plan.json", PLAN)
    raw, audit = s13.m.read_prices(s13.s9.RAW)
    past = [r for r in raw if r["date"] < s13.m.START]
    previous = s13.s7.load_forecasts(s13.EARLY_PREDICTIONS, past, s13.WARMUP, s13.EARLY_END)
    previous += s13.s7.load_forecasts(s13.s9.HISTORY_FORECAST, past, s13.m.VALID_START, s13.HISTORY_END)
    historical_direct = s13.utility_decisions(past, s13.HISTORY_START, s13.HISTORY_END)
    choices = decisions_for(past, previous, historical_direct, s13.HISTORY_START, s13.HISTORY_END)
    runs = replay(past, choices)
    historical = summarize(past, choices, runs)
    history_months = s14.monthly_comparison(runs)
    folds = {}
    for year, (first, last) in s13.FOLDS.items():
        yearly = decisions_for(past, previous, historical_direct, first, last)
        folds[year] = summarize(past, yearly, replay(past, yearly))
    measures = {}
    tol = s13.TOLERANCE
    for name in (CONSENSUS, CONFIRMED, TREND):
        wins = sum(r[name]["total_return"] > r[BASE]["total_return"] + tol for r in folds.values())
        vs_consensus = sum(r[name]["total_return"] > r[CONSENSUS]["total_return"] + tol for r in folds.values())
        passed = (wins >= 4 and historical[name]["total_return"] > historical[BASE]["total_return"] + tol
                  and historical[name]["max_drawdown"] <= historical[BASE]["max_drawdown"] + tol)
        measures[name] = {"annual_wins_vs_tcn": wins, "annual_wins_vs_previous_consensus": vs_consensus,
                          "passes_existing_history_criteria": passed}
    selected = max((n for n, r in measures.items() if r["passes_existing_history_criteria"]),
                   key=lambda n: history_months[n]["worst_leave_one_month_relative_equity_advantage"], default=BASE)
    save("history_protocol.json", {"plan": PLAN, "selected": selected, "historical": historical, "folds": folds,
                                   "eligibility": measures, "monthly_audit": history_months,
                                   "index_sha256": audit["sha256"]})
    print(f"History selection: {selected} {measures}", flush=True)

    first, last = s13.m.START, s13.m.END
    rows = [r for r in raw if r["date"] <= last]
    forecasts = s13.s7.load_forecasts(s13.s9.FORMAL_FORECAST, rows, first, last)
    direct = s13.utility_decisions(rows, first, last)
    choices = decisions_for(rows, forecasts, direct, first, last)
    middle = len(forecasts) // 2
    cutoff = forecasts[middle - 1]["target_date"]
    prefix_rows = [r for r in rows if str(r["date"]) <= cutoff]
    prefix = decisions_for(prefix_rows, forecasts[:middle], direct[:middle], first, cutoff)
    if any(prefix[name] != records[:middle] for name, records in choices.items()):
        raise ValueError("Future data changed earlier decisions")
    runs = replay(rows, choices)
    for name, run in runs.items():
        if name == BASE:
            s13.s7.verify_ledger(rows, run)
        else:
            verify_consensus_ledger(rows, choices[name], run)
        s13.m.write_csv(OUT / f"{name}_daily.csv", run["curve"])
        s13.m.write_csv(OUT / f"{name}_trades.csv", run["trades"], s13.s7.TRADE_FIELDS)
        s13.m.write_csv(OUT / f"{name}_signals.csv", choices[name])
    formal = summarize(rows, choices, runs)
    monthly = formal_months(runs)
    s13.m.write_csv(OUT / "monthly_audit.csv", [{"strategy": n, **r} for n, a in monthly.items() for r in a["months"]])
    save("formal_results.json", {"selected_in_history": selected, "formal": formal, "monthly_audit": monthly,
                                 "ledger_and_prefix_checks": "passed", "previously_viewed_evaluation": True})
    for period, data in (("history", historical), ("formal", formal)):
        print(period, {n: [r["total_return"], r["max_drawdown"], r["order_count"],
                           r["longest_without_any_order"]] for n, r in data.items()})
    for name, data in monthly.items():
        print(name, data["worst_leave_one_month"], data["worst_leave_one_month_relative_equity_advantage"],
              data["relative_equity_excluding_fixed_march_april"])


if __name__ == "__main__":
    main()
