"""Test whether broad participation warrants retaining exposure below MA20."""
import json
import math

import step13_strategy_directions as s13
import step15_consensus_refinements as s15
import step18_sideways_experiment as s18
from step14_month_audit import month_audit
from step15_trend_vote import trend_vote_decisions

s9 = s13.s9
OUT = s13.m.ROOT / "outputs/step19"
BASE, MA20 = s13.BASELINE, s15.TREND
SIZES = {"breadth_quarter": 0.25, "breadth_half": 0.50}
PLAN = {
    "reference": MA20,
    "trigger": "disagreement and the original MA20 rule sets zero; up-share at least 60% today and 50% on average over five sessions",
    "weights_on_trigger": SIZES,
    "coverage": s9.MIN_VALID_QUOTES,
    "fee_each_side": s13.s7.FEE,
    "comparison_history": "2023 through 2025-10-09, the common period with point-in-time breadth data",
    "history_selection": "beats MA20 in full-period net return and at least two of three segments, with no higher full-period drawdown",
    "execution": "decide after close; next open; long-only; T+1; adjust on target changes; scheduled final liquidation",
    "course_period_previously_viewed": True,
}


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def decisions_for(rows, forecasts, direct, breadth, first, last):
    original = s13.s7.weighted_forecasts(s13.subset(forecasts, first, last), "buffer")
    trend = trend_vote_decisions(rows, original, s13.subset(direct, first, last))
    variants = {name: [] for name in SIZES}
    for signal in trend:
        info = breadth.get(signal["asof_date"], {})
        daily = info.get("breadth")
        mean = info.get("breadth5")
        blocked = signal["strategy_weight"] == 0 and signal["raw_consensus_weight"] == 0.5
        trigger = (blocked and info.get("valid_quotes", 0) >= s9.MIN_VALID_QUOTES
                   and daily is not None and daily >= s9.MIN_BREADTH
                   and mean is not None and mean >= s9.MIN_AVG_BREADTH)
        for name, size in SIZES.items():
            variants[name].append({**signal, "model": name, "ma20_target": signal["strategy_weight"],
                                   "strategy_weight": size if trigger else signal["strategy_weight"],
                                   "breadth": daily, "breadth5": mean, "valid_quotes": info.get("valid_quotes"),
                                   "ma20_blocked_half": blocked, "breadth_trigger": trigger})
    return {BASE: original, MA20: trend, **variants}


def summarize(rows, choices, runs):
    results = s18.summarize(rows, choices, runs)
    prices = {str(r["date"]): r for r in rows}
    next_day = {str(a["date"]): str(b["date"]) for a, b in zip(rows, rows[1:])}
    for name in SIZES:
        signals = choices[name]
        triggered = [r["target_date"] for r in signals if r["breadth_trigger"]]
        # Following-open comparison is a descriptive price check, not a one-day strategy.
        last = signals[-1]["target_date"]
        changes = [prices[next_day[d]]["open"] / prices[d]["open"] - 1 for d in triggered if d < last]
        monthly = month_audit(runs[name]["curve"], runs[MA20]["curve"])
        results[name]["breadth_trigger_dates"] = triggered
        results[name]["trigger_next_open_move"] = {
            "count": len(changes), "positive": sum(x > 0 for x in changes),
            "mean": math.fsum(changes) / len(changes) if changes else None,
        }
        results[name]["vs_ma20"] = {
            k: monthly[k] for k in ("total_relative_equity_advantage", "months_beating_baseline",
                                   "worst_leave_one_month", "worst_leave_one_month_relative_equity_advantage")
        }
    return results


def run_segment(rows, tcn, direct, breadth, first, last):
    decisions = decisions_for(rows, tcn, direct, breadth, first, last)
    runs = s18.replay(rows, decisions, verify=True)
    return summarize(rows, decisions, runs), decisions, runs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("plan.json", PLAN)
    raw, metadata = s13.m.read_prices(s9.RAW)
    history = [r for r in raw if r["date"] < s13.m.START]
    first, last = s13.m.VALID_START, s13.HISTORY_END
    prior_breadth = s9.load_breadth(s13.m.START)
    tcn = s13.s7.load_forecasts(s9.HISTORY_FORECAST, history, first, last)
    # Preserve the historical refit cadence used in Steps 15–18.
    direct = s13.utility_decisions(history, s13.HISTORY_START, last)
    historical = run_segment(history, tcn, direct, prior_breadth, first, last)[0]
    folds = {label: run_segment(history, tcn, direct, prior_breadth, a, b)[0] for label, a, b in s9.FOLDS}
    tol = s13.TOLERANCE
    eligibility = {}
    for name in SIZES:
        wins = sum(r[name]["total_return"] > r[MA20]["total_return"] + tol for r in folds.values())
        eligibility[name] = {"wins": wins, "passed": wins >= 2
                            and historical[name]["total_return"] > historical[MA20]["total_return"] + tol
                            and historical[name]["max_drawdown"] <= historical[MA20]["max_drawdown"] + tol}
    selected = max((name for name, r in eligibility.items() if r["passed"]),
                   key=lambda name: historical[name]["total_return"], default=MA20)
    save("history_protocol.json", {"plan": PLAN, "history": historical, "folds": folds, "selected": selected,
                                  "eligibility": eligibility, "index_sha256": metadata["sha256"],
                                  "breadth_before_cutoff_sha256": s9.fingerprint(prior_breadth)})

    first, last = s13.m.START, s13.m.END
    rows = [r for r in raw if r["date"] <= last]
    breadth = s9.load_breadth()
    if {day: info for day, info in breadth.items() if day < str(first)} != prior_breadth:
        raise ValueError("Later breadth data affected historical inputs")
    tcn = s13.s7.load_forecasts(s9.FORMAL_FORECAST, rows, first, last)
    direct = s13.utility_decisions(rows, first, last)
    formal, choices, runs = run_segment(rows, tcn, direct, breadth, first, last)
    cut = tcn[len(tcn) // 2 - 1]["target_date"]
    prefix = decisions_for([r for r in rows if str(r["date"]) <= cut], tcn, direct,
                           {d: r for d, r in breadth.items() if d < cut}, first, cut)
    if any(prefix[name] != s13.subset(records, first, cut) for name, records in choices.items()):
        raise ValueError("Later data affected earlier signals")
    for name, run in runs.items():
        s13.m.write_csv(OUT / f"{name}_daily.csv", run["curve"])
        s13.m.write_csv(OUT / f"{name}_trades.csv", run["trades"], s13.s7.TRADE_FIELDS)
        s13.m.write_csv(OUT / f"{name}_signals.csv", choices[name])
    for name in (BASE, MA20):
        previous = json.loads((s15.OUT / "formal_results.json").read_text(encoding="utf-8"))["formal"][name]
        for key in ("total_return", "max_drawdown", "order_count", "fees_paid"):
            if not math.isclose(formal[name][key], previous[key], abs_tol=1e-10):
                raise ValueError(f"Step 15 benchmark changed: {name}/{key}")
    audited = s15.formal_months(runs)
    s13.m.write_csv(OUT / "course_monthly_audit.csv",
                    [{"strategy": n, **r} for n, a in audited.items() for r in a["months"]])
    save("course_results.json", {"formal": formal, "monthly_audit": audited, "selected_in_history": selected,
                                 "ledger_and_prefix_checks": "passed"})
    for label, values in (("history", historical), *folds.items(), ("course", formal)):
        print(label, [(n, f"{r['total_return']:+.2%}", f"{r['max_drawdown']:.2%}", r["order_count"],
                       len(r.get("breadth_trigger_dates", []))) for n, r in values.items()])
    print("History selection:", selected, eligibility)


if __name__ == "__main__":
    main()
