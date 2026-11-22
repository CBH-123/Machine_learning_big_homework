"""Evaluate fixed dual-trend refinements and a separate extension window."""
import argparse
import csv
from datetime import date
import json
import math

import step6_tcn as tcn
import step13_strategy_directions as s13
import step14_month_resilience as s14
import step15_consensus_refinements as s15
from step14_consensus import verify_consensus_ledger
from step15_trend_vote import trend_vote_decisions
from step16_forward_market import CALENDAR, CSV, ENDPOINT, FIRST, LAST, OUT, REQUEST_PARAMS, validate
from step16_trend_refinements import PLAN as REFINEMENT_PLAN, trend_refinement_decisions

BASE, TREND = s13.BASELINE, s15.TREND
VARIANTS = ("dual_horizon_gate", "aligned_trend_weight")
FROZEN_FILES = (
    "step1_baseline.py", "step2_models.py", "step3_robustness.py", "step6_tcn.py", "step7_net_return.py",
    "step13_utility_policy.py", "step14_consensus.py", "step15_trend_vote.py",
    "step16_trend_refinements.py", "step16_forward_market.py", "step16_refinement_experiment.py",
)
PLAN = {
    "variants": REFINEMENT_PLAN, "history_selection": s13.PLAN["candidate_requirements"],
    "fee_each_side": s13.s7.FEE,
    "extension": {"first": str(FIRST), "last": str(LAST), "calendar": [str(d) for d in CALENDAR],
                  "source": ENDPOINT, "request": REQUEST_PARAMS,
                  "account": "starts with 100000 cash; pre-window data warms signals",
                  "end": "regular trading at last open, closing mark without liquidation",
                  "refit": "continue original 20-session schedule from course period start",
                  "status": "short retrospective extension; the assignment return excludes it"},
}


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze_state():
    return {name: s13.sha(s13.m.ROOT / name) for name in FROZEN_FILES}


def choose(rows, forecasts, direct, first, last, carry=False):
    if not carry:
        forecasts = s13.subset(forecasts, first, last)
        direct = s13.subset(direct, first, last)
    baseline = s13.s7.weighted_forecasts(forecasts, "buffer")
    all_signals = {BASE: baseline, TREND: trend_vote_decisions(rows, baseline, direct),
                   **trend_refinement_decisions(rows, baseline, direct)}
    return {name: s13.subset(records, first, last) for name, records in all_signals.items()}


def replay(rows, choices, liquidate=True):
    return {name: s13.s3.replay(rows, signals, {"name": name, "kind": "forecast_weight"},
                               rebalance_on_weight_change_only=True, liquidate_at_end=liquidate)
            for name, signals in choices.items()}


def with_market(rows, choices, liquidate=True):
    runs = replay(rows, choices, liquidate)
    for name, run in runs.items():
        verify_consensus_ledger(rows, choices[name], run, liquidate_at_end=liquidate)
    measures = s14.summarize(rows, runs)
    by_date = {r["date"].isoformat(): r for r in rows}
    first, last = choices[BASE][0]["target_date"], choices[BASE][-1]["target_date"]
    measures["benchmark_index_gross"] = by_date[last]["close"] / by_date[first]["open"] - 1
    return measures, runs


def prepare(raw, data):
    if CSV.exists():
        raise ValueError("External snapshot already exists; use --forward for the frozen protocol.")
    past = [r for r in raw if r["date"] < s13.m.START]
    forecasts = s13.s7.load_forecasts(s13.EARLY_PREDICTIONS, past, s13.WARMUP, s13.EARLY_END)
    forecasts += s13.s7.load_forecasts(s13.s9.HISTORY_FORECAST, past, s13.m.VALID_START, s13.HISTORY_END)
    direct = s13.utility_decisions(past, s13.HISTORY_START, s13.HISTORY_END)
    choices = choose(past, forecasts, direct, s13.HISTORY_START, s13.HISTORY_END)
    historical, _ = with_market(past, choices)
    yearly = {y: s14.summarize(past, replay(past, choose(past, forecasts, direct, first, last)))
              for y, (first, last) in s13.FOLDS.items()}
    eligible = {}
    for name in (TREND, *VARIANTS):
        wins = sum(fold[name]["total_return"] > fold[BASE]["total_return"] + s13.TOLERANCE
                   for fold in yearly.values())
        passed = (wins >= 4 and historical[name]["total_return"] > historical[BASE]["total_return"] + s13.TOLERANCE
                  and historical[name]["max_drawdown"] <= historical[BASE]["max_drawdown"] + s13.TOLERANCE)
        eligible[name] = {"yearly_wins": wins, "passed": passed}
    selected = max((name for name, r in eligible.items() if r["passed"]),
                   key=lambda n: historical[n]["total_return"], default=BASE)
    save("history_protocol.json", {"plan": PLAN, "historical": historical, "yearly": yearly,
                                   "selected": selected, "eligible": eligible})

    start, end = s13.m.START, s13.m.END
    course = [r for r in raw if r["date"] <= end]
    forecasts = s13.s7.load_forecasts(s13.s9.FORMAL_FORECAST, course, start, end)
    direct = s13.utility_decisions(course, start, end)
    choices = choose(course, forecasts, direct, start, end)
    measures, runs = with_market(course, choices)
    audits = s15.formal_months(runs)
    save("course_results.json", {"formal": measures, "monthly_audit": audits})
    s13.m.write_csv(OUT / "course_monthly_audit.csv",
                    [{"strategy": n, **row} for n, a in audits.items() for row in a["months"]])
    save("forward_protocol.json", {"stage": "before requesting extension bars", "plan": PLAN,
                                   "code_sha256": freeze_state(), "index_sha256": data["sha256"],
                                   "course_predictions_sha256": s13.sha(s13.s9.FORMAL_FORECAST),
                                   "selected_from_cutoff_history": selected})
    print("Historical selection:", selected, eligible)
    print("Course returns:", {n: v["total_return"] for n, v in measures.items() if isinstance(v, dict)})
    print("Frozen extension setup:", FIRST, LAST, len(CALENDAR))


def read_external():
    with CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = [{k: date.fromisoformat(v) if k == "date" else float(v) for k, v in row.items()}
                for row in csv.DictReader(f)]
    validate(rows)
    return rows


def forward(raw, data):
    protocol = json.loads((OUT / "forward_protocol.json").read_text(encoding="utf-8"))
    if (protocol["code_sha256"] != freeze_state() or protocol["index_sha256"] != data["sha256"]
            or protocol["course_predictions_sha256"] != s13.sha(s13.s9.FORMAL_FORECAST)):
        raise ValueError("The extension protocol differs from the frozen code or inputs")
    new = read_external()
    if raw[-1]["date"] >= new[0]["date"]:
        raise ValueError("External bars overlap the original data")
    rows = raw + new
    tcn.torch.set_num_threads(2)
    tcn.torch.use_deterministic_algorithms(True)
    forecasts, fits, _ = tcn.rolling(rows, s13.m.START, LAST, tcn.torch.device("cpu"))
    original = s13.s7.load_forecasts(s13.s9.FORMAL_FORECAST, rows, s13.m.START, s13.m.END)
    for saved, computed in zip(original, forecasts):
        if any(saved[k] != computed[k] for k in saved):
            raise ValueError(f"Course forecast differs from the frozen original on {saved['target_date']}")
    direct = s13.utility_decisions(rows, s13.m.START, LAST)
    choices = choose(rows, forecasts, direct, FIRST, LAST, carry=True)
    measures, runs = with_market(rows, choices, liquidate=False)
    for name, run in runs.items():
        s13.m.write_csv(OUT / f"forward_{name}_daily.csv", run["curve"])
        s13.m.write_csv(OUT / f"forward_{name}_trades.csv", run["trades"], s13.s7.TRADE_FIELDS)
        s13.m.write_csv(OUT / f"forward_{name}_signals.csv", choices[name])
    save("forward_results.json", {"dates": [str(d) for d in CALENDAR], "external_sha256": s13.sha(CSV),
                                  "protocol_sha256": s13.sha(OUT / "forward_protocol.json"),
                                  "latest_fits": fits[-2:], "metrics": measures})
    print("Forward returns:", {n: v["total_return"] for n, v in measures.items() if isinstance(v, dict)})
    print("Forward closing index benchmark gross:", measures["benchmark_index_gross"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, data = s13.m.read_prices(s13.s9.RAW)
    if args.forward:
        forward(rows, data)
    else:
        prepare(rows, data)


if __name__ == "__main__":
    main()
