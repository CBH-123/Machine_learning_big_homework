"""Diagnose the sideways return gap and compare two fixed filters across periods."""
from collections import defaultdict
import csv
from datetime import date
import json
import math

import step13_strategy_directions as s13
import step15_consensus_refinements as s15
import step16_refinement_experiment as s16
import step17_early_stress as s17
from step14_consensus import verify_consensus_ledger
from step15_trend_vote import trend_vote_decisions
from step18_sideways_rules import PLAN as FILTER_PLAN, sideways_decisions

OUT = s13.m.ROOT / "outputs/step18"
BASE, MA20 = s13.BASELINE, s15.TREND
FEE = s13.s7.FEE
PLAN = {
    "controls": [BASE, MA20],
    "variants": FILTER_PLAN,
    "diagnosis": "same order targets, prices and terminal policy, 0 versus 5 bp fees each side",
    "phase_labels": "ex post benchmark monthly returns, same ±5% classification as Step 17",
    "history_selection": s13.PLAN["candidate_requirements"],
    "already_reviewed_samples": ["course", "external 15 days", "2016–18 baseline stress"],
}


def save(filename, data):
    (OUT / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def choose(rows, forecasts, direct, first, last, carry=False):
    if not carry:
        forecasts = s13.subset(forecasts, first, last)
        direct = s13.subset(direct, first, last)
    original = s13.s7.weighted_forecasts(forecasts, "buffer")
    variants = sideways_decisions(rows, original, direct)
    choices = {BASE: original, MA20: trend_vote_decisions(rows, original, direct), **variants}
    return {name: s13.subset(signals, first, last) for name, signals in choices.items()}


def replay(rows, choices, fee=FEE, liquidate=True, verify=False):
    runs = {name: s13.s3.replay(rows, signals, {"name": name, "kind": "forecast_weight"}, fee_rate=fee,
                               rebalance_on_weight_change_only=True, liquidate_at_end=liquidate)
            for name, signals in choices.items()}
    if verify:
        for name, run in runs.items():
            verify_consensus_ledger(rows, choices[name], run, liquidate_at_end=liquidate)
    return runs


def summarize(rows, choices, runs):
    results = s16.s14.summarize(rows, runs)
    for name, result in results.items():
        result["target_days_different_from_ma20"] = sum(a["strategy_weight"] != b["strategy_weight"]
            for a, b in zip(choices[name], choices[MA20], strict=True))
        result["traded_notional"] = math.fsum(t["units"] * t["price"] for t in runs[name]["trades"])
    return results


def fee_decomposition(with_fee, without_fee, labeled_months, choices):
    month_phase = {month: classification["phase"] for month, classification in labeled_months.items()}
    logs = {name: {phase: defaultdict(list) for phase in ("up", "sideways", "down")} for name in with_fee}
    for name, runs in with_fee.items():
        net, gross = runs["curve"][1:], without_fee[name]["curve"][1:]
        for a, b, signal in zip(net, gross, choices[name], strict=True):
            day = a["timestamp"][:10]
            if a["timestamp"] != b["timestamp"] or day != signal["target_date"]:
                raise ValueError("Dates differ between the fee experiment and its control")
            bucket = logs[name][month_phase[day[:7]]]
            bucket["net"].append(math.log1p(a["daily_return"]))
            bucket["gross"].append(math.log1p(b["daily_return"]))
            bucket["open_weights"].append(a["open_weight_after_trade"])
        for fill in runs["trades"]:
            bucket = logs[name][month_phase[fill["date"][:7]]]
            bucket["notional"].append(fill["units"] * fill["price"])
            bucket["fees"].append(fill["fee"])
            bucket["orders"].append(1)
    output = {}
    for phase in ("up", "sideways", "down"):
        measures = {}
        base = logs[BASE][phase]
        index_factors = [d["benchmark_index_gross"] for d in labeled_months.values() if d["phase"] == phase]
        for name, grouping in logs.items():
            r = grouping[phase]
            net_log, gross_log = math.fsum(r["net"]), math.fsum(r["gross"])
            relative_gross = gross_log - math.fsum(base["gross"])
            relative_net = net_log - math.fsum(base["net"])
            measures[name] = {
                "trading_days": len(r["net"]), "net_return": math.expm1(net_log),
                "zero_fee_return": math.expm1(gross_log),
                "net_relative_log_growth": relative_net,
                "zero_fee_relative_log_growth": relative_gross,
                "differential_fee_log_effect": relative_net - relative_gross,
                "mean_open_weight": math.fsum(r["open_weights"]) / len(r["open_weights"]),
                "fees_paid": math.fsum(r["fees"]), "orders": len(r["orders"]),
                "traded_notional": math.fsum(r["notional"]),
            }
        output[phase] = {"months": len(index_factors), "index_gross_return": math.prod(1 + x for x in index_factors) - 1,
                         "strategies": measures}
    for name, result in with_fee.items():
        actual = math.prod(1 + group["strategies"][name]["net_return"] for group in output.values()) - 1
        if not math.isclose(actual, result["metrics"]["total_return"], abs_tol=1e-12):
            raise ValueError("Market phase net attribution does not reconcile")
    return output


def dump_runs(label, runs, choices):
    for name, run in runs.items():
        s13.m.write_csv(OUT / f"{label}_{name}_daily.csv", run["curve"])
        s13.m.write_csv(OUT / f"{label}_{name}_trades.csv", run["trades"], s13.s7.TRADE_FIELDS)
        s13.m.write_csv(OUT / f"{label}_{name}_signals.csv", choices[name])


def sideways_trade_drivers(rows, result, signals, labeled_months):
    sideways_months = {month for month, row in labeled_months.items() if row["phase"] == "sideways"}
    prices = {str(r["date"]): r for r in rows}
    opening_fills = {t["date"] for t in result["trades"] if t["time"] == "09:30"}
    crossings = 0
    crossing_with_fill = 0
    for prior, current in zip(signals, signals[1:]):
        day = current["target_date"]
        if day[:7] not in sideways_months or current["trend_holds"] == prior["trend_holds"]:
            continue
        crossings += 1
        if (prior["raw_consensus_weight"] == current["raw_consensus_weight"] == 0.5
                and day in opening_fills):
            crossing_with_fill += 1
    blocked = [prices[r["target_date"]] for r in signals if r["target_date"][:7] in sideways_months
               and r["raw_consensus_weight"] == 0.5 and r["strategy_weight"] == 0.0]
    return {"raw_ma_flip_count": crossings, "ma_flip_fills_between_two_disagreement_signals": crossing_with_fill,
            "half_position_filtered_days": len(blocked),
            "index_open_to_close_up_days_of_filtered": sum(r["close"] > r["open"] for r in blocked)}


def verify_benchmark_fills(name, label, ref):
    with (OUT / f"{label}_{name}_trades.csv").open(encoding="utf-8-sig", newline="") as a, \
            ref.open(encoding="utf-8-sig", newline="") as b:
        for new, old in zip(csv.DictReader(a), csv.DictReader(b), strict=True):
            for key in s13.s7.TRADE_FIELDS:
                matched = (math.isclose(float(new[key]), float(old[key]), rel_tol=1e-12, abs_tol=1e-10)
                           if key in ("units", "price", "fee") else new[key] == old[key])
                if not matched:
                    raise ValueError(f"The previously reported benchmark {label}/{name} changed")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("plan.json", PLAN)
    raw, audit = s13.m.read_prices(s13.s9.RAW)
    early_ref = json.loads((s17.OUT / "results.json").read_text(encoding="utf-8"))
    first, last = (date.fromisoformat(early_ref["continuous"][k]) for k in ("first", "last"))
    early_rows = [r for r in raw if r["date"] <= last]
    early_forecasts = s13.s7.load_forecasts(s17.OUT / "tcn_forecasts.csv", early_rows, first, last)
    early_direct = s13.utility_decisions(early_rows, first, last)
    early_choices = choose(early_rows, early_forecasts, early_direct, first, last)
    early_runs = replay(early_rows, early_choices, verify=True)
    early = summarize(early_rows, early_choices, early_runs)
    diagnostic = fee_decomposition(early_runs, replay(early_rows, early_choices, fee=0), early_ref["months"], early_choices)
    diagnostic["sideways"]["original_ma20_trade_drivers"] = sideways_trade_drivers(
        early_rows, early_runs[MA20], early_choices[MA20], early_ref["months"])
    dump_runs("early", early_runs, early_choices)
    for name in (BASE, MA20):
        verify_benchmark_fills(name, "early", s17.OUT / f"continuous_{name}_trades.csv")

    past = [r for r in raw if r["date"] < s13.m.START]
    latest = s13.s7.load_forecasts(s13.EARLY_PREDICTIONS, past, s13.WARMUP, s13.EARLY_END)
    latest += s13.s7.load_forecasts(s13.s9.HISTORY_FORECAST, past, s13.m.VALID_START, s13.HISTORY_END)
    direct = s13.utility_decisions(past, s13.HISTORY_START, s13.HISTORY_END)
    historical_choice = choose(past, latest, direct, s13.HISTORY_START, s13.HISTORY_END)
    historical = summarize(past, historical_choice, replay(past, historical_choice))
    folds = {}
    for year, (first_year, end_year) in s13.FOLDS.items():
        ch = choose(past, latest, direct, first_year, end_year)
        folds[year] = summarize(past, ch, replay(past, ch))
    tol = s13.TOLERANCE
    qualified = {}
    for name, result in historical.items():
        if name == BASE:
            continue
        wins = sum(r[name]["total_return"] > r[BASE]["total_return"] + tol for r in folds.values())
        qualified[name] = {"annual_wins": wins, "passes": wins >= 4
            and result["total_return"] > historical[BASE]["total_return"] + tol
            and result["max_drawdown"] <= historical[BASE]["max_drawdown"] + tol}
    selected = max((n for n, r in qualified.items() if r["passes"]),
                   key=lambda n: historical[n]["total_return"], default=BASE)
    save("history_protocol.json", {"plan": PLAN, "early": early, "early_fee_attribution": diagnostic,
                                   "historical": historical, "folds": folds, "selection": selected,
                                   "qualified": qualified, "source_sha256": audit["sha256"],
                                   "rules_sha256": s13.sha(s13.m.ROOT / "step18_sideways_rules.py")})
    print("Historical selection:", selected, qualified, flush=True)

    start, end = s13.m.START, s13.m.END
    rows = [r for r in raw if r["date"] <= end]
    forecasts = s13.s7.load_forecasts(s13.s9.FORMAL_FORECAST, rows, start, end)
    direct = s13.utility_decisions(rows, start, end)
    choices = choose(rows, forecasts, direct, start, end)
    mid = len(forecasts) // 2
    cutoff = forecasts[mid - 1]["target_date"]
    prefix = [r for r in rows if str(r["date"]) <= cutoff]
    short = choose(prefix, forecasts[:mid], direct[:mid], start, cutoff)
    if any(short[name] != records[:mid] for name, records in choices.items()):
        raise ValueError("Later prices changed earlier filter signals")
    runs = replay(rows, choices, verify=True)
    formal = summarize(rows, choices, runs)
    dump_runs("course", runs, choices)
    for name in (BASE, MA20):
        verify_benchmark_fills(name, "course", s15.OUT / f"{name}_trades.csv")
    audited = s15.formal_months(runs)
    s13.m.write_csv(OUT / "course_monthly_audit.csv",
                    [{"strategy": n, **r} for n, a in audited.items() for r in a["months"]])
    save("course_results.json", {"formal": formal, "monthly_audit": audited, "selected_from_history": selected})
    for phase, figures in diagnostic.items():
        data = figures["strategies"]
        print(phase, figures["months"], {n: [round(v["net_return"], 5), round(v["zero_fee_return"], 5),
                                             v["orders"], round(v["fees_paid"], 2)] for n, v in data.items()})
    for label, values in (("early", early), ("history", historical), ("course", formal)):
        print(label, {n: [r["total_return"], r["max_drawdown"], r["order_count"]] for n, r in values.items()})


if __name__ == "__main__":
    main()
