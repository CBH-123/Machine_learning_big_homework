"""Paired sell-timing and price-risk experiments on the frozen split-entry strategy."""
import argparse
import json
from datetime import date
from hashlib import sha256
from itertools import product
from math import expm1, fsum, isclose, log1p
from pathlib import Path

import step16_refinement_experiment as forward
from step32_entry_votes import PERIODS, ROOT, build, ledger, m, month_audit, qualifies, read, seq, sources, tcn


CONTROL = "immediate"
EXIT_WEIGHTS = {"immediate": .475, "half_step": .7125, "wait_one": .95, "fast_step": .2375}
VARIANTS = [(rule, cap) for cap in (False, True) for rule in EXIT_WEIGHTS]
ARCHIVE = ROOT / "step32_entry_votes"


def risk_states(rows):
    """A drawdown measured in recent daily price ranges, observed after each close."""
    ranges = [row["high"] - row["low"] for row in rows]
    for i in range(1, len(rows)):
        ranges[i] = max(rows[i]["high"], rows[i - 1]["close"]) - min(rows[i]["low"], rows[i - 1]["close"])
    active = False
    states = {}
    for i in range(19, len(rows)):
        closes = [r["close"] for r in rows[i - 19:i + 1]]
        mean = fsum(closes) / 20
        atr = fsum(ranges[i - 19:i + 1]) / 20
        if closes[-1] >= mean:
            active = False
        elif max(closes) - closes[-1] >= 2 * atr:
            active = True
        states[str(rows[i]["date"])] = active
    return states


def decisions(rows, raw):
    base = build(rows, raw, {})[0]["split_staged"]
    risk = risk_states(rows)
    variants = {}
    for rule, cap in VARIANTS:
        name = rule + ("_cap" if cap else "")
        records = []
        for signal in base:
            split_exit = signal["prior_raw"] == 1 and signal["raw_target"] == .5
            cap_active = cap and risk[signal["asof_date"]]
            reference = signal["strategy_weight"]
            target = EXIT_WEIGHTS[rule] if split_exit else reference
            if cap_active:
                target = min(.475, target)
            records.append({
                "asof_date": signal["asof_date"], "target_date": signal["target_date"],
                "fit_asof": signal["fit_asof"], "max_training_label_date": signal["max_training_label_date"],
                "vote_type": signal["vote_type"], "raw_target": signal["raw_target"], "split_exit": split_exit,
                "cap_active": cap_active, "reference_weight": reference, "strategy_weight": target,
            })
        variants[name] = records
    return variants


def evaluate(phase, all_rows, first, last, raw, expected, out):
    rows = [r for r in all_rows if str(r["date"]) <= last]
    if [r["target_date"] for r in raw] != [str(r["date"]) for r in rows if str(r["date"]) >= first]:
        raise ValueError(f"Calendar mismatch in {phase}")
    variants = decisions(rows, raw)
    count = len(raw) // 2
    cut = raw[count - 1]["target_date"]
    prefix = decisions([r for r in rows if str(r["date"]) <= cut], raw[:count])
    results = {}
    curves = {}
    for name, records in variants.items():
        if records[:count] != prefix[name]:
            raise ValueError("A later observation changed an earlier decision")
        score, result = ledger(rows, name, records)
        trades = result["trades"]
        stressed = seq.replay_engine.replay(rows, records, {"name": name, "kind": "forecast_weight"},
                                            fee_rate=2*m.FEE_RATE, rebalance_on_weight_change_only=True)
        seq.verify_consensus_ledger(rows, records, stressed)
        curve = result["curve"]
        trough = min(range(len(curve)), key=lambda i: curve[i]["drawdown"])
        peak = max(range(trough + 1), key=lambda i: curve[i]["equity"])
        score.update(phase=phase, first=raw[0]["target_date"], last=raw[-1]["target_date"],
                     double_fee_return=stressed["metrics"]["total_return"],
                     ex_april_contribution=expm1(fsum(log1p(r["daily_return"]) for r in curve[1:]
                                                     if r["timestamp"][5:7] != "04")),
                     changed_days=sum(r["strategy_weight"] != r["reference_weight"] for r in records),
                     downgrades=sum(r["split_exit"] for r in records),
                     peak_date=curve[peak]["timestamp"][:10], trough_date=curve[trough]["timestamp"][:10])
        if name == CONTROL:
            for key in ("total_return", "max_drawdown", "order_count", "fees_paid", "longest_flat_full_sessions"):
                if not isclose(score[key], float(expected[key]), abs_tol=1e-9):
                    raise ValueError(f"Archived control mismatch in {phase}/{key}")
        results[name] = score
        curves[name] = curve
        if phase == "course" and name != CONTROL:
            for ending, entries in (("signals", records), ("daily", curve), ("trades", trades)):
                m.write_csv(out / f"{name}_{ending}.csv", entries)
    if phase == "course":
        checks = {name: month_audit(curve, curves[CONTROL]) for name, curve in curves.items() if name != CONTROL}
        tcn.write_json(out / "monthly_checks.json", checks)
    return results


def follow_up(out, prices, selected):
    """Bound the paired comparison on archived post-course prices with missing breadth."""
    folder = ROOT / "step17"
    archive = json.loads((folder / "forward_update.json").read_text(encoding="utf-8"))
    prior = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    if (sha256((folder / "new_session.csv").read_bytes()).hexdigest() != archive["new_session_sha256"] or
            sha256(forward.CSV.read_bytes()).hexdigest() != prior["previous_external_snapshot_sha256"]):
        raise ValueError("Follow-up price snapshots changed")
    last_bar, = read(folder / "new_session.csv")
    extension = forward.read_external() + [
        {k: date.fromisoformat(v) if k == "date" else float(v) for k, v in last_bar.items()}
    ]
    rows = prices + extension
    raw = read(folder / "forward_trend_vote_signals.csv")
    days = [str(r["date"]) for r in extension]
    if days != archive["dates"] or days != [r["target_date"] for r in raw] or prices[-1]["date"] >= extension[0]["date"]:
        raise ValueError("Follow-up signal calendar mismatch")
    last_breadth = read(m.ROOT / "data/breadth/chinext_daily_breadth.csv")[-1]["date"]
    if last_breadth >= raw[0]["asof_date"]:
        raise ValueError("Breadth is available; use observed data for the relevant dates")
    unknown = [i for i, r in enumerate(raw)
               if float(r["raw_consensus_weight"]) == .5 and float(r["strategy_weight"]) == 0]
    pairs = []
    for flags in product((False, True), repeat=len(unknown)):
        signals = [dict(r) for r in raw]
        for i, trigger in zip(unknown, flags):
            signals[i]["strategy_weight"] = .5 if trigger else 0.
        choices = decisions(rows, signals)
        base, _ = ledger(rows, CONTROL, choices[CONTROL])
        trial, _ = ledger(rows, selected, choices[selected])
        pairs.append({"breadth_flags": "".join("1" if flag else "0" for flag in flags),
                      "control_return": base["total_return"], "selected_return": trial["total_return"],
                      "control_drawdown": base["max_drawdown"], "selected_drawdown": trial["max_drawdown"],
                      "return_delta": trial["total_return"] - base["total_return"],
                      "drawdown_reduction": base["max_drawdown"] - trial["max_drawdown"]})
    summary = {key: [min(r[key] for r in pairs), max(r[key] for r in pairs)] for key in pairs[0]
               if key != "breadth_flags"}
    tcn.write_json(out / "post_course_bounds.json", {
        "first": days[0], "last": days[-1], "sessions": len(days), "control": CONTROL, "selected": selected,
        "unknown_breadth_asof_dates": [raw[i]["asof_date"] for i in unknown], "cases": len(pairs),
        "paired_ranges": summary, "both_improved_cases": sum(r["return_delta"] > 1e-10 and
                                                            r["drawdown_reduction"] > 1e-10 for r in pairs),
        "scope": "Short, previously archived price period. Start in cash, final close liquidation, same fees. "
                 "Each unknown breadth trigger can be on or off; these are bounds, not probabilities.",
    })
    m.write_csv(out / "post_course_scenarios.csv", pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "step33_exit_timing")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    prices, metadata = m.read_prices(m.ROOT / "data/raw/SZ#399006.txt")
    historical = json.loads((ARCHIVE / "protocol.json").read_text(encoding="utf-8"))
    if historical["price_sha256"] != metadata["sha256"]:
        raise ValueError("Underlying price data changed")
    tcn.write_json(out / "protocol.json", {
        "asof": "Previous close; execute at next open. Start in cash. Liquidate at scheduled final close.",
        "base": "Step 32 split_staged. Change the sell decision only on the first day of a raw 1 to 0.5 downgrade. "
                "On the following day use the fresh signal. Target zero always gives zero.",
        "first_downgrade_target": EXIT_WEIGHTS,
        "independent_risk_cap": "Cross each rule with a fixed .475 cap. Enter cap state below MA20 when the "
                                "distance from the 20-session highest close is at least twice the 20-session mean "
                                "true range. Release on a close at or above MA20. Initialize from prior prices.",
        "selection": "Select an improvement over split_staged only if both 2018-2022 and 2023 to cutoff have "
                     "strictly higher net return, no greater maximum drawdown, superior double-fee return and at "
                     "least two consecutive whole sessions in cash. Pick highest recent historical net return if "
                     "multiple qualify. Also check this winner against the published strategy separately.",
        "stress": "2016-2017 is an extra stress period outside selection; point-in-time breadth unavailable then.",
        "formal_has_been_viewed_before": True, "fee_each_side": m.FEE_RATE, "price_sha256": metadata["sha256"],
    })
    refs = {(r["phase"], r["name"]): r for r in read(ARCHIVE / "comparison.csv")}
    scores = {}
    def run(phase):
        first, last = PERIODS[phase]
        rows = [r for r in prices if r["date"] <= last]
        raw = sources(phase, rows)["repo"]
        return evaluate(phase, rows, str(first), str(last), raw, refs[phase, "split_staged"], out)

    for phase in ("early", "history"):
        scores[phase] = run(phase)
    eligible = {}
    for name in scores["history"]:
        if name == CONTROL:
            continue
        eligible[name] = {}
        for phase in ("early", "history"):
            pub = {k: float(refs[phase, "published"][k]) for k in ("total_return", "max_drawdown", "double_fee_return")}
            r = scores[phase][name]
            eligible[name][phase] = {"vs_split": qualifies(r, scores[phase][CONTROL]), "vs_published": qualifies(r, pub)}
    passing = [n for n, periods in eligible.items() if all(v["vs_split"] for v in periods.values())]
    selected = max(passing, key=lambda n: scores["history"][n]["total_return"], default=CONTROL)
    promotion = [n for n in passing if all(v["vs_published"] for v in eligible[n].values())]
    tcn.write_json(out / "selection_before_course.json", {
        "checks": eligible, "passed_vs_split": passing, "selected_vs_split": selected,
        "passed_vs_both": promotion, "selected_vs_both":
        max(promotion, key=lambda n: scores["history"][n]["total_return"], default="published"),
    })
    stress_ref = json.loads((ARCHIVE / "diagnostics.json").read_text(encoding="utf-8"))["early_stress"]["split_staged"]
    old_raw = read(ROOT / "step17_history/continuous_trend_vote_signals.csv")
    old_raw = [r for r in old_raw if r["target_date"] < "2018-01-01"]
    scores["stress"] = evaluate("stress", prices, "2016-01-01", "2017-12-31", old_raw, stress_ref, out)
    scores["course"] = run("course")
    m.write_csv(out / "comparison.csv", [r for group in scores.values() for r in group.values()])
    follow_up(out, prices, selected)
    for phase, group in scores.items():
        print(phase, [(n, f"{s['total_return']:.2%}", f"{s['max_drawdown']:.2%}") for n, s in group.items()])
    print("Selected against split entry:", selected, "Passed against both:", promotion)


if __name__ == "__main__":
    main()
