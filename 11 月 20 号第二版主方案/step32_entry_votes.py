"""Compare ways to execute increases in the existing two-model index strategy."""
import argparse
import json
from collections import deque
from math import expm1, fsum, isclose, log1p
from pathlib import Path
from statistics import median

from step14_month_audit import month_audit
from step31_position_transitions import PERIODS, ROOT, ledger, m, read, seq, sources, tcn


RULES = {
    "published": "Execute the original target.",
    "immediate95": "Execute the original target, scaled by 95%.",
    "staged95": "On every increase, execute half the increase on the first day.",
    "split_staged": "Stage an increase only when the two models disagree.",
    "consensus_staged": "Stage an increase only when both models vote to hold.",
    "confirmed_vote": "Stage an increase unless a model voting to hold also voted to hold one session earlier.",
    "learned_entry": "Stage unless the median first open-to-open return of this type of entry is positive. "
                     "Use the last 30 matured events of the same vote type, with at least 12 observations.",
}
CANDIDATES = tuple(RULES)[3:]
CONTROLS = {"published": "base_100", "immediate95": "base_95", "staged95": "stage_buy_95"}
GROUPS = ("both", "tcn_only", "utility_only")


def vote(value):
    if value in (True, "True"):
        return True
    if value in (False, "False"):
        return False
    raise ValueError(f"Invalid model vote {value!r}")


def build(rows, raw, seed):
    dates = {str(row["date"]): i for i, row in enumerate(rows)}
    history = {group: deque(seed.get(group, ()), maxlen=30) for group in GROUPS}
    pending = deque()
    decisions = {name: [] for name in RULES}
    prior_weight = 0.0
    prior_votes = (False, False)
    last_date = None
    for signal in raw:
        day, target = signal["asof_date"], signal["target_date"]
        i = dates[day]
        if dates[target] != i + 1 or (last_date is not None and last_date != day):
            raise ValueError("Signal calendar mismatch")
        if not signal["max_training_label_date"] <= signal["fit_asof"] <= day < target:
            raise ValueError("Training labels unavailable at decision time")
        # The target day's open is unknown at decision time. An event at i+1 matures at i+2.
        while pending and pending[0][0] <= i:
            end, category = pending.popleft()
            move = rows[end]["open"] / rows[end - 1]["open"] - 1
            history[category].append(move)
        weight = float(signal["strategy_weight"])
        votes = (vote(signal["tcn_holds"]), vote(signal["utility_holds"]))
        group = { (True, True): "both", (True, False): "tcn_only",
                  (False, True): "utility_only", (False, False): "neither"}[votes]
        if weight not in (0, .5, 1) or (weight > 0 and sum(votes) != 2 * weight):
            raise ValueError("Unexpected vote/weight pair")
        increased = weight > prior_weight
        samples = history.get(group, ())
        typical = median(samples) if len(samples) >= 12 else None
        repeated = any(a and b for a, b in zip(votes, prior_votes))
        stage = {
            "published": False, "immediate95": False, "staged95": True,
            "split_staged": group != "both", "consensus_staged": group == "both",
            "confirmed_vote": not repeated, "learned_entry": typical is None or typical <= 0,
        }
        for name, apply_stage in stage.items():
            first_step = (weight + prior_weight) / 2 if increased and apply_stage else weight
            decisions[name].append({
                "asof_date": day, "target_date": target, "fit_asof": signal["fit_asof"],
                "max_training_label_date": signal["max_training_label_date"],
                "vote_type": group, "prior_raw": prior_weight, "raw_target": weight,
                "increase": increased, "staged_today": bool(increased and apply_stage),
                "matured_sample_count": len(samples), "matured_median": typical,
                "strategy_weight": first_step if name == "published" else .95 * first_step,
            })
        if increased:
            pending.append((i + 2, group))
        prior_votes, prior_weight, last_date = votes, weight, target
    # Carry only examples already observed at the last decision close across segment boundaries.
    return decisions, {group: list(samples) for group, samples in history.items()}


def evaluate(phase, prices, seed, archive, out):
    first, last = PERIODS[phase]
    rows = [row for row in prices if row["date"] <= last]
    raw = sources(phase, rows)["repo"]
    if [r["target_date"] for r in raw] != [str(r["date"]) for r in rows if r["date"] >= first]:
        raise ValueError(f"Missing {phase} trading dates")
    decisions, next_seed = build(rows, raw, seed)
    k = len(raw) // 2
    prefix, _ = build([r for r in rows if str(r["date"]) <= raw[k - 1]["target_date"]], raw[:k], seed)
    scores = {}
    for name, records in decisions.items():
        if records[:k] != prefix[name]:
            raise ValueError("A future observation affected a past decision")
        metrics, account = ledger(rows, name, records)
        stressed = seq.replay_engine.replay(rows, records, {"name": name, "kind": "forecast_weight"},
                                            fee_rate=2 * m.FEE_RATE, rebalance_on_weight_change_only=True)
        seq.verify_consensus_ledger(rows, records, stressed)
        marks = account["curve"][1:]
        excluding_april = expm1(fsum(log1p(r["daily_return"]) for r in marks if r["timestamp"][5:7] != "04"))
        metrics.update(phase=phase, first=raw[0]["target_date"], last=raw[-1]["target_date"],
                       increases=sum(r["increase"] for r in records), staged=sum(r["staged_today"] for r in records),
                       double_fee_return=stressed["metrics"]["total_return"], ex_april_contribution=excluding_april)
        if name in CONTROLS:
            reference = archive[phase, CONTROLS[name]]
            for key in ("total_return", "max_drawdown", "order_count", "fees_paid", "double_fee_return"):
                if not isclose(metrics[key], reference[key], abs_tol=1e-9):
                    raise ValueError(f"{phase}/{name}/{key} disagrees with archive")
        scores[name] = metrics
        if phase == "course":
            for suffix, table in (("signals", records), ("daily", account["curve"]), ("trades", account["trades"])):
                m.write_csv(out / f"{name}_{suffix}.csv", table)
    return scores, next_seed


def qualifies(candidate, control):
    return (candidate["total_return"] > control["total_return"] + 1e-10
            and candidate["max_drawdown"] <= control["max_drawdown"] + 1e-10
            and candidate["double_fee_return"] > control["double_fee_return"] + 1e-10
            and candidate["longest_flat_full_sessions"] >= 2)


def diagnostics(out, prices, price_hash):
    """Stress the fixed split rule without feeding these diagnostics back into selection."""
    archive = ROOT / "step17_history"
    if json.loads((archive / "protocol.json").read_text(encoding="utf-8"))["source_sha256"] != price_hash:
        raise ValueError("Early stress sample has a different index input")
    rows = [r for r in prices if str(r["date"]) < "2018-01-01"]
    raw = [r for r in read(archive / "continuous_trend_vote_signals.csv") if r["target_date"] < "2018-01-01"]
    if [r["target_date"] for r in raw] != [str(r["date"]) for r in rows if str(r["date"]) >= "2016-01-01"]:
        raise ValueError("Incomplete 2016-2017 calendar")
    choices, _ = build(rows, raw, {})
    stress = {name: ledger(rows, name, choices[name])[0] for name in ("published", "staged95", "split_staged")}
    split_curve = read(out / "split_staged_daily.csv")
    attribution = {name: month_audit(split_curve, read(out / f"{name}_daily.csv"))
                   for name in ("staged95", "published")}
    tcn.write_json(out / "diagnostics.json", {
        "early_stress_dates": [raw[0]["target_date"], raw[-1]["target_date"]],
        "early_stress_note": "Archived MA20 votes. Breadth was unavailable then. Fresh account, "
                             "no overlap with the two pre-course selection periods.",
        "early_stress": stress,
        "course_monthly_attribution": attribution,
        "attribution_note": "Compounding recorded daily returns with one month omitted, without re-running trades.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "step32_entry_votes")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    prices, metadata = m.read_prices(m.ROOT / "data/raw/SZ#399006.txt")
    if metadata["sha256"] != json.loads((ROOT / "step31_position_transitions/protocol.json").read_text())["price_sha256"]:
        raise ValueError("Price data changed")
    archive = {(r["phase"], r["name"]): r for r in json.loads((ROOT / "step31c_cash_reserve/comparison.json").read_text())
               if r["family"] == "repo"}
    tcn.write_json(out / "protocol.json", {
        "rules": RULES, "reference": "published and staged95", "fee_each_side": m.FEE_RATE,
        "scope": "All non-published targets multiplied by .95. Sells are immediate. Only target changes trigger trades.",
        "selection": "Require higher net return, no deeper drawdown, higher double-fee return and >=2 consecutive "
                     "full flat sessions against BOTH controls in BOTH pre-course segments. Break ties by recent "
                     "pre-course return. Otherwise retain published strategy.",
        "course_previously_viewed": True, "price_sha256": metadata["sha256"],
        "calibration": "Start each phase in cash without carried votes. Carry only mature training examples across phases.",
    })
    scores, seed = {}, {}
    for phase in ("early", "history"):
        scores[phase], seed = evaluate(phase, prices, seed, archive, out)
    verdict = {name: {phase: {ref: qualifies(scores[phase][name], scores[phase][ref])
                              for ref in ("published", "staged95")} for phase in ("early", "history")}
               for name in CANDIDATES}
    passing = [name for name in CANDIDATES if all(all(v.values()) for v in verdict[name].values())]
    selected = max(passing, key=lambda name: scores["history"][name]["total_return"], default="published")
    tcn.write_json(out / "selection_before_course.json", {"checks": verdict, "passing": passing, "selected": selected})
    scores["course"], _ = evaluate("course", prices, seed, archive, out)
    m.write_csv(out / "comparison.csv", [r for phase in scores.values() for r in phase.values()])
    diagnostics(out, prices, metadata["sha256"])
    for phase, group in scores.items():
        print(phase, [(k, f"{v['total_return']:.2%}", f"{v['max_drawdown']:.2%}") for k, v in group.items()], flush=True)
    print("Selected before course:", selected)


if __name__ == "__main__":
    main()
