"""用成分股上涨占比对照原 TCN 的空仓恢复交易规则。"""
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path

import step2_models as m
import step3_robustness as s3
import step7_net_return as s7
from step8_auxiliary import cash_audit
from step9_download_breadth import DATA, requested_days

OUT = m.ROOT / "outputs" / "step9"
BREADTH_DATA = DATA / "chinext_daily_breadth.csv"
RAW = m.ROOT / "data/raw/SZ#399006.txt"
HISTORY_FORECAST = s7.SOURCE / "validation_predictions.csv"
FORMAL_FORECAST = s7.SOURCE / "test_predictions.csv"
BASELINE = "original_tcn"
EXPERIMENT = "breadth_reentry"
MIN_CASH_SESSIONS = 5
MIN_VALID_QUOTES = 90
MIN_BREADTH = 0.60
MIN_AVG_BREADTH = 0.50
FOLDS = (
    ("2023", date(2023, 1, 1), date(2023, 12, 31)),
    ("2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("2025_before_cutoff", date(2025, 1, 1), m.START - timedelta(days=1)),
)
PLAN = {
    "signal_asof": "20:00 Asia/Shanghai, next session open trade",
    "fee_each_side": s7.FEE, "normal_entry": s7.ENTRY, "exit_at_or_below": 0,
    "extra_entry": {"min_prior_cash_sessions": MIN_CASH_SESSIONS,
                    "tcn_score_strictly_above": 0, "min_today_up_share": MIN_BREADTH,
                    "min_up_share_5day_mean": MIN_AVG_BREADTH, "min_daily_valid_quotes": MIN_VALID_QUOTES},
    "history_end": str(m.START - timedelta(days=1)),
    "historical_requirement": "higher total net return, no worse drawdown, and wins in at least 2 of 3 subperiods",
    "formal_replay_is_exploratory": True,
}


def save(filename, payload):
    (OUT / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_breadth(stop_before=None):
    schedule = requested_days()
    expected = {day: regime for day, regime in schedule.items() if stop_before is None or day < stop_before}
    records = {}
    recent = []
    with BREADTH_DATA.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            day = date.fromisoformat(row["date"])
            if stop_before is not None and day >= stop_before:
                continue
            if day not in expected or day in records:
                raise ValueError(f"Unexpected breadth date: {day}")
            effective, members = expected[day]
            n, up, flat, down = (int(row[col]) for col in ("valid_quotes", "up", "flat", "down"))
            if (row["membership_effective"] != str(effective)
                    or not 0 <= n <= int(row["members"]) == len(members) == 100
                    or min(up, flat, down) < 0 or up + flat + down != n):
                raise ValueError(f"Invalid breadth counts on {day}")
            fraction = up / n if n else None
            if n and not math.isclose(fraction, float(row["breadth"])):
                raise ValueError(f"Invalid breadth fraction on {day}")
            recent.append(fraction if n >= MIN_VALID_QUOTES else None)
            recent = recent[-5:]
            prior_mean = sum(recent) / 5 if len(recent) == 5 and all(x is not None for x in recent) else None
            records[day] = {"breadth": fraction, "breadth5": prior_mean, "valid_quotes": n}
    if list(records) != list(expected):
        raise ValueError("Breadth rows must match official trading dates in calendar order.")
    return {str(day): record for day, record in records.items()}


def decision_weights(forecasts, breadth):
    held = False
    cash_streak = 0
    decisions = []
    for f in forecasts:
        aux = breadth.get(f["asof_date"], {})
        score = f["predicted_return"]
        additional = (not held and cash_streak >= MIN_CASH_SESSIONS and 0 < score <= s7.ENTRY
                      and aux.get("breadth5") is not None and aux["breadth5"] >= MIN_AVG_BREADTH
                      and aux["breadth"] >= MIN_BREADTH)
        if score > s7.ENTRY or additional:
            held = True
        elif score <= 0:
            held = False
        decisions.append({**f, **aux, "strategy_weight": int(held),
                          "cash_sessions_before_signal": cash_streak, "breadth_entry": additional})
        cash_streak = 0 if held else cash_streak + 1
    return decisions


def compare(rows, forecasts, breadth, validate_ledger=False):
    choices = decision_weights(forecasts, breadth)
    without_data = [d["strategy_weight"] for d in decision_weights(forecasts, {})]
    original_weights = [d["strategy_weight"] for d in s7.weighted_forecasts(forecasts, "buffer")]
    if without_data != original_weights:
        raise ValueError("Missing auxiliary data must reproduce the original rules")
    baseline = s7.replay(rows, forecasts, "buffer")
    variant = s3.replay(rows, choices, {"name": EXPERIMENT, "kind": "forecast_weight"}, fee_rate=s7.FEE)
    fills = {t["date"] for t in variant["trades"] if t["side"] == "BUY"}
    signals = [c for c in choices if c["breadth_entry"] and c["target_date"] in fills]
    results = {}
    for label, run in ((BASELINE, baseline), (EXPERIMENT, variant)):
        if validate_ledger:
            s7.verify_ledger(rows, run)
        results[label] = {**run["metrics"], **cash_audit(rows, run)}
    results[EXPERIMENT]["additional_entries"] = len(signals)
    return results, baseline, variant, choices, signals


def fingerprint(records):
    return hashlib.sha256(json.dumps(records, allow_nan=False, sort_keys=True).encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("plan.json", PLAN)
    rows, _ = m.read_prices(RAW)
    history = [r for r in rows if r["date"] < m.START]
    history_breadth = load_breadth(m.START)
    history_forecasts = s7.load_forecasts(HISTORY_FORECAST, history, m.VALID_START, history[-1]["date"])
    history_stats, *_ = compare(history, history_forecasts, history_breadth)
    per_fold = {}
    for year, first, last in FOLDS:
        selected = [f for f in history_forecasts if str(first) <= f["target_date"] <= str(last)]
        per_fold[year], *_ = compare(history, selected, history_breadth)
    wins = sum(r[EXPERIMENT]["total_return"] > r[BASELINE]["total_return"] for r in per_fold.values())
    passed = (wins >= 2 and history_stats[EXPERIMENT]["total_return"] > history_stats[BASELINE]["total_return"]
              and history_stats[EXPERIMENT]["max_drawdown"] <= history_stats[BASELINE]["max_drawdown"])
    save("history_protocol.json", {"plan": PLAN, "rules_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   "breadth_history_sha256": fingerprint(history_breadth), "history": history_stats,
                                   "folds": per_fold, "wins": wins, "passed": passed})

    breadth = load_breadth()
    if {d: r for d, r in breadth.items() if d < str(m.START)} != history_breadth:
        raise ValueError("Later quotes affected earlier breadth values")
    forecasts = s7.load_forecasts(FORMAL_FORECAST, rows, m.START, m.END)
    results, baseline, variant, choices, additional = compare(rows, forecasts, breadth, validate_ledger=True)
    idx = len(choices) // 2
    if decision_weights(forecasts[:idx], breadth) != choices[:idx]:
        raise ValueError("Later forecasts affected earlier trades")
    for name, run in ((BASELINE, baseline), (EXPERIMENT, variant)):
        m.write_csv(OUT / f"{name}_daily.csv", run["curve"])
        m.write_csv(OUT / f"{name}_trades.csv", run["trades"], s7.TRADE_FIELDS)
    m.write_csv(OUT / "breadth_signals.csv", choices, sorted({k for row in choices for k in row}))
    save("formal_results.json", {"protocol": PLAN, "breadth_sha256": fingerprint(breadth),
                                 "historical_passed": passed, "formal": results,
                                 "additional_entries": additional,
                                 "coverage": {"minimum": min(r["valid_quotes"] for r in breadth.values()),
                                              "days_under_minimum": sum(r["valid_quotes"] < MIN_VALID_QUOTES for r in breadth.values())}})
    for section, stats in (("history", history_stats), *per_fold.items(), ("formal", results)):
        print(section, [(k, round(v["total_return"] * 100, 3), round(v["max_drawdown"] * 100, 3),
                         v.get("additional_entries"), v["longest_without_any_order"]) for k, v in stats.items()])
    print(f"Historical segments won: {wins}/3. Passed: {passed}.")


if __name__ == "__main__":
    main()
