"""比较独立的上涨占比入场信号与按实际持有区间训练的 TCN。"""
from pathlib import Path
import hashlib
import json
import math

import step2_models as m
import step3_robustness as s3
import step6_tcn as tcn
import step7_net_return as s7
from step8_auxiliary import cash_audit
import step9_breadth_reentry as s9
import numpy as np

OUT = m.ROOT / "outputs/step10"
BASELINE, BREADTH, TRADE_TARGET = "original_tcn", "independent_breadth", "open_to_open_tcn"
PLAN = {
    "strategies": {
        BASELINE: "TCN next close / as-of close return, original buffer",
        BREADTH: "original buffer plus independent breadth reentry after at least 5 cash sessions",
        TRADE_TARGET: "TCN next open to following open return, original buffer",
    },
    "entry_threshold": s7.ENTRY,
    "exit_threshold": 0.0,
    "breadth_rule": {"cash_sessions": s9.MIN_CASH_SESSIONS, "daily": s9.MIN_BREADTH,
                     "mean_last_5": s9.MIN_AVG_BREADTH, "min_daily_quotes": s9.MIN_VALID_QUOTES},
    "fee_each_side": s7.FEE,
    "train_window": m.TRAIN_WINDOW,
    "refit_every": m.REFIT_EVERY,
    "seed": tcn.SEED,
    "history_selection": "higher net return, maximum drawdown no larger, and beats baseline in >=2 of 3 subperiods",
    "formal_period_previously_viewed": True,
}


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def opening_forecasts(rows, first, last, device, *, model_factory=tcn.TinyTCN, epochs=tcn.EPOCHS):
    x = m.build_features(rows)
    windows = np.lib.stride_tricks.sliding_window_view(x, tcn.LOOKBACK, axis=0)
    shift = tcn.LOOKBACK - 1
    eligible = np.flatnonzero(np.isfinite(windows).all(axis=(1, 2))) + shift
    opens = np.array([r["open"] for r in rows])
    # 输入是第 i 日收盘时的特征。标签的两个价格是第 i+1 日和第 i+2 日的开盘。
    target_returns = opens[2:] / opens[1:-1] - 1
    signals = [i for i in eligible if i + 1 < len(rows) and first <= rows[i + 1]["date"] <= last]
    forecasts, refits = [], []
    for number, i in enumerate(signals):
        asof = str(rows[i]["date"])
        if number % m.REFIT_EVERY == 0:
            prior = eligible[eligible + 2 <= i][-m.TRAIN_WINDOW:]
            if len(prior) != m.TRAIN_WINDOW:
                raise ValueError(f"Only {len(prior)} mature samples on {asof}")
            model, mean, scale, _ = tcn.fit_window(windows[prior - shift], target_returns[prior], device,
                                                  epochs, model_factory=model_factory)
            fit = {"fit_asof": asof, "max_training_label_date": str(rows[prior[-1] + 2]["date"]),
                   "last_training_signal_date": str(rows[prior[-1]]["date"]), "n_train": len(prior)}
            if not fit["last_training_signal_date"] < fit["max_training_label_date"] <= asof:
                raise ValueError("Training label is unavailable at fit time")
            refits.append(fit)
            if len(refits) == 1 or len(refits) % 10 == 0:
                print(f"Open-to-open fit {len(refits)} on {asof}", flush=True)
        pred = tcn.predict(model, windows[i - shift:i - shift + 1], mean, scale, device)
        if not math.isfinite(pred):
            raise ValueError(f"Invalid model forecast: {asof}")
        forecasts.append({"asof_date": asof, "target_date": str(rows[i + 1]["date"]),
                          "predicted_return": pred, "fit_asof": fit["fit_asof"],
                          "max_training_label_date": fit["max_training_label_date"]})
    expected = [str(r["date"]) for r in rows if first <= r["date"] <= last]
    if [r["target_date"] for r in forecasts] != expected:
        raise ValueError("Opening forecasts do not match the trading calendar")
    return forecasts, refits


def independent_breadth(forecasts, breadth):
    held, cash_days = False, 0
    decisions = []
    for f in forecasts:
        signal = breadth.get(f["asof_date"], {})
        score = f["predicted_return"]
        confirm = (not held and cash_days >= s9.MIN_CASH_SESSIONS and score <= s7.ENTRY
                   and signal.get("breadth5") is not None and signal["breadth5"] >= s9.MIN_AVG_BREADTH
                   and signal["breadth"] >= s9.MIN_BREADTH)
        if score > s7.ENTRY or confirm:
            held = True
        elif score <= 0:
            held = False
        decisions.append({**f, **signal, "strategy_weight": float(held), "breadth_only_entry": confirm,
                          "previous_cash_sessions": cash_days})
        cash_days = 0 if held else cash_days + 1
    return decisions


def simulate(rows, forecasts, open_forecasts, breadth, fee=s7.FEE):
    decisions = {
        BASELINE: s7.weighted_forecasts(forecasts, "buffer"),
        BREADTH: independent_breadth(forecasts, breadth),
        TRADE_TARGET: s7.weighted_forecasts(open_forecasts, "buffer"),
    }
    results = {name: s3.replay(rows, signals, {"name": name, "kind": "forecast_weight"}, fee_rate=fee)
               for name, signals in decisions.items()}
    return results, decisions


def summarize(rows, runs, choices):
    metrics = {name: {**run["metrics"], **cash_audit(rows, run)} for name, run in runs.items()}
    breadth_days = {r["target_date"] for r in choices[BREADTH] if r["breadth_only_entry"]}
    metrics[BREADTH]["breadth_only_entries"] = [
        trade["date"] for trade in runs[BREADTH]["trades"]
        if trade["side"] == "BUY" and trade["date"] in breadth_days
    ]
    return metrics


def in_period(forecasts, first, last):
    return [r for r in forecasts if str(first) <= r["target_date"] <= str(last)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save("plan.json", PLAN)
    raw, audit = m.read_prices(s9.RAW)
    history = [r for r in raw if r["date"] < m.START]
    last_before = history[-1]["date"]
    tcn.torch.set_num_threads(2)
    tcn.torch.use_deterministic_algorithms(True)
    device = tcn.torch.device("cpu")

    historical_breadth = s9.load_breadth(m.START)
    original = s7.load_forecasts(s9.HISTORY_FORECAST, history, m.VALID_START, last_before)
    opening, fits = opening_forecasts(history, m.VALID_START, last_before, device)
    m.write_csv(OUT / "history_open_tcn_fits.csv", fits)
    hist = summarize(history, *simulate(history, original, opening, historical_breadth))
    folds = {}
    for name, first, last in s9.FOLDS:
        folds[name] = summarize(history, *simulate(history, in_period(original, first, last),
                                                 in_period(opening, first, last), historical_breadth))
    eligible = {}
    for name in (BREADTH, TRADE_TARGET):
        wins = sum(f[name]["total_return"] > f[BASELINE]["total_return"] for f in folds.values())
        eligible[name] = {"wins": wins, "passed": wins >= 2 and hist[name]["total_return"] > hist[BASELINE]["total_return"]
                          and hist[name]["max_drawdown"] <= hist[BASELINE]["max_drawdown"]}
    qualifying = [name for name, result in eligible.items() if result["passed"]]
    selected = max(qualifying, key=lambda name: hist[name]["total_return"], default=BASELINE)
    save("history_protocol.json", {"plan": PLAN, "rules_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   "raw_index_sha256": audit["sha256"],
                                   "history_breadth_sha256": s9.fingerprint(historical_breadth),
                                   "history": hist, "folds": folds, "eligible": eligible, "selected": selected})
    print(f"Historical selection: {selected}. {eligible}", flush=True)

    rows = [r for r in raw if r["date"] <= m.END]
    breadth = s9.load_breadth()
    if {d: r for d, r in breadth.items() if d < str(m.START)} != historical_breadth:
        raise ValueError("Breadth history changed during evaluation")
    original = s7.load_forecasts(s9.FORMAL_FORECAST, rows, m.START, m.END)
    opening, fits = opening_forecasts(rows, m.START, m.END, device)
    if fits[0]["max_training_label_date"] >= str(m.START):
        raise ValueError("First model fit uses a label from the assignment evaluation period")
    # 给首个预测仅保留日历和截至信号日的真实行情，核对未来价格对预测没有影响。
    first_test_row = next(r for r in rows if r["date"] == m.START)
    boundary_rows = history + [{**history[-1], "date": first_test_row["date"]}]
    boundary_prediction, _ = opening_forecasts(boundary_rows, m.START, m.START, device)
    if boundary_prediction != opening[:1]:
        raise ValueError("Future prices affected the initial opening forecast")
    m.write_csv(OUT / "formal_open_tcn_fits.csv", fits)
    m.write_csv(OUT / "formal_open_tcn_forecasts.csv", opening)
    runs, choices = simulate(rows, original, opening, breadth)
    midpoint = len(original) // 2
    if independent_breadth(original[:midpoint], breadth) != choices[BREADTH][:midpoint]:
        raise ValueError("Later breadth signals changed earlier decisions")
    for name, run in runs.items():
        s7.verify_ledger(rows, run)
        m.write_csv(OUT / f"{name}_trades.csv", run["trades"], s7.TRADE_FIELDS)
        m.write_csv(OUT / f"{name}_daily.csv", run["curve"])
    m.write_csv(OUT / "formal_breadth_signals.csv", choices[BREADTH], sorted({k for r in choices[BREADTH] for k in r}))
    formal = summarize(rows, runs, choices)
    sensitivity = []
    for fee in (0, 0.00025, s7.FEE, 0.001):
        r = runs if fee == s7.FEE else simulate(rows, original, opening, breadth, fee)[0]
        sensitivity.extend({"strategy": name, **run["metrics"]} for name, run in r.items())
    m.write_csv(OUT / "cost_sensitivity.csv", sensitivity)
    save("formal_results.json", {"selected_before_replay": selected, "formal": formal,
                                 "previously_viewed_evaluation": True, "ledger_check": "passed"})
    for period, stats in (("history", hist), *folds.items(), ("formal", formal)):
        print(period, [(name, round(r["total_return"] * 100, 3), round(r["max_drawdown"] * 100, 3),
                        r["order_count"], r.get("breadth_only_entries")) for name, r in stats.items()])


if __name__ == "__main__":
    main()
