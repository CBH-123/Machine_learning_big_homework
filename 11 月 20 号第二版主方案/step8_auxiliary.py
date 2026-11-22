"""在同一小型 TCN 和同一交易规则下，比较辅助指数日线的增量。"""
import argparse
import csv
import hashlib
import json
from datetime import date
from itertools import groupby

import step2_models as m
import step6_tcn as tcn
import step7_net_return as s7
from step8_download_aux import OUT as AUX, SOURCE, START as AUX_START, SYMBOLS
import numpy as np

OUT = m.ROOT / "outputs" / "step8"
RAW = m.ROOT / "data" / "raw" / "SZ#399006.txt"
GROUPS = {
    "target_only": (),
    "plus_csi300": ("csi300",),
    "plus_chinext_composite": ("chinext_composite",),
    "plus_both": tuple(SYMBOLS),
}
FEATURES = ("return_1", "return_5", "return_20", "volatility_20", "relative_return_5_vs_target")
FOLDS = (
    ("2023", date(2023, 1, 1), date(2023, 12, 31)),
    ("2024", date(2024, 1, 1), date(2024, 12, 31)),
    ("2025_before_cutoff", date(2025, 1, 1), date(2025, 10, 9)),
)
PLAN = {
    "source": SOURCE, "symbols": SYMBOLS, "source_first": str(AUX_START),
    "features": FEATURES, "groups": GROUPS, "features_asof": "20:00 Asia/Shanghai same-day closes",
    "first_fit_before": str(m.START), "history_evaluation_start": str(FOLDS[0][1]),
    "fee_each_side": s7.FEE, "rule": s7.RULES["buffer"], "model": "step6_tcn",
    "selection": "beats baseline historical net return, MDD no larger, wins >= 2 of 3 subperiods; then max net return",
    "formal_data_previously_viewed": True,
}


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_close(path, first, stop):
    with path.open(encoding="utf-8-sig", newline="") as f:
        obs = [(date.fromisoformat(r["date"]), float(r["close"])) for r in csv.DictReader(f)]
    report_days = [d for d, _ in obs]
    if (not obs or report_days != sorted(set(report_days)) or not first <= report_days[0] <= report_days[-1] < stop
            or any(not np.isfinite(p) or p <= 0 for _, p in obs)):
        raise ValueError(f"Invalid closes: {path}")
    return dict(obs)


def feature_block(rows, aux):
    close = np.array([aux.get(r["date"], np.nan) for r in rows])
    target = np.array([r["close"] for r in rows])
    ret = np.r_[np.nan, close[1:] / close[:-1] - 1]
    features = np.full((len(close), len(FEATURES)), np.nan)
    for i in range(20, len(close)):
        if np.isfinite(close[i-20:i+1]).all():
            r5 = close[i] / close[i-5] - 1
            features[i] = [ret[i], r5, close[i] / close[i-20] - 1, np.std(ret[i-19:i+1], ddof=1),
                           r5 - (target[i] / target[i-5] - 1)]
    return features


def aux_blocks(rows, symbols, formal=False):
    blocks, audit = {}, {}
    expected = {r["date"] for r in rows if AUX_START <= r["date"]}
    for name in symbols:
        hp = AUX / f"history_{name}.csv"
        series = load_close(hp, AUX_START, m.START)
        paths = {"history_sha256": sha(hp)}
        if formal:
            fp = AUX / f"formal_{name}.csv"
            series.update(load_close(fp, m.START, date.fromordinal(m.END.toordinal() + 1)))
            paths["formal_sha256"] = sha(fp)
        missing, unexpected = sorted(expected - series.keys()), sorted(series.keys() - expected)
        if missing or unexpected:
            raise ValueError(f"{name}: missing {len(missing)} {missing[:5]}; unexpected {len(unexpected)} {unexpected[:5]}")
        block = feature_block(rows, series)
        prefix = len(rows) - 25
        np.testing.assert_array_equal(block[:prefix], feature_block(rows[:prefix], series))
        blocks[name] = block
        audit[name] = {"count": len(series), "first": str(min(series)), "last": str(max(series)), **paths}
    return blocks, audit


def cash_audit(rows, result):
    """Use actual positions and fills. An exit at the open counts only in after-open runs."""
    curve, trades = result["curve"][1:], result["trades"]
    px = {str(r["date"]): r["open"] for r in rows}
    orders = {r["date"] for r in trades}
    flags = [(r["timestamp"][:10], float(r["open_weight_after_trade"]) == 0 and float(r["units"]) == 0)
             for r in curve]

    def runs(days):
        return [{ "start": g[0][0], "end": g[-1][0], "days": len(g)}
                for k, group in groupby(days, key=lambda item: item[1]) if k
                for g in [list(group)]]

    post_open = runs(flags)
    no_trades = runs([(d, flat and d not in orders) for d, flat in flags])
    best = max(post_open, key=lambda r: r["days"], default=None)
    extra = {}
    if best:
        first_next_buy = next((r for r in trades if r["date"] > best["end"] and r["side"] == "BUY"), None)
        if first_next_buy:
            d = first_next_buy["date"]
            extra = {"next_entry": d, "index_return_from_first_cash_open_to_reentry": px[d] / px[best["start"]] - 1}
    return {"cash_days_after_open": sum(flat for _, flat in flags), "after_open_spells": len(post_open),
            "longest_after_open": best, "longest_without_any_order": max(no_trades, key=lambda r: r["days"], default=None),
            **extra}


def analyze(rows, first, last, name, blocks, device):
    extra = np.column_stack([blocks[s] for s in GROUPS[name]]) if GROUPS[name] else None
    prediction, fits, _ = tcn.rolling(rows, first, last, device, extra=extra)
    for f in prediction:
        f["prediction_time"] = f["asof_date"] + " 20:00:00"
    expected = [str(r["date"]) for r in rows if first <= r["date"] <= last]
    if [f["target_date"] for f in prediction] != expected:
        raise ValueError("Forecast and official target calendars differ")
    if first == m.START and not fits[0]["max_training_label_date"] < str(m.START):
        raise ValueError("First fit used a formal-period label")
    result = s7.replay(rows, prediction, "buffer")
    _, errors = m.evaluate_forecasts(rows, prediction)
    return prediction, fits, result, {**result["metrics"], **errors, **cash_audit(rows, result)}


def historical(rows, device):
    save("plan.json", PLAN)
    rows = [r for r in rows if r["date"] < m.START]
    blocks, audit = aux_blocks(rows, SYMBOLS)
    stats = {}
    details = []
    for name in GROUPS:
        print(f"History: {name}", flush=True)
        pred, fits, result, summary = analyze(rows, FOLDS[0][1], FOLDS[-1][2], name, blocks, device)
        stats[name] = summary
        details.append({"features": name, "period": "full_history", **summary})
        for label, first, last in FOLDS:
            subset = [f for f in pred if str(first) <= f["target_date"] <= str(last)]
            # Independent portfolios for subperiods. Refit cadence stays the same across all versions.
            r = s7.replay(rows, subset, "buffer")
            details.append({"features": name, "period": label, **r["metrics"], **cash_audit(rows, r)})
        m.write_csv(OUT / f"history_{name}_forecasts.csv", pred)
        m.write_csv(OUT / f"history_{name}_fits.csv", fits)
    by = {(r["features"], r["period"]): r for r in details}
    base = stats["target_only"]
    eligibility = {}
    for name in list(GROUPS)[1:]:
        wins = sum(by[name, fold]["total_return"] > by["target_only", fold]["total_return"] for fold, *_ in FOLDS)
        r = stats[name]
        eligibility[name] = {"wins": wins, "eligible": (wins >= 2 and r["total_return"] > base["total_return"]
                                                        and r["max_drawdown"] <= base["max_drawdown"])}
    candidates = [name for name in eligibility if eligibility[name]["eligible"]]
    selected = max(candidates, key=lambda name: (stats[name]["total_return"], -stats[name]["max_drawdown"]), default=None)
    m.write_csv(OUT / "history_comparison.csv", details, sorted({k for r in details for k in r}))
    frozen = {"plan_sha256": sha(OUT / "plan.json"), "script_sha256": sha(__file_path()),
              "tcn_sha256": sha(m.ROOT / "step6_tcn.py"), "official_data_sha256": sha(RAW),
              "history_data": audit, "selected_auxiliary": selected, "selection": eligibility, "results": details}
    save("history_protocol.json", frozen)
    print(json.dumps({"selected_auxiliary": selected, "eligibility": eligibility,
                      "historical": {k: [v["total_return"], v["max_drawdown"]] for k, v in stats.items()}}, indent=2))


def __file_path():
    return m.ROOT / "step8_auxiliary.py"


def formal(rows, device):
    frozen = json.loads((OUT / "history_protocol.json").read_text(encoding="utf-8"))
    for key, path in (("plan_sha256", OUT / "plan.json"), ("script_sha256", __file_path()),
                      ("tcn_sha256", m.ROOT / "step6_tcn.py"), ("official_data_sha256", RAW)):
        if sha(path) != frozen[key]:
            raise ValueError("Previously frozen plan, code or target data changed.")
    selected = frozen["selected_auxiliary"]
    if not selected:
        print("No auxiliary candidate passed the historical acceptance rule.")
        return
    rows = [r for r in rows if r["date"] <= m.END]
    blocks, audit = aux_blocks(rows, GROUPS[selected], formal=True)
    if any(d["history_sha256"] != frozen["history_data"][k]["history_sha256"] for k, d in audit.items()):
        raise ValueError("Previously frozen auxiliary history changed.")
    stats = {}
    for name in ("target_only", selected):
        print(f"Formal replay: {name}", flush=True)
        pred, fits, result, summary = analyze(rows, m.START, m.END, name, blocks, device)
        stats[name] = summary
        m.write_csv(OUT / f"formal_{name}_forecasts.csv", pred)
        m.write_csv(OUT / f"formal_{name}_fits.csv", fits)
        m.write_csv(OUT / f"formal_{name}_daily.csv", result["curve"])
        m.write_csv(OUT / f"formal_{name}_trades.csv", result["trades"], s7.TRADE_FIELDS)
    save("formal_results.json", {"selected": selected, "data": audit, "results": stats})
    print(json.dumps(stats, indent=2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--formal", action="store_true")
    args = cli.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, _ = m.read_prices(RAW)
    tcn.torch.set_num_threads(2)
    tcn.torch.use_deterministic_algorithms(True)
    device = tcn.torch.device("cpu")
    if args.formal:
        formal(rows, device)
    else:
        historic_cash = {name: cash_audit(rows, {
            "curve": s7.s3.read_csv(s7.OUT / f"{name}_daily.csv"),
            "trades": s7.s3.read_csv(s7.OUT / f"{name}_trades.csv"),
        }) for name in ("original", "selected")}
        save("step7_cash_audit.json", historic_cash)
        historical(rows, device)


if __name__ == "__main__":
    main()
