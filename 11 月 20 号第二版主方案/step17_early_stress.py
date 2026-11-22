"""Replay the frozen Step 16 strategies on teacher data through 2018."""
from collections import defaultdict
from datetime import date
from hashlib import sha256
import json
import math

import step1_baseline as s1
import step16_refinement_experiment as s16


s13 = s16.s13
tcn = s16.tcn
OUT = s1.ROOT / "outputs/step17_history"
FIRST = date(2016, 1, 1)
LAST = date(2018, 12, 31)
YEARS = range(FIRST.year, LAST.year + 1)
MARKET_CUTOFF = 0.05
REGIME_LOOKBACK = 60


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def eligible_start(rows):
    """Find a common first target date without shortening either training window."""
    features = s13.m.build_features(rows)
    windows = tcn.np.lib.stride_tricks.sliding_window_view(features, tcn.LOOKBACK, axis=0)
    shift = tcn.LOOKBACK - 1
    valid_windows = tcn.np.isfinite(windows).all(axis=(1, 2))
    for i in range(1, len(rows)):
        if not FIRST <= rows[i]["date"] <= LAST:
            continue
        asof = i - 1
        if asof - shift < s13.m.TRAIN_WINDOW or not valid_windows[asof - shift]:
            continue
        tcn_training_count = valid_windows[:asof - shift].sum()
        utility_end = asof - 1
        utility_start = utility_end - s13.m.TRAIN_WINDOW
        if (tcn_training_count >= s13.m.TRAIN_WINDOW and utility_start >= 0
                and tcn.np.isfinite(features[utility_start:utility_end]).all()):
            return rows[i]["date"]
    raise ValueError("No target has sufficient mature data in the early evaluation period.")


def save_replay(name, rows, forecasts, direct, first, last):
    choices = s16.choose(rows, forecasts, direct, first, last)
    metrics, runs = s16.with_market(rows, choices)
    selected = [r for r in rows if first <= r["date"] <= last]
    marks, buy_hold, trades = s1.buy_and_hold(selected, fee_rate=s13.s7.FEE)
    metrics["buy_hold_fee_adjusted"] = buy_hold
    for method, run in runs.items():
        s1.write_csv(OUT / f"{name}_{method}_daily.csv", run["curve"])
        s1.write_csv(OUT / f"{name}_{method}_trades.csv", run["trades"], s13.s7.TRADE_FIELDS)
        s1.write_csv(OUT / f"{name}_{method}_signals.csv", choices[method])
    s1.write_csv(OUT / f"{name}_buy_hold_daily.csv", marks)
    s1.write_csv(OUT / f"{name}_buy_hold_trades.csv", trades)
    return {
        "first": str(selected[0]["date"]),
        "last": str(selected[-1]["date"]),
        "trading_days": len(selected),
        "metrics": metrics,
    }, runs


def realized_months(rows, runs, first, last):
    # The benchmark starts at the first open and otherwise uses close-to-close returns.
    # Use benchmark month outcomes only for retrospective diagnostics.
    months = defaultdict(lambda: defaultdict(list))
    previous = next(r["open"] for r in rows if r["date"] == first)
    for r in rows:
        if first <= r["date"] <= last:
            months[str(r["date"])[:7]]["benchmark"].append(math.log(r["close"] / previous))
            previous = r["close"]
    for name, run in runs.items():
        for mark in run["curve"][1:]:
            months[mark["timestamp"][:7]][name].append(math.log1p(mark["daily_return"]))
    return {
        month: {"phase": "up" if market >= MARKET_CUTOFF else "down" if market <= -MARKET_CUTOFF else "sideways",
                "benchmark_index_gross": market,
                "results": {name: math.expm1(math.fsum(returns)) for name, returns in changes.items() if name != "benchmark"}}
        for month, changes in sorted(months.items())
        for market in [math.expm1(math.fsum(changes["benchmark"]))]
    }


def causal_regimes(rows, runs, first, last):
    """Attribute each session to the market regime observable before its open."""
    rates = defaultdict(lambda: defaultdict(list))
    exposures = defaultdict(lambda: defaultdict(list))
    positions = {row["date"]: i for i, row in enumerate(rows)}
    marks = {name: {mark["timestamp"][:10]: mark for mark in run["curve"][1:]}
             for name, run in runs.items()}
    for r in rows:
        if not first <= r["date"] <= last:
            continue
        i = positions[r["date"]]
        signal = rows[i - 1]["close"]
        trend = signal / rows[i - 1 - REGIME_LOOKBACK]["close"] - 1
        phase = "bullish" if trend > MARKET_CUTOFF else "bearish" if trend < -MARKET_CUTOFF else "neutral"
        rates[phase]["benchmark"].append(r["close"] / (r["open"] if r["date"] == first else signal) - 1)
        for name, strategy in marks.items():
            mark = strategy[str(r["date"])]
            rates[phase][name].append(mark["daily_return"])
            exposures[phase][name].append(mark["open_weight_after_trade"])
    return {
        phase: {
            "sessions": len(values["benchmark"]),
            "mean_daily_returns": {name: math.fsum(series) / len(series) for name, series in values.items()},
            "compounded_attribution": {name: math.expm1(math.fsum(math.log1p(day) for day in series))
                                      for name, series in values.items()},
            "mean_open_weights": {name: math.fsum(series) / len(series)
                                  for name, series in exposures[phase].items()},
        }
        for phase, values in sorted(rates.items())
    }


def main():
    raw, data = s1.read_prices(s13.s9.RAW)
    rows = [r for r in raw if r["date"] <= LAST]
    first = eligible_start(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {
        "purpose": "early stress replay; separate from the coursework and new market data",
        "source": str(s13.s9.RAW),
        "source_sha256": data["sha256"],
        "last_row_available_to_models": str(rows[-1]["date"]),
        "requested_first": str(FIRST),
        "first_common_eligible_target": str(first),
        "refit_start": str(first),
        "training_size": s13.m.TRAIN_WINDOW,
        "refit_every": s13.m.REFIT_EVERY,
        "rules": "the four frozen Step 16 rules, including the original TCN",
        "balance": "continuous account and independent fresh accounts per calendar year",
        "reset_rules_at_year_boundary_for_fresh_account": True,
        "annual_refit_cadence_restarts": False,
        "course_order": "first opening to final closing liquidation, long-only, T+1",
        "fee_each_side": s13.s7.FEE,
        "monthly_market_phases": {
            "up": "gross index rose at least 5%",
            "sideways": "gross index change between -5% and +5%, exclusive",
            "down": "gross index fell at least 5%",
            "role": "ex post labels only, not a training feature or trading filter",
        },
        "causal_session_regimes": {
            "signal": "as-of close divided by close 60 trading sessions before as-of, minus one",
            "threshold": MARKET_CUTOFF,
            "role": "attribution based only on pre-open values; does not change strategy decisions",
        },
        "rule_source_sha256": {
            name: sha256((s1.ROOT / name).read_bytes()).hexdigest()
            for name in (*s16.FROZEN_FILES, "step17_early_stress.py")
        },
    })
    tcn.torch.set_num_threads(2)
    tcn.torch.use_deterministic_algorithms(True)
    device = tcn.torch.device("cpu")
    forecasts, fits, _ = tcn.rolling(rows, first, LAST, device)
    direct = s13.utility_decisions(rows, first, LAST)
    dates = [str(r["date"]) for r in rows if first <= r["date"] <= LAST]
    for decisions in (forecasts, direct):
        if [row["target_date"] for row in decisions] != dates or any(
            not (d["max_training_label_date"] <= d["fit_asof"] <= d["asof_date"] < d["target_date"])
            for d in decisions
        ):
            raise ValueError("Early model date alignment or label maturity failed.")
    if any(f["n_train"] != s13.m.TRAIN_WINDOW for f in fits):
        raise ValueError("The early history changed the TCN training size.")

    check = [dict(r) for r in rows if r["date"] <= first]
    if check[-1]["date"] != first:
        raise ValueError("First prediction does not have a preceding observation.")
    for field in ("open", "high", "low", "close", "volume", "amount"):
        check[-1][field] *= 1.05
    first_forecast, *_ = tcn.rolling(check, first, first, device)
    if first_forecast[0] != forecasts[0] or s13.utility_decisions(check, first, first)[0] != direct[0]:
        raise ValueError("The unknown first target day's data changed the first signal.")

    s1.write_csv(OUT / "tcn_forecasts.csv", forecasts)
    s1.write_csv(OUT / "tcn_fits.csv", [{k: v for k, v in f.items() if k != "seconds"} for f in fits])
    s1.write_csv(OUT / "utility_forecasts.csv", direct)
    continuous, runs = save_replay("continuous", rows, forecasts, direct, first, LAST)
    annual = {}
    for y in YEARS:
        start, end = max(first, date(y, 1, 1)), date(y, 12, 31)
        if start <= end:
            annual[str(y)], _ = save_replay(str(y), rows, forecasts, direct, start, end)
    months = realized_months(rows, runs, first, LAST)
    regimes = causal_regimes(rows, runs, first, LAST)
    for name, run in runs.items():
        monthly_growth = math.prod(1 + month["results"][name] for month in months.values())
        regime_growth = math.prod(1 + regime["compounded_attribution"][name] for regime in regimes.values())
        if not all(math.isclose(growth - 1, run["metrics"]["total_return"], abs_tol=1e-12)
                   for growth in (monthly_growth, regime_growth)):
            raise ValueError(f"Month or causal regime attribution does not reconcile for {name}.")
    save(OUT / "results.json", {"continuous": continuous, "annual": annual, "months": months,
                                "causal_session_regimes": regimes,
                                "first_fit_asof": fits[0]["fit_asof"], "number_of_tcn_fits": len(fits),
                                "ledger_and_first_prediction_checks": "passed"})
    for name, period in {"continuous": continuous, **annual}.items():
        print(name, period["first"], period["last"], period["trading_days"],
              {k: (v["total_return"], v["max_drawdown"]) for k, v in period["metrics"].items()
               if isinstance(v, dict)}, flush=True)


if __name__ == "__main__":
    main()
