"""Learn next session's open-to-close and subsequent overnight returns separately."""
import argparse
import json
from dataclasses import dataclass
from itertools import groupby
from math import expm1, fsum, isclose, log1p
from pathlib import Path

from step1_baseline import LongOnlyT1Account, buy_and_hold
from step32_entry_votes import PERIODS, ROOT, build, ledger, m, month_audit, qualifies, read, sources, tcn
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

BASE = "split_staged"
NAMES = (BASE, "open_policy", "open_overlay", "all_close", "time_buy", "time_sell", "time_both",
         "night_rescue", "timing_and_rescue")
CANDIDATES = tuple(n for n in NAMES if n not in (BASE, "all_close"))
COST = log1p(m.FEE_RATE) - log1p(-m.FEE_RATE)
INPUTS = ("return_1", "return_5", "return_20", "ma_gap_20", "volatility_20", "price_position_20",
          "candle_range", "upper_shadow", "lower_shadow", "amount_ratio_20")


def forecasts(rows, first, last):
    opening, close = (np.array([r[k] for r in rows]) for k in ("open", "close"))
    dates = [str(r["date"]) for r in rows]
    intra = np.log(close / opening)
    gap = np.r_[np.nan, np.log(opening[1:] / close[:-1])]
    x = m.build_features(rows)[:, [m.FEATURE_NAMES.index(k) for k in INPUTS]]
    extra = []
    for series in (intra, gap):
        for w in (1, 5, 20):
            averages = np.convolve(series, np.ones(w)/w, mode="valid")
            extra.append(np.r_[np.full(w-1, np.nan), averages])
    x = np.column_stack([x, *extra])
    # Both answers for feature day j are available at j+2 open.
    labels = np.column_stack((intra[1:-1], gap[2:]))
    target_days = [i for i, day in enumerate(dates) if first <= day <= last]
    result = []
    for offset, target in enumerate(target_days):
        i = target - 1
        if offset % m.REFIT_EVERY == 0:
            start, end = i - 1 - m.TRAIN_WINDOW, i - 1
            if start < 0 or not np.isfinite(x[start:end]).all() or not np.isfinite(labels[start:end]).all():
                raise ValueError("Incomplete mature training window")
            model = make_pipeline(StandardScaler(), Ridge(alpha=100.))
            model.fit(x[start:end], labels[start:end])
            fitted, label_day = dates[i], dates[end - 1 + 2]
        daylight, night = model.predict(x[i:i+1])[0]
        if not label_day <= fitted <= dates[i] < dates[target]:
            raise ValueError("Forecast clock violation")
        result.append(dict(asof_date=dates[i], target_date=dates[target], fit_asof=fitted,
                           max_training_label_date=label_day, intraday=float(daylight), overnight=float(night)))
    return result


def schedules(base, preds):
    result = {name: [] for name in NAMES}
    previous_base = independent = 0.
    for ref, pred in zip(base, preds, strict=True):
        if (ref["asof_date"], ref["target_date"]) != (pred["asof_date"], pred["target_date"]):
            raise ValueError("Models refer to different sessions")
        day, night = pred["intraday"], pred["overnight"]
        target = ref["strategy_weight"]
        if day + night > COST:
            independent = .95
        elif day + night < 0:
            independent = 0.
        overlay = target
        if day > 0 and night > 0 and day + night > COST:
            overlay = min(.95, target + .2375)
        elif day < 0 and night < 0 and day + night < -COST:
            overlay = max(0., target - .2375)
        delay_buy = target > previous_base and day < -COST
        delay_sell = target < previous_base and day > COST
        rescue = .475 if target == 0 and night > COST else target
        for name in NAMES:
            closing = target
            opening = target
            if name in ("open_policy", "open_overlay"):
                opening = closing = independent if name == "open_policy" else overlay
            else:
                if (name == "all_close" or
                    (name in ("time_buy", "time_both", "timing_and_rescue") and delay_buy) or
                    (name in ("time_sell", "time_both", "timing_and_rescue") and delay_sell)):
                    opening = previous_base
                if name in ("night_rescue", "timing_and_rescue"):
                    closing = rescue
            # A delayed exit and an overnight rescue could otherwise create a fresh buy at the open
            # followed by a same-day reduction. Only carry previously held shares in a delayed exit.
            if opening > closing and name == "timing_and_rescue" and delay_sell:
                opening = min(opening, result[name][-1]["close_target"] if result[name] else 0.)
            result[name].append({**pred, "fit_asof": max(ref["fit_asof"], pred["fit_asof"]),
                                 "max_training_label_date": max(ref["max_training_label_date"], pred["max_training_label_date"]),
                                 "base_target": target, "open_target": opening, "close_target": closing})
        previous_base = target
    return result


@dataclass
class Replay:
    metrics: dict
    curve: list
    trades: list


def simulate(bars, plan, name, fee):
    account = LongOnlyT1Account(100_000., [r["date"] for r in bars], fee)
    trades, curve = [], []
    peak = previous = account.cash
    prior_target = 0.
    for i, (bar, s) in enumerate(zip(bars, plan, strict=True)):
        on = bar["date"]
        if not s["max_training_label_date"] <= s["fit_asof"] <= s["asof_date"] < str(on) == s["target_date"]:
            raise ValueError("Trade scheduled after the cutoff")
        terminal = i == len(plan)-1
        for clock, field in (("09:30", "open"), ("15:00", "close")):
            target = 0. if terminal and field == "close" else s[field+"_target"]
            assert 0. <= target <= 1.
            price = bar[field]
            if target != prior_target or terminal and field == "close":
                delta = (account.cash + account.units*price)*target/price - account.units
                qty = abs(delta)/(1 + (fee if delta > 0 else -fee)*target)
                if qty > 1e-8 and (delta < 0 or not terminal):
                    side = "BUY" if delta > 0 else "SELL"
                    if delta > 0:
                        account.buy(on, price, qty)
                    else:
                        account.sell(on, price, qty)
                    trades.append(dict(date=str(on), time=clock, side=side, units=qty, price=price,
                                       fee=qty*price*fee, signal_asof=s["asof_date"]))
            prior_target = target
            if field == "open":
                open_units = account.units
        equity = account.cash + account.units*bar["close"]
        peak = max(peak, equity)
        curve.append(dict(timestamp=f"{on} 15:00:00", equity=equity, cash=account.cash, units=account.units,
                          open_units=open_units, daily_return=equity/previous-1, drawdown=equity/peak-1))
        previous = equity
    if account.units > 1e-8:
        raise ValueError("Unliquidated shares")
    fully_flat = (r["units"] <= 1e-8 and r["open_units"] <= 1e-8 for r in curve)
    flat = max((sum(1 for _ in group) for empty, group in groupby(fully_flat) if empty), default=0)
    return Replay(dict(name=name, total_return=previous/100_000.-1, max_drawdown=-min(r["drawdown"] for r in curve),
                       order_count=len(trades), fees_paid=fsum(f["fee"] for f in trades),
                       longest_flat_full_sessions=flat, ex_april=expm1(fsum(log1p(r["daily_return"]) for r in curve
                                                                           if r["timestamp"][5:7] != "04"))),
                  curve, trades)


def audit(result, bars, fee):
    """Independently reconstruct settled inventory, cash and closing equity from the fills."""
    by_day = {d: list(group) for d, group in groupby(result.trades, key=lambda r: r["date"])}
    cash, units = 100_000., 0.
    for b, daily in zip(bars, result.curve, strict=True):
        day = str(b["date"])
        sellable = units
        fills = by_day.pop(day, [])
        if [f["time"] for f in fills] != sorted(set(f["time"] for f in fills)):
            raise ValueError("Invalid auction sequence")
        for f in fills:
            p = b[{"09:30":"open", "15:00":"close"}[f["time"]]]
            q = f["units"]
            if not (f["signal_asof"] < day and p == f["price"] and q > 0 and isclose(f["fee"], p*q*fee)):
                raise ValueError("Bad fill")
            if f["side"] == "SELL":
                if q > sellable + 1e-8:
                    raise ValueError("Intraday lot sold")
                sellable -= q
                q = -q
            units += q
            cash -= q*p + f["fee"]
            if cash < -1e-7 or units < -1e-8:
                raise ValueError("Leverage or short position")
        if not isclose(cash + units*b["close"], daily["equity"], abs_tol=1e-7):
            raise ValueError("Daily mark mismatch")
    if by_day or abs(units) > 1e-8:
        raise ValueError("Fill calendar or closing inventory mismatch")


def evaluate(phase, prices, out):
    first, last = PERIODS[phase]
    rows = [r for r in prices if r["date"] <= last]
    bars = [r for r in rows if r["date"] >= first]
    base = build(rows, sources(phase, rows)["repo"], {})[0][BASE]
    preds = forecasts(rows, str(first), str(last))
    cut = max(2, len(preds)//2)
    # Replace even the last target bar by a dummy. All predictions through that open must survive.
    prefix_rows = rows[:len(rows)-len(bars)+cut]
    prefix_rows[-1] = {**prefix_rows[-2], "date": prefix_rows[-1]["date"]}
    if forecasts(prefix_rows, str(first), preds[cut-1]["target_date"]) != preds[:cut]:
        raise ValueError("Future price changed the forecast")
    plans = schedules(base, preds)
    benchmark = buy_and_hold(bars, fee_rate=m.FEE_RATE)[1]["total_return"]
    known = next(r for r in read(ROOT / "step32_entry_votes/comparison.csv") if r["phase"] == phase and r["name"] == BASE)
    control, account = ledger(rows, BASE, base)
    metrics, curves = {}, {}
    for name, plan in plans.items():
        run = simulate(bars, plan, name, m.FEE_RATE)
        stress = simulate(bars, plan, name, 2*m.FEE_RATE)
        audit(run, bars, m.FEE_RATE)
        audit(stress, bars, 2*m.FEE_RATE)
        s = run.metrics
        s.update(phase=phase, double_fee_return=stress.metrics["total_return"], benchmark=benchmark,
                 excess_return_difference=s["total_return"]-benchmark, trading_days=len(bars))
        if name == BASE:
            keys = ("total_return", "max_drawdown", "fees_paid", "order_count", "double_fee_return")
            if any(not isclose(s[k], float(known[k]), abs_tol=1e-9) for k in keys):
                raise ValueError("Archived baseline failed to reproduce")
            if any(not isclose(a["equity"], b["equity"], abs_tol=1e-8) for a, b in zip(run.curve, account["curve"][1:], strict=True)):
                raise ValueError("Original daily ledger failed to reproduce")
        metrics[name], curves[name] = s, run.curve
        if phase == "course":
            for stem, data in (("signals", plan), ("daily", run.curve), ("trades", run.trades)):
                m.write_csv(out / f"{name}_{stem}.csv", data)
    if phase == "course":
        tcn.write_json(out / "monthly.json", {n: month_audit(c, curves[BASE]) for n, c in curves.items() if n != BASE})
        # Only the segment with both label endpoints inside the coursework window is scored.
        iday = np.log([b["close"]/b["open"] for b in bars])
        night = np.log([b["open"]/a["close"] for a, b in zip(bars, bars[1:])])
        score = {}
        for key, actual in (("intraday", iday), ("overnight", night)):
            pred = np.array([p[key] for p in preds[:len(actual)]])
            score[key] = {"n":len(actual), "direction_accuracy":float(np.mean(np.sign(pred)==np.sign(actual))),
                          "mae_log_return":float(np.abs(pred-actual).mean()), "zero_mae":float(np.abs(actual).mean())}
        tcn.write_json(out / "forecast_checks.json", score)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "step34_session_policy")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    prices, info = m.read_prices(m.ROOT / "data/raw/SZ#399006.txt")
    tcn.write_json(out / "protocol.json", {
        "prices_sha256":info["sha256"], "training":"Ridge alpha 100, standardized only on latest 1260 mature samples, "
        "refit every 20 sessions; both future segment labels mature at j+2 open. All orders scheduled the previous close.",
        "features":[*INPUTS, "intraday and gap log return means over 1, 5 and 20 sessions"],
        "rules": list(NAMES), "hurdle_log_return":COST, "fees_each_side":m.FEE_RATE,
        "schedule":"Prior split-entry signals. Delay target increases to close on negative intraday forecast below "
        "-hurdle; delay decreases on forecast above hurdle. Overnight rescue when base flat and predicted overnight "
        "above hurdle: buy to 47.5% at close. Independent and overlay alternatives trade only at open. "
        "No last-day buys; precommitted final liquidation. Full-cash streak counts open and close after fills.",
        "selection":"Both pre-course segments must beat split_staged net and at double fees, drawdown no deeper, "
        "and at least two completely flat sessions. Among passes pick best recent return. Otherwise retain prior.",
        "formal_already_viewed":True, "benchmark":"Buy at first open, sell at last close; same fees.",
    })
    with threadpool_limits(limits=1):
        scores = {phase:evaluate(phase, prices, out) for phase in ("early", "history")}
        checks = {n:{p:qualifies(group[n], group[BASE]) for p, group in scores.items()} for n in CANDIDATES}
        passing = [n for n, c in checks.items() if all(c.values())]
        selected = max(passing, key=lambda n:scores["history"][n]["total_return"], default=BASE)
        tcn.write_json(out / "selection_before_course.json", {"checks":checks, "passed":passing, "selected":selected})
        scores["course"] = evaluate("course", prices, out)
    m.write_csv(out / "comparison.csv", [s for group in scores.values() for s in group.values()])
    for phase, group in scores.items():
        print(phase, [(n, f"{s['total_return']:.2%}", f"{s['max_drawdown']:.2%}") for n,s in group.items()])
    print("Selected:", selected)


if __name__ == "__main__":
    main()
