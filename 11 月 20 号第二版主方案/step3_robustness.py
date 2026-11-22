"""第三步：同仓位诊断、逐年历史验证、仅用过去选择风控。

保留第二步全部结果。正式区间沿用第二步已经保存的逐日预测，不重新挑模型参数。
事后同仓位与循环移位仅作解释，绝不进入规则选择。
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import step2_models as m
import numpy as np

OUT = m.ROOT / "outputs/step3"
PROFILES = ("raw", "vol15", "trend20", "trend20_vol15")
THRESHOLDS = (0.0, 0.001, 0.002)
MODEL_NAMES = ("ridge", "random_forest")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_forecasts(path):
    """只保留预测时已知列；不把事后真实收盘或涨跌幅读入决策数据。"""
    keep = ("model", "asof_date", "prediction_time", "target_date", "close_asof",
            "predicted_return", "predicted_close", "annual_volatility_20", "ma_gap_20",
            "fit_id", "fit_asof", "max_training_label_date")
    numeric = {"close_asof", "predicted_return", "predicted_close", "annual_volatility_20", "ma_gap_20"}
    result = {name: [] for name in MODEL_NAMES}
    for r in read_csv(path):
        if r["model"] in result:
            result[r["model"]].append({k: float(r[k]) if k in numeric else r[k] for k in keep})
    return result


def policy_weights(forecasts, spec):
    kind = spec.get("kind", "ml")
    if kind == "forecast_weight":
        return np.array([float(f["strategy_weight"]) for f in forecasts])
    if kind == "fixed":
        return np.full(len(forecasts), spec["weight"], dtype=float)
    if kind == "ma20":
        return np.array([float(f["ma_gap_20"] > 0) for f in forecasts])
    weights = np.array([float(f["predicted_return"] > spec["threshold"]) for f in forecasts])
    profile = spec.get("profile", "raw")
    if profile not in PROFILES:
        raise ValueError("未知风控方案")
    if "trend20" in profile:
        weights *= np.array([float(f["ma_gap_20"] > 0) for f in forecasts])
    if "vol15" in profile:
        weights *= np.array([min(1.0, 0.15/max(f["annual_volatility_20"], 1e-8)) for f in forecasts])
    return weights


def replay(rows, forecasts, spec, fee_rate=m.FEE_RATE, override_weights=None,
           provenance="causal", capital=100_000, rebalance_on_weight_change_only=False,
           liquidate_at_end=True):
    """与第二步相同的撮合口径，另记录开盘调仓后实际仓位用于对照。

    provenance=ex_post 的外部持仓序列仅用于诊断，不能伪装成当时可执行的决策。
    可选择只在目标仓位变化时调仓；持有期间实际仓位随价格变动。
    默认在最后一天禁增仓并收盘清算。观察继续运行的账户时可只按最后收盘估值。
    """
    if not forecasts or provenance not in ("causal", "ex_post"):
        raise ValueError("预测或来源标记无效")
    if override_weights is not None and provenance != "ex_post":
        raise ValueError("外部事后持仓必须明确标为 ex_post")
    if spec.get("ex_post", False) and provenance != "ex_post":
        raise ValueError("事后同仓位规则不能标为可执行决策")
    weights = np.asarray(override_weights if override_weights is not None else policy_weights(forecasts, spec))
    if len(weights) != len(forecasts) or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1+1e-12):
        raise ValueError("目标仓位必须是0到1之间的有限数")
    weights = np.clip(weights, 0, 1)
    by_date = {str(r["date"]): r for r in rows}
    days = [by_date[f["target_date"]] for f in forecasts]
    account = m.LongOnlyT1Account(capital, [r["date"] for r in days], fee_rate)
    curve = [{"timestamp": f'{days[0]["date"]} 09:29:00', "equity": capital,
              "cash": capital, "units": 0, "target_weight": 0, "open_weight_after_trade": 0,
              "close_weight": 0, "daily_return": 0, "drawdown": 0}]
    trades = []
    peak = previous = capital
    for i, (row, f, weight) in enumerate(zip(days, forecasts, weights)):
        on, price = row["date"], row["open"]
        terminal = liquidate_at_end and i == len(days) - 1
        if not f["asof_date"] < str(on):
            raise ValueError("交易信号不能使用当天完整行情")
        if f.get("max_training_label_date") and not f["max_training_label_date"] <= f["fit_asof"] <= f["asof_date"]:
            raise ValueError("训练答案越过了预测时点")
        difference = (account.cash+account.units*price)*float(weight)/price-account.units
        if rebalance_on_weight_change_only:
            if i and weight == weights[i-1]:
                difference = 0
            elif difference > 0:
                difference /= 1 + float(weight) * fee_rate
            elif difference < 0:
                difference /= 1 - float(weight) * fee_rate

        def record(side, q, p, clock, signal):
            trades.append({"date": str(on), "time": clock, "side": side, "units": q,
                           "price": p, "fee": q*p*fee_rate, "signal_asof": signal,
                           "provenance": provenance})

        signal = f["asof_date"] if provenance == "causal" else "EX_POST_DIAGNOSTIC"
        if difference > 1e-8 and not terminal:
            q = min(difference, account.cash/(price*(1+fee_rate)))
            if q > 1e-8:
                account.buy(on, price, q)
                record("BUY", q, price, "09:30", signal)
        elif difference < -1e-8:
            q = min(-difference, account.sellable(on))
            if q > 1e-8:
                account.sell(on, price, q)
                record("SELL", q, price, "09:30", signal)
        open_equity = account.cash+account.units*price
        open_weight = account.units*price/open_equity
        if terminal and account.units > 1e-8:
            q = account.units
            account.sell(on, row["close"], q)
            record("SELL", q, row["close"], "15:00", "PRECOMMITTED_END")
        equity = account.cash+account.units*row["close"]
        peak = max(peak, equity)
        curve.append({"timestamp": f"{on} 15:00:00", "equity": equity,
                      "cash": account.cash, "units": account.units, "target_weight": float(weight),
                      "open_weight_after_trade": open_weight,
                      "close_weight": account.units*row["close"]/equity,
                      "daily_return": equity/previous-1, "drawdown": equity/peak-1})
        previous = equity
    daily = np.array([r["daily_return"] for r in curve[1:]])
    std = np.std(daily, ddof=1) if len(daily) > 1 else 0
    metrics = {"name": spec["name"], "provenance": provenance, "trading_days": len(days),
               "fee_rate_each_side": fee_rate, "final_equity": curve[-1]["equity"],
               "total_return": curve[-1]["equity"]/capital-1,
               "annualized_return": (curve[-1]["equity"]/capital)**(252/len(days))-1,
               "max_drawdown": -min(r["drawdown"] for r in curve),
               "sharpe_rf_zero": float(np.mean(daily)/std*np.sqrt(252)) if std > 1e-12 else 0.0,
               "average_open_weight": float(np.mean([r["open_weight_after_trade"] for r in curve[1:]])),
               "average_close_weight": float(np.mean([r["close_weight"] for r in curve[1:]])),
               "order_count": len(trades), "fees_paid": sum(t["fee"] for t in trades)}
    return {"curve": curve, "trades": trades, "metrics": metrics}


def select_before(rows_before, predictions_before, decision_date, thresholds=THRESHOLDS):
    """内层仅接受决策日前的真实数据，按最差半段收益减回撤选择风控。"""
    cutoff = str(decision_date)
    if any(r["date"] >= decision_date for r in rows_before):
        raise ValueError("规则选择收到了决策日或之后的行情")
    if any(f["target_date"] >= cutoff or f.get("provenance", "causal") != "causal"
           for fs in predictions_before.values() for f in fs):
        raise ValueError("规则选择只能使用过去的真实预测，不能使用事后诊断")
    inner_start = date(decision_date.year-2, decision_date.month, decision_date.day)
    selected_forecasts = {name: [f for f in predictions_before[name] if str(inner_start) <= f["target_date"] < cutoff]
                          for name in MODEL_NAMES}
    model_metrics = []
    for name in MODEL_NAMES:
        fs = selected_forecasts[name]
        if len(fs) < 252:
            raise ValueError("内层历史验证不足一个交易年")
        _, metric = m.evaluate_forecasts(rows_before, fs)
        model_metrics.append(metric)
    model = min(model_metrics, key=lambda r: (r["return_rmse"], r["model"]))["model"]
    fs = selected_forecasts[model]
    midpoint = len(fs)//2
    candidates = []
    for threshold in thresholds:
        for profile in PROFILES:
            spec = {"name": f"{profile}_t{threshold}", "kind": "ml", "profile": profile,
                    "threshold": threshold}
            full = replay(rows_before, fs, spec)["metrics"]
            halves = [replay(rows_before, part, spec)["metrics"] for part in (fs[:midpoint], fs[midpoint:])]
            score = min(h["annualized_return"]-h["max_drawdown"] for h in halves)
            candidates.append({"profile": profile, "threshold": threshold,
                               "full_return": full["total_return"], "full_drawdown": full["max_drawdown"],
                               "half1_return": halves[0]["total_return"], "half1_drawdown": halves[0]["max_drawdown"],
                               "half2_return": halves[1]["total_return"], "half2_drawdown": halves[1]["max_drawdown"],
                               "worst_half_score": score})
    best = max(candidates, key=lambda r: r["worst_half_score"])
    choice = {"decision_date": cutoff, "inner_first_target": fs[0]["target_date"],
              "inner_last_target": fs[-1]["target_date"], "inner_predictions": len(fs),
              "half1_last_target": fs[midpoint-1]["target_date"],
              "half2_first_target": fs[midpoint]["target_date"],
              "model": model, "profile": best["profile"], "threshold": best["threshold"],
              "selection_rule": "minimum of two half-period (annualized return - max drawdown)",
              "worst_half_score": best["worst_half_score"]}
    return choice, model_metrics, candidates


def subset_forecasts(predictions, first=None, before=None, last=None):
    return {name: [f for f in fs if (first is None or f["target_date"] >= str(first))
                   and (before is None or f["target_date"] < str(before))
                   and (last is None or f["target_date"] <= str(last))]
            for name, fs in predictions.items()}


def matched_exposure(rows, forecasts, reference, name):
    """用全期实际平均开盘仓位确定对照，仅供事后诊断，不能参与选策略。"""
    weight = reference["metrics"]["average_open_weight"]
    spec = {"name": name, "kind": "fixed", "weight": weight, "ex_post": True}
    result = replay(rows, forecasts, spec, provenance="ex_post")
    result["metrics"]["matched_target_weight"] = weight
    return result


def shift_diagnostic(rows, forecasts, reference):
    schedule = np.array([r["open_weight_after_trade"] for r in reference["curve"][1:]])
    # 只平移完整序列，保留多空/持有段结构；实际交易仍检查T+1和末日禁止加仓。
    shifts = []
    for shift in range(1, len(schedule)):
        result = replay(rows, forecasts, {"name": f"shift_{shift}"},
                        override_weights=np.roll(schedule, shift), provenance="ex_post")
        shifts.append({"shift": shift, **result["metrics"]})
    original = reference["metrics"]["total_return"]
    count = sum(r["total_return"] >= original for r in shifts)
    summary = {"original_return": original, "shift_count": len(shifts),
               "shifts_at_least_as_good": count, "fraction_at_least_as_good": count/len(shifts),
               "median_shift_return": float(np.median([r["total_return"] for r in shifts])),
               "diagnostic_only": True, "formal_p_value": False,
               "note": "ex-post circular shifts; no causal deployability or formal significance claim"}
    return shifts, summary


def zero_cost_controls(rows, forecasts, old_spec, reference):
    """主结果之后补充的成本解释，不参与选模；同仓位比例仍是事后信息。"""
    old = replay(rows, forecasts, {**old_spec, "name": "old_zero_cost"}, fee_rate=0)
    matched = replay(rows, forecasts,
                     {"name": "matched_zero_cost", "kind": "fixed", "ex_post": True,
                      "weight": reference["metrics"]["average_open_weight"]},
                     fee_rate=0, provenance="ex_post")
    return {"old_zero_cost": old, "matched_zero_cost": matched}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stage2 = json.loads((m.ROOT/"outputs/step2/summary.json").read_text(encoding="utf-8"))
    old = stage2["protocol"]
    rows_all, audit = m.read_prices(m.ROOT/"data/raw/SZ#399006.txt")
    assert audit["sha256"] == stage2["source_sha256"]
    assert old["feature_names"] == m.FEATURE_NAMES
    assert old["train_window"] == m.TRAIN_WINDOW and old["refit_every_trading_days"] == m.REFIT_EVERY
    for key, value in old["forest"].items():
        assert m.make_model("random_forest").get_params()[key] == value
    assert m.make_model("ridge").named_steps["ridge"].alpha == old["ridge_alpha"]
    assert old["fee_rate_each_side"] == m.FEE_RATE
    history = [r for r in rows_all if r["date"] < m.START]
    plan = {"source_sha256": audit["sha256"], "stage2_protocol_id": old["protocol_id"],
            "history_forecast_start": "2017-01-01", "outer_years": list(range(2019, 2026)),
            "outer_2025_ends": "2025-10-09", "inner_window_calendar_years": 2,
            "models": list(MODEL_NAMES), "profiles": list(PROFILES), "thresholds": list(THRESHOLDS),
            "selection_score": "worst of two half-period (annualized return - max drawdown)",
            "fee_rate_each_side": m.FEE_RATE, "train_window": m.TRAIN_WINDOW, "refit_every": m.REFIT_EVERY,
            "official_replay_start": str(m.START), "official_replay_end": str(m.END),
            "matched_exposure_and_circular_shifts": "ex_post_diagnostics_only; never select using them",
            "test_was_previously_viewed": True,
            "not_a_historical_preregistration": True}
    plan["plan_id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    (OUT/"research_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'本轮研究计划已记录 {plan["plan_id"]}；候选集合不再按正式区间成绩修改。', flush=True)

    historical, fit_logs, flat_forecasts = {}, [], []
    for model in MODEL_NAMES:
        fs, fits, _ = m.rolling_forecast(history, date(2017, 1, 1), history[-1]["date"], model)
        historical[model] = fs
        flat_forecasts += fs
        fit_logs += fits
        print(f'历史滚动预测完成 {model}: {len(fs)} 日, {len(fits)} 次更新', flush=True)
    m.write_csv(OUT/"historical_predictions.csv", flat_forecasts)
    m.write_csv(OUT/"historical_fit_audit.csv", fit_logs)

    manifest, runs, fold_metrics, choices, candidate_rows, model_rows = [], {}, [], [], [], []

    def save_run(name, result):
        result["metrics"]["name"] = name
        runs[name] = result
        m.write_csv(OUT/f"{name}_daily.csv", result["curve"])
        m.write_csv(OUT/f"{name}_trades.csv", result["trades"],
                    ["date", "time", "side", "units", "price", "fee", "signal_asof", "provenance"])
        manifest.append({"name": name, "curve_file": f"{name}_daily.csv", "trades_file": f"{name}_trades.csv",
                         **result["metrics"]})

    for year in range(2019, 2026):
        decision = date(year, 1, 1)
        prior = [r for r in history if r["date"] < decision]
        past_predictions = subset_forecasts(historical, before=decision)
        choice, model_metrics, candidates = select_before(prior, past_predictions, decision)
        label = str(year) if year < 2025 else "2025_to_1009"
        choice["fold"] = label
        choices.append(choice)
        candidate_rows += [{"fold": label, "decision_date": str(decision), **r} for r in candidates]
        model_rows += [{"fold": label, **r} for r in model_metrics]
        outer = subset_forecasts(historical, first=decision, last=date(year, 12, 31))[choice["model"]]
        specs = [{"name": "robust", "profile": choice["profile"], "threshold": choice["threshold"]},
                 {"name": "buy_hold", "kind": "fixed", "weight": 1.0},
                 {"name": "ma20", "kind": "ma20"}]
        for spec in specs:
            result = replay(history, outer, spec)
            fold_metrics.append({"fold": label, "model": choice["model"], "profile": choice["profile"],
                                 "threshold": choice["threshold"], **result["metrics"]})
            save_run(f'fold_{label}_{spec["name"]}', result)
        print(f'历史外层 {label}: {choice["model"]}, {choice["profile"]}, 门槛 {choice["threshold"]:.2%}', flush=True)

    choice, model_metrics, candidates = select_before(history, historical, m.START)
    candidate_rows += [{"fold": "final", "decision_date": str(m.START), **r} for r in candidates]
    model_rows += [{"fold": "final", **r} for r in model_metrics]
    final = {**choice, "plan_id": plan["plan_id"], "stage2_protocol_id": old["protocol_id"],
             "final_replay_is_exploratory": True, "selection_excludes_matched_and_shift_diagnostics": True}
    final["protocol_id"] = hashlib.sha256(json.dumps(final, sort_keys=True).encode()).hexdigest()[:16]
    (OUT/"protocol.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT/"fold_choices.json").write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")
    m.write_csv(OUT/"validation_candidate_scores.csv", candidate_rows)
    m.write_csv(OUT/"validation_model_metrics.csv", model_rows)
    m.write_csv(OUT/"historical_fold_metrics.csv", fold_metrics)
    print(f'正式回放规则已锁定 {final["protocol_id"]}: {choice["model"]}, {choice["profile"]}, {choice["threshold"]:.2%}', flush=True)

    # 至此才读取正式区间预测；沿用第二步的快照，不重新生成或回填。
    snapshot_path = m.ROOT/"outputs/step2/test_predictions.csv"
    predictions = load_forecasts(snapshot_path)
    rows = [r for r in rows_all if r["date"] <= m.END]
    for fs in predictions.values():
        assert len(fs) == 252 and all(f["target_date"] >= str(m.START) for f in fs)
    old_fs, new_fs = predictions[old["selected_model"]], predictions[choice["model"]]
    old_profile = "raw" if old["vol_target"] is None else "vol15"
    old_run = replay(rows, old_fs, {"name": "old_ml", "profile": old_profile, "threshold": old["threshold"]})
    old_metrics = next(r for r in stage2["test_strategy_metrics"] if r["strategy"] == "selected")
    assert abs(old_run["metrics"]["final_equity"]-old_metrics["final_equity"]) < 1e-5
    assert abs(old_run["metrics"]["max_drawdown"]-old_metrics["max_drawdown"]) < 1e-10
    new_run = replay(rows, new_fs, {"name": "robust_ml", "profile": choice["profile"], "threshold": choice["threshold"]})
    for name, result in [("old_ml", old_run), ("robust_ml", new_run),
                         ("buy_hold", replay(rows, old_fs, {"name": "buy_hold", "kind": "fixed", "weight": 1.0})),
                         ("cash", replay(rows, old_fs, {"name": "cash", "kind": "fixed", "weight": 0.0})),
                         ("ma20", replay(rows, old_fs, {"name": "ma20", "kind": "ma20"})),
                         ("matched_old", matched_exposure(rows, old_fs, old_run, "matched_old")),
                         ("matched_robust", matched_exposure(rows, new_fs, new_run, "matched_robust"))]:
        save_run(name, result)
    controls = zero_cost_controls(rows, old_fs,
                                  {"profile": old_profile, "threshold": old["threshold"]}, old_run)
    for name, result in controls.items():
        save_run(name, result)
    shifts, shift_summary = shift_diagnostic(rows, old_fs, old_run)
    m.write_csv(OUT/"circular_shift_diagnostics.csv", shifts)
    m.write_csv(OUT/"final_prediction_inputs.csv", predictions["ridge"]+predictions["random_forest"])
    m.write_csv(OUT/"final_fit_audit.csv", [r for r in read_csv(m.ROOT/"outputs/step2/fit_audit.csv") if r["phase"] == "test"])
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    final_metrics = [runs[name]["metrics"] for name in ("buy_hold", "cash", "ma20", "old_ml", "matched_old", "robust_ml", "matched_robust")]
    summary = {"plan": plan, "protocol": final, "fold_choices": choices,
               "historical_fold_metrics": fold_metrics, "final_metrics": final_metrics,
               "shift_diagnostic": shift_summary,
               "zero_cost_controls": [r["metrics"] for r in controls.values()],
               "zero_cost_check_added_after_primary_results": True,
               "prediction_snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
               "same_exposure_note": "matched weights use full replay exposure; diagnostic only, not an ex-ante benchmark"}
    (OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    from verify_step3_outputs import verify
    verify()
    from step3_reporting import make_plots, write_results
    make_plots(summary, runs, shifts, OUT)
    write_results(summary, OUT)
    print(json.dumps({"final_choice": final, "final_metrics": final_metrics,
                      "shift_diagnostic": shift_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
