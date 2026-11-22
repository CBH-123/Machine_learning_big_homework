"""第二步：因果量价特征、岭回归/随机森林、滚动预测与 T+1 回测。

运行 python step2_models.py。所有模型/策略选择只使用 2025-10-10 之前的验证期。
输出 protocol.json 后，才进行规定投资区间的滚动模拟；不会按测试成绩重选方案。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from step1_baseline import ROOT, START, END, LongOnlyT1Account, read_prices, write_csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VALID_START = date(2023, 1, 1)
TRAIN_WINDOW = 1260
REFIT_EVERY = 20
FEE_RATE = 0.0005  # 单边综合成本 0.05%，教学假设，不是实际市场费率。
SEED = 2026
MODEL_NAMES = {"naive": "价格不变基准", "ridge": "岭回归", "random_forest": "随机森林"}
FEATURE_INFO = [
    *[(f"return_{w}", f"过去{w}个交易日涨跌幅", f"close[t]/close[t-{w}]-1") for w in (1, 2, 5, 10, 20)],
    *[(f"lag_return_{k}", f"向前{k}天的单日涨跌幅", f"close[t-{k}]/close[t-{k+1}]-1") for k in (1, 2, 3)],
    *[(f"ma_gap_{w}", f"收盘价相对{w}日均价的偏离", f"close[t]/mean(close[t-{w-1}:t])-1") for w in (5, 10, 20, 60)],
    *[(f"volatility_{w}", f"过去{w}日收益波动率", f"std(last {w} daily returns, ddof=1)") for w in (5, 10, 20)],
    ("candle_body", "当天开盘至收盘的涨跌", "(close-open)/open"),
    ("candle_range", "当天最高与最低的振幅", "(high-low)/close"),
    ("upper_shadow", "当天上影线相对长度", "(high-max(open,close))/close"),
    ("lower_shadow", "当天下影线相对长度", "(min(open,close)-low)/close"),
    *[(f"volume_ratio_{w}", f"成交量相对{w}日均值", f"volume[t]/mean(last {w} volumes)-1") for w in (5, 20)],
    ("log_volume_change", "成交量对数变化", "log((volume[t]+1)/(volume[t-1]+1))"),
    ("amount_ratio_20", "成交额相对20日均值", "amount[t]/mean(last 20 amounts)-1"),
    ("log_amount_change", "成交额对数变化", "log((amount[t]+1)/(amount[t-1]+1))"),
    ("price_position_20", "收盘价在近期高低区间的位置", "(close-min(last 20 lows))/(max(last 20 highs)-min(last 20 lows))"),
]
FEATURE_NAMES = [row[0] for row in FEATURE_INFO]


def build_features(rows: list[dict]) -> np.ndarray:
    """第 i 行仅访问 rows[:i+1]；保留前60行 NaN，绝不用未来数据回填。"""
    n = len(rows)
    x = np.full((n, len(FEATURE_NAMES)), np.nan)
    arrays = {k: np.array([r[k] for r in rows], dtype=float)
              for k in ("open", "high", "low", "close", "volume", "amount")}
    c, o, h, low, v, a = [arrays[k] for k in ("close", "open", "high", "low", "volume", "amount")]
    ret = np.r_[np.nan, c[1:] / c[:-1] - 1]
    for i in range(60, n):
        f = [c[i] / c[i-w] - 1 for w in (1, 2, 5, 10, 20)]
        f += [ret[i-k] for k in (1, 2, 3)]
        f += [c[i] / np.mean(c[i-w+1:i+1]) - 1 for w in (5, 10, 20, 60)]
        f += [np.std(ret[i-w+1:i+1], ddof=1) for w in (5, 10, 20)]
        f += [(c[i]-o[i])/o[i], (h[i]-low[i])/c[i],
              (h[i]-max(o[i], c[i]))/c[i], (min(o[i], c[i])-low[i])/c[i]]
        f += [v[i]/max(np.mean(v[i-w+1:i+1]), 1e-12)-1 for w in (5, 20)]
        f += [np.log((v[i]+1)/(v[i-1]+1)),
              a[i]/max(np.mean(a[i-19:i+1]), 1e-12)-1,
              np.log((a[i]+1)/(a[i-1]+1))]
        lo, hi = np.min(low[i-19:i+1]), np.max(h[i-19:i+1])
        f += [(c[i]-lo)/max(hi-lo, 1e-12)]
        x[i] = f
    return x


def make_model(name: str):
    if name == "ridge":
        # 标准化器随每次模型拟合，只在当次训练窗口上估计均值与方差。
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=100, max_depth=4,
                                     min_samples_leaf=20, max_features=0.7,
                                     max_samples=0.8, random_state=SEED, n_jobs=2)
    raise ValueError(f"未知模型：{name}")


def rolling_forecast(rows: list[dict], first_target: date, last_target: date,
                     model_name: str, train_window=TRAIN_WINDOW,
                     refit_every=REFIT_EVERY, min_train=252):
    """先用已成熟标签拟合，再预测；返回预测时不附带目标日的真实答案。

    i 是已收盘的特征日，预测目标是 i+1。可训练样本 j 必须满足 j+1<=i。
    即使全体历史存于磁盘，模型也只会收到这些可用样本。
    """
    if refit_every < 1 or train_window < min_train:
        raise ValueError("更新间隔和训练窗口无效")
    x = build_features(rows)
    closes = np.array([r["close"] for r in rows])
    y = np.r_[closes[1:]/closes[:-1]-1, np.nan]
    valid = np.isfinite(x).all(axis=1)
    indexes = [i for i in range(len(rows)-1)
               if valid[i] and first_target <= rows[i+1]["date"] <= last_target]
    model = None
    forecasts, fits, importance = [], [], []
    fit_info = {}
    for number, i in enumerate(indexes):
        asof = rows[i]["date"]
        if model_name != "naive" and (model is None or number % refit_every == 0):
            eligible = np.flatnonzero(valid & (np.arange(len(rows)) < i))[-train_window:]
            if len(eligible) < min_train:
                raise ValueError("可用训练样本太少")
            latest_label = rows[int(eligible[-1])+1]["date"]
            assert latest_label <= asof
            model = make_model(model_name)
            model.fit(x[eligible], y[eligible])
            fit_info = {"model": model_name, "fit_id": len(fits)+1,
                        "fit_asof": str(asof), "n_train": len(eligible),
                        "first_training_asof": str(rows[int(eligible[0])]["date"]),
                        "last_training_asof": str(rows[int(eligible[-1])]["date"]),
                        "max_training_label_date": str(latest_label)}
            fits.append(dict(fit_info))
            if not importance:
                scores = (model.feature_importances_ if model_name == "random_forest"
                          else np.abs(model.named_steps["ridge"].coef_))
                importance = [{"feature": name, "score": float(score),
                               "model": model_name, "fit_asof": str(asof),
                               "method": "impurity importance" if model_name == "random_forest" else "absolute standardized coefficient"}
                              for name, score in zip(FEATURE_NAMES, scores)]
                importance.sort(key=lambda r: r["score"], reverse=True)
        pred = 0.0 if model_name == "naive" else float(model.predict(x[i:i+1])[0])
        record = {"model": model_name, "asof_date": str(asof),
                  "prediction_time": f"{asof} 15:00:00",
                  "target_date": str(rows[i+1]["date"]),
                  "close_asof": float(closes[i]), "predicted_return": pred,
                  "predicted_close": float(closes[i]*(1+pred)),
                  "annual_volatility_20": float(x[i, FEATURE_NAMES.index("volatility_20")]*np.sqrt(252)),
                  "ma_gap_20": float(x[i, FEATURE_NAMES.index("ma_gap_20")]),
                  "fit_id": fit_info.get("fit_id", ""),
                  "fit_asof": fit_info.get("fit_asof", ""),
                  "max_training_label_date": fit_info.get("max_training_label_date", "")}
        if model_name != "naive":
            assert record["max_training_label_date"] <= record["asof_date"] < record["target_date"]
        forecasts.append(record)
    return forecasts, fits, importance


def evaluate_forecasts(rows, forecasts):
    """预测全部产生后，才把目标日真实行情附加到评价表中。"""
    by_date = {str(r["date"]): r for r in rows}
    scored = []
    for f in forecasts:
        actual_close = by_date[f["target_date"]]["close"]
        actual_return = actual_close / f["close_asof"] - 1
        scored.append({**f, "actual_close": actual_close,
                       "actual_return": actual_return,
                       "price_error": f["predicted_close"]-actual_close})
    if not scored:
        raise ValueError("没有可评价的预测")
    err = np.array([r["predicted_return"]-r["actual_return"] for r in scored])
    price_err = np.array([r["price_error"] for r in scored])
    directional = (None if scored[0]["model"] == "naive" else
                   float(np.mean([np.sign(r["predicted_return"]) == np.sign(r["actual_return"]) for r in scored])))
    metrics = {"model": scored[0]["model"], "n_predictions": len(scored),
               "return_mae": float(np.mean(np.abs(err))),
               "return_rmse": float(np.sqrt(np.mean(err**2))),
               "price_mae_points": float(np.mean(np.abs(price_err))),
               "price_rmse_points": float(np.sqrt(np.mean(price_err**2))),
               "direction_accuracy": directional}
    return scored, metrics


def weight_from_forecast(forecast, spec):
    kind = spec.get("kind", "ml")
    if kind == "buy_hold":
        weight = 1.0
    elif kind == "ma20":
        weight = float(forecast["ma_gap_20"] > 0)
    else:
        weight = float(forecast["predicted_return"] > spec["threshold"])
    vol_target = spec.get("vol_target")
    if vol_target is not None:
        weight *= min(1.0, vol_target/max(forecast["annual_volatility_20"], 1e-8))
    return weight


def backtest(rows, forecasts, spec, fee_rate=FEE_RATE, capital=100_000):
    """前一交易日收盘信号，次日开盘调仓；日终估值，预定结束日清算。

    最后一天禁止加仓，保证当日收盘清算不会违反 T+1。
    """
    if not forecasts:
        raise ValueError("预测为空")
    by_date = {str(r["date"]): r for r in rows}
    trading_rows = [by_date[f["target_date"]] for f in forecasts]
    account = LongOnlyT1Account(capital, [r["date"] for r in trading_rows], fee_rate)
    curve, trades = [], []
    first = trading_rows[0]["date"]
    curve.append({"timestamp": f"{first} 09:29:00", "equity": capital, "cash": capital,
                  "units": 0, "target_weight": 0, "actual_weight_close": 0,
                  "drawdown": 0, "daily_return": 0, "event": "INITIAL"})
    peak = previous = capital
    blocked = 0

    def record_trade(row, side, units, price, clock, asof):
        trades.append({"date": str(row["date"]), "time": clock, "side": side,
                       "units": units, "price": price, "fee": units*price*fee_rate,
                       "signal_asof": asof})

    for k, (row, f) in enumerate(zip(trading_rows, forecasts)):
        on, p = row["date"], row["open"]
        if not f["asof_date"] < str(on):
            raise ValueError("开盘交易信号必须来自更早的交易日")
        weight = weight_from_forecast(f, spec)
        if not 0 <= weight <= 1:
            raise ValueError("持仓权重必须在0到1之间")
        wealth_open = account.cash + account.units*p
        desired_units = wealth_open*weight/p
        difference = desired_units-account.units
        event = "HOLD"
        if difference > 1e-8 and k != len(trading_rows)-1:
            units = min(difference, account.cash/(p*(1+fee_rate)))
            if units > 1e-8:
                account.buy(on, p, units)
                record_trade(row, "BUY", units, p, "09:30", f["asof_date"])
                event = "BUY_AT_OPEN"
        elif difference < -1e-8:
            units = min(-difference, account.sellable(on))
            blocked += int(units < -difference-1e-8)
            if units > 1e-8:
                account.sell(on, p, units)
                record_trade(row, "SELL", units, p, "09:30", f["asof_date"])
                event = "SELL_AT_OPEN"
        if k == len(trading_rows)-1 and account.units > 1e-8:
            units = account.units
            account.sell(on, row["close"], units)
            record_trade(row, "SELL", units, row["close"], "15:00", "PRECOMMITTED_END")
            event = "FINAL_LIQUIDATION"
        equity = account.cash + account.units*row["close"]
        peak = max(peak, equity)
        curve.append({"timestamp": f"{on} 15:00:00", "equity": equity,
                      "cash": account.cash, "units": account.units,
                      "target_weight": weight,
                      "actual_weight_close": account.units*row["close"]/equity,
                      "drawdown": equity/peak-1, "daily_return": equity/previous-1,
                      "event": event})
        assert account.cash >= -1e-7 and account.units >= -1e-7
        previous = equity
    returns = np.array([r["daily_return"] for r in curve[1:]])
    n = len(trading_rows)
    annualized = (curve[-1]["equity"]/capital)**(252/n)-1
    std = np.std(returns, ddof=1) if n > 1 else 0
    summary = {"strategy": spec["name"], "final_equity": curve[-1]["equity"],
               "total_return": curve[-1]["equity"]/capital-1,
               "annualized_return": float(annualized),
               "max_drawdown": float(-min(r["drawdown"] for r in curve)),
               "sharpe_rf_zero": float(np.mean(returns)/std*np.sqrt(252)) if std > 1e-12 else 0.0,
               "order_count": len(trades), "fees_paid": sum(t["fee"] for t in trades),
               "fee_rate_each_side": fee_rate, "trading_days": n,
               "average_close_weight": float(np.mean([r["actual_weight_close"] for r in curve[1:]])),
               "t1_blocked_sells": blocked, "final_units": account.units,
               "selection_score": float(annualized-0.5*(-min(r["drawdown"] for r in curve)))}
    return curve, trades, summary


def select_protocol(validation_metrics, predictions, history, fee_rate):
    """只接受测试前历史；先按验证收益率RMSE选ML模型，再选择交易门槛/风控。"""
    if any(r["date"] >= START for r in history):
        raise ValueError("方案选择不允许收到测试期行情")
    eligible_metrics = [r for r in validation_metrics if r["model"] != "naive"]
    selected = min(eligible_metrics, key=lambda r: (r["return_rmse"], r["model"]))["model"]
    forecasts = predictions[selected]
    if any(date.fromisoformat(f["target_date"]) >= START for f in forecasts):
        raise ValueError("验证预测包含测试日期")
    candidates = []
    for threshold in (0.0, 0.001, 0.002):
        for vol_target in (None, 0.15):
            spec = {"name": f"threshold={threshold};vol_target={vol_target}",
                    "kind": "ml", "threshold": threshold, "vol_target": vol_target}
            _, _, metrics = backtest(history, forecasts, spec, fee_rate)
            candidates.append({**spec, **metrics})
    best = max(candidates, key=lambda r: r["selection_score"])
    protocol = {"initial_training_before": str(START), "validation_first_target": str(VALID_START),
                "validation_last_target": str(history[-1]["date"]),
                "selected_model": selected, "threshold": best["threshold"],
                "vol_target": best["vol_target"], "train_window": TRAIN_WINDOW,
                "refit_every_trading_days": REFIT_EVERY,
                "feature_names": FEATURE_NAMES, "fee_rate_each_side": fee_rate,
                "model_selection": "lowest validation return RMSE among ML models",
                "strategy_selection": "validation annualized return minus 0.5 * max drawdown",
                "candidate_thresholds": [0.0, 0.001, 0.002],
                "candidate_vol_targets": [None, 0.15],
                "ridge_alpha": 10.0,
                "forest": {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 20,
                           "max_features": 0.7, "max_samples": 0.8, "random_state": SEED},
                "target": "next trading day close / asof close - 1",
                "execution": "asof close signal; next trading day open rebalance",
                "last_day_rule": "no position increases; precommitted close liquidation",
                "online_training": "labels must be realized no later than each fit_asof"}
    protocol["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    return protocol, candidates


def save_samples(rows, out):
    x = build_features(rows)
    samples = []
    for i in range(60, len(rows)-1):
        r, target = rows[i], rows[i+1]
        group = ("test" if target["date"] >= START else
                 "validation" if target["date"] >= VALID_START else "pre_validation_history")
        samples.append({"asof_date": r["date"], "target_date": target["date"],
                        "label_available_at": f'{target["date"]} 15:00:00',
                        "group_by_target_date": group,
                        **{name: float(value) for name, value in zip(FEATURE_NAMES, x[i])},
                        "target_return": target["close"]/r["close"]-1})
    write_csv(out/"learning_samples.csv", samples)
    write_csv(out/"feature_dictionary.csv", [{"feature": a, "meaning": b, "formula": c}
                                             for a, b, c in FEATURE_INFO])
    return {"feature_count": len(FEATURE_NAMES), "total_labeled_samples": len(samples),
            "initial_available_samples": sum(r["target_date"] < START for r in samples),
            "validation_samples": sum(r["group_by_target_date"] == "validation" for r in samples),
            "test_samples": sum(r["group_by_target_date"] == "test" for r in samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fee-rate", type=float, default=FEE_RATE)
    args = parser.parse_args()
    if not 0 <= args.fee_rate < 1:
        parser.error("费率必须在 [0, 1) 内")
    all_rows, source_audit = read_prices(ROOT/"data/raw/SZ#399006.txt")
    history = [r for r in all_rows if r["date"] < START]
    out = ROOT/"outputs/step2"
    out.mkdir(parents=True, exist_ok=True)
    validation_forecasts, validation_scored, validation_metrics, fit_logs = {}, [], [], []
    for name in MODEL_NAMES:
        forecasts, fits, _ = rolling_forecast(history, VALID_START, history[-1]["date"], name)
        scored, metrics = evaluate_forecasts(history, forecasts)
        validation_forecasts[name] = forecasts
        validation_scored += scored
        validation_metrics.append(metrics)
        fit_logs += [{"phase": "validation", **f} for f in fits]
        print(f'验证完成 {name}: return RMSE={metrics["return_rmse"]:.6f}', flush=True)
    protocol, candidate_scores = select_protocol(validation_metrics, validation_forecasts, history, args.fee_rate)
    # 先落盘锁定方案；之后才构建/运行正式测试的模型与交易评价。
    (out/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out/"validation_model_metrics.csv", validation_metrics)
    write_csv(out/"validation_strategy_candidates.csv", candidate_scores)
    write_csv(out/"validation_predictions.csv", validation_scored)
    print(f'方案已冻结 {protocol["protocol_id"]}: {protocol["selected_model"]}, '
          f'threshold={protocol["threshold"]}, vol_target={protocol["vol_target"]}', flush=True)

    rows = [r for r in all_rows if r["date"] <= END]
    sample_summary = save_samples(rows, out)
    test_forecasts, test_scored, test_metrics, initial_importance = {}, [], [], []
    for name in MODEL_NAMES:
        forecasts, fits, importance = rolling_forecast(rows, START, END, name)
        assert len(forecasts) == 252
        if fits:
            assert fits[0]["max_training_label_date"] < str(START)
        scored, metrics = evaluate_forecasts(rows, forecasts)
        test_forecasts[name] = forecasts
        test_scored += scored
        test_metrics.append(metrics)
        fit_logs += [{"phase": "test", **f} for f in fits]
        if name == protocol["selected_model"]:
            initial_importance = importance
        print(f'测试预测完成 {name}: return RMSE={metrics["return_rmse"]:.6f}', flush=True)

    selected_spec = {"name": "selected", "kind": "ml", "threshold": protocol["threshold"],
                     "vol_target": protocol["vol_target"]}
    definitions = [
        ({"name": "buy_hold", "kind": "buy_hold"}, "naive"),
        ({"name": "ma20", "kind": "ma20"}, "naive"),
        ({"name": "ridge_basic", "kind": "ml", "threshold": 0.0}, "ridge"),
        ({"name": "random_forest_basic", "kind": "ml", "threshold": 0.0}, "random_forest"),
        (selected_spec, protocol["selected_model"]),
    ]
    strategies, strategy_metrics = {}, []
    for spec, name in definitions:
        curve, trades, metrics = backtest(rows, test_forecasts[name], spec, args.fee_rate)
        strategies[spec["name"]] = {"curve": curve, "trades": trades, "summary": metrics}
        strategy_metrics.append(metrics)
        write_csv(out/f'{spec["name"]}_daily.csv', curve)
        write_csv(out/f'{spec["name"]}_trades.csv', trades,
                  ["date", "time", "side", "units", "price", "fee", "signal_asof"])
    cost_sensitivity = []
    for cost in (0.0, FEE_RATE, 0.001):
        for spec, name in [(definitions[0][0], "naive"), (selected_spec, protocol["selected_model"])]:
            _, _, metrics = backtest(rows, test_forecasts[name], spec, cost)
            cost_sensitivity.append(metrics)
    baseline_return = strategies["buy_hold"]["summary"]["total_return"]
    for metrics in strategy_metrics:
        metrics["excess_return_vs_buy_hold"] = metrics["total_return"]-baseline_return
    summary = {"source_sha256": source_audit["sha256"], "sample_counts": sample_summary,
               "protocol": protocol, "validation_model_metrics": validation_metrics,
               "test_model_metrics": test_metrics, "test_strategy_metrics": strategy_metrics,
               "cost_sensitivity": cost_sensitivity,
               "test_was_viewed_in_step1": True,
               "evaluation_note": "chronological replay, not a never-seen blind test or future guarantee"}
    for filename, records in [("test_predictions.csv", test_scored), ("test_model_metrics.csv", test_metrics),
                               ("test_strategy_metrics.csv", strategy_metrics), ("fit_audit.csv", fit_logs),
                               ("initial_feature_importance.csv", initial_importance),
                               ("cost_sensitivity.csv", cost_sensitivity)]:
        write_csv(out/filename, records)
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    from verify_step2_outputs import verify
    verify()
    from step2_reporting import make_plots, write_results
    make_plots(rows, test_scored, initial_importance, strategies, summary, out)
    write_results(summary, out)
    print(json.dumps({"sample_counts": sample_summary, "selected": strategies["selected"]["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
