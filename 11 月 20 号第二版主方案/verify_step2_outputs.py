"""独立核对第二步导出结果：不调用模型或回测账户实现。"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"outputs/step2"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def verify(out=OUT):
    out = Path(out)
    summary = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    assert summary["protocol"] == protocol
    calendar = [r["date"] for r in read_csv(ROOT/"data/processed/test_period.csv")]
    prices = {r["date"]: float(r["close"]) for r in read_csv(ROOT/"data/processed/prices_until_end.csv")}
    forecasts = read_csv(out/"test_predictions.csv")
    by_model = defaultdict(list)
    for r in forecasts:
        assert r["asof_date"] < r["target_date"] <= "2026-10-23"
        if r["model"] != "naive":
            assert r["max_training_label_date"] <= r["fit_asof"] <= r["asof_date"]
        by_model[r["model"]].append(r)
    for records in by_model.values():
        assert [r["target_date"] for r in records] == calendar
    fits = read_csv(out/"fit_audit.csv")
    for fit in fits:
        assert fit["last_training_asof"] < fit["max_training_label_date"] <= fit["fit_asof"]
        if fit["phase"] == "validation" or fit["fit_id"] == "1":
            assert fit["max_training_label_date"] < "2025-10-10"

    reconstructed = []
    for name in [r["strategy"] for r in summary["test_strategy_metrics"]]:
        curve = read_csv(out/f"{name}_daily.csv")
        trades = read_csv(out/f"{name}_trades.csv")
        daily_trades = defaultdict(list)
        for trade in trades:
            daily_trades[trade["date"]].append(trade)
            if trade["signal_asof"] != "PRECOMMITTED_END":
                assert trade["signal_asof"] < trade["date"]
            else:
                assert trade["date"] == calendar[-1] and trade["time"] == "15:00"
        cash, peak, mdd, fees = 100000.0, 100000.0, 0.0, 0.0
        lots = []
        for row in curve[1:]:
            day = row["timestamp"][:10]
            for t in daily_trades[day]:
                q, p, fee = float(t["units"]), float(t["price"]), float(t["fee"])
                assert q > 0 and p > 0 and fee >= 0
                assert math.isclose(fee, q*p*protocol["fee_rate_each_side"], abs_tol=1e-7)
                fees += fee
                if t["side"] == "BUY":
                    cash -= q*p+fee
                    lots.append([day, q])
                else:
                    assert t["side"] == "SELL"
                    eligible = sum(qty for bought, qty in lots if bought < day)
                    assert q <= eligible+1e-7, "卖出没有持有或尚未满足T+1的份额"
                    remaining = q
                    for lot in lots:
                        if lot[0] < day:
                            take = min(remaining, lot[1])
                            lot[1] -= take
                            remaining -= take
                    cash += q*p-fee
                assert cash >= -1e-6
            units = sum(q for _, q in lots)
            equity = cash+units*prices[day]
            peak = max(peak, equity)
            mdd = max(mdd, 1-equity/peak)
            assert math.isclose(equity, float(row["equity"]), abs_tol=1e-5)
            assert math.isclose(units, float(row["units"]), abs_tol=1e-7)
        reported = next(r for r in summary["test_strategy_metrics"] if r["strategy"] == name)
        assert math.isclose(cash, reported["final_equity"], abs_tol=1e-5)
        assert math.isclose(fees, reported["fees_paid"], abs_tol=1e-5)
        assert math.isclose(mdd, reported["max_drawdown"], abs_tol=1e-9)
        assert sum(q for _, q in lots) < 1e-7
        reconstructed.append({"strategy": name, "reconstructed_final_cash": cash,
                              "reconstructed_max_drawdown": mdd, "trades_checked": len(trades)})
    audit = {"status": "passed", "protocol_id": protocol["protocol_id"],
             "models_checked": list(by_model), "predictions_checked": len(forecasts),
             "fit_records_checked": len(fits), "trading_days_per_model": len(calendar),
             "strategies_independently_reconstructed": reconstructed,
             "checks": ["training label availability", "complete target dates",
                        "no shorting or borrowing", "T+1 by purchase lot",
                        "pre-open signal timing", "costs on both sides",
                        "daily mark-to-market and max drawdown", "end liquidation"]}
    (out/"audit_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    verify(parser.parse_args().out)
