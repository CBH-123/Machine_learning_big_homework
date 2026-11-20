"""统一复现预测基线、策略消融和费用敏感性实验。"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import features as m
from account import simulate, verify_consensus_ledger
from baselines import RIDGE_ALPHA, prediction_metrics, rolling_ridge, zero_predictions

ROOT = Path(__file__).resolve().parent
FEE_SCENARIOS = (0.0, m.FEE_RATE, 0.001)
FLOAT_FIELDS = ("strategy_weight", "predicted_close", "predicted_return", "utility_score",
                "ma20", "breadth", "breadth5")
BOOL_FIELDS = ("tcn_holds", "utility_holds", "breadth_trigger")


def load_archived_signals(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        records = list(csv.DictReader(source))
    if not records:
        raise ValueError("归档信号为空")
    for rec in records:
        rec.setdefault("model", "tiny_tcn")
        for key in FLOAT_FIELDS:
            rec[key] = float(rec[key]) if rec[key] else None
        for key in BOOL_FIELDS:
            if rec[key] not in ("True", "False"):
                raise ValueError(f"无效的布尔信号：{key}={rec[key]}")
            rec[key] = rec[key] == "True"
    return records


def strategy_variants(rows, signals):
    """只替换目标仓位，所有方案共用同一账户、费用和成交时点。"""
    prices = {str(row["date"]): row for row in rows}

    def replace(rec, weight):
        return {**rec, "strategy_weight": float(weight)}

    variants = {name: [] for name in ("buy_hold", "ma20", "tcn_only", "utility_only",
                                      "dual_consensus", "dual_ma20", "full")}
    for rec in signals:
        tcn, utility = bool(rec["tcn_holds"]), bool(rec["utility_holds"])
        trend = prices[rec["asof_date"]]["close"] > rec["ma20"]
        variants["buy_hold"].append(replace(rec, 1.0))
        variants["ma20"].append(replace(rec, trend))
        variants["tcn_only"].append(replace(rec, tcn))
        variants["utility_only"].append(replace(rec, utility))
        variants["dual_consensus"].append(replace(rec, tcn and utility))
        variants["dual_ma20"].append(replace(rec, float(tcn) if tcn == utility else .5 if trend else 0.0))
        variants["full"].append(replace(rec, rec["strategy_weight"]))
    return variants


def run_strategy_variants(rows, signals, fee_rate=m.FEE_RATE):
    output = []
    for name, decisions in strategy_variants(rows, signals).items():
        result = simulate(rows, decisions, fee_rate=fee_rate)
        verify_consensus_ledger(rows, decisions, result)
        metrics = result["metrics"]
        metrics["name"] = name
        output.append(metrics)
    benchmark = next(rec["total_return"] for rec in output if rec["name"] == "buy_hold")
    for rec in output:
        rec["excess_return_vs_buy_hold"] = rec["total_return"] - benchmark
    return output


def run_fee_sensitivity(rows, signals):
    output = []
    for fee_rate in FEE_SCENARIOS:
        result = simulate(rows, signals, fee_rate=fee_rate)
        verify_consensus_ledger(rows, signals, result)
        metrics = result["metrics"]
        output.append({key: metrics[key] for key in (
            "fee_rate_each_side", "final_equity", "total_return", "max_drawdown",
            "sharpe_rf_zero", "order_count", "fees_paid")})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=m.START,
                        help="首个目标交易日，YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, default=m.END,
                        help="最后目标交易日，YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation_output")
    parser.add_argument("--recompute", action="store_true",
                        help="重新训练 TCN 和效用模型，不读取正式期归档信号")
    args = parser.parse_args()
    if args.end < args.start:
        raise ValueError("结束日期不能早于开始日期")

    rows, _ = m.read_prices(ROOT / "data" / "SZ#399006.txt")
    rows = [row for row in rows if row["date"] <= args.end]
    archived = not args.recompute and (args.start, args.end) == (m.START, m.END)
    if archived:
        signals = load_archived_signals(ROOT / "results" / "signals.csv")
        source = "results/signals.csv"
    else:
        import torch
        from run import decide, load_breadth
        from tcn import rolling
        from utility import utility_decisions

        torch.set_num_threads(2)
        breadth = load_breadth(ROOT / "data" / "breadth.csv")
        forecasts, _, _ = rolling(rows, args.start, args.end, torch.device("cpu"))
        scores = utility_decisions(rows, args.start, args.end)
        signals = decide(rows, forecasts, scores, breadth)
        source = "recomputed"

    ridge = rolling_ridge(rows, args.start, args.end)
    signal_dates = [(rec["asof_date"], rec["target_date"]) for rec in signals]
    if signal_dates != [(rec["asof_date"], rec["target_date"]) for rec in ridge]:
        raise ValueError("基线与策略信号的日期不一致")
    predictions = [prediction_metrics(rows, signals, model="tiny_tcn"),
                   prediction_metrics(rows, zero_predictions(rows, signals)),
                   prediction_metrics(rows, ridge)]
    strategies = run_strategy_variants(rows, signals)
    fees = run_fee_sensitivity(rows, signals)

    args.out.mkdir(parents=True, exist_ok=True)
    m.write_csv(args.out / "prediction_metrics.csv", predictions)
    m.write_csv(args.out / "ridge_predictions.csv", ridge)
    m.write_csv(args.out / "strategy_metrics.csv", strategies)
    m.write_csv(args.out / "cost_sensitivity.csv", fees)
    summary = {"start": str(args.start), "end": str(args.end), "signal_source": source,
               "ridge_alpha": RIDGE_ALPHA, "fee_scenarios": FEE_SCENARIOS}
    (args.out / "experiment.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"prediction_metrics": predictions, "strategy_metrics": strategies,
                      "cost_sensitivity": fees}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
