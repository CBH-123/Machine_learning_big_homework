"""从原始日线训练双模型，按指数点位模拟题目一。"""
import argparse
import csv
import json
from collections import deque
from pathlib import Path
from statistics import fmean

import torch
import features
from account import simulate, verify_consensus_ledger
from tcn import rolling
from utility import utility_decisions

ROOT = Path(__file__).resolve().parent


def load_breadth(path):
    output = {}
    recent = deque(maxlen=5)
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            day = row["date"]
            count, up, flat, down = (int(row[key]) for key in ("valid_quotes", "up", "flat", "down"))
            if (day in output or row["membership_effective"] > day or
                    not 0 <= count <= int(row["members"]) or min(up, down, flat) < 0 or up + down + flat != count):
                raise ValueError(f"市场宽度记录无效：{day}")
            share = up / count if count >= 90 else None
            recent.append(share)
            average = sum(recent) / 5 if len(recent) == 5 and None not in recent else None
            output[day] = (share, average)
    if list(output) != sorted(output):
        raise ValueError("市场宽度日期必须升序")
    return output


def decide(rows, forecasts, scores, breadth):
    indices = {str(row["date"]): index for index, row in enumerate(rows)}
    decisions = []
    holding = False
    for pred, score in zip(forecasts, scores, strict=True):
        asof, target = pred["asof_date"], pred["target_date"]
        index = indices[asof]
        if ((asof, target) != (score["asof_date"], score["target_date"]) or indices[target] != index + 1
                or not all(rec["max_training_label_date"] <= rec["fit_asof"] <= asof < target for rec in (pred, score))):
            raise ValueError("预测与交易日或训练标签不匹配")
        if pred["predicted_return"] > 0.001:
            holding = True
        elif pred["predicted_return"] <= 0:
            holding = False
        utility = bool(score["strategy_weight"])
        ma20 = fmean(row["close"] for row in rows[index-19:index+1])
        trend = rows[index]["close"] > ma20
        day, average = breadth.get(asof, (None, None))
        trigger = holding != utility and not trend and day is not None and average is not None and day >= .60 and average >= .50
        weight = float(holding) if holding == utility else .5 if trend or trigger else 0.
        decisions.append({"asof_date": asof, "target_date": target, "strategy_weight": weight,
                          "predicted_close": pred["predicted_close"], "predicted_return": pred["predicted_return"],
                          "tcn_holds": holding, "utility_score": score["utility_score"], "utility_holds": utility,
                          "ma20": ma20, "breadth": day, "breadth5": average, "breadth_trigger": trigger,
                          "fit_asof": max(pred["fit_asof"], score["fit_asof"]),
                          "max_training_label_date": max(pred["max_training_label_date"], score["max_training_label_date"])})
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run_output")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows, _ = features.read_prices(ROOT / "data" / "SZ#399006.txt")
    rows = [row for row in rows if row["date"] <= features.END]
    breadth = load_breadth(ROOT / "data/breadth.csv")
    forecasts, _, _ = rolling(rows, features.START, features.END, torch.device("cpu"))
    scores = utility_decisions(rows, features.START, features.END)
    signals = decide(rows, forecasts, scores, breadth)
    result = simulate(rows, signals)
    verify_consensus_ledger(rows, signals, result)
    metrics = result["metrics"]
    price = {str(row["date"]): row for row in rows}
    metrics["direction_accuracy"] = fmean((row["predicted_return"] > 0) ==
        (price[row["target_date"]]["close"] > price[row["asof_date"]]["close"]) for row in signals)
    metrics["price_mae_points"] = fmean(abs(row["predicted_close"] - price[row["target_date"]]["close"]) for row in signals)
    buy = rows[next(index for index, row in enumerate(rows) if row["date"] == features.START)]["open"]
    metrics["buy_hold_net_return"] = rows[-1]["close"] / buy * (1-features.FEE_RATE) / (1+features.FEE_RATE) - 1
    args.out.mkdir(exist_ok=True, parents=True)
    for filename, records in (("signals", signals), ("daily", result["curve"]), ("trades", result["trades"])):
        features.write_csv(args.out / (filename + ".csv"), records)
    content = json.dumps(metrics, indent=2, allow_nan=False)
    (args.out / "metrics.json").write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
