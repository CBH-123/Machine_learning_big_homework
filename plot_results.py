"""Rebuild the README figure from archived daily equity and the course index file."""
import csv
import json
from math import ceil, floor
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "results/daily.csv").open(encoding="utf-8-sig", newline="") as stream:
        curve = list(csv.DictReader(stream))
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    raw = (ROOT / "data/SZ#399006.txt").read_text(encoding="gb18030")
    prices = {fields[0].replace("/", "-"): (float(fields[1]), float(fields[4]))
              for line in raw.splitlines() if (fields := line.split()) and "/" in fields[0] and len(fields) == 7
              and fields[0][:4].isdecimal()}
    dates = [rec["timestamp"][:10] for rec in curve]
    fee = metrics["fee_rate_each_side"]
    capital = float(curve[0]["equity"])
    units = capital / ((1 + fee) * prices[dates[1]][0])
    benchmark = [1.] + [units * prices[day][1] / capital for day in dates[1:]]
    benchmark[-1] *= 1 - fee
    strategy = [float(row["equity"]) / capital for row in curve]
    drawdown = [100 * float(row["drawdown"]) for row in curve]
    low = floor(10 * min(benchmark + strategy))/10
    high = ceil(10 * max(benchmark + strategy))/10
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420" viewBox="0 0 900 420" role="img">',
             '<title>策略与指数净值及策略最大回撤</title>',
             '<rect width="900" height="420" fill="#fff"/>',
             '<g fill="#24394e" font-family="sans-serif" font-size="15">']

    def line(values, color, ymap):
        points = " ".join(f"{68 + index * 800/(len(values)-1):.1f},{ymap(val):.1f}"
                          for index, val in enumerate(values))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>')

    def label(x, y, value, color="#24394e"):
        parts.append(f'<text x="{x}" y="{y}" fill="{color}">{value}</text>')

    def tick(y, value):
        parts.append(f'<path d="M68 {y}H868" stroke="#dbe4ea"/>')
        label(12, y + 5, value)

    y_net = lambda value: 238 - 183*(value-low)/(high-low)
    for count in range(round(low*10), round(high*10)+1, 2):
        tick(y_net(count/10), f"{count/10:.1f}")
    label(68, 30, "净值（期初 = 1）")
    label(512, 30, "━━ 双模型策略", "#087a65")
    label(690, 30, "━━ 买入并持有", "#536db2")
    line(benchmark, "#536db2", y_net)
    line(strategy, "#087a65", y_net)
    label(68, 275, "策略回撤（按每日收盘）")
    depth = max(5, ceil(-min(drawdown)/5)*5)
    y_dd = lambda value: 296 - 74*value/depth
    for value in (0, -depth/2, -depth):
        tick(y_dd(value), f"{value:g}%")
    line(drawdown, "#b34847", y_dd)
    label(68, 399, dates[1])
    label(440, 399, dates[len(dates)//2])
    label(788, 399, dates[-1])
    parts.append("</g></svg>")
    (ROOT / "docs/performance.svg").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
