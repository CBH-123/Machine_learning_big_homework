"""第一步：检查创业板日线，绘图，计算只做多且遵守 T+1 的买入持有基准。

运行：python step1_baseline.py
这一步不训练模型、不选择参数。指数点位仅用于教学模拟，允许小数份额。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 本次使用目录内依赖，不改动电脑的全局 Python 环境。
if (ROOT / ".deps").exists():
    sys.path.insert(0, str(ROOT / ".deps"))

START = date(2025, 10, 10)
END = date(2026, 10, 23)
FIELDS = ("date", "open", "high", "low", "close", "volume", "amount")


def read_prices(path: Path) -> tuple[list[dict], dict]:
    """跳过通达信说明行；逐条校验，不静默填补或删除异常行情。"""
    raw = path.read_bytes()
    rows = []
    ignored = []
    for number, line in enumerate(raw.decode("gb18030").splitlines(), 1):
        if not re.match(r"^\d{4}/\d{2}/\d{2}", line.strip()):
            if line.strip():
                ignored.append(number)
            continue
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f"第 {number} 行应有 7 列")
        row = dict(zip(FIELDS, [datetime.strptime(parts[0], "%Y/%m/%d").date(),
                                *map(float, parts[1:])]))
        values = [row[k] for k in FIELDS[1:]]
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"第 {number} 行存在缺失值或无穷值")
        if min(values[:4]) <= 0 or min(values[4:]) < 0:
            raise ValueError(f"第 {number} 行价格或成交数据无效")
        if not (row["low"] <= min(row["open"], row["close"])
                <= max(row["open"], row["close"]) <= row["high"]):
            raise ValueError(f"第 {number} 行开高低收不一致")
        rows.append(row)
    if not rows:
        raise ValueError("没有读到行情")
    if len({r["date"] for r in rows}) != len(rows):
        raise ValueError("存在重复日期")
    if any(a["date"] >= b["date"] for a, b in zip(rows, rows[1:])):
        raise ValueError("日期未严格递增")
    audit = {
        "source": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": "gb18030", "raw_rows": len(rows),
        "first_date": rows[0]["date"].isoformat(),
        "last_date": rows[-1]["date"].isoformat(),
        "ignored_non_price_line_numbers": ignored,
        "duplicate_dates": 0, "missing_or_nonfinite_values": 0,
        "invalid_ohlc_rows": 0,
    }
    return rows, audit


def split_by_time(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    history = [r for r in rows if r["date"] < START]
    evaluation = [r for r in rows if START <= r["date"] <= END]
    excluded = [r for r in rows if r["date"] > END]
    return history, evaluation, excluded


@dataclass
class Lot:
    bought_on: date
    units: float


class LongOnlyT1Account:
    """小型教学账户：持仓非负、不借钱；只有更早交易日买入的份额可卖。

    按批次记录买入日，因此已有老持仓时，当天新增的份额不会冻结老持仓。
    这里只检查日线层面的约束，不模拟涨跌停、交易手数或真实 ETF 成交。
    """

    def __init__(self, capital: float, trading_dates: list[date], fee_rate: float = 0):
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError("初始资金必须为正数")
        if not math.isfinite(fee_rate) or not 0 <= fee_rate < 1:
            raise ValueError("费率应在 [0, 1) 内")
        if not trading_dates or trading_dates != sorted(set(trading_dates)):
            raise ValueError("交易日历必须严格递增且非空")
        self.cash = capital
        self.fee_rate = fee_rate
        self.calendar = set(trading_dates)
        self.lots: list[Lot] = []
        self.last_action_date: date | None = None

    @property
    def units(self) -> float:
        return sum(lot.units for lot in self.lots)

    def sellable(self, on: date) -> float:
        return sum(lot.units for lot in self.lots if lot.bought_on < on)

    def _check_order(self, on: date, price: float, units: float):
        if on not in self.calendar:
            raise ValueError("该日期不在数据交易日历中")
        if self.last_action_date and on < self.last_action_date:
            raise ValueError("不能倒退到过去成交")
        if not all(math.isfinite(x) and x > 0 for x in (price, units)):
            raise ValueError("成交价和份额必须为正数")

    def buy(self, on: date, price: float, units: float):
        self._check_order(on, price, units)
        cost = price * units * (1 + self.fee_rate)
        if cost > self.cash + 1e-8:
            raise ValueError("现金不足，禁止借钱买入")
        self.cash = max(0.0, self.cash - cost)
        self.lots.append(Lot(on, units))
        self.last_action_date = on

    def sell(self, on: date, price: float, units: float):
        self._check_order(on, price, units)
        if units > self.units + 1e-8:
            raise ValueError("卖出超过总持仓，禁止做空")
        if units > self.sellable(on) + 1e-8:
            raise ValueError("违反 T+1：当天买入的份额不能当天卖出")
        remaining = units
        for lot in self.lots:
            if lot.bought_on < on:
                taken = min(lot.units, remaining)
                lot.units -= taken
                remaining -= taken
        self.lots = [lot for lot in self.lots if lot.units > 1e-10]
        self.cash += price * units * (1 - self.fee_rate)
        self.last_action_date = on


def buy_and_hold(rows: list[dict], capital: float = 100_000,
                 fee_rate: float = 0) -> tuple[list[dict], dict, list[dict]]:
    if len(rows) < 2 or rows[0]["date"] >= rows[-1]["date"]:
        raise ValueError("T+1 买入持有基准至少需要两个不同交易日")
    account = LongOnlyT1Account(capital, [r["date"] for r in rows], fee_rate)
    entry, exit_ = rows[0], rows[-1]
    units = capital / (entry["open"] * (1 + fee_rate))
    account.buy(entry["date"], entry["open"], units)
    trades = [{"date": entry["date"], "time": "09:30", "side": "BUY",
               "price": entry["open"], "units": units,
               "fee": units * entry["open"] * fee_rate}]
    curve = [{"timestamp": f'{entry["date"]} 09:29:00', "equity": capital,
              "cash": capital, "units": 0, "sellable_units": 0,
              "daily_return": 0, "cumulative_return": 0,
              "peak_equity": capital, "drawdown": 0, "event": "INITIAL"}]
    peak = capital
    peak_time = curve[0]["timestamp"]
    max_drawdown = 0
    mdd_peak_time = mdd_trough_time = peak_time
    previous = capital
    for i, row in enumerate(rows):
        event = "BUY_AT_OPEN" if i == 0 else "HOLD"
        if i == len(rows) - 1:
            # 退出日期和收盘成交规则事先固定，不根据期末价格挑卖点。
            sell_units = account.units
            account.sell(row["date"], row["close"], sell_units)
            trades.append({"date": row["date"], "time": "15:00", "side": "SELL",
                           "price": row["close"], "units": sell_units,
                           "fee": sell_units * row["close"] * fee_rate})
            event = "SELL_AT_CLOSE"
        equity = account.cash + account.units * row["close"]
        timestamp = f'{row["date"]} 15:00:00'
        if equity > peak:
            peak, peak_time = equity, timestamp
        drawdown = equity / peak - 1
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            mdd_peak_time, mdd_trough_time = peak_time, timestamp
        curve.append({"timestamp": timestamp, "equity": equity,
                      "cash": account.cash, "units": account.units,
                      "sellable_units": account.sellable(row["date"]),
                      "daily_return": equity / previous - 1,
                      "cumulative_return": equity / capital - 1,
                      "peak_equity": peak, "drawdown": drawdown, "event": event})
        previous = equity
    summary = {
        "initial_capital": capital, "start_date": str(entry["date"]),
        "end_date": str(exit_["date"]), "entry_price_open": entry["open"],
        "exit_price_close": exit_["close"], "units_bought": units,
        "final_equity": curve[-1]["equity"],
        "profit": curve[-1]["equity"] - capital,
        "total_return": curve[-1]["cumulative_return"],
        "max_drawdown": -max_drawdown,
        "max_drawdown_peak_at": mdd_peak_time,
        "max_drawdown_trough_at": mdd_trough_time,
        "trading_days": len(rows), "fee_rate_each_side": fee_rate,
        "fees_total": sum(t["fee"] for t in trades),
        "long_only": True, "t_plus_1_checked": True,
        "valuation": "daily close plus initial capital; excludes intraday drawdown",
        "execution": "precommitted first-day open buy and last-day close sell",
        "instrument_assumption": "hypothetical index units, fractional units allowed",
        "model_trained": False,
    }
    return curve, summary, trades


def write_csv(path: Path, rows: list[dict], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows, history, evaluation, curve, summary, out):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "svg.fonttype": "none"})

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    ax.plot([r["date"] for r in history], [r["close"] for r in history],
            color="#3467a4", lw=1.0, label="测试开始前的历史数据（尚未训练模型）")
    ax.plot([r["date"] for r in evaluation], [r["close"] for r in evaluation],
            color="#bb641a", lw=1.2, label="规定投资区间：用于最终评价")
    ax.axvspan(START, END, color="#bb641a", alpha=0.09)
    ax.axvline(START, color="#7a532e", lw=0.8)
    ax.set(title="创业板指数收盘价与时间边界", xlabel="日期", ylabel="收盘价（点）")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, "01_price_history")

    fig, ax = plt.subplots(figsize=(11, 4.8), layout="constrained")
    ax.plot([r["date"] for r in evaluation], [r["close"] for r in evaluation],
            color="#3467a4", lw=1.4, label="每日收盘价")
    ax.scatter([START], [summary["entry_price_open"]], marker="^", s=55,
               color="#397342", label=f'期初开盘买入 {summary["entry_price_open"]:,.2f} 点', zorder=4)
    ax.scatter([END], [summary["exit_price_close"]], marker="v", s=55,
               color="#bb641a", label=f'期末收盘卖出 {summary["exit_price_close"]:,.2f} 点', zorder=4)
    ax.set(title="规定投资区间的价格与基准交易", xlabel="日期", ylabel="指数点位（点）")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, fontsize=9)
    save(fig, "02_test_prices")

    times = [datetime.fromisoformat(r["timestamp"]) for r in curve]
    fig, (ax, dd) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    ax.plot(times, [r["equity"] / 10000 for r in curve], color="#3467a4", lw=1.5)
    ax.axhline(summary["initial_capital"] / 10000, color="#777777", lw=0.9, ls="--", label="起始本金")
    ax.set(ylabel="账户总资产（万元）", title="买入并持有基准：资金曲线与回撤")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.text(0.02, 0.97,
            f'期末 {summary["final_equity"]:,.2f} 元   区间收益 {summary["total_return"]:.2%}\n'
            f'最大回撤 {summary["max_drawdown"]:.2%}   单边费率 {summary["fee_rate_each_side"]:.2%}',
            transform=ax.transAxes, va="top", fontsize=10)
    dd.fill_between(times, [r["drawdown"] * 100 for r in curve], 0, color="#bb641a", alpha=0.24)
    dd.plot(times, [r["drawdown"] * 100 for r in curve], color="#bb641a", lw=1)
    dd.set(xlabel="日期", ylabel="相对历史高点（%）")
    dd.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    save(fig, "03_buy_hold_equity_drawdown")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/SZ#399006.txt")
    parser.add_argument("--fee-rate", type=float, default=0,
                        help="单边综合成本率；默认 0 为未扣费教学基准，不代表实际市场费率")
    args = parser.parse_args()
    rows, audit = read_prices(args.input)
    history, evaluation, excluded = split_by_time(rows)
    if not evaluation or evaluation[0]["date"] != START or evaluation[-1]["date"] != END:
        raise ValueError("数据未完整覆盖指定起止交易日")
    audit.update({"history_rows": len(history), "evaluation_rows": len(evaluation),
                  "excluded_after_end": [str(r["date"]) for r in excluded]})
    curve, summary, trades = buy_and_hold(evaluation, fee_rate=args.fee_rate)
    processed = ROOT / "data/processed"
    out = ROOT / "outputs/step1"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(processed / "prices_until_end.csv", history + evaluation, FIELDS)
    write_csv(processed / "history_before_test.csv", history, FIELDS)
    write_csv(processed / "test_period.csv", evaluation, FIELDS)
    write_csv(out / "buy_hold_daily.csv", curve)
    write_csv(out / "trades.csv", trades)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "chart_data.json").write_text(json.dumps(curve, ensure_ascii=False), encoding="utf-8")
    make_plots(history + evaluation, history, evaluation, curve, summary, out)
    print(json.dumps({"audit": audit, "baseline": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
